#!/usr/bin/env python3
"""Recall-cost characterization: decode cost vs spilled-set size (review M7, P5).

The paper currently supports "the recurring recall cost grows with the spilled
set" with two points (33.0 tok/s @ 511 cells, 28.5 @ 1535). Two points fix a
line through anything, and the shape is what decides whether the policy
survives the 10^4-10^5-token workloads the discussion invokes.

Three measurements:

  A  per-step curve.  One long policy-3 run at fixed C, prompt + a large decode
     budget, with UNIKV_LOG on. The per-call CSV now carries n_spilled, and
     call_ms is bracketed by synchronize() on both sides, so each decode step is
     one (n_spill, ms) sample. A single run yields thousands of points spanning
     n_spill = 0 to (prompt + budget - C).

  B  thermal control.  The same C, prompt and budget under policy 1, whose
     window is pinned at C with nothing spilled: work per step is constant, so
     any drift in ms/step across that run is time-correlated thermal droop. In
     run A n_spill grows monotonically with time, so droop would alias straight
     onto the slope -- this control measures the alias and lets it be
     subtracted. It doubles as the same-resident-size, GPU-only reference that
     isolates the fixed cost of entering the two-tier path.

  C  e2e cross-check.  Uninstrumented policy-3 runs at several decode budgets,
     randomized-block with cooldowns, giving wall-clock tok/s against final
     n_spill. Confirms the per-step shape is not an artifact of the per-step
     synchronize() pair.

Protocol everywhere: -fa off, -fit off, greedy (temp 0), seed 123,
--ignore-eos, -b/-ub 512, alpha 0. flash_attn is read back out of each run's
own log and recorded, not assumed.
"""

import csv
import datetime
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN_DIR        = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / "recall_cost"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTX           = 1024
PROMPT_TOKENS = 512
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
SPILL_CAP     = 16384          # cells; 16384 * 128 KiB = 2 GiB host tier
CEILING       = 58.0
FIXED_TOKEN   = " token"
SHUFFLE_SEED  = 20260803

# A/B: long run reaching n_spill = 512 + LONG_GEN - CTX
LONG_GEN   = int(os.environ.get("UNIKV_RC_LONG_GEN", "10752"))     # -> n_spill 10240
# C: e2e budgets -> final n_spill = 512 + gen - CTX
E2E_GENS   = [int(x) for x in os.environ.get(
                  "UNIKV_RC_E2E_GENS", "1536,3072,6144").split(",")]
E2E_TRIALS = int(os.environ.get("UNIKV_RC_TRIALS", "2"))
COOLDOWN_S = int(os.environ.get("UNIKV_RC_COOLDOWN", "120"))
PHASES     = os.environ.get("UNIKV_RC_PHASES", "AB,C")


def count_tokens(path: Path) -> int:
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def ensure_prompt() -> Path:
    path = PROMPTS_DIR / f"prompt_{PROMPT_TOKENS}tok.txt"
    path.write_text(FIXED_TOKEN * max(PROMPT_TOKENS - 1, 0))
    got = count_tokens(path)
    if got != PROMPT_TOKENS:
        raise RuntimeError(f"prompt token mismatch: want {PROMPT_TOKENS} got {got}")
    return path


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def run(prompt: Path, policy: int, gen: int, tag: str, steplog: bool) -> dict:
    e2e_csv  = LOGS_DIR / f"e2e_{tag}.csv"
    step_csv = LOGS_DIR / f"step_{tag}.csv"
    log_txt  = LOGS_DIR / f"gen_{tag}.txt"
    e2e_csv.unlink(missing_ok=True)
    step_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(gen), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({
        "UNIKV_POLICY": str(policy),
        "UNIKV_ALPHA": "0",
        "UNIKV_E2E_LOG": str(e2e_csv),
        "UNIKV_SPILL_CAP": str(SPILL_CAP),
    })
    if steplog:
        env["UNIKV_LOG"] = str(step_csv)

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    log_txt.write_text("=== STDOUT ===\n" + done.stdout +
                       "\n=== STDERR ===\n" + done.stderr[-20000:])

    dec_tokens, tok_s = None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec_tokens = int(rows[-1]["decode_tokens"])
            tok_s = float(rows[-1]["tok_per_sec"])

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", done.stderr)
    retained = 0
    rm = re.findall(r"retained=(\d+)", done.stderr)
    if rm:
        retained = int(rm[-1])

    return {
        "tag": tag, "policy": policy, "gen": gen, "rc": done.returncode,
        "decode_tokens": dec_tokens, "tok_per_sec": tok_s,
        "flash_attn": mfa.group(1) if mfa else "UNPARSED",
        "retained": retained, "step_csv": step_csv,
    }


def summarize_steps(path: Path, label: str) -> None:
    if not path.exists():
        print(f"  ({label}: no step log)")
        return
    rows = [r for r in csv.DictReader(path.open()) if int(r["step"]) > 1]
    if not rows:
        return
    pts = [(int(r["n_spilled"]), float(r["ttft_ms"])) for r in rows]
    spilling = [(n, ms) for n, ms in pts if n > 0]
    resident = [ms for n, ms in pts if n == 0]
    print(f"  {label}: {len(pts)} steps, n_spill 0..{max(n for n, _ in pts)}")
    if resident:
        print(f"    pre-spill steps      : n={len(resident):5d}  mean {statistics.mean(resident):7.2f} ms")
    if len(spilling) > 2:
        xs = [n for n, _ in spilling]
        ys = [ms for _, ms in spilling]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        print(f"    spilling steps       : n={len(spilling):5d}  "
              f"{slope * 1000:.3f} us/cell, intercept {my - slope * mx:.2f} ms")


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    prompt = ensure_prompt()
    flags = []

    print(f"recall-cost sweep: C={CTX}, prompt {PROMPT_TOKENS}, -fa off, seed {SEED}")
    print(f"  phases={PHASES}  long_gen={LONG_GEN} (-> n_spill "
          f"{PROMPT_TOKENS + LONG_GEN - CTX})  e2e gens={E2E_GENS} x{E2E_TRIALS}")
    print()

    if "AB" in PHASES:
        print("== A/B: per-step curve + thermal control (instrumented) ==")
        order = [("A_p3_long", 3), ("B_p1_ctrl", 1)]
        for i, (tag, policy) in enumerate(order):
            if COOLDOWN_S > 0 and i > 0:
                print(f"  cooldown {COOLDOWN_S}s ...", flush=True)
                time.sleep(COOLDOWN_S)
            t0 = time.time()
            print(f"  {tag} (policy {policy}, n={LONG_GEN}) start={iso(t0)} ...",
                  end="", flush=True)
            r = run(prompt, policy, LONG_GEN, tag, steplog=True)
            print(f" dec={r['decode_tokens']} {r['tok_per_sec']} tok/s "
                  f"fa={r['flash_attn']} ({time.time() - t0:.0f}s)")
            if r["flash_attn"] != "disabled":
                flags.append(f"{tag}: flash_attn={r['flash_attn']}")
            if r["decode_tokens"] != LONG_GEN:
                flags.append(f"{tag}: decoded {r['decode_tokens']} != {LONG_GEN}")
            dest = RESULTS_DIR / f"recall_cost_steps_{tag}.csv"
            if r["step_csv"].exists():
                shutil.copyfile(r["step_csv"], dest)
                summarize_steps(dest, tag)
                print(f"    wrote {dest}")
        print()

    if "C" in PHASES:
        print("== C: e2e cross-check (uninstrumented, randomized block) ==")
        out = RESULTS_DIR / "recall_cost_e2e.csv"
        plan = [(g, t) for g in E2E_GENS for t in range(1, E2E_TRIALS + 1)]
        random.Random(SHUFFLE_SEED).shuffle(plan)
        print("  order:", " ".join(f"g{g}t{t}" for g, t in plan))
        with out.open("w", newline="") as fh:
            csv.writer(fh).writerow(
                ["order_idx", "gen", "trial", "ctx", "n_spill_final", "rc",
                 "decode_tokens", "tok_per_sec", "flash_attn", "retained",
                 "t_start_iso", "duration_s"])
        per_gen: dict[int, list[float]] = {g: [] for g in E2E_GENS}
        for idx, (gen, trial) in enumerate(plan, 1):
            if COOLDOWN_S > 0:
                print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
                time.sleep(COOLDOWN_S)
            t0 = time.time()
            tag = f"C_g{gen}_t{trial}"
            print(f"  [{idx:2d}/{len(plan)}] gen={gen} trial={trial} ...", end="", flush=True)
            r = run(prompt, 3, gen, tag, steplog=False)
            dur = time.time() - t0
            nsp = PROMPT_TOKENS + gen - CTX
            print(f" {r['tok_per_sec']} tok/s  n_spill={r['retained']} ({dur:.0f}s)")
            if r["flash_attn"] != "disabled":
                flags.append(f"{tag}: flash_attn={r['flash_attn']}")
            if r["decode_tokens"] != gen:
                flags.append(f"{tag}: decoded {r['decode_tokens']} != {gen}")
            if r["tok_per_sec"] is not None and r["tok_per_sec"] >= CEILING:
                flags.append(f"{tag}: {r['tok_per_sec']} >= {CEILING} ceiling")
            elif r["tok_per_sec"] is not None:
                per_gen[gen].append(r["tok_per_sec"])
            with out.open("a", newline="") as fh:
                csv.writer(fh).writerow(
                    [idx, gen, trial, CTX, nsp, r["rc"], r["decode_tokens"],
                     r["tok_per_sec"], r["flash_attn"], r["retained"],
                     iso(t0), f"{dur:.1f}"])

        print(f"\n  {'gen':>6s} {'n_spill':>8s} {'n':>2s} {'mean':>7s} {'sd':>6s}")
        for g in E2E_GENS:
            v = per_gen[g]
            if v:
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                print(f"  {g:6d} {PROMPT_TOKENS + g - CTX:8d} {len(v):2d} "
                      f"{statistics.mean(v):7.2f} {sd:6.2f}")
        print(f"  wrote {out}")

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
        return 1
    print("\nno flags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
