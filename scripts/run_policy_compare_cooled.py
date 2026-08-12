#!/usr/bin/env python3
"""Paired policy comparison p0 / p1 / p3 at C in {1024, 2048}, cooled
randomized-block protocol (review items M1/M6, DA-3).

Two things this fixes relative to `run_policy3_reruns.py` (Run 1):

  1. Policy 1 (rolling-window shift) is included. Run 1 compared UniKV only
     against the error-out baseline (policy 0), which is the comparator the
     review calls cherry-picked: policy 1 also completes the full budget, so
     the token-count advantage over it is 1x and the real difference is what
     gets generated, not how much.
  2. The block is cooled and randomized (the protocol used for the cooled
     alpha sweep) instead of one sequential uncooled pass, on a device
     documented to swing ~21% from heat.

Protocol, identical across every arm: 512-token prompt + 2048-token decode
budget, -fa off, -fit off, greedy (temp 0), seed 123, --ignore-eos,
-b 512 -ub 512, UNIKV_ALPHA=0. The flash_attn line is parsed out of each run's
own log and recorded per row as evidence rather than trusted from this source.

Throughput runs are UNINSTRUMENTED (no UNIKV_LOG) so the per-decode-call
synchronize() pair does not inflate the wall clock; that makes these numbers
directly comparable to the cooled alpha sweep's alpha=0 anchor. Run with
UNIKV_PC_STEPLOG=1 for a separate instrumented pass that records per-arm
shift/spill counts (counts are deterministic under greedy + fixed seed, so one
pass is enough).

env knobs:
  UNIKV_PC_TRIALS    trials per arm            (default 3)
  UNIKV_PC_COOLDOWN  seconds before every run  (default 150)
  UNIKV_PC_ARMS      comma-separated arm tags  (default all six)
  UNIKV_PC_TAG       output filename suffix
  UNIKV_PC_STEPLOG   1 = set UNIKV_LOG and count shift/spill events
"""

import csv
import datetime
import os
import random
import re
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
ART_DIR     = ROOT / "artifacts" / "r3_policy_compare"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

PROMPT_TOKENS = 512
GEN_TOKENS    = 2048
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
SPILL_CAP     = 8192
CEILING       = 58.0
FIXED_TOKEN   = " token"
SHUFFLE_SEED  = 20260803

TRIALS     = int(os.environ.get("UNIKV_PC_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_PC_COOLDOWN", "150"))
TAG        = os.environ.get("UNIKV_PC_TAG", "")
STEPLOG    = os.environ.get("UNIKV_PC_STEPLOG", "") == "1"
_suffix    = f"_{TAG}" if TAG else ""

MASTER_CSV = RESULTS_DIR / f"r3_policy_compare{_suffix}_master.csv"

# tag -> (ctx, policy, role)
ALL_ARMS = {
    "p0_c1024": (1024, 0, "stock baseline, errors at cache-full"),
    "p1_c1024": (1024, 1, "rolling-window shift (llama.cpp --context-shift analogue)"),
    "p3_c1024": (1024, 3, "UniKV lossless spill-and-recall"),
    "p0_c2048": (2048, 0, "stock baseline, errors at cache-full"),
    "p1_c2048": (2048, 1, "rolling-window shift"),
    "p3_c2048": (2048, 3, "UniKV lossless spill-and-recall"),
    "p4_c1024": (1024, 4, "H2O fixed-budget eviction (prior-art comparator)"),
    "p4_c2048": (2048, 4, "H2O fixed-budget eviction (prior-art comparator)"),
}


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


def run_one(prompt: Path, tag: str, trial: int) -> dict:
    ctx, policy, _ = ALL_ARMS[tag]
    e2e_csv  = LOGS_DIR / f"e2e_{tag}_t{trial}.csv"
    step_csv = LOGS_DIR / f"step_{tag}_t{trial}.csv"
    log_txt  = LOGS_DIR / f"gen_{tag}_t{trial}.txt"
    e2e_csv.unlink(missing_ok=True)
    step_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
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
    if STEPLOG:
        env["UNIKV_LOG"] = str(step_csv)

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    log_txt.write_text("=== STDOUT ===\n" + done.stdout +
                       "\n=== STDERR ===\n" + done.stderr)

    dec_tokens, tok_s = None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec_tokens = int(rows[-1]["decode_tokens"])
            tok_s = float(rows[-1]["tok_per_sec"])

    # evidence, not assumption: read the setting back out of the run's own log
    mfa = re.search(r"flash_attn\s*=\s*(\w+)", done.stderr)
    flash_attn = mfa.group(1) if mfa else "UNPARSED"

    kv_mib = None
    mk = re.search(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", done.stderr)
    if mk:
        kv_mib = float(mk.group(1))

    # policy 3 logs each spill at INFO; policy 1's shift is DEBUG-only, so its
    # count comes from the instrumented (UNIKV_PC_STEPLOG=1) pass.
    spill_lines = len(re.findall(r"unikv: spilled \d+ cells", done.stderr))
    evict_lines = len(re.findall(r"unikv: h2o evicted \d+ cells", done.stderr))
    if evict_lines:
        spill_lines = evict_lines   # one demotion-event column for every policy
    retained = 0
    rm = re.findall(r"retained=(\d+)", done.stderr)
    if rm:
        retained = int(rm[-1])

    step_events = None
    if STEPLOG and step_csv.exists():
        with step_csv.open(newline="") as fh:
            step_events = sum(1 for r in csv.DictReader(fh)
                              if int(r.get("shift_events", 0) or 0) > 0)

    return {
        "rc": done.returncode, "decode_tokens": dec_tokens, "tok_per_sec": tok_s,
        "flash_attn": flash_attn, "kv_device_mib": kv_mib,
        "spill_log_lines": spill_lines, "retained": retained,
        "step_shift_events": step_events,
    }


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    want = os.environ.get("UNIKV_PC_ARMS", "")
    arms = [a.strip() for a in want.split(",") if a.strip()] if want else list(ALL_ARMS)
    for a in arms:
        if a not in ALL_ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {', '.join(ALL_ARMS)}")

    prompt = ensure_prompt()

    plan = [(a, t) for a in arms for t in range(1, TRIALS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print(f"policy comparison: {len(arms)} arms x {TRIALS} trials = {len(plan)} runs")
    print(f"  prompt {PROMPT_TOKENS} tok, decode budget {GEN_TOKENS}, -fa off, seed {SEED}")
    print(f"  randomized block (seed {SHUFFLE_SEED}), cooldown {COOLDOWN_S}s, "
          f"step log {'ON' if STEPLOG else 'off'}")
    print("  order:", " ".join(f"{a}t{t}" for a, t in plan))
    print()

    MASTER_CSV.unlink(missing_ok=True)
    fields = ["order_idx", "arm", "ctx", "policy", "trial", "rc", "decode_tokens",
              "tok_per_sec", "flash_attn", "kv_device_mib", "spill_log_lines",
              "retained", "step_shift_events", "t_start_iso", "duration_s"]
    with MASTER_CSV.open("w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    per_arm: dict[str, list[float]] = {a: [] for a in arms}
    flags = []
    for idx, (tag, trial) in enumerate(plan, 1):
        ctx, policy, _ = ALL_ARMS[tag]
        if COOLDOWN_S > 0:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] {tag} trial={trial} start={iso(t0)} ...",
              end="", flush=True)
        r = run_one(prompt, tag, trial)
        dur = time.time() - t0
        print(f" {r['tok_per_sec']} tok/s  dec={r['decode_tokens']}  "
              f"fa={r['flash_attn']}  ({dur:.0f}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"{tag}t{trial}: flash_attn={r['flash_attn']} — protocol requires off")
        if r["tok_per_sec"] is None:
            flags.append(f"{tag}t{trial}: no tok/s recorded")
        elif r["tok_per_sec"] >= CEILING:
            flags.append(f"{tag}t{trial}: {r['tok_per_sec']} >= {CEILING} ceiling — STOP")
        else:
            per_arm[tag].append(r["tok_per_sec"])

        with MASTER_CSV.open("a", newline="") as fh:
            csv.writer(fh).writerow([
                idx, tag, ctx, policy, trial, r["rc"], r["decode_tokens"],
                r["tok_per_sec"], r["flash_attn"], r["kv_device_mib"],
                r["spill_log_lines"], r["retained"], r["step_shift_events"],
                iso(t0), f"{dur:.1f}"])

    print("\n" + "=" * 78)
    print(f"{'arm':12s} {'C':>5s} {'pol':>3s} {'n':>2s} {'tokens':>7s} "
          f"{'mean':>7s} {'sd':>6s}  role")
    print("-" * 78)
    for tag in arms:
        ctx, policy, role = ALL_ARMS[tag]
        vals = per_arm[tag]
        toks = "n/a"
        with MASTER_CSV.open(newline="") as fh:
            got = [r["decode_tokens"] for r in csv.DictReader(fh) if r["arm"] == tag]
        if got:
            toks = got[0] if len(set(got)) == 1 else f"VARIES{set(got)}"
        if vals:
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{tag:12s} {ctx:5d} {policy:3d} {len(vals):2d} {toks:>7s} "
                  f"{mean:7.2f} {sd:6.2f}  {role}")
        else:
            print(f"{tag:12s} {ctx:5d} {policy:3d}  0 {toks:>7s} "
                  f"{'n/a':>7s} {'n/a':>6s}  {role}")
    print("=" * 78)

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    else:
        print(f"\nall {len(plan)} runs: flash attention off, under the {CEILING} tok/s ceiling.")
    print(f"\nwrote {MASTER_CSV}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
