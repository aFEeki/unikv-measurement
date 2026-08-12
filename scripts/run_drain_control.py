#!/usr/bin/env python3
"""alpha=0-with-drain control (review M4, DA-5).

The alpha sweep applies llama_context::synchronize() only at alpha>0, so the
alpha=0 arm does not pay the pipeline drain that every alpha>0 arm pays. The
reported alpha=0 offset below the alpha>0 fit is therefore partly by
construction: instrumentation differs across the treatment boundary, so the
offset cannot by itself be evidence of a non-blocking/blocking transition.

UNIKV_DRAIN (added for this control) decouples the two:
  unset -> drain iff alpha > 0   (the published behavior)
  1     -> drain even at alpha = 0
  0     -> never drain, even at alpha > 0

Five conditions, randomized-block with cooldowns, otherwise identical to the
cooled alpha sweep (policy 3, C=1024, 512-token prompt, 2048 decode, -fa off,
seed 123, greedy, uninstrumented so the per-step logging synchronize() pair does
not itself drain):

  a0_nodrain    alpha 0.00, drain off  -- the published alpha=0 anchor
  a0_drain      alpha 0.00, drain ON   -- drain cost alone, no modeled transfer
  a025_drain    alpha 0.25, drain ON   -- the published alpha=0.25 point
  a025_nodrain  alpha 0.25, drain off  -- modeled transfer alone, no drain
  a1_nodrain    alpha 1.00, drain off  -- checks whether drain cost is
                                          alpha-independent across the range

Decomposition: drain cost = a0_nodrain - a0_drain; modeled-transfer cost at
alpha=0.25 = a0_nodrain - a025_nodrain. If the two sum to a0_nodrain - a025_drain
the effects are additive and the alpha=0 offset splits cleanly.
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

RESULTS_DIR = ROOT / "alpha_results"
ART_DIR     = ROOT / "artifacts" / "drain_control"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTX           = 1024
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

TRIALS     = int(os.environ.get("UNIKV_DC_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_DC_COOLDOWN", "200"))

# tag -> (alpha, drain_env or None, note)
CONDITIONS = {
    "a0_nodrain":   (0.0,  None, "published alpha=0 anchor (no drain)"),
    "a0_drain":     (0.0,  "1",  "drain only — no modeled transfer"),
    "a025_drain":   (0.25, None, "published alpha=0.25 (drain + transfer)"),
    "a025_nodrain": (0.25, "0",  "modeled transfer only — no drain"),
    "a1_nodrain":   (1.0,  "0",  "alpha=1 without drain (drain-cost constancy)"),
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


def run_one(prompt: Path, cond: str, trial: int) -> dict:
    alpha, drain, _ = CONDITIONS[cond]
    e2e_csv = LOGS_DIR / f"e2e_{cond}_t{trial}.csv"
    log_txt = LOGS_DIR / f"gen_{cond}_t{trial}.txt"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({
        "UNIKV_POLICY": "3",
        "UNIKV_ALPHA": f"{alpha:g}",
        "UNIKV_E2E_LOG": str(e2e_csv),
        "UNIKV_SPILL_CAP": str(SPILL_CAP),
    })
    env.pop("UNIKV_DRAIN", None)
    if drain is not None:
        env["UNIKV_DRAIN"] = drain

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    log_txt.write_text("=== STDOUT ===\n" + done.stdout +
                       "\n=== STDERR ===\n" + done.stderr[-6000:])

    dec, tps, ms = None, None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec = int(rows[-1]["decode_tokens"])
            tps = float(rows[-1]["tok_per_sec"])
            ms  = float(rows[-1]["decode_ms"])

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", done.stderr)
    return {"rc": done.returncode, "decode_tokens": dec, "tok_per_sec": tps,
            "decode_ms": ms, "flash_attn": mfa.group(1) if mfa else "UNPARSED"}


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    prompt = ensure_prompt()
    conds = list(CONDITIONS)
    plan = [(c, t) for c in conds for t in range(1, TRIALS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)

    print(f"drain control: {len(conds)} conditions x {TRIALS} trials = {len(plan)} runs")
    print(f"  policy 3, C={CTX}, {PROMPT_TOKENS}-token prompt, {GEN_TOKENS} decode, "
          f"-fa off, uninstrumented")
    print(f"  randomized block (seed {SHUFFLE_SEED}), cooldown {COOLDOWN_S}s")
    print("  order:", " ".join(f"{c}t{t}" for c, t in plan))
    print()

    out = RESULTS_DIR / "p3_drain_control_master.csv"
    out.unlink(missing_ok=True)
    with out.open("w", newline="") as fh:
        csv.writer(fh).writerow(["order_idx", "condition", "alpha", "drain", "trial",
                                 "rc", "decode_tokens", "decode_ms", "tok_per_sec",
                                 "flash_attn", "t_start_iso", "duration_s"])

    vals: dict[str, list[float]] = {c: [] for c in conds}
    flags = []
    for idx, (cond, trial) in enumerate(plan, 1):
        alpha, drain, _ = CONDITIONS[cond]
        if COOLDOWN_S > 0:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] {cond:13s} trial={trial} ...", end="", flush=True)
        r = run_one(prompt, cond, trial)
        dur = time.time() - t0
        print(f" {r['tok_per_sec']} tok/s  dec={r['decode_tokens']} ({dur:.0f}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"{cond}t{trial}: flash_attn={r['flash_attn']}")
        if r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"{cond}t{trial}: decoded {r['decode_tokens']} != {GEN_TOKENS}")
        if r["tok_per_sec"] is None:
            flags.append(f"{cond}t{trial}: no tok/s")
        elif r["tok_per_sec"] >= CEILING:
            flags.append(f"{cond}t{trial}: {r['tok_per_sec']} >= {CEILING} ceiling")
        else:
            vals[cond].append(r["tok_per_sec"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, cond, alpha, drain if drain is not None
                                     else "default", trial, r["rc"],
                                     r["decode_tokens"], r["decode_ms"],
                                     r["tok_per_sec"], r["flash_attn"],
                                     iso(t0), f"{dur:.1f}"])

    print("\n" + "=" * 76)
    print(f"{'condition':14s} {'alpha':>5s} {'drain':>7s} {'n':>2s} {'mean':>7s} {'sd':>6s}  note")
    print("-" * 76)
    means = {}
    for c in conds:
        alpha, drain, note = CONDITIONS[c]
        v = vals[c]
        if v:
            means[c] = statistics.mean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"{c:14s} {alpha:5g} {str(drain if drain is not None else 'dflt'):>7s} "
                  f"{len(v):2d} {means[c]:7.2f} {sd:6.2f}  {note}")
        else:
            print(f"{c:14s} {alpha:5g} {'--':>7s}  0 {'n/a':>7s} {'n/a':>6s}  {note}")
    print("=" * 76)

    if {"a0_nodrain", "a0_drain", "a025_drain", "a025_nodrain"} <= means.keys():
        drain_cost = means["a0_nodrain"] - means["a0_drain"]
        xfer_cost = means["a0_nodrain"] - means["a025_nodrain"]
        both = means["a0_nodrain"] - means["a025_drain"]
        print("\nDecomposition of the published alpha=0 -> alpha=0.25 step "
              f"({both:+.2f} tok/s):")
        print(f"  pipeline drain alone            : {drain_cost:+.2f} tok/s")
        print(f"  modeled transfer alone (a=0.25) : {xfer_cost:+.2f} tok/s")
        print(f"  sum of parts                    : {drain_cost + xfer_cost:+.2f} tok/s "
              f"(vs {both:+.2f} measured together)")

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
