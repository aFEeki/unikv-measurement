#!/usr/bin/env python3
"""UniKV R2 alpha sweep on policy 3 (recall-cost model). Supersedes the §3
policy-1 alpha numbers.

alpha now prices the PER-STEP RECALL of the spilled tier, not the shift:
each decode step with n_spilled > 0 injects delay = alpha * (n_spilled *
per_cell_KV_bytes) / 64 GB/s, with synchronize() before the sleep. alpha=0 is
the un-synced unified-memory reference (no per-step drain); alpha>0 models a
blocking discrete-memory recall (drain + delay). The alpha=0 -> alpha>0 step is
real physics (non-blocking shared memory -> blocking PCIe), not an artifact.

Methodology mirrors the original sweep: 7 alphas x 5 trials = 35 runs in
randomized-block order (fixed shuffle seed, order printed for audit) so
time-correlated thermal drift decorrelates from alpha instead of aliasing onto
it. Records e2e tok/s + a wall-clock timing log per run.

Config: policy 3, C=1024, 512-token prompt + 2048 decode, -fa off, seed 123,
--ignore-eos. Every run must complete the full 2048 (lossless) and sit under the
58 tok/s ceiling.
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
ART_DIR     = ROOT / "artifacts" / "r2_policy3"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "alpha_logs"

# Optional cooldown between runs (seconds) to hold thermal steady, and trial
# count + filename tag, all via env so the original flat run stays reproducible.
COOLDOWN_S = int(os.environ.get("UNIKV_SWEEP_COOLDOWN", "0"))
TAG        = os.environ.get("UNIKV_SWEEP_TAG", "")
_suffix    = f"_{TAG}" if TAG else ""

MASTER_CSV  = RESULTS_DIR / f"p3_alpha_sweep{_suffix}_master.csv"
TIMING_CSV  = RESULTS_DIR / f"p3_alpha_timing{_suffix}_log.csv"

CTX          = 1024
PROMPT_TOKENS = 512
GEN_TOKENS   = 2048
THREADS      = 10
GPU_LAYERS   = 999
BATCH        = 512
UBATCH       = 512
SEED         = 123
SPILL_CAP    = 8192
FIXED_TOKEN  = " token"

ALPHAS       = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TRIALS       = int(os.environ.get("UNIKV_SWEEP_TRIALS", "5"))
SHUFFLE_SEED = 20260710
CEILING      = 58.0


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


def run_one(prompt: Path, alpha: float, trial: int) -> dict:
    e2e_csv = LOGS_DIR / f"e2e_a{alpha:g}_t{trial}.csv"
    log_txt = LOGS_DIR / f"gen_a{alpha:g}_t{trial}.txt"
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

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    log_txt.write_text("=== STDOUT ===\n" + done.stdout + "\n=== STDERR ===\n" + done.stderr[-4000:])

    dec_tokens, tok_s = None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec_tokens = int(rows[-1]["decode_tokens"])
            tok_s = float(rows[-1]["tok_per_sec"])

    return {"rc": done.returncode, "decode_tokens": dec_tokens, "tok_per_sec": tok_s}


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    prompt = ensure_prompt()

    plan = [(a, t) for a in ALPHAS for t in range(1, TRIALS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print(f"policy 3 alpha sweep: {len(ALPHAS)} alphas x {TRIALS} trials = {len(plan)} runs, "
          f"C={CTX}, -fa off, randomized-block (seed {SHUFFLE_SEED})")
    print("order:", " ".join(f"a{a:g}t{t}" for a, t in plan))
    print()

    MASTER_CSV.unlink(missing_ok=True)
    TIMING_CSV.unlink(missing_ok=True)
    with MASTER_CSV.open("w", newline="") as fh:
        csv.writer(fh).writerow(["order_idx", "alpha", "trial", "rc",
                                 "decode_tokens", "tok_per_sec"])
    with TIMING_CSV.open("w", newline="") as fh:
        csv.writer(fh).writerow(["order_idx", "alpha", "trial", "t_start_iso",
                                 "t_end_iso", "duration_s", "tok_per_sec"])

    if COOLDOWN_S > 0:
        print(f"cooldown = {COOLDOWN_S}s before every run (thermal steady-state)\n")

    per_alpha: dict[float, list[float]] = {a: [] for a in ALPHAS}
    flags = []
    for idx, (alpha, trial) in enumerate(plan, 1):
        if COOLDOWN_S > 0:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] alpha={alpha:<4g} trial={trial} start={iso(t0)} ...",
              end="", flush=True)
        r = run_one(prompt, alpha, trial)
        t1 = time.time()
        tps = r["tok_per_sec"]
        print(f" {tps} tok/s  dec={r['decode_tokens']}  ({t1 - t0:.0f}s)")

        if r["rc"] != 0:
            flags.append(f"a{alpha:g}t{trial}: rc={r['rc']}")
        if r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"a{alpha:g}t{trial}: decoded {r['decode_tokens']} != {GEN_TOKENS} (not lossless!)")
        if tps is None:
            flags.append(f"a{alpha:g}t{trial}: no tok/s")
        elif tps >= CEILING:
            flags.append(f"a{alpha:g}t{trial}: {tps} >= {CEILING} ceiling")
        elif tps is not None:
            per_alpha[alpha].append(tps)

        with MASTER_CSV.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, f"{alpha:g}", trial, r["rc"],
                                     r["decode_tokens"], tps])
        with TIMING_CSV.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, f"{alpha:g}", trial, iso(t0), iso(t1),
                                     f"{t1 - t0:.1f}", tps])

    print("\n" + "=" * 56)
    print(f"{'alpha':>6s} {'n':>3s} {'mean':>8s} {'std':>7s}")
    print("-" * 56)
    for a in ALPHAS:
        vals = per_alpha[a]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"{a:6g} {len(vals):3d} {mean:8.2f} {std:7.2f}")
        else:
            print(f"{a:6g}   0      n/a     n/a")
    print("=" * 56)

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    else:
        print(f"\nall {len(plan)} runs: lossless (2048 tokens), < 58 tok/s ceiling.")
    print(f"\nwrote {MASTER_CSV}\nwrote {TIMING_CSV}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
