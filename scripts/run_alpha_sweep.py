#!/usr/bin/env python3
"""
Phase 4 alpha sweep — measures how UniKV's continuous-generation advantage
degrades as transfer cost (PCIe simulation) increases.

Setup mirrors the Phase 3B stress benchmark: 512-token prompt + 2048 decode
tokens at ctx=1024 (the confirmed-overflow case).

For each alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] we run UNIKV_POLICY=1
once. We additionally run UNIKV_POLICY=0 once as a "stops at cache full"
reference. After all runs, we print:
  alpha | mean_tok_per_sec | total_shifts | notes

The key paper claim (and what we want to see): mean tok/s should decrease
monotonically as alpha increases, demonstrating the crossover where the
unified-memory advantage shrinks under simulated PCIe cost.
"""

from __future__ import annotations

import csv
import datetime
import os
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
LLAMA_DIR     = ROOT / "llama.cpp"
BUILD_DIR     = LLAMA_DIR / "build-m4pro-metal"
BIN_DIR       = BUILD_DIR / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN  = BIN_DIR / "llama-tokenize"
MODEL_PATH    = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
RESULTS_DIR   = ROOT / "alpha_results"
ARTIFACT_DIR  = ROOT / "artifacts" / "alpha_sweep"
PROMPTS_DIR   = ARTIFACT_DIR / "prompts"
LOGS_DIR      = ARTIFACT_DIR / "logs"

CTX_SIZE         = 1024
PROMPT_TOKENS    = 512
GEN_TOKENS       = 2048
THREADS          = 10
THREADS_BATCH    = 10
GPU_LAYERS       = 999
BATCH_SIZE       = 512
UBATCH_SIZE      = 512
SEED             = 123
FIXED_PROMPT_TOKEN = " token"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TRIALS_PER_ALPHA = 5

# Fixed seed => reproducible shuffle; the resulting order is printed for audit.
# Randomizing (alpha, trial) order decorrelates any time-varying perturbation
# (thermal drift, background load) from alpha, so it shows up as noise across
# alphas rather than masquerading as "throughput falls with alpha".
SHUFFLE_SEED = 20260702

# Hard physical sanity ceiling (M4 Pro ~273 GB/s over ~4.7 GB weights ~= 58 tok/s).
# Any end-to-end tok/s at or above this is non-physical => flag, do not report.
CEILING_TOK_PER_SEC = 58.0

TIMING_LOG = RESULTS_DIR / "alpha_timing_log.csv"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_paths() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_capture(
    args: list[str],
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        args,
        cwd=LLAMA_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    log_path.write_text(completed.stdout)
    return completed.returncode


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def count_tokens(prompt_path: Path) -> int:
    args = [
        str(TOKENIZE_BIN),
        "-m", str(MODEL_PATH),
        "-f", str(prompt_path),
        "--show-count",
        "--log-disable",
    ]
    completed = subprocess.run(
        args,
        cwd=LLAMA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=True,
    )
    match = re.search(r"Total number of tokens:\s*(\d+)", completed.stdout)
    if not match:
        raise RuntimeError(f"Could not parse token count:\n{completed.stdout}")
    return int(match.group(1))


def ensure_prompt(n_tokens: int) -> Path:
    prompt_path = PROMPTS_DIR / f"alpha_prompt_{n_tokens}tok.txt"
    text = FIXED_PROMPT_TOKEN * max(n_tokens - 1, 0)
    prompt_path.write_text(text)
    actual = count_tokens(prompt_path)
    if actual != n_tokens:
        raise RuntimeError(
            f"Prompt token mismatch: requested {n_tokens}, got {actual}"
        )
    return prompt_path


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

def completion_args(prompt_path: Path) -> list[str]:
    return [
        str(COMPLETION_BIN),
        "-m",  str(MODEL_PATH),
        "-f",  str(prompt_path),
        "-n",  str(GEN_TOKENS),
        "-c",  str(CTX_SIZE),
        "-ngl", str(GPU_LAYERS),
        "-t",  str(THREADS),
        "-tb", str(THREADS_BATCH),
        "-b",  str(BATCH_SIZE),
        "-ub", str(UBATCH_SIZE),
        "-fa", "on",
        "-fit", "off",
        "--temp", "0",
        "--seed", str(SEED),
        "--ignore-eos",
        "--no-warmup",
        "--perf",
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",
    ]


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def log_timing(order_idx: int, alpha: float, trial: int,
               t_start: float, t_end: float, e2e: float) -> None:
    """Append per-trial wall-clock timestamps so a low cluster can be checked
    against a time window rather than assumed to be a real alpha effect."""
    new = not TIMING_LOG.exists()
    with TIMING_LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["order_idx", "alpha", "trial", "t_start_iso",
                        "t_end_iso", "duration_s", "e2e_tok_per_sec"])
        w.writerow([order_idx, f"{alpha:g}", trial, _iso(t_start),
                    _iso(t_end), f"{t_end - t_start:.2f}", f"{e2e:.3f}"])


def run_alpha_trial(alpha: float, trial: int, prompt_path: Path,
                    order_idx: int, n_total: int) -> None:
    csv_path = RESULTS_DIR / f"alpha_sweep_{alpha:g}_t{trial}.csv"
    e2e_path = RESULTS_DIR / f"alpha_sweep_{alpha:g}_t{trial}_e2e.csv"
    log_path = LOGS_DIR / f"alpha_sweep_{alpha:g}_t{trial}.log"
    csv_path.unlink(missing_ok=True)
    e2e_path.unlink(missing_ok=True)

    extra_env = {
        "UNIKV_POLICY":  "1",
        "UNIKV_ALPHA":   f"{alpha:g}",
        "UNIKV_LOG":     str(csv_path),
        "UNIKV_E2E_LOG": str(e2e_path),
    }
    t_start = time.time()
    print(f"  [{order_idx:>2}/{n_total}] alpha={alpha:<5g} trial={trial}  "
          f"start={_iso(t_start)} ...", flush=True)
    rc = run_capture(completion_args(prompt_path), log_path, extra_env)
    t_end = time.time()
    if rc != 0:
        print(f"      [WARNING] policy run exited rc={rc} — see {log_path}", flush=True)
    e2e = read_e2e_tps(e2e_path)
    print(f"      end={_iso(t_end)}  dur={t_end - t_start:6.1f}s  "
          f"e2e={e2e:6.2f} tok/s", flush=True)
    if e2e >= CEILING_TOK_PER_SEC:
        print(f"      [FLAG] e2e {e2e:.2f} >= {CEILING_TOK_PER_SEC} tok/s ceiling "
              f"— non-physical, do not report", flush=True)
    log_timing(order_idx, alpha, trial, t_start, t_end, e2e)


def run_baseline(prompt_path: Path) -> Path:
    csv_path = RESULTS_DIR / "alpha_sweep_baseline.csv"
    log_path = LOGS_DIR / "alpha_sweep_baseline.log"
    csv_path.unlink(missing_ok=True)

    e2e_path = RESULTS_DIR / "alpha_sweep_baseline_e2e.csv"
    e2e_path.unlink(missing_ok=True)
    extra_env = {
        "UNIKV_POLICY":  "0",
        "UNIKV_ALPHA":   "0",
        "UNIKV_LOG":     str(csv_path),
        "UNIKV_E2E_LOG": str(e2e_path),
    }
    print(f"  baseline (UNIKV_POLICY=0) → {csv_path.name} ...", flush=True)
    rc = run_capture(completion_args(prompt_path), log_path, extra_env)
    if rc != 0:
        print(f"    [expected] baseline exited rc={rc} (KV cache full → return 1)", flush=True)
    return csv_path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def mean_decode_tps(rows: list[dict[str, str]]) -> float:
    decode = rows[1:]
    if not decode:
        return 0.0
    return sum(float(r["tok_per_sec"]) for r in decode) / len(decode)


def read_e2e_tps(path: Path) -> float:
    """End-to-end decode throughput (single GPU sync after the loop). HEADLINE."""
    rows = read_csv_rows(path)
    return float(rows[-1]["tok_per_sec"]) if rows else 0.0


def total_shifts(rows: list[dict[str, str]]) -> int:
    """shift_events is per-call (0 or 1). Total shifts = rows where it fires."""
    if not rows:
        return 0
    return sum(1 for r in rows if int(r.get("shift_events", 0)) > 0)


def trial_tps_shifts(alpha: float, trial: int) -> tuple[float, int, int]:
    """Returns (end-to-end tok/s [headline], total shifts, per-step row count)."""
    rows = read_csv_rows(RESULTS_DIR / f"alpha_sweep_{alpha:g}_t{trial}.csv")
    e2e  = read_e2e_tps(RESULTS_DIR / f"alpha_sweep_{alpha:g}_t{trial}_e2e.csv")
    return e2e, total_shifts(rows), len(rows)


def aggregate_alpha(alpha: float) -> tuple[float, int, int]:
    """Return (mean_tps_avg, mean_total_shifts, mean_rows) over trials."""
    tps_list, shifts_list, rows_list = [], [], []
    for t in range(1, TRIALS_PER_ALPHA + 1):
        tps, sh, rows = trial_tps_shifts(alpha, t)
        if rows == 0:
            continue
        tps_list.append(tps)
        shifts_list.append(sh)
        rows_list.append(rows)
    if not tps_list:
        return 0.0, 0, 0
    return (
        sum(tps_list) / len(tps_list),
        round(sum(shifts_list) / len(shifts_list)),
        round(sum(rows_list) / len(rows_list)),
    )


def summarize() -> None:
    print("\n" + "=" * 96)
    print(f"ALPHA SWEEP SUMMARY (ctx={CTX_SIZE}, {PROMPT_TOKENS}-tok prompt, "
          f"{GEN_TOKENS} decode tokens, {TRIALS_PER_ALPHA} trials/alpha, "
          f"end-to-end tok/s)")
    print("=" * 96)
    print(f"{'alpha':>8} | {'mean':>7} | {'std':>6} | {'min':>7} | {'max':>7} | "
          f"{'n':>2} | {'shifts':>7} | notes")
    print("-" * 96)

    over_ceiling: list[tuple] = []
    summary_rows = []
    for alpha in ALPHAS:
        tps_list, shifts_list = [], []
        for t in range(1, TRIALS_PER_ALPHA + 1):
            tps, sh, rows = trial_tps_shifts(alpha, t)
            if rows > 0:
                tps_list.append(tps)
                shifts_list.append(sh)
                if tps >= CEILING_TOK_PER_SEC:
                    over_ceiling.append((alpha, t, tps))
        if not tps_list:
            continue
        mean = statistics.mean(tps_list)
        std  = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
        note = ""
        if alpha == 0.0:
            note = "unified memory (M4 Pro reality)"
        elif alpha == 1.0:
            note = "A100 PCIe 4.0 x16 equivalent"
        print(f"{alpha:>8g} | {mean:>7.2f} | {std:>6.2f} | {min(tps_list):>7.2f} | "
              f"{max(tps_list):>7.2f} | {len(tps_list):>2d} | "
              f"{round(sum(shifts_list) / len(shifts_list)):>7d} | {note}")
        summary_rows.append((alpha, mean, std))

    base_rows = read_csv_rows(RESULTS_DIR / "alpha_sweep_baseline.csv")
    base_tps  = read_e2e_tps(RESULTS_DIR / "alpha_sweep_baseline_e2e.csv")
    base_shifts = total_shifts(base_rows)
    if base_tps >= CEILING_TOK_PER_SEC:
        over_ceiling.append(("baseline", 1, base_tps))
    print(f"{'baseline':>8} | {base_tps:>7.2f} | {'--':>6} | {base_tps:>7.2f} | "
          f"{base_tps:>7.2f} | {1:>2d} | {base_shifts:>7d} | "
          f"UNIKV_POLICY=0, stops at cache full")

    # Report the trend as measured; do NOT smooth toward monotonicity. Mark an
    # uptick only as "within noise" or "EXCEEDS noise" relative to the pooled
    # std of the two adjacent alphas, so the reader judges significance.
    print("\nTrend (mean +/- std e2e tok/s vs alpha; noise NOT smoothed):")
    prev = None
    for alpha, mean, std in summary_rows:
        marker = ""
        if prev is not None:
            p_alpha, p_mean, p_std = prev
            gap = mean - p_mean
            noise = max(std, p_std, 1e-9)
            if gap > 1e-6:
                sig = "within noise" if gap <= noise else "EXCEEDS noise"
                marker = f"   up {gap:+.2f} vs a={p_alpha:g} ({sig})"
            elif gap < -1e-6:
                marker = f"   down {gap:+.2f}"
        print(f"  alpha={alpha:<5g}  {mean:6.2f} +/- {std:4.2f}{marker}")
        prev = (alpha, mean, std)

    # Macro endpoints: is the full-range drop outside noise?
    if len(summary_rows) >= 2:
        a0, m0, s0 = summary_rows[0]
        aN, mN, sN = summary_rows[-1]
        drop = m0 - mN
        bands = "DO NOT overlap" if (m0 - s0) > (mN + sN) else "overlap"
        print(f"\nMacro: alpha={a0:g} -> alpha={aN:g}: {m0:.2f} -> {mN:.2f} tok/s "
              f"(drop {drop:.2f}, {-100.0 * drop / m0:+.1f}%); "
              f"[min@{a0:g}={m0 - s0:.2f}] vs [max@{aN:g}={mN + sN:.2f}] "
              f"-> bands {bands}.")

    if over_ceiling:
        print(f"\n[FLAG] Non-physical readings at/over the {CEILING_TOK_PER_SEC} "
              f"tok/s ceiling -- investigate, DO NOT report:")
        for a, t, tps in over_ceiling:
            print(f"    alpha={a} trial={t}: {tps:.2f} tok/s")
    else:
        print(f"\nSanity: all trials below the {CEILING_TOK_PER_SEC} tok/s ceiling. OK.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_paths()
    must_exist(COMPLETION_BIN)
    must_exist(TOKENIZE_BIN)
    must_exist(MODEL_PATH)

    print("Alpha sweep: Phase 4 — transfer-cost simulation")
    print(f"  ctx={CTX_SIZE}  prompt={PROMPT_TOKENS}  decode={GEN_TOKENS}")
    print(f"  alphas: {ALPHAS}")
    print()

    print(f"Building prompt ({PROMPT_TOKENS} tokens)...")
    prompt_path = ensure_prompt(PROMPT_TOKENS)
    print(f"  {prompt_path}\n")

    # Randomized block design: TRIALS_PER_ALPHA rounds; each round runs every
    # alpha exactly once in a freshly shuffled order. Guarantees each alpha gets
    # one trial in every time block, so no alpha can absorb a monotonic drift
    # (running 0->2 monotonically would alias drift onto increasing alpha).
    # Strictly stronger decorrelation than a plain shuffle of the full set; the
    # trial index equals the round number. Order printed below for audit.
    rng = random.Random(SHUFFLE_SEED)
    work = []
    for rnd in range(1, TRIALS_PER_ALPHA + 1):
        round_alphas = ALPHAS[:]
        rng.shuffle(round_alphas)
        work.extend((a, rnd) for a in round_alphas)
    n_total = len(work)

    print(f"Randomized block execution order (seed={SHUFFLE_SEED}, "
          f"{TRIALS_PER_ALPHA} rounds x {len(ALPHAS)} alphas = {n_total} runs, "
          f"then baseline):")
    for i, (a, t) in enumerate(work, 1):
        print(f"  {i:>2}. alpha={a:<5g} trial={t}")
    print()

    TIMING_LOG.unlink(missing_ok=True)
    for i, (a, t) in enumerate(work, 1):
        run_alpha_trial(a, t, prompt_path, i, n_total)
    run_baseline(prompt_path)

    summarize()
    print(f"\nAll CSV files: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
