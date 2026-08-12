#!/usr/bin/env python3
"""
Phase 3B stress benchmark — exercises the UniKV shift policy by forcing KV cache
overflow during decode, not prefill.

Design: 512-token prompt + 2048 decode tokens at ctx=[1024, 2048, 4096].
  ctx=1024: overflows at decode step ~513 (512 prompt + 512 decode fills cache)
  ctx=2048: overflows at decode step ~1537 (512 prompt + 1536 decode fills cache)
  ctx=4096: no overflow (512+2048=2560 < 4096)

Each context size runs twice:
  UNIKV_POLICY=0  (baseline, expected to fail/stop at overflow)
  UNIKV_POLICY=1  (UniKV, shifts and continues)
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
LLAMA_DIR     = ROOT / "llama.cpp"
BUILD_DIR     = LLAMA_DIR / "build-m4pro-metal"
BIN_DIR       = BUILD_DIR / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN  = BIN_DIR / "llama-tokenize"
MODEL_PATH    = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
RESULTS_DIR   = ROOT / "stress_results"
ARTIFACT_DIR  = ROOT / "artifacts" / "stress_benchmark"
PROMPTS_DIR   = ARTIFACT_DIR / "prompts"
LOGS_DIR      = ARTIFACT_DIR / "logs"

CONTEXT_SIZES    = [1024, 2048, 4096]
PROMPT_TOKENS    = 512
GEN_TOKENS       = 2048
THREADS          = 10
THREADS_BATCH    = 10
GPU_LAYERS       = 999
BATCH_SIZE       = 512
UBATCH_SIZE      = 512
SEED             = 123
FIXED_PROMPT_TOKEN = " token"


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
) -> tuple[int, str]:
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
    return completed.returncode, completed.stdout


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
    """Build a prompt of exactly n_tokens tokens (BOS + repeated ' token' filler)."""
    prompt_path = PROMPTS_DIR / f"stress_prompt_{n_tokens}tok.txt"
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

def completion_args(ctx_size: int, prompt_path: Path) -> list[str]:
    return [
        str(COMPLETION_BIN),
        "-m",  str(MODEL_PATH),
        "-f",  str(prompt_path),
        "-n",  str(GEN_TOKENS),
        "-c",  str(ctx_size),
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


def run_stress(ctx_size: int, policy: int, prompt_path: Path) -> tuple[int, Path]:
    tag      = "baseline" if policy == 0 else "policy"
    csv_path = RESULTS_DIR / f"stress_{tag}_{ctx_size}.csv"
    e2e_path = RESULTS_DIR / f"stress_{tag}_{ctx_size}_e2e.csv"
    log_path = LOGS_DIR / f"stress_{tag}_ctx{ctx_size}.log"

    # Remove stale files so the subprocess always writes fresh CSVs.
    csv_path.unlink(missing_ok=True)
    e2e_path.unlink(missing_ok=True)

    args = completion_args(ctx_size, prompt_path)
    extra_env = {
        "UNIKV_POLICY":  str(policy),
        "UNIKV_LOG":     str(csv_path),
        "UNIKV_E2E_LOG": str(e2e_path),
    }

    print(f"  policy={policy}  ctx={ctx_size:>5}  → {csv_path.name} ...", flush=True)
    rc, _ = run_capture(args, log_path, extra_env)

    if policy == 0 and rc != 0:
        print(f"    [expected] baseline exited rc={rc} (KV cache full → return 1)", flush=True)
    elif rc != 0:
        print(f"    [WARNING]  policy run exited rc={rc} — see {log_path}", flush=True)

    return rc, csv_path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_e2e_tps(path: Path) -> float:
    """End-to-end decode throughput (single GPU sync after the loop)."""
    rows = read_csv_rows(path)
    return float(rows[-1]["tok_per_sec"]) if rows else 0.0


def analyze(ctx_size: int) -> None:
    baseline_rows = read_csv_rows(RESULTS_DIR / f"stress_baseline_{ctx_size}.csv")
    policy_rows   = read_csv_rows(RESULTS_DIR / f"stress_policy_{ctx_size}.csv")

    print(f"\n  ctx={ctx_size}")
    print(f"    rows in CSV — baseline: {len(baseline_rows)}, policy: {len(policy_rows)}")

    # -- shift events (policy) --
    shifted = [r for r in policy_rows if int(r.get("shift_events", 0)) > 0]
    if shifted:
        first = shifted[0]
        print(f"    shift_events > 0 : YES")
        print(f"      first shift    : step={first['step']}  "
              f"kv_used={first['kv_used']}  kv_ratio={float(first['kv_ratio']):.4f}")
        print(f"      total shifted  : {len(shifted)} steps")
    else:
        print(f"    shift_events > 0 : NO  (cache never overflowed during this run)")

    # -- mean decode tok/s (skip row 0 = prefill) --
    def mean_tps(rows: list[dict[str, str]]) -> float:
        decode = rows[1:]
        if not decode:
            return 0.0
        return sum(float(r["tok_per_sec"]) for r in decode) / len(decode)

    # -- end-to-end decode tok/s (HEADLINE: single GPU sync after the loop) --
    b_e2e = read_e2e_tps(RESULTS_DIR / f"stress_baseline_{ctx_size}_e2e.csv")
    p_e2e = read_e2e_tps(RESULTS_DIR / f"stress_policy_{ctx_size}_e2e.csv")
    print(f"    end-to-end tok/s  — baseline: {b_e2e:.2f},  policy: {p_e2e:.2f}   [HEADLINE]")

    b_tps = mean_tps(baseline_rows)
    p_tps = mean_tps(policy_rows)
    print(f"    per-step tok/s    — baseline: {b_tps:.2f},  policy: {p_tps:.2f}   "
          f"(per-step syncs => slightly pessimistic)")

    # -- max kv_ratio --
    def max_ratio(rows: list[dict[str, str]]) -> float:
        return max((float(r["kv_ratio"]) for r in rows), default=0.0)

    b_max = max_ratio(baseline_rows)
    p_max = max_ratio(policy_rows)
    print(f"    max kv_ratio      — baseline: {b_max:.4f},  policy: {p_max:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_paths()
    must_exist(COMPLETION_BIN)
    must_exist(TOKENIZE_BIN)
    must_exist(MODEL_PATH)

    overflow_at = PROMPT_TOKENS + GEN_TOKENS
    print(f"Stress benchmark: {PROMPT_TOKENS}-tok prompt, {GEN_TOKENS} decode tokens")
    print(f"Overflow expected for ctx < {overflow_at}  "
          f"(contexts {[c for c in CONTEXT_SIZES if c < overflow_at]} will overflow)\n")

    print(f"Building prompt ({PROMPT_TOKENS} tokens)...")
    prompt_path = ensure_prompt(PROMPT_TOKENS)
    print(f"  {prompt_path}\n")

    print("Running (baseline first, then policy, per context size)...")
    for ctx_size in CONTEXT_SIZES:
        print(f"\nContext {ctx_size}:")
        run_stress(ctx_size, 0, prompt_path)
        run_stress(ctx_size, 1, prompt_path)

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    for ctx_size in CONTEXT_SIZES:
        analyze(ctx_size)

    print(f"\nAll CSV files: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
