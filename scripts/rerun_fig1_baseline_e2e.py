#!/usr/bin/env python3
"""
Re-run Figure 1's baseline decode-throughput-vs-context curve through the SAME
fixed end-to-end harness used by the stress and alpha sweeps, so every figure
reports the identical primary metric.

Figure 1 workload (unchanged from run_m4pro_baseline.py): a near-full context
is filled by a prompt of (ctx - GEN_TOKENS) tokens, then GEN_TOKENS are decoded.
UNIKV_POLICY=0 (baseline). Contexts C in {4096, 8192, 16384, 32768}.

Metric: end-to-end decode tok/s = decode tokens / decode wall-clock, with one
synchronize() after the generation loop and prefill excluded (via UNIKV_E2E_LOG,
captured by llama-completion). We ALSO record common_perf_print's eval-time
tok/s as a cross-check -- that path was already GPU-synced (t_eval_us is
accumulated inside synchronize()), so it is the old Figure 1 number and should
sit slightly ABOVE e2e (e2e additionally includes per-token sampling overhead).

Output: baseline_m4pro_e2e.csv  (one row per context).
Guardrail: any e2e tok/s >= CEILING is flagged and NOT reported.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN_DIR        = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUT_CSV        = ROOT / "artifacts" / "m4pro_baseline" / "e2e" / "baseline_m4pro_e2e.csv"
ARTIFACT_DIR   = ROOT / "artifacts" / "m4pro_baseline"
PROMPTS_DIR    = ARTIFACT_DIR / "prompts"
LOGS_DIR       = ARTIFACT_DIR / "logs"
RESULTS_DIR    = ROOT / "alpha_results"  # not used; e2e logs go beside OUT_CSV

# Match run_m4pro_baseline.py exactly.
CONTEXT_SIZES = [4096, 8192, 16384, 32768]
GEN_TOKENS    = 256
THREADS       = 10
THREADS_BATCH = 10
GPU_LAYERS    = 999
BATCH_SIZE    = 2048
UBATCH_SIZE   = 512
SEED          = 123
FIXED_PROMPT_TOKEN = " token"

CEILING_TOK_PER_SEC = 58.0
E2E_DIR = ROOT / "artifacts" / "m4pro_baseline" / "e2e"


def must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def count_tokens(prompt_path: Path) -> int:
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(prompt_path),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True,
    ).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse token count:\n{out}")
    return int(m.group(1))


def ensure_prompt(ctx_size: int, prompt_tokens: int) -> Path:
    p = PROMPTS_DIR / f"e2e_prompt_ctx_{ctx_size}.txt"
    p.write_text(FIXED_PROMPT_TOKEN * max(prompt_tokens - 1, 0))
    actual = count_tokens(p)
    if actual != prompt_tokens:
        raise RuntimeError(
            f"Prompt token mismatch for ctx={ctx_size}: requested {prompt_tokens}, got {actual}")
    return p


def completion_args(ctx_size: int, prompt_path: Path) -> list[str]:
    return [
        str(COMPLETION_BIN),
        "-m", str(MODEL_PATH),
        "-f", str(prompt_path),
        "-n", str(GEN_TOKENS),
        "-c", str(ctx_size),
        "-ngl", str(GPU_LAYERS),
        "-t", str(THREADS),
        "-tb", str(THREADS_BATCH),
        "-b", str(BATCH_SIZE),
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


def parse_eval_tps(output: str) -> tuple[float, int]:
    """common_perf_print eval time -> (eval_tok_per_sec, generated_tokens).
    This path is GPU-synced (t_eval_us set inside synchronize()); it is the old
    Figure 1 metric, kept as a cross-check."""
    m_ms = re.search(r"^common_perf_print:\s+eval time =\s+([0-9.]+) ms /\s+(\d+) runs",
                     output, re.MULTILINE)
    if not m_ms:
        return 0.0, 0
    eval_ms = float(m_ms.group(1))
    runs = int(m_ms.group(2))
    return (runs / (eval_ms / 1000.0) if eval_ms > 0 else 0.0), runs


def run_ctx(ctx_size: int) -> dict:
    prompt_tokens = ctx_size - GEN_TOKENS
    prompt_path = ensure_prompt(ctx_size, prompt_tokens)
    e2e_path = E2E_DIR / f"baseline_ctx_{ctx_size}_e2e.csv"
    log_path = LOGS_DIR / f"e2e_baseline_ctx_{ctx_size}.log"
    e2e_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["UNIKV_POLICY"]  = "0"          # baseline, measured identically
    env["UNIKV_E2E_LOG"] = str(e2e_path)

    t0 = time.time()
    print(f"  ctx={ctx_size:>6}  prompt={prompt_tokens:>6}  decode={GEN_TOKENS}  "
          f"start={time.strftime('%H:%M:%S')} ...", flush=True)
    completed = subprocess.run(
        completion_args(ctx_size, prompt_path), cwd=LLAMA_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    log_path.write_text(completed.stdout)
    dur = time.time() - t0
    if completed.returncode != 0:
        raise RuntimeError(f"ctx={ctx_size} exited rc={completed.returncode}; see {log_path}")

    rows = list(csv.DictReader(e2e_path.open())) if e2e_path.exists() else []
    e2e_tps = float(rows[-1]["tok_per_sec"]) if rows else 0.0
    dec_tokens = int(rows[-1]["decode_tokens"]) if rows else 0
    eval_tps, eval_runs = parse_eval_tps(completed.stdout)

    flag = e2e_tps >= CEILING_TOK_PER_SEC
    print(f"      dur={dur:6.1f}s  e2e={e2e_tps:6.2f} tok/s  "
          f"(eval-time xcheck={eval_tps:6.2f})  decode_tokens={dec_tokens}"
          + ("  [FLAG >= CEILING]" if flag else ""), flush=True)
    return {
        "context_length_tokens": ctx_size,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": dec_tokens,
        "e2e_tok_per_sec": round(e2e_tps, 3),
        "eval_time_tok_per_sec_xcheck": round(eval_tps, 3),
        "duration_s": round(dur, 2),
        "e2e_csv": str(e2e_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "over_ceiling": flag,
    }


def main() -> None:
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        must_exist(p)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    E2E_DIR.mkdir(parents=True, exist_ok=True)

    print("Figure 1 baseline re-run (end-to-end harness, UNIKV_POLICY=0)")
    print(f"  contexts={CONTEXT_SIZES}  decode={GEN_TOKENS}  prompt=ctx-{GEN_TOKENS}\n")

    results = [run_ctx(c) for c in CONTEXT_SIZES]

    fields = ["context_length_tokens", "prompt_tokens", "decode_tokens",
              "e2e_tok_per_sec", "eval_time_tok_per_sec_xcheck", "duration_s",
              "e2e_csv", "log"]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}")
    print(f"{'ctx':>7} | {'e2e_tok/s':>10} | {'eval xcheck':>11}")
    print("-" * 36)
    for r in results:
        print(f"{r['context_length_tokens']:>7} | {r['e2e_tok_per_sec']:>10.2f} | "
              f"{r['eval_time_tok_per_sec_xcheck']:>11.2f}")

    flagged = [r for r in results if r["over_ceiling"]]
    if flagged:
        print(f"\n[FLAG] {len(flagged)} context(s) at/over the {CEILING_TOK_PER_SEC} "
              f"tok/s ceiling -- investigate, DO NOT report:")
        for r in flagged:
            print(f"    ctx={r['context_length_tokens']}: {r['e2e_tok_per_sec']:.2f} tok/s")
    else:
        print(f"\nSanity: all contexts below the {CEILING_TOK_PER_SEC} tok/s ceiling. OK.")
    print("Done.")


if __name__ == "__main__":
    main()
