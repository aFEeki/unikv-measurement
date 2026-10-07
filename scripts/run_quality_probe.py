#!/usr/bin/env python3
"""Earliest passkey probe: retention (policy 2) against the rolling window (policy 1).

Policy 2 is an early retention variant that the paper does not evaluate; the
fork still implements it. This probe and quality_probe.csv record that first
comparison.

Design: long-range passkey retrieval. The prompt is an introduction, a short
filler, the passkey line placed early, a long filler and a closing question
("... The pass key is"). The same prompt is decoded greedily under two arms:

  arm A  UNIKV_POLICY=1, -c 1024   the prompt exceeds the cache, so the rolling
                                   shift evicts the early passkey
  arm B  UNIKV_POLICY=2, -c 4096   the whole sequence fits; nothing is evicted

Metric: exact match, whether the greedy completion contains the passkey. Arm A's
per-step UNIKV_LOG must show shift events, as evidence that eviction happened.

Same input, differing only in (policy, ctx). Results go to quality_results/;
generated text and logs go to artifacts/quality_probe/. Deterministic; reports
no tok/s.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BUILD_DIR      = LLAMA_DIR / "build-m4pro-metal"
BIN_DIR        = BUILD_DIR / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR    = ROOT / "quality_results"
ARTIFACT_DIR   = ROOT / "artifacts" / "quality_probe"
PROMPTS_DIR    = ARTIFACT_DIR / "prompts"
LOGS_DIR       = ARTIFACT_DIR / "logs"

# --- probe content ---------------------------------------------------------
PASSKEY      = "48291"
FILLER       = ("The grass is green. The sky is blue. The sun is yellow. "
                "Here we go. There and back again. ")
PRE_REPEAT   = 7     # keep the passkey early (within about the first 200 tokens)
POST_REPEAT  = 150   # pad the tail so total prompt ~3k tokens
INTRO        = ("There is an important piece of information hidden inside the "
                "text below. Find it and remember it, because you will be asked "
                "about it at the end.\n\n")
PASSKEY_LINE = f"\nThe pass key is {PASSKEY}. Remember it. {PASSKEY} is the pass key.\n\n"
QUESTION     = "\nQuestion: What is the pass key?\nAnswer: The pass key is"

# --- run params ------------------------------------------------------------
GEN_TOKENS   = 32
THREADS      = 10
GPU_LAYERS   = 999
SEED         = 123
# (policy, ctx) arms: same prompt, differ only here.
ARMS         = [(1, 1024), (2, 4096)]


def ensure_paths() -> None:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def count_tokens(text: str, tmp_name: str) -> int:
    tmp = PROMPTS_DIR / tmp_name
    tmp.write_text(text)
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(tmp),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True,
    ).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse token count:\n{out}")
    return int(m.group(1))


def build_prompt() -> tuple[Path, int, int]:
    """Write the probe prompt; return (path, prompt_tokens, passkey_end_pos)."""
    prefix = INTRO + (FILLER * PRE_REPEAT) + PASSKEY_LINE  # up to & incl passkey
    full   = prefix + (FILLER * POST_REPEAT) + QUESTION
    prompt_path = PROMPTS_DIR / "passkey_prompt.txt"
    prompt_path.write_text(full)

    prompt_tokens   = count_tokens(full,   "_count_full.txt")
    passkey_end_pos = count_tokens(prefix, "_count_prefix.txt")
    return prompt_path, prompt_tokens, passkey_end_pos


def completion_args(ctx_size: int, prompt_path: Path) -> list[str]:
    return [
        str(COMPLETION_BIN),
        "-m",   str(MODEL_PATH),
        "-f",   str(prompt_path),
        "-n",   str(GEN_TOKENS),
        "-c",   str(ctx_size),
        "-ngl", str(GPU_LAYERS),
        "-t",   str(THREADS),
        "-fa",  "on",
        "-fit", "off",
        "--temp", "0",
        "--seed", str(SEED),
        "--no-warmup",
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",
    ]


def read_shift_events(csv_path: Path) -> tuple[int, str]:
    """Return (max shift_events, first step where shift_events>0 or '')."""
    if not csv_path.exists():
        return 0, ""
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    max_se = max((int(r.get("shift_events", 0) or 0) for r in rows), default=0)
    first  = next((r["step"] for r in rows if int(r.get("shift_events", 0) or 0) > 0), "")
    return max_se, first


def run_arm(policy: int, ctx: int, prompt_path: Path) -> dict:
    tag       = f"policy{policy}_ctx{ctx}"
    step_csv  = LOGS_DIR / f"unikv_log_{tag}.csv"
    out_log   = LOGS_DIR / f"gen_{tag}.txt"
    step_csv.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_LOG": str(step_csv)})

    print(f"  arm: policy={policy} ctx={ctx} ...", flush=True)
    completed = subprocess.run(
        completion_args(ctx, prompt_path),
        cwd=LLAMA_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace",
    )
    gen_text = completed.stdout
    out_log.write_text("=== STDOUT (generated) ===\n" + gen_text +
                       "\n=== STDERR (tail) ===\n" + completed.stderr[-4000:])

    found = PASSKEY in gen_text
    max_se, first_shift = read_shift_events(step_csv)
    snippet = " ".join(gen_text.split())[:160]

    return {
        "policy": policy, "ctx": ctx, "gen_tokens": GEN_TOKENS,
        "passkey": PASSKEY, "passkey_found": int(found),
        "shift_events": max_se, "first_shift_step": first_shift,
        "rc": completed.returncode, "output_snippet": snippet,
    }


def main() -> None:
    ensure_paths()
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        must_exist(p)

    prompt_path, prompt_tokens, passkey_end_pos = build_prompt()
    small_ctx = min(c for _, c in ARMS)
    print("Passkey retrieval probe")
    print(f"  prompt tokens        : {prompt_tokens}")
    print(f"  passkey ends at ~tok : {passkey_end_pos} (early)")
    print(f"  arms                 : {ARMS}")

    # Sanity: arm A must overflow (evict), arm B must fit, and the passkey must
    # fall outside arm A's retained tail window, so it is actually evicted.
    big_ctx = max(c for _, c in ARMS)
    checks = {
        "arm A overflows (prompt > small ctx)":        prompt_tokens > small_ctx,
        "arm B fits (prompt+gen < big ctx)":           prompt_tokens + GEN_TOKENS < big_ctx,
        "passkey outside arm A retained tail window":  passkey_end_pos < (prompt_tokens - small_ctx),
    }
    print("  design sanity:")
    for k, v in checks.items():
        print(f"    [{'ok' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("Design preconditions not met — adjust PRE/POST_REPEAT or ctx.")

    rows = [run_arm(policy, ctx, prompt_path) for policy, ctx in ARMS]

    for r in rows:
        r["prompt_tokens"] = prompt_tokens
        r["passkey_end_pos"] = passkey_end_pos

    fields = ["policy", "ctx", "prompt_tokens", "gen_tokens", "passkey",
              "passkey_end_pos", "passkey_found", "shift_events",
              "first_shift_step", "rc", "output_snippet"]
    out_csv = RESULTS_DIR / "quality_probe.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    for r in rows:
        verdict = "RETRIEVED" if r["passkey_found"] else "LOST"
        print(f"  policy={r['policy']} ctx={r['ctx']:>4}  passkey={verdict:9} "
              f"shift_events={r['shift_events']:>4} "
              f"first_shift={r['first_shift_step'] or '-'}  rc={r['rc']}")
        print(f"      out: {r['output_snippet']!r}")
    print(f"\nCSV: {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
