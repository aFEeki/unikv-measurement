#!/usr/bin/env python3
"""UniKV R2 re-runs on policy 3 (lossless spill-and-recall).

Run 1  policy comparison, C in {1024, 2048}: baseline p0 vs UniKV p3, on a
       512-token prompt + 2048-token decode budget. Baseline halts at cache
       full; p3 completes the full budget. Records decoded-token count and
       e2e tok/s for both arms, plus p3 spill events + retained cells.

Run 3  no-overhead reference, C=4096: p3 idle (nothing spills) vs p0 stock,
       2048 decode rows. The real idle overhead of policy 3, replacing 0.77%.

All arms: -fa off, greedy, seed 123, --ignore-eos, single clean process each,
run sequentially so thermal state does not leak between arms.

Does NOT touch the alpha model (alpha=0 everywhere here). Does NOT overwrite the
locked stress_results CSVs -- writes r2_* names.
"""

import csv
import os
import re
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
ART_DIR     = ROOT / "artifacts" / "r2_policy3"
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


def run_arm(prompt: Path, ctx: int, policy: int, tag: str) -> dict:
    step_csv = LOGS_DIR / f"step_{tag}.csv"
    e2e_csv  = LOGS_DIR / f"e2e_{tag}.csv"
    log_txt  = LOGS_DIR / f"gen_{tag}.txt"
    for f in (step_csv, e2e_csv):
        f.unlink(missing_ok=True)

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
        "UNIKV_LOG": str(step_csv),
        "UNIKV_E2E_LOG": str(e2e_csv),
        "UNIKV_SPILL_CAP": str(SPILL_CAP),
        "UNIKV_MEM_BREAKDOWN": "1",
    })

    t0 = time.time()
    print(f"  {tag:20s} (policy {policy} @ C={ctx}) ...", end="", flush=True)
    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    log_txt.write_text("=== STDOUT ===\n" + done.stdout + "\n=== STDERR ===\n" + done.stderr)

    dec_tokens, tok_s = None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec_tokens = int(rows[-1]["decode_tokens"])
            tok_s = float(rows[-1]["tok_per_sec"])

    spill_events = 0
    if step_csv.exists():
        with step_csv.open(newline="") as fh:
            spill_events = sum(1 for r in csv.DictReader(fh)
                               if int(r.get("shift_events", 0) or 0) > 0)

    retained = 0
    rm = re.findall(r"retained=(\d+)", done.stderr)
    if rm:
        retained = int(rm[-1])

    kv_mib = None
    mk = re.search(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", done.stderr)
    if mk:
        kv_mib = float(mk.group(1))

    print(f" dec_tokens={dec_tokens} tok/s={tok_s} spills={spill_events} "
          f"retained={retained} kv={kv_mib}MiB ({dur:.0f}s)")

    flag = tok_s is not None and tok_s >= CEILING
    return {
        "tag": tag, "ctx": ctx, "policy": policy, "rc": done.returncode,
        "decode_tokens": dec_tokens, "tok_per_sec": tok_s,
        "spill_events": spill_events, "retained": retained,
        "kv_device_mib": kv_mib, "ceiling_flag": flag,
    }


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    prompt = ensure_prompt()
    print(f"prompt = {PROMPT_TOKENS} tok, decode budget = {GEN_TOKENS}, -fa off, seed {SEED}\n")

    print("== Run 1: policy comparison p0 vs p3, C in {1024, 2048} ==")
    run1 = []
    for ctx in (1024, 2048):
        run1.append(run_arm(prompt, ctx, 0, f"r1_p0_c{ctx}"))
        run1.append(run_arm(prompt, ctx, 3, f"r1_p3_c{ctx}"))

    print("\n== Run 3: no-overhead reference, C=4096, p3 idle vs p0 ==")
    run3 = []
    run3.append(run_arm(prompt, 4096, 0, "r3_p0_c4096"))
    run3.append(run_arm(prompt, 4096, 3, "r3_p3_c4096"))

    out1 = RESULTS_DIR / "r2_policy_compare.csv"
    with out1.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run1[0].keys()))
        w.writeheader()
        w.writerows(run1)

    out3 = RESULTS_DIR / "r2_no_overhead.csv"
    with out3.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run3[0].keys()))
        w.writeheader()
        w.writerows(run3)

    # no-overhead: idle p3 vs p0 at C=4096
    p0 = next(r for r in run3 if r["policy"] == 0)
    p3 = next(r for r in run3 if r["policy"] == 3)
    overhead = None
    if p0["tok_per_sec"] and p3["tok_per_sec"]:
        overhead = 100.0 * (p0["tok_per_sec"] - p3["tok_per_sec"]) / p0["tok_per_sec"]

    print("\n" + "=" * 72)
    print("Run 1 (token count = generated before halt/budget):")
    for r in run1:
        print(f"  C={r['ctx']:5d} policy {r['policy']}: {r['decode_tokens']:5} tokens "
              f"@ {r['tok_per_sec']} tok/s  spills={r['spill_events']} retained={r['retained']}")
    print(f"\nRun 3 no-overhead @ C=4096: p0 {p0['tok_per_sec']} vs p3 {p3['tok_per_sec']} tok/s "
          f"-> overhead = {overhead:.2f}%" if overhead is not None else "\nRun 3 overhead: n/a")
    print(f"  (p3 spill_events at C=4096 must be 0: {p3['spill_events']})")

    flags = [r["tag"] for r in run1 + run3 if r["ceiling_flag"]]
    if flags:
        print(f"\n!! CEILING BREACH (>= {CEILING} tok/s): {flags} — STOP, investigate")
        return 1
    print(f"\nAll arms < {CEILING} tok/s ceiling.")
    print(f"wrote {out1}\nwrote {out3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
