#!/usr/bin/env python3
"""Interleaved A/B/C for the device-visible spilled tier (B1 verification).

The +24% and the resulting sub-10% exactness premium against H2O currently rest
on single uncooled runs taken sequentially, on a machine that was in active use.
Sequential single runs cannot distinguish a treatment effect from drift between
them.

This does not fix that by cooling -- it fixes it by COUNTERBALANCING. Three arms
are run in a rotated order every round, so each arm occupies each position in the
round roughly equally and any monotone drift (thermal or load) cancels across
rounds instead of loading onto whichever arm ran last. That makes the DIFFERENCES
robust without waiting for an idle machine; the absolute levels are still
uncooled and are not protocol numbers.

Arms, all at C=1024, -fa off, uninstrumented:
  A  p3 CPU-pinned spilled tier   (the published configuration)
  B  p3 device-visible tier       (UNIKV_SPILL_DEV=1)
  C  p4 H2O                       (the comparator the premium is measured against)

Round orders rotate: ABC, BCA, CAB, CBA.
"""

import csv
import datetime
import os
import re
import statistics as st
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
ART_DIR     = ROOT / "artifacts" / "b1_gpu_tier"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTX           = 1024
PROMPT_TOKENS = 512
GEN_TOKENS    = int(os.environ.get("UNIKV_AB_GEN", "1536"))
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
SPILL_CAP     = 4096
CEILING       = 58.0
FIXED_TOKEN   = " token"
COOLDOWN_S    = int(os.environ.get("UNIKV_AB_COOLDOWN", "0"))

# tag -> (policy, spill_dev, label)
ARMS = {
    "A_p3_cpu": (3, "0", "p3 spilled tier pinned to CPU (published)"),
    "B_p3_dev": (3, "1", "p3 spilled tier device-visible"),
    "C_p4_h2o": (4, "0", "H2O fixed-budget eviction"),
}
ROUNDS = [
    ["A_p3_cpu", "B_p3_dev", "C_p4_h2o"],
    ["B_p3_dev", "C_p4_h2o", "A_p3_cpu"],
    ["C_p4_h2o", "A_p3_cpu", "B_p3_dev"],
    ["C_p4_h2o", "B_p3_dev", "A_p3_cpu"],
]


def ensure_prompt() -> Path:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPTS_DIR / f"prompt_{PROMPT_TOKENS}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(PROMPT_TOKENS - 1, 0))
        out = subprocess.run(
            [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
             "--show-count", "--log-disable"],
            cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", check=True).stdout
        m = re.search(r"Total number of tokens:\s*(\d+)", out)
        if not m or int(m.group(1)) != PROMPT_TOKENS:
            raise RuntimeError("prompt token mismatch")
    return path


def run_one(prompt: Path, tag: str, rnd: int, pos: int) -> dict:
    policy, spill_dev, _ = ARMS[tag]
    e2e_csv = LOGS_DIR / f"e2e_{tag}_r{rnd}.csv"
    log_txt = LOGS_DIR / f"gen_{tag}_r{rnd}.txt"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": spill_dev,
                "UNIKV_E2E_LOG": str(e2e_csv),
                "UNIKV_SPILL_CAP": str(SPILL_CAP)})
    env.pop("UNIKV_LOG", None)      # uninstrumented: this produces a tok/s number

    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    log_txt.write_text("=== STDOUT ===\n" + d.stdout + "\n=== STDERR ===\n" + d.stderr[-8000:])

    dec, tps = None, None
    if e2e_csv.exists():
        rows = list(csv.DictReader(e2e_csv.open()))
        if rows:
            dec = int(rows[-1]["decode_tokens"])
            tps = float(rows[-1]["tok_per_sec"])

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    on_dev = bool(re.search(r"spilled tier -> .*device-visible", d.stderr))
    return {
        "arm": tag, "round": rnd, "position": pos, "rc": d.returncode,
        "decode_tokens": dec, "tok_per_sec": tps,
        "flash_attn": mfa.group(1) if mfa else "UNPARSED",
        "tier_device_visible": on_dev,
        "duration_s": round(dur, 1),
        "t_start": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
    }


def main() -> int:
    for d in (RESULTS_DIR, LOGS_DIR, PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt()

    print(f"interleaved A/B/C: {len(ROUNDS)} rounds x {len(ARMS)} arms, "
          f"C={CTX}, {GEN_TOKENS} decode, -fa off, uninstrumented")
    print("  counterbalanced order, not cooled: differences are robust to drift, "
          "absolute levels are not protocol numbers")
    for i, r in enumerate(ROUNDS, 1):
        print(f"  round {i}: {' '.join(r)}")
    print()

    rows = []
    for rnd, order in enumerate(ROUNDS, 1):
        for pos, tag in enumerate(order, 1):
            if COOLDOWN_S:
                time.sleep(COOLDOWN_S)
            print(f"  r{rnd} p{pos} {tag:10s} ...", end="", flush=True)
            r = run_one(prompt, tag, rnd, pos)
            rows.append(r)
            print(f" {r['tok_per_sec']} tok/s  dec={r['decode_tokens']} "
                  f"fa={r['flash_attn']} dev={r['tier_device_visible']} ({r['duration_s']}s)")

    out = RESULTS_DIR / "b1_interleaved_ab.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 74)
    print(f"{'arm':10s} {'n':>2s} {'mean':>7s} {'sd':>6s} {'min':>7s} {'max':>7s}  role")
    print("-" * 74)
    means = {}
    for tag in ARMS:
        v = [r["tok_per_sec"] for r in rows if r["arm"] == tag and r["tok_per_sec"]]
        if v:
            means[tag] = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"{tag:10s} {len(v):2d} {means[tag]:7.2f} {sd:6.2f} {min(v):7.2f} "
                  f"{max(v):7.2f}  {ARMS[tag][2]}")
    print("=" * 74)

    # paired within-round differences: immune to between-round drift
    print("\nwithin-round paired differences (drift-immune):")
    for x, y in (("B_p3_dev", "A_p3_cpu"), ("C_p4_h2o", "B_p3_dev"),
                 ("C_p4_h2o", "A_p3_cpu")):
        diffs = []
        for rnd in range(1, len(ROUNDS) + 1):
            a = next((r["tok_per_sec"] for r in rows if r["arm"] == x and r["round"] == rnd), None)
            b = next((r["tok_per_sec"] for r in rows if r["arm"] == y and r["round"] == rnd), None)
            if a and b:
                diffs.append(100.0 * (a / b - 1.0))
        if diffs:
            sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
            print(f"  {x} vs {y}: {st.mean(diffs):+6.2f}% +/- {sd:.2f} "
                  f"(per-round: {' '.join(f'{d:+.1f}' for d in diffs)})")

    flags = [f"{r['arm']} r{r['round']}: fa={r['flash_attn']}" for r in rows
             if r["flash_attn"] != "disabled"]
    flags += [f"{r['arm']} r{r['round']}: decoded {r['decode_tokens']}" for r in rows
              if r["decode_tokens"] != GEN_TOKENS]
    flags += [f"{r['arm']} r{r['round']}: tier_device_visible mismatch" for r in rows
              if r["tier_device_visible"] != (ARMS[r["arm"]][1] == "1")]
    flags += [f"{r['arm']} r{r['round']}: {r['tok_per_sec']} >= {CEILING}" for r in rows
              if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING]
    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
