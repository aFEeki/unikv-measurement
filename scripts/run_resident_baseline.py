#!/usr/bin/env python3
"""Resident-attention baseline for the per-cell term. COOLED, IDLE MACHINE.

The question: is delta, the per-cell cost of exact retention, a cost of
retention at all, or just what the unmodified runtime pays to attend over one
more token with the fused kernel off? Upstream runs elsewhere in the archive put
that resident cost near 2 us/cell, close to delta_dev = 2.153, but they come from
other blocks, other context ranges and single uncooled runs. This measures it
under the isochronal design itself.

Arms, per model:
  p0_resident  upstream (policy 0), flash attention off, C = 16384 so the whole
               context stays resident. Prompts of 512, 1536, 2048, 3072, 5120
               and 9216 tokens: the SAME prompt files the two-tier blocks used,
               so the total context at each point matches a two-tier target of
               0, 512, 1024, 2048, 4096 and 8192.
  p3_dev       exact retention, device-visible tier, C = 1024, targets 512..8192.
               Only with UNIKV_RB_PAIRED=1. It puts the two-tier arm in the SAME
               block as its baseline, so the excess is a within-block contrast
               rather than a comparison across blocks run weeks apart.

Same instrumentation as the two-tier blocks: per-step wall clock from the step
log, 128-step burst, first 32 dropped. Randomised complete block, 3 rounds, 200 s
cooldowns, flash attention verified per run from its own log.

  UNIKV_MODEL=<gguf> UNIKV_RB_TAG=llama UNIKV_RB_PAIRED=1 \\
      python3 scripts/run_resident_baseline.py
"""

import csv
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = os.environ.get("UNIKV_RB_TAG", "llama")
os.environ["UNIKV_TAG"] = f"resid_{TAG}"          # logs go to artifacts/resid_<tag>_cooled
sys.path.insert(0, str(ROOT / "scripts"))
import run_b2_cooled as b2                          # noqa: E402  (launch, ensure_prompt)

# Reuse the prompt files of the two-tier block for this model, so both arms see
# byte-identical prompts.
PROMPT_SRC = {"llama": "b2_cooled", "qwen": "qwen_cooled",
              "l3b": "l3b_cooled", "l1b": "l1b_cooled"}
b2.PROMPTS_DIR = ROOT / "artifacts" / PROMPT_SRC[TAG] / "prompts"

PAIRED   = os.environ.get("UNIKV_RB_PAIRED", "0") == "1"
# the spilling targets each model's own two-tier block used
TARGETS  = ([0, 512, 2048, 8192] if TAG == "l1b"
            else [0, 512, 1024, 2048, 4096, 8192])
RES_CTX  = 16384
ISO_CTX  = 1024
ROUNDS   = 3
COOLDOWN = int(os.environ.get("UNIKV_RB_COOLDOWN", "200"))
SEED     = 20260923
BURST, WARM = b2.ISO_BURST, b2.ISO_WARMUP

SMOKE = os.environ.get("UNIKV_RB_SMOKE", "0") == "1"
OUT = ROOT / "stress_results" / (f"resident_baseline_{TAG}_SMOKE.csv" if SMOKE
                                 else f"resident_baseline_{TAG}.csv")
FIELDS = ["order_idx", "arm", "policy", "ctx", "target", "prompt_tokens", "round",
          "rc", "measured_steps", "kv_used_mean", "n_spill_mean", "context_mean",
          "ms_mean", "ms_median", "ms_sd", "flash_attn", "duration_s", "t_start"]


def prompt_tokens(target):
    return ISO_CTX + target if target > 0 else ISO_CTX // 2


def one(arm, target):
    ptok = prompt_tokens(target)
    prompt = b2.ensure_prompt(ptok)
    if arm == "p0_resident":
        pol, dev, ctx = 0, "0", RES_CTX
    else:
        pol, dev, ctx = 3, "1", ISO_CTX
    tag = f"{arm}_n{target}"
    r = b2.launch(prompt, ctx, pol, dev, BURST, tag, True, max(target + 2048, 4096))
    srows = list(csv.DictReader(r["step_csv"].open())) if r["step_csv"].exists() else []
    burst = srows[-BURST:] if len(srows) >= BURST else srows
    meas = burst[WARM:]
    ms = [float(x["ttft_ms"]) for x in meas]
    kv = [int(x["kv_used"]) for x in meas]
    ns = [int(x.get("n_spilled", 0) or 0) for x in meas]
    ctxm = [a + b for a, b in zip(kv, ns)]
    return {
        "arm": arm, "policy": pol, "ctx": ctx, "target": target,
        "prompt_tokens": ptok, "rc": r["rc"], "measured_steps": len(meas),
        "kv_used_mean": round(st.mean(kv), 1) if kv else None,
        "n_spill_mean": round(st.mean(ns), 1) if ns else None,
        "context_mean": round(st.mean(ctxm), 1) if ctxm else None,
        "ms_mean": round(st.mean(ms), 3) if ms else None,
        "ms_median": round(st.median(ms), 3) if ms else None,
        "ms_sd": round(st.stdev(ms), 3) if len(ms) > 1 else 0.0,
        "flash_attn": r["flash_attn"], "duration_s": r["duration_s"],
    }


def ols(xs, ys):
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return b, a, (s2 / sxx) ** 0.5


def main():
    cells = [("p0_resident", t) for t in TARGETS]
    if PAIRED:
        cells += [("p3_dev", t) for t in TARGETS if t > 0]
    rng, plan = random.Random(SEED), []
    for rd in range(1, ROUNDS + 1):
        c = cells[:]; rng.shuffle(c)
        plan += [(a, t, rd) for a, t in c]
    if SMOKE:                       # one run per arm, no cooldown, not for the paper
        plan = [("p0_resident", 512, 1)] + ([("p3_dev", 512, 1)] if PAIRED else [])
    b2.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"resident baseline [{TAG}] model={b2.MODEL_PATH.name}")
    print(f"  {len(plan)} runs, complete block {ROUNDS} x {len(cells)}, "
          f"seed {SEED}, {COOLDOWN}s cooldowns, paired={PAIRED}\n", flush=True)

    with OUT.open("w", newline="") as fh:
        csv.writer(fh).writerow(FIELDS)
    rows, flags = [], []
    for i, (arm, target, rd) in enumerate(plan, 1):
        if COOLDOWN and not SMOKE:
            time.sleep(COOLDOWN)
        t0 = time.time()
        rec = one(arm, target)
        rec.update(order_idx=i, round=rd, t_start=b2.iso_now(t0))
        rows.append(rec)
        with OUT.open("a", newline="") as fh:
            csv.writer(fh).writerow([rec[k] for k in FIELDS])
        print(f"  [{i:2d}/{len(plan)}] {arm:11s} target={target:5d} r{rd}  "
              f"context={rec['context_mean']}  {rec['ms_mean']} ms/step  "
              f"rc={rec['rc']} fa={rec['flash_attn']} ({rec['duration_s']}s)", flush=True)
        if rec["flash_attn"] != "disabled":
            flags.append(f"{arm} n{target} r{rd}: flash_attn={rec['flash_attn']}")
        if rec["rc"] != 0 or rec["measured_steps"] != BURST - WARM:
            flags.append(f"{arm} n{target} r{rd}: rc={rec['rc']} steps={rec['measured_steps']}")

    print("\n" + "=" * 72)
    for arm in ("p0_resident", "p3_dev"):
        pts = [r for r in rows if r["arm"] == arm and r["ms_mean"] and r["target"] >= 512]
        if len(pts) < 3:
            continue
        b, a, se = ols([r["context_mean"] for r in pts], [r["ms_mean"] for r in pts])
        print(f"  {arm:11s} slope over targets >= 512: {b*1000:.4f} +/- {se*1000:.4f} "
              f"us per context cell (n={len(pts)})")
    print("FLAGS:", flags or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
