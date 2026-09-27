#!/usr/bin/env python3
"""Demotion-copy gate: how much of the fixed charge is the per-step copy.
COOLED, IDLE MACHINE.

Every decode step on the plateau demotes one cell, and the copy runs per layer:
K as one row, V (transposed with flash attention off) as one strided column,
which on a Metal buffer without a 2-D copy becomes one call per V element per
layer, each way. The copy therefore sits inside the fixed charge on both tiers,
and nothing so far has separated it.

UNIKV_COPY_REPS (fork) runs the copy 0, 1 or 2 times per demotion. At 2 the
repeat rewrites identical bytes, so the output stays exact; at 0 the spilled
tier keeps zeros and the output is invalid, an instrument only. Two readings of
the same quantity:
  by difference   step time at reps 1 minus reps 0, and reps 2 minus reps 1,
                  within this block; the two agree if the copy is additive
  directly        copy_us, the wall time of the copy loop in each step,
                  written by the fork to the step log

Cells, per model: both tiers at target 128 (the plateau) with reps 0, 1 and 2,
plus each tier's no-spill reference (target 0), so the plateau jump is
re-measured inside the same block. Same launch, prompts, instrumentation and
burst as the plateau blocks (run_b2_cooled.py block 1). Randomized complete
block, 3 rounds, 200 s cooldowns, flash attention verified per run.

  UNIKV_MODEL=<gguf> UNIKV_CG_TAG=llama python3 scripts/run_copy_gate.py
  UNIKV_CG_SMOKE=1 ...   one run per cell, no cooldown, not for the paper
"""

import csv
import hashlib
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = os.environ.get("UNIKV_CG_TAG", "llama")
os.environ["UNIKV_TAG"] = f"copy_{TAG}"            # logs go to artifacts/copy_<tag>_cooled
sys.path.insert(0, str(ROOT / "scripts"))
import run_b2_cooled as b2                           # noqa: E402  (launch, ensure_prompt)

# the plateau blocks' own prompt files, so the prompts are byte-identical
PROMPT_SRC = {"llama": "llamasmall_cooled", "qwen": "qwensmall_cooled",
              "l3b": "l3bsmall_cooled", "l1b": "l1b_cooled"}
b2.PROMPTS_DIR = ROOT / "artifacts" / PROMPT_SRC[TAG] / "prompts"

TARGET   = 128
ROUNDS   = 3
COOLDOWN = int(os.environ.get("UNIKV_CG_COOLDOWN", "200"))
SEED     = 20260926
SMOKE    = os.environ.get("UNIKV_CG_SMOKE", "0") == "1"
# smoke only: raw logit dumps for the reps 1 / reps 2 bitwise comparison
LOGIT_DIR = Path(os.environ.get("UNIKV_CG_LOGIT_DIR", ROOT / "artifacts" / f"copy_{TAG}_logits"))
BURST, WARM = b2.ISO_BURST, b2.ISO_WARMUP

CELLS = ([(dev, 0, 1) for dev in ("0", "1")] +
         [(dev, TARGET, reps) for dev in ("0", "1") for reps in (0, 1, 2)])

OUT = ROOT / "stress_results" / (f"copy_gate_{TAG}_SMOKE.csv" if SMOKE
                                 else f"copy_gate_{TAG}.csv")
FIELDS = ["order_idx", "spill_dev", "target", "reps", "round", "prompt_tokens",
          "rc", "measured_steps", "n_spill_mean", "ms_mean", "ms_median", "ms_sd",
          "copy_us_mean", "copy_us_median", "out_hash", "flash_attn",
          "tier_device_visible", "duration_s", "t_start"]


def output_hash(log_txt: Path) -> str:
    """Hash of the generated text, to check that reps 2 reproduces reps 1."""
    s = log_txt.read_text(errors="replace")
    body = s.split("=== STDOUT ===\n", 1)[-1].split("\n=== STDERR ===", 1)[0]
    body = "\n".join(l for l in body.splitlines() if not l.startswith("UNIKV_E2E"))
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def one(dev, target, reps, rd):
    ptok = b2.ISO_CTX + target if target > 0 else b2.ISO_CTX // 2
    prompt = b2.ensure_prompt(ptok)
    tag = f"cg_dev{dev}_n{target}_r{reps}_t{rd}"
    os.environ["UNIKV_COPY_REPS"] = str(reps)
    if SMOKE and target > 0:                         # bitwise exactness check, smoke only
        os.environ["UNIKV_LOGIT_LOG"] = str(LOGIT_DIR / f"logits_{tag}.bin")
    try:
        r = b2.launch(prompt, b2.ISO_CTX, 3, dev, BURST, tag, True,
                      max(target + 2048, 4096))
    finally:
        os.environ.pop("UNIKV_COPY_REPS", None)
        os.environ.pop("UNIKV_LOGIT_LOG", None)

    srows = list(csv.DictReader(r["step_csv"].open())) if r["step_csv"].exists() else []
    burst = srows[-BURST:] if len(srows) >= BURST else srows
    meas = burst[WARM:]
    ms = [float(x["ttft_ms"]) for x in meas]
    nsp = [int(x.get("n_spilled", 0) or 0) for x in meas]
    cus = [int(x["copy_us"]) for x in meas if x.get("copy_us") not in (None, "")]
    return {
        "spill_dev": dev, "target": target, "reps": reps, "round": rd,
        "prompt_tokens": ptok, "rc": r["rc"], "measured_steps": len(meas),
        "n_spill_mean": round(st.mean(nsp), 1) if nsp else None,
        "ms_mean": round(st.mean(ms), 3) if ms else None,
        "ms_median": round(st.median(ms), 3) if ms else None,
        "ms_sd": round(st.stdev(ms), 3) if len(ms) > 1 else 0.0,
        "copy_us_mean": round(st.mean(cus), 1) if cus else None,
        "copy_us_median": st.median(cus) if cus else None,
        "out_hash": output_hash(b2.LOGS_DIR / f"gen_{tag}.txt"),
        "flash_attn": r["flash_attn"],
        "tier_device_visible": r["tier_device_visible"],
        "duration_s": r["duration_s"],
    }


def main():
    rng, plan = random.Random(SEED), []
    for rd in range(1, ROUNDS + 1):
        c = CELLS[:]; rng.shuffle(c)
        plan += [(dev, t, reps, rd) for dev, t, reps in c]
    if SMOKE:
        plan = [(dev, t, reps, 1) for dev, t, reps in CELLS]
    b2.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if SMOKE:
        LOGIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"copy gate [{TAG}] model={b2.MODEL_PATH.name}")
    print(f"  {len(plan)} runs, complete block {ROUNDS} x {len(CELLS)}, seed {SEED}, "
          f"{0 if SMOKE else COOLDOWN}s cooldowns, smoke={SMOKE}\n", flush=True)

    with OUT.open("w", newline="") as fh:
        csv.writer(fh).writerow(FIELDS)
    rows, flags = [], []
    for i, (dev, target, reps, rd) in enumerate(plan, 1):
        if COOLDOWN and not SMOKE:
            time.sleep(COOLDOWN)
        t0 = time.time()
        rec = one(dev, target, reps, rd)
        rec.update(order_idx=i, t_start=b2.iso_now(t0))
        rows.append(rec)
        with OUT.open("a", newline="") as fh:
            csv.writer(fh).writerow([rec[k] for k in FIELDS])
        print(f"  [{i:2d}/{len(plan)}] dev={dev} n={target:3d} reps={reps} r{rd}  "
              f"{rec['ms_mean']} ms/step  copy {rec['copy_us_mean']} us  "
              f"n_spill={rec['n_spill_mean']}  out={rec['out_hash']}  "
              f"fa={rec['flash_attn']} ({rec['duration_s']}s)", flush=True)

        where = f"dev{dev} n{target} reps{reps} r{rd}"
        if rec["flash_attn"] != "disabled":
            flags.append(f"{where}: flash_attn={rec['flash_attn']}")
        if rec["rc"] != 0 or rec["measured_steps"] != BURST - WARM:
            flags.append(f"{where}: rc={rec['rc']} steps={rec['measured_steps']}")
        if target > 0 and rec["tier_device_visible"] != (dev == "1"):
            flags.append(f"{where}: tier mode mismatch")
        if (target == 0 or reps == 0) and (rec["copy_us_mean"] or 0) > 5:
            flags.append(f"{where}: copy_us {rec['copy_us_mean']} where no copy runs")

    # ---- summary: the copy by difference and directly, per tier ----
    print("\n" + "=" * 72)
    for dev, name in (("0", "CPU-pinned"), ("1", "device-visible")):
        g = lambda t, r, k="ms_mean": [x[k] for x in rows if x["spill_dev"] == dev
                                       and x["target"] == t and x["reps"] == r and x[k] is not None]
        ref, r0, r1, r2 = g(0, 1), g(TARGET, 0), g(TARGET, 1), g(TARGET, 2)
        if not (ref and r0 and r1 and r2):
            continue
        m = st.mean
        print(f"  {name}: ref {m(ref):.3f}  reps0 {m(r0):.3f}  reps1 {m(r1):.3f}  "
              f"reps2 {m(r2):.3f} ms")
        print(f"    plateau jump (reps1 - ref) {m(r1)-m(ref):+.3f} ms; copy by difference "
              f"{m(r1)-m(r0):+.3f} (1-0), {m(r2)-m(r1):+.3f} (2-1) ms; "
              f"copy_us at reps 1 / 2: {m(g(TARGET, 1, 'copy_us_mean')):.0f} / "
              f"{m(g(TARGET, 2, 'copy_us_mean')):.0f}")
        h1 = {x["out_hash"] for x in rows if x["spill_dev"] == dev and x["target"] == TARGET and x["reps"] == 1}
        h2 = {x["out_hash"] for x in rows if x["spill_dev"] == dev and x["target"] == TARGET and x["reps"] == 2}
        if h1 != h2 or len(h1) != 1:
            flags.append(f"dev{dev}: reps 2 output differs from reps 1 ({h1} vs {h2})")
    if SMOKE:
        # the repeat rewrites identical bytes, so reps 2 must match reps 1 bit for bit;
        # reps 0 leaves zeros in the tier and must not
        for dev in ("0", "1"):
            f = lambda r: (LOGIT_DIR / f"logits_cg_dev{dev}_n{TARGET}_r{r}_t1.bin").read_bytes()
            try:
                same12, same01 = f(1) == f(2), f(0) == f(1)
            except FileNotFoundError as e:
                flags.append(f"dev{dev}: logit dump missing ({e.filename})"); continue
            print(f"  dev{dev} logits: reps2 == reps1 bitwise: {same12};  reps0 == reps1: {same01}")
            if not same12:
                flags.append(f"dev{dev}: reps 2 logits differ from reps 1")
    print("FLAGS:", flags or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
