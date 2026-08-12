#!/usr/bin/env python3
"""B2 — protocol numbers for the two spilled-tier modes. OVERNIGHT, IDLE MACHINE.

Nothing from B1 goes in the paper: it was smoke. This produces the
protocol-grade numbers. B2 does NOT re-measure what B1 settled (the ceilings,
the device-memory accounting, the 65k capacity workload).

BLOCK 1 — isochronal recall cost, both modes.
  gamma and delta per mode, and specifically whether delta DIFFERS between them.
  Targets n_spill in {512, 1024, 2048, 4096, 8192} x {CPU-pinned,
  device-visible}, plus a no-spill reference per mode. 3 trials each, EXCEPT
  n_spill=4096 CPU-pinned which gets 5: that point ranged 44.0-60.9 ms in the
  earlier sweep where every other point repeated under 0.5 ms, and background
  machine activity is a plausible cause. This either deletes the mystery or
  confirms it on a quiet machine.

  Reports per mode: affine fit, slope standard error, residual SD. Then tests
  whether the two slopes are separable given those standard errors. "delta fell
  by half" has to survive that test; if it does not, the analysis says the
  direction is unresolved rather than quoting a difference the fits cannot
  support.

  Also checks that the two no-spill references agree. With nothing spilled the
  tier is inactive and the modes should be identical; a disagreement would mean
  the buffer allocation does something at idle, which would need explaining.

  This block is INSTRUMENTED by necessity -- per-step wall clock IS the
  measurement here. It reports ms/step, not a tok/s headline, so the
  uninstrumented rule (which exists to protect tok/s numbers) is not violated.

BLOCK 2 — policy comparison, five arms, C in {1024, 2048}, 3 trials.
  upstream (halts) / rolling window / H2O / exact retention CPU-pinned /
  exact retention device-visible. Supersedes the 8-arm block; not spliced
  against it or against the interleaved A/B.

  The CPU-pinned arm is POSITION-BALANCED rather than randomised: it drifted
  -23% across the A/B session against -6% and -10% for the GPU-resident arms, so
  an unlucky draw would land the most thermally fragile arm in the trough. Its
  six runs are spread one per sixth of the block; every other arm is randomised
  normally. Uninstrumented.

Protocol: seed 123, temp 0, greedy, EOS disabled, flash attention off
everywhere and verified per run from its own log, 200 s cooldowns.

Run:  python3 scripts/run_b2_cooled.py             (both blocks, ~5 h)
      UNIKV_B2=1 python3 scripts/run_b2_cooled.py  (block 1 only, ~2.5 h)
      UNIKV_B2=2 python3 scripts/run_b2_cooled.py  (block 2 only, ~2.1 h)

Before running: machine idle, screensaver off, Spotlight and Time Machine
paused. Contention lands asymmetrically on the CPU-pinned arm, which biases
toward confirming the hypothesis.
"""

import csv
import datetime
import os
import random
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
ART_DIR     = ROOT / "artifacts" / "b2_cooled"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

THREADS     = 10
GPU_LAYERS  = 999
BATCH       = 512
UBATCH      = 512
SEED        = 123
CEILING     = 58.0
FIXED_TOKEN = " token"

WHICH      = os.environ.get("UNIKV_B2", "1,2")
COOLDOWN_S = int(os.environ.get("UNIKV_B2_COOLDOWN", "200"))
SHUFFLE    = 20260807

# ---- block 1 ---------------------------------------------------------------
ISO_CTX     = 1024
ISO_TARGETS = [0, 512, 1024, 2048, 4096, 8192]
ISO_BURST   = 128
ISO_WARMUP  = 32          # drops the one-off graph-realloc step after prefill
ISO_TRIALS  = 3
ISO_EXTRA   = {("0", 4096): 5}   # (spill_dev, target) -> trial count override

# ---- block 2 ---------------------------------------------------------------
POL_CTXS      = [1024, 2048]
POL_TRIALS    = 3
PROMPT_TOKENS = 512
GEN_TOKENS    = 2048
BALANCED_ARM  = "p3_cpu"
# arm -> (policy, spill_dev, completes_budget, role)
POLICY_ARMS = {
    "p0_upstream": (0, "0", False, "upstream, halts at cache-full"),
    "p1_window":   (1, "0", True,  "rolling window (context shift)"),
    "p4_h2o":      (4, "0", True,  "H2O fixed-budget eviction"),
    "p3_cpu":      (3, "0", True,  "exact retention, CPU-pinned tier"),
    "p3_dev":      (3, "1", True,  "exact retention, device-visible tier"),
}


def ensure_prompt(n: int) -> Path:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPTS_DIR / f"prompt_{n}tok.txt"
    if not path.exists():
        path.write_text(FIXED_TOKEN * max(n - 1, 0))
        out = subprocess.run(
            [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
             "--show-count", "--log-disable"],
            cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", check=True).stdout
        m = re.search(r"Total number of tokens:\s*(\d+)", out)
        if not m or int(m.group(1)) != n:
            raise RuntimeError(f"prompt token mismatch for {n}: "
                               f"got {m.group(1) if m else '?'}")
    return path


def iso_now(ts):
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def ols(xs, ys):
    """slope, intercept, se(slope), residual sd."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s = (sum(r * r for r in resid) / (n - 2)) ** 0.5
    return b, a, s / sxx ** 0.5, s


def launch(prompt, ctx, policy, dev, gen, tag, steplog, spill_cap):
    e2e_csv  = LOGS_DIR / f"e2e_{tag}.csv"
    step_csv = LOGS_DIR / f"step_{tag}.csv"
    log_txt  = LOGS_DIR / f"gen_{tag}.txt"
    e2e_csv.unlink(missing_ok=True)
    step_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(gen), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": dev, "UNIKV_SPILL_CAP": str(spill_cap),
                "UNIKV_E2E_LOG": str(e2e_csv)})
    env.pop("UNIKV_LOG", None)
    env.pop("UNIKV_H2O_TRACE", None)
    if steplog:
        env["UNIKV_LOG"] = str(step_csv)

    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    log_txt.write_text("=== STDOUT ===\n" + d.stdout +
                       "\n=== STDERR ===\n" + d.stderr[-12000:])

    dec, tps = None, None
    if e2e_csv.exists():
        rows = list(csv.DictReader(e2e_csv.open()))
        if rows:
            dec = int(rows[-1]["decode_tokens"])
            tps = float(rows[-1]["tok_per_sec"])
    mfa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    return {
        "rc": d.returncode, "decode_tokens": dec, "tok_per_sec": tps,
        "flash_attn": mfa.group(1) if mfa else "UNPARSED",
        "tier_device_visible": bool(re.search(r"spilled tier -> .*device-visible", d.stderr)),
        "step_csv": step_csv, "duration_s": round(dur, 1),
    }


# ============================ BLOCK 1 =======================================
def block1():
    print("== BLOCK 1: isochronal recall cost, both tier modes ==")
    plan = []
    for dev in ("0", "1"):
        for target in ISO_TARGETS:
            n = ISO_EXTRA.get((dev, target), ISO_TRIALS)
            plan += [(dev, target, t) for t in range(1, n + 1)]
    random.Random(SHUFFLE).shuffle(plan)
    print(f"  {len(plan)} runs, randomized (seed {SHUFFLE}), {COOLDOWN_S}s cooldowns")
    print(f"  n_spill=4096 CPU-pinned gets {ISO_EXTRA[('0', 4096)]} trials "
          f"(re-testing the 44.0-60.9 ms anomaly)")
    print("  instrumented: per-step wall clock is the measurement here\n")

    out = RESULTS_DIR / "b2_isochronal_both_modes.csv"
    fields = ["order_idx", "spill_dev", "target", "trial", "prompt_tokens", "rc",
              "measured_steps", "transient_ms", "n_spill_mean", "ms_mean",
              "ms_median", "ms_sd", "flash_attn", "tier_device_visible",
              "duration_s", "t_start"]
    with out.open("w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    rows, flags = [], []
    for idx, (dev, target, trial) in enumerate(plan, 1):
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        ptok = ISO_CTX + target if target > 0 else ISO_CTX // 2
        prompt = ensure_prompt(ptok)
        tag = f"b2iso_dev{dev}_n{target}_t{trial}"
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] dev={dev} n_spill={target:5d} t{trial} ...",
              end="", flush=True)
        r = launch(prompt, ISO_CTX, 3, dev, ISO_BURST, tag, True,
                   max(target + 2048, 4096))

        srows = list(csv.DictReader(r["step_csv"].open())) if r["step_csv"].exists() else []
        burst = srows[-ISO_BURST:] if len(srows) >= ISO_BURST else srows
        meas  = burst[ISO_WARMUP:]
        ms    = [float(x["ttft_ms"]) for x in meas]
        nsp   = [int(x.get("n_spilled", 0) or 0) for x in meas]
        transient = max((float(x["ttft_ms"]) for x in burst[:ISO_WARMUP]), default=None)

        rec = {
            "order_idx": idx, "spill_dev": dev, "target": target, "trial": trial,
            "prompt_tokens": ptok, "rc": r["rc"], "measured_steps": len(meas),
            "transient_ms": round(transient, 1) if transient else None,
            "n_spill_mean": round(st.mean(nsp), 1) if nsp else None,
            "ms_mean": round(st.mean(ms), 3) if ms else None,
            "ms_median": round(st.median(ms), 3) if ms else None,
            "ms_sd": round(st.stdev(ms), 3) if len(ms) > 1 else 0.0,
            "flash_attn": r["flash_attn"],
            "tier_device_visible": r["tier_device_visible"],
            "duration_s": r["duration_s"], "t_start": iso_now(t0),
        }
        rows.append(rec)
        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([rec[k] for k in fields])
        print(f" {rec['ms_mean']} ms/step (n_spill={rec['n_spill_mean']}) "
              f"({rec['duration_s']}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"dev{dev} n{target} t{trial}: flash_attn={r['flash_attn']}")
        if len(meas) != ISO_BURST - ISO_WARMUP:
            flags.append(f"dev{dev} n{target} t{trial}: {len(meas)} measured steps")
        if r["tier_device_visible"] != (dev == "1" and target > 0):
            # the tier only reports device-visible once something has spilled
            if target > 0:
                flags.append(f"dev{dev} n{target} t{trial}: tier mode mismatch")

    # ---------------- fits ----------------
    print("\n" + "=" * 78)
    fits, bases = {}, {}
    for dev in ("0", "1"):
        label = "device-visible" if dev == "1" else "CPU-pinned"
        pts = {}
        for r in rows:
            if r["spill_dev"] == dev and r["ms_mean"]:
                pts.setdefault(r["target"], []).append(r["ms_mean"])
        print(f"\n-- {label} --")
        print(f"   {'n_spill':>8s} {'n':>2s} {'ms/step':>9s} {'sd':>6s}")
        for n in sorted(pts):
            v = pts[n]
            print(f"   {n:8d} {len(v):2d} {st.mean(v):9.3f} "
                  f"{st.stdev(v) if len(v) > 1 else 0.0:6.3f}")
        bases[dev] = st.mean(pts[0]) if 0 in pts else None
        # fit on every individual spilling run, not on the per-target means
        xs = [r["target"] for r in rows
              if r["spill_dev"] == dev and r["target"] > 0 and r["ms_mean"]]
        ys = [r["ms_mean"] for r in rows
              if r["spill_dev"] == dev and r["target"] > 0 and r["ms_mean"]]
        f = ols(xs, ys)
        if f:
            fits[dev] = f
            b, a, seb, s = f
            print(f"   fit (n={len(xs)}): t = {a:.3f} ms + {b*1000:.4f} us/cell")
            print(f"        slope SE = {seb*1000:.4f} us/cell, residual SD = {s:.3f} ms")
            if bases[dev]:
                print(f"        no-spill {bases[dev]:.3f} ms -> gamma = {a - bases[dev]:.3f} ms")

    # ---------------- the two questions this block exists to answer ----------
    print("\n" + "=" * 78)
    print("Q1. Do the no-spill references agree? (tier inactive -> modes identical)")
    if bases.get("0") and bases.get("1"):
        d0 = [r["ms_mean"] for r in rows if r["spill_dev"] == "0" and r["target"] == 0]
        d1 = [r["ms_mean"] for r in rows if r["spill_dev"] == "1" and r["target"] == 0]
        diff = bases["1"] - bases["0"]
        pooled = ((st.stdev(d0) ** 2 / len(d0) + st.stdev(d1) ** 2 / len(d1)) ** 0.5
                  if len(d0) > 1 and len(d1) > 1 else 0.0)
        print(f"   CPU-pinned {bases['0']:.3f} ms vs device-visible {bases['1']:.3f} ms")
        print(f"   difference {diff:+.3f} ms, SE {pooled:.3f}"
              + (f", t = {diff/pooled:+.2f}" if pooled else ""))
        verdict = ("AGREE — the tier is inert at idle, as expected"
                   if not pooled or abs(diff) < 2 * pooled else
                   "DISAGREE — the buffer allocation is doing something at idle "
                   "and needs explaining")
        print(f"   -> {verdict}")

    print("\nQ2. Are the two slopes separable given their standard errors?")
    if "0" in fits and "1" in fits:
        b0, _, se0, _ = fits["0"]
        b1, _, se1, _ = fits["1"]
        d = (b0 - b1) * 1000
        sed = ((se0 ** 2 + se1 ** 2) ** 0.5) * 1000
        t = d / sed if sed else float("inf")
        print(f"   delta CPU-pinned      = {b0*1000:.4f} +/- {se0*1000:.4f} us/cell")
        print(f"   delta device-visible  = {b1*1000:.4f} +/- {se1*1000:.4f} us/cell")
        print(f"   difference            = {d:+.4f} +/- {sed:.4f} us/cell, t = {t:+.2f}")
        if abs(t) >= 2.0:
            print(f"   -> SEPARABLE at |t| >= 2. delta differs between modes; "
                  f"device-visible is {100*(1-b1/b0):.1f}% lower.")
        else:
            print("   -> NOT SEPARABLE at |t| >= 2. Report that delta also fell "
                  "but that the magnitude is unresolved by these fits; do NOT "
                  "quote a ratio.")

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    print(f"\nwrote {out}\n")


# ============================ BLOCK 2 =======================================
def build_block2_plan():
    """Randomised, except the CPU-pinned arm which is position-balanced."""
    balanced, others = [], []
    for ctx in POL_CTXS:
        for arm in POLICY_ARMS:
            for t in range(1, POL_TRIALS + 1):
                (balanced if arm == BALANCED_ARM else others).append((arm, ctx, t))

    rng = random.Random(SHUFFLE + 1)
    rng.shuffle(others)
    rng.shuffle(balanced)

    total = len(balanced) + len(others)
    seg = total / len(balanced)          # one balanced run per segment
    slots = sorted(rng.randrange(int(i * seg), int((i + 1) * seg))
                   for i in range(len(balanced)))

    plan, bi, oi = [], 0, 0
    for i in range(total):
        if bi < len(balanced) and i in slots:
            plan.append(balanced[bi]); bi += 1
        else:
            plan.append(others[oi]); oi += 1
    plan += balanced[bi:] + others[oi:]
    return plan


def block2():
    print("== BLOCK 2: policy comparison, five arms ==")
    prompt = ensure_prompt(PROMPT_TOKENS)
    plan = build_block2_plan()
    pos = [i for i, (a, _, _) in enumerate(plan, 1) if a == BALANCED_ARM]
    print(f"  {len(plan)} runs, {COOLDOWN_S}s cooldowns")
    print(f"  {BALANCED_ARM} position-balanced at slots {pos} of {len(plan)}; "
          f"all other arms randomised")
    print("  order:", " ".join(f"{a}@{c}t{t}" for a, c, t in plan), "\n")

    out = RESULTS_DIR / "b2_policy_block.csv"
    fields = ["order_idx", "arm", "ctx", "policy", "spill_dev", "trial", "rc",
              "decode_tokens", "tok_per_sec", "flash_attn",
              "tier_device_visible", "duration_s", "t_start"]
    with out.open("w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    vals, flags = {}, []
    for idx, (arm, ctx, trial) in enumerate(plan, 1):
        policy, dev, completes, _ = POLICY_ARMS[arm]
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] {arm:12s} C={ctx:4d} t{trial} ...",
              end="", flush=True)
        r = launch(prompt, ctx, policy, dev, GEN_TOKENS,
                   f"b2pol_{arm}_c{ctx}_t{trial}", False, 8192)
        print(f" {r['tok_per_sec']} tok/s dec={r['decode_tokens']} "
              f"fa={r['flash_attn']} ({r['duration_s']}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"{arm}@{ctx}t{trial}: flash_attn={r['flash_attn']}")
        if completes and r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"{arm}@{ctx}t{trial}: decoded {r['decode_tokens']} != {GEN_TOKENS}")
        if r["tier_device_visible"] != (dev == "1"):
            flags.append(f"{arm}@{ctx}t{trial}: tier mode mismatch")
        if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
            flags.append(f"{arm}@{ctx}t{trial}: {r['tok_per_sec']} >= {CEILING}")
        elif r["tok_per_sec"]:
            vals.setdefault((arm, ctx), []).append(r["tok_per_sec"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, arm, ctx, policy, dev, trial, r["rc"],
                                     r["decode_tokens"], r["tok_per_sec"],
                                     r["flash_attn"], r["tier_device_visible"],
                                     r["duration_s"], iso_now(t0)])

    print("\n" + "=" * 80)
    print(f"{'arm':12s} {'C':>5s} {'n':>2s} {'tokens':>7s} {'mean':>7s} {'sd':>6s}  role")
    print("-" * 80)
    with out.open(newline="") as fh:
        allrows = list(csv.DictReader(fh))
    means = {}
    for ctx in POL_CTXS:
        for arm in POLICY_ARMS:
            v = vals.get((arm, ctx), [])
            toks = {r["decode_tokens"] for r in allrows
                    if r["arm"] == arm and int(r["ctx"]) == ctx}
            if v:
                means[(arm, ctx)] = st.mean(v)
                print(f"{arm:12s} {ctx:5d} {len(v):2d} "
                      f"{(toks.pop() if len(toks) == 1 else '?'):>7s} "
                      f"{means[(arm, ctx)]:7.2f} "
                      f"{st.stdev(v) if len(v) > 1 else 0.0:6.2f}  {POLICY_ARMS[arm][3]}")
        print()
    print("=" * 80)

    print("\nexactness premium (H2O over exact retention), per context:")
    for ctx in POL_CTXS:
        h = means.get(("p4_h2o", ctx))
        for arm in ("p3_cpu", "p3_dev"):
            m = means.get((arm, ctx))
            if h and m:
                print(f"  C={ctx}: {arm:7s} {m:6.2f} vs H2O {h:6.2f} "
                      f"-> {100*(h/m-1):+6.2f}%")

    # position audit for the balanced arm
    bal = [(int(r["order_idx"]), float(r["tok_per_sec"]))
           for r in allrows if r["arm"] == BALANCED_ARM and r["tok_per_sec"]]
    if bal:
        print(f"\n{BALANCED_ARM} position audit (guards against a thermal-trough draw):")
        print("  " + "  ".join(f"slot{o}={v:.2f}" for o, v in sorted(bal)))

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    print(f"\nwrote {out}\n")


def main() -> int:
    for d in (RESULTS_DIR, LOGS_DIR, PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)
    print("B2 cooled blocks — requires an IDLE machine (see module docstring)\n")
    if "1" in WHICH:
        block1()
    if "2" in WHICH:
        block2()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
