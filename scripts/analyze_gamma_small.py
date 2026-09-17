#!/usr/bin/env python3
"""gamma from directly measured small targets vs gamma from the published extrapolation.

gamma is currently obtained by extrapolating the affine fit back to n_spill = 0
and subtracting the no-spill mean, from blocks whose lowest spilling target is
512. That is an extrapolation into a region with no data, computed from the
noisiest cells in the block.

This compares, per model and per tier:
  gamma_local      plateau of the small targets MINUS THAT BLOCK'S OWN no-spill
                   reference. Slope-free, because the small region is measured
                   flat (see the 1B block); a local OLS slope there is noise.
  gamma_published  published global-fit intercept MINUS THAT BLOCK'S OWN
                   no-spill reference.

BOTH ARE WITHIN-BLOCK DIFFERENCES, so each block's level offset cancels before
the two are compared. No level is ever spliced across sessions.

The affine-form check is then simply whether the two agree: if the extrapolation
lands where the plateau actually is, the form holds to zero; if not, the
published gamma is partly a fit artifact.
"""
import csv, math, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "stress_results"
PAIRS = {  # model -> (small-target block, published block)
    "Qwen2.5 7B":   ("qwensmall_isochronal_both_modes.csv",  "qwen_isochronal_both_modes.csv"),
    "Llama 3.2 3B": ("l3bsmall_isochronal_both_modes.csv",   "l3b_isochronal_both_modes.csv"),
    "Llama 3.1 8B": ("llamasmall_isochronal_both_modes.csv", "b2_isochronal_both_modes.csv"),
}
SMALL_MAX = 250.0   # n_spill; the measured plateau region (targets 8/32/128 -> 87.5/111.5/207.5)

def load(p):
    d = {}
    for r in csv.DictReader((ROOT / p).open()):
        if r["rc"] != "0" or not r["ms_mean"]:
            continue
        if r["flash_attn"] != "disabled":
            raise SystemExit(f"STOP: flash attention on in {p}")
        d.setdefault((r["spill_dev"], float(r["n_spill_mean"])), []).append(
            (float(r["ms_mean"]), float(r["ms_median"])))
    return d

def ols(x, y):
    n = len(x); mx, my = st.mean(x), st.mean(y); sxx = sum((v-mx)**2 for v in x)
    b = sum((u-mx)*(v-my) for u, v in zip(x, y))/sxx; a = my - b*mx
    s2 = sum((v-(a+b*u))**2 for u, v in zip(x, y))/(n-2)
    # SE of the intercept
    se_a = math.sqrt(s2*(1.0/n + mx*mx/sxx))
    return b*1000, a, se_a, math.sqrt(s2/sxx)*1000

for model, (sp, pp) in PAIRS.items():
    S, P = load(sp), load(pp)
    print(f"\n{'='*78}\n{model}\n  small block: {sp}\n  published  : {pp}\n{'='*78}")
    for dev, nm in (("0", "CPU-pinned"), ("1", "device-visible")):
        for j, resp in ((0, "ms_mean"), (1, "ms_median")):
            # --- small block: own reference + plateau ---
            ref_s = [v[j] for k, vs in S.items() if k[0] == dev and k[1] == 0.0 for v in vs]
            sm = {k[1]: [v[j] for v in vs] for k, vs in S.items()
                  if k[0] == dev and 0 < k[1] <= SMALL_MAX}
            if not ref_s or not sm: continue
            plateau_vals = [v for vs in sm.values() for v in vs]
            g_loc = st.mean(plateau_vals) - st.mean(ref_s)
            se_loc = math.hypot(st.stdev(plateau_vals)/math.sqrt(len(plateau_vals)),
                                st.stdev(ref_s)/math.sqrt(len(ref_s)))
            # --- published block: own reference + global fit ---
            ref_p = [v[j] for k, vs in P.items() if k[0] == dev and k[1] == 0.0 for v in vs]
            big = [(k[1], v) for k, vs in P.items() if k[0] == dev and k[1] > 250 for v in vs]
            b, a, se_a, se_b = ols([x for x, _ in big], [v[j] for _, v in big])
            g_pub = a - st.mean(ref_p)
            se_pub = math.hypot(se_a, st.stdev(ref_p)/math.sqrt(len(ref_p)))
            d = g_loc - g_pub; se_d = math.hypot(se_loc, se_pub)
            print(f"  {nm:15} [{resp:9}]")
            print(f"     gamma_local     {g_loc:7.3f} +/- {se_loc:.3f} ms   "
                  f"(plateau {st.mean(plateau_vals):.3f} - own no-spill {st.mean(ref_s):.3f})")
            print(f"     gamma_published {g_pub:7.3f} +/- {se_pub:.3f} ms   "
                  f"(intercept {a:.3f} - own no-spill {st.mean(ref_p):.3f}, "
                  f"delta {b:.3f} us/cell)")
            print(f"     difference      {d:+7.3f} +/- {se_d:.3f} ms  t = {d/se_d:+6.2f}"
                  f"   {'AGREE -> gamma rescued' if abs(d/se_d) < 2 else 'DISAGREE -> affine form does not reach zero'}")
            if abs(g_pub) > 1e-9:
                print(f"     relative error of the published value: {100*d/g_pub:+.1f}%")
    # is the small region flat?
    print(f"  -- flatness of the small region (this block) --")
    for dev, nm in (("0", "CPU-pinned"), ("1", "device-visible")):
        xs = sorted(k[1] for k in S if k[0] == dev and 0 < k[1] <= SMALL_MAX)
        if len(xs) < 2: continue
        lo = [v[0] for v in S[(dev, xs[0])]]; hi = [v[0] for v in S[(dev, xs[-1])]]
        d = st.mean(hi) - st.mean(lo)
        se = math.hypot(st.stdev(hi)/math.sqrt(len(hi)), st.stdev(lo)/math.sqrt(len(lo)))
        print(f"     {nm:15} {xs[0]:.1f} -> {xs[-1]:.1f} cells: {d:+.3f} +/- {se:.3f} ms, "
              f"t = {d/se:+.2f}  {'flat' if abs(d/se) < 2 else 'NOT flat'}")
