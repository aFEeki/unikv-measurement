#!/usr/bin/env python3
"""Corrected recall-cost fit — subtract the measured thermal drift, per target.

Every quantity here comes from stress_results/thermal_control_block.csv, which
contains BOTH the isochronal arms and their matched constant-work controls in one
randomised complete block. Nothing is combined across blocks.

  thermal(dev, target, round) = ctrl_ms(dev, target, round) - ctrl_ms(dev, 0, round)

      the control does identical work per step at every target; the only thing
      that differs is how long the machine has been under load when its band is
      measured. So this difference is drift, not work.

  corrected(run) = iso_ms(run) - thermal(dev, target, round of that run)

      paired WITHIN round, so a round that ran warm corrects with its own control.

The fit then follows artifacts/b2_cooled/REGRESSION_SPEC.txt: per-run
observations, measured n_spill_mean as the regressor, target > 0 only, gamma =
intercept minus that arm's own no-spill reference.
"""

import csv, math, statistics as st, sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else "stress_results/thermal_control_block.csv")


def ols(xs, ys):
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return b, a, math.sqrt(s2 / sxx), math.sqrt(s2), n


def main():
    rows = [r for r in csv.DictReader(SRC.open()) if r["rc"] == "0" and r["ms_mean"]]
    fa = {r["flash_attn"] for r in rows}
    if fa != {"disabled"}:
        raise SystemExit(f"STOP: flash attention not off everywhere: {fa}")
    rounds = sorted({int(r["round"]) for r in rows})
    print(f"{SRC.name}: {len(rows)} usable runs, rounds {rounds}, flash attention off\n")

    ctrl, iso = {}, {}
    for r in rows:
        key = (r["spill_dev"], int(r["target"]), int(r["round"]))
        (ctrl if r["kind"] == "ctrl" else iso).setdefault(key, []).append(r)

    # ---- the drift the control actually saw --------------------------------
    print("=" * 82)
    print("MEASURED THERMAL DRIFT — constant work, only elapsed time differs")
    print("=" * 82)
    drift = {}
    for dev, name in (("0", "CPU-pinned"), ("1", "device-visible")):
        print(f"\n-- {name} (control band times matched to this arm's timeline) --")
        print(f"   {'target':>7} {'n':>2} {'ctrl ms/step':>13} {'drift vs target 0':>19}")
        base = {}
        for rnd in rounds:
            k = (dev, 0, rnd)
            if k in ctrl: base[rnd] = st.mean([float(x["ms_mean"]) for x in ctrl[k]])
        for t in (0, 512, 1024, 2048, 4096, 8192):
            vals, ds = [], []
            for rnd in rounds:
                k = (dev, t, rnd)
                if k in ctrl and rnd in base:
                    v = st.mean([float(x["ms_mean"]) for x in ctrl[k]])
                    vals.append(v); ds.append(v - base[rnd])
                    drift[(dev, t, rnd)] = v - base[rnd]
            if vals:
                sd = st.stdev(ds) if len(ds) > 1 else 0.0
                print(f"   {t:7d} {len(vals):2d} {st.mean(vals):13.3f} "
                      f"{st.mean(ds):+13.3f} ms  (sd {sd:.3f})")

    # ---- raw vs corrected fits ---------------------------------------------
    print("\n" + "=" * 82)
    print("RECALL-COST FIT, RAW vs THERMALLY CORRECTED (per REGRESSION_SPEC)")
    print("=" * 82)
    res = {}
    for dev, name in (("0", "CPU-pinned"), ("1", "device-visible")):
        xs, y_raw, y_cor = [], [], []
        base_raw, base_cor = [], []
        for (d, t, rnd), rs in iso.items():
            if d != dev: continue
            for r in rs:
                dr = drift.get((d, t, rnd), 0.0)
                if t > 0:
                    xs.append(float(r["n_spill_mean"]))
                    y_raw.append(float(r["ms_mean"]))
                    y_cor.append(float(r["ms_mean"]) - dr)
                else:
                    base_raw.append(float(r["ms_mean"]))
                    base_cor.append(float(r["ms_mean"]) - dr)
        if len(xs) < 3: continue
        braw, araw, seraw, sdraw, n = ols(xs, y_raw)
        bcor, acor, secor, sdcor, _ = ols(xs, y_cor)
        gr, gc = araw - st.mean(base_raw), acor - st.mean(base_cor)
        res[dev] = (braw * 1000, bcor * 1000, seraw * 1000, secor * 1000, gr, gc)
        print(f"\n-- {name} --  (n={n}, df={n-2})")
        print(f"   RAW        t = {araw:7.3f} ms + {braw*1000:.4f} us/cell "
              f"(SE {seraw*1000:.4f}, resid SD {sdraw:.3f})  gamma {gr:.3f} ms")
        print(f"   CORRECTED  t = {acor:7.3f} ms + {bcor*1000:.4f} us/cell "
              f"(SE {secor*1000:.4f}, resid SD {sdcor:.3f})  gamma {gc:.3f} ms")
        share = 100 * (braw - bcor) / braw
        print(f"   -> thermal share of delta: {share:.2f}%   "
              f"(delta falls {braw*1000:.4f} -> {bcor*1000:.4f} us/cell)")

    # ---- the headline: is the residual near the ~6% the arithmetic predicts?
    print("\n" + "=" * 82)
    print("IS THE RESIDUAL NEAR THE ~6% THE PAPER'S ARITHMETIC BOUNDS IT AT?")
    print("=" * 82)
    for dev, name in (("0", "CPU-pinned"), ("1", "device-visible")):
        if dev not in res: continue
        braw, bcor, seraw, secor, gr, gc = res[dev]
        share = 100 * (braw - bcor) / braw
        verdict = ("AT the bound" if 4 <= share <= 8 else
                   "BELOW the bound — the bound was conservative" if share < 4 else
                   "ABOVE the bound — the bound was too generous")
        print(f"  {name:15s} thermal share {share:5.2f}%  -> {verdict}")
        print(f"                  delta {braw:.4f} -> {bcor:.4f} us/cell, "
              f"an UPPER BOUND becomes an ESTIMATE")
    if "0" in res and "1" in res:
        b0r, b0c = res["0"][0], res["0"][1]
        b1r, b1c = res["1"][0], res["1"][1]
        ser = math.hypot(res["0"][2], res["1"][2]); sec = math.hypot(res["0"][3], res["1"][3])
        print(f"\n  slope separability  RAW: t = {(b0r-b1r)/ser:.2f}   "
              f"CORRECTED: t = {(b0c-b1c)/sec:.2f}")
        print(f"  device-visible reduction in delta  RAW {100*(1-b1r/b0r):.1f}%   "
              f"CORRECTED {100*(1-b1c/b0c):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
