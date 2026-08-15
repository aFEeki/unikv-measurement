#!/usr/bin/env python3
"""Fit gamma and delta from an isochronal block, per artifacts/b2_cooled/REGRESSION_SPEC.txt.

That spec is the authority on the fit and this script implements it literally, so
a second model's coefficients are produced by the same procedure as the paper's:

  filter     rc == 0 and target > 0   (target == 0 rows are the no-spill
                                       references; they set gamma, they are not
                                       points on the affine branch)
  unit       one RUN is one observation — NOT a fit on the six cell means
  regressor  n_spill_mean, the MEASURED spilled-set size from the run's own log
  response   ms_mean for the published coefficients, ms_median for the
             estimator-sensitivity check
  gamma      intercept minus that arm's OWN no-spill reference

Note that run_b2_cooled.py's inline printout fits against the harness's TARGET
instead, which is why its intercepts differ from the paper's by slope x 79.5
cells. REGRESSION_SPEC explains that; this script deliberately reproduces the
PAPER's version.

  python3 scripts/analyze_isochronal.py stress_results/b2_isochronal_both_modes.csv
  python3 scripts/analyze_isochronal.py <qwen.csv> --compare <llama.csv>

--compare puts two INDEPENDENTLY FITTED blocks side by side and tests the
paper's formulas as predictions. It never pools the rows: the two models are
separate blocks and merging them would destroy exactly the property that makes
this a prediction test.
"""

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

# KiB of KV per cache cell, parsed from each runtime's own startup log
# (llama_kv_cache: size = X MiB (N cells...)) and recorded here so the predicted
# ratios below are traceable rather than assumed.
KV_KIB_PER_CELL = {
    "llama": 128.0,   # 32 layers x (1024+1024) x 2 B  = 131072 B
    "qwen":   56.0,   # 28 layers x ( 512+ 512) x 2 B  =  57344 B
}


def ols(xs, ys):
    """slope, intercept, se(slope), residual sd, n."""
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return b, a, math.sqrt(s2 / sxx), math.sqrt(s2), n


def load(path):
    rows = list(csv.DictReader(Path(path).open()))
    bad = [r for r in rows if r["rc"] != "0"]
    fa = {r["flash_attn"] for r in rows}
    return rows, bad, fa


def fit_block(path, label):
    rows, bad, fa = load(path)
    print(f"\n{'='*78}\n{label}   ({Path(path).name})\n{'='*78}")
    print(f"  {len(rows)} runs, rc!=0: {len(bad)}, flash_attn values: {sorted(fa)}")
    if fa != {"disabled"}:
        raise SystemExit(f"STOP: flash attention not off on every run of {path}: {fa}")

    out = {}
    for dev, name in (("0", "CPU-pinned"), ("1", "device-visible")):
        sel = [r for r in rows if r["spill_dev"] == dev and r["rc"] == "0"]
        spill = [r for r in sel if float(r["target"]) > 0]
        base  = [float(r["ms_mean"]) for r in sel if float(r["target"]) == 0]

        print(f"\n-- {name} --")
        print(f"   {'target':>7} {'n':>2} {'n_spill':>9} {'ms/step':>9} {'sd':>7}")
        by = {}
        for r in sel:
            by.setdefault(int(r["target"]), []).append(r)
        for t in sorted(by):
            v = [float(r["ms_mean"]) for r in by[t]]
            ns = st.mean([float(r["n_spill_mean"]) for r in by[t]])
            print(f"   {t:7d} {len(v):2d} {ns:9.1f} {st.mean(v):9.3f} "
                  f"{(st.stdev(v) if len(v) > 1 else 0.0):7.3f}")

        x = [float(r["n_spill_mean"]) for r in spill]
        b, a, seb, sd, n = ols(x, [float(r["ms_mean"]) for r in spill])
        bm, am, sebm, sdm, _ = ols(x, [float(r["ms_median"]) for r in spill])
        ref = st.mean(base)
        gamma = a - ref
        print(f"   fit on ms_mean   (n={n}, df={n-2}): "
              f"t = {a:.3f} ms + {b*1000:.4f} us/cell")
        print(f"        slope SE {seb*1000:.4f} us/cell, residual SD {sd:.3f} ms")
        print(f"   fit on ms_median (n={n}, df={n-2}): "
              f"t = {am:.3f} ms + {bm*1000:.4f} us/cell")
        print(f"   no-spill reference {ref:.3f} ms (n={len(base)}, "
              f"sd {(st.stdev(base) if len(base)>1 else 0.0):.3f})"
              f"  ->  gamma = {gamma:.3f} ms")
        out[dev] = dict(slope=b*1000, intercept=a, se=seb*1000, sd=sd, n=n,
                        ref=ref, gamma=gamma, slope_med=bm*1000, name=name)

    b0, b1 = out["0"]["slope"], out["1"]["slope"]
    se = math.hypot(out["0"]["se"], out["1"]["se"])
    t = (b0 - b1) / se
    print(f"\n   slope separability: ({b0:.4f} - {b1:.4f}) / {se:.4f} = t = {t:.2f}"
          f"  -> {'SEPARABLE' if abs(t) >= 2 else 'NOT SEPARABLE'}")
    print(f"   device-visible reduction:  gamma {100*(1-out['1']['gamma']/out['0']['gamma']):.1f}%"
          f"   delta(means) {100*(1-b1/b0):.1f}%"
          f"   delta(medians) {100*(1-out['1']['slope_med']/out['0']['slope_med']):.1f}%")
    out["t_sep"] = t
    return out


def compare(new, old, new_key, old_key):
    kn, ko = KV_KIB_PER_CELL[new_key], KV_KIB_PER_CELL[old_key]
    pred = kn / ko
    print(f"\n{'='*78}\nPREDICTION TEST — two independently fitted blocks, never pooled\n{'='*78}")
    print(f"  KV per cell: {old_key} {ko:.1f} KiB, {new_key} {kn:.1f} KiB"
          f"  -> bytes-cost prediction for the delta ratio = {pred:.4f}")
    print(f"  (a per-cell overhead independent of size would predict 1.0)\n")

    print(f"  {'quantity':30} {'%s'%old_key:>12} {'%s'%new_key:>12} {'ratio':>8}  verdict")
    print("  " + "-" * 74)
    for dev in ("0", "1"):
        nm = old[dev]["name"]
        for key, unit in (("slope", "us/cell"), ("gamma", "ms"), ("ref", "ms")):
            o, n = old[dev][key], new[dev][key]
            r = n / o
            if key == "slope":
                d_bytes = abs(r - pred)
                d_flat = abs(r - 1.0)
                verdict = ("tracks BYTES" if d_bytes < d_flat / 2 else
                           "tracks CELL COUNT" if d_flat < d_bytes / 2 else
                           "BETWEEN the two")
            elif key == "gamma":
                verdict = ("model-independent" if 0.8 <= r <= 1.25
                           else "MODEL-DEPENDENT")
            else:
                verdict = ""
            label = f"{nm} {key}" + (f" ({unit})" if unit else "")
            print(f"  {label:30} {o:12.4f} {n:12.4f} {r:8.4f}  {verdict}")
        print()

    print("  device-visible reduction (paper: 69% of gamma, 24.5-30.5% of delta)")
    for tag, blk in ((old_key, old), (new_key, new)):
        print(f"    {tag:6}  gamma {100*(1-blk['1']['gamma']/blk['0']['gamma']):5.1f}%"
              f"   delta(means) {100*(1-blk['1']['slope']/blk['0']['slope']):5.1f}%"
              f"   delta(medians) {100*(1-blk['1']['slope_med']/blk['0']['slope_med']):5.1f}%")

    print(f"\n  slope separability:  {old_key} t = {old['t_sep']:.2f}"
          f"   {new_key} t = {new['t_sep']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--key", default=None,
                    help="model key for the KV-per-cell table; inferred from "
                         "the filename when omitted")
    ap.add_argument("--compare-key", default=None)
    a = ap.parse_args()

    # Infer the label from the filename rather than defaulting to a model
    # name: a hardcoded default silently mislabels the other model's block,
    # which is the one thing this script must never do.
    def infer(path, given):
        if given:
            return given
        stem = Path(path).name.lower()
        for k in KV_KIB_PER_CELL:
            if k in stem:
                return k
        # the paper's own block predates the cross-model work and so carries
        # no model name in its filename
        if stem.startswith("b2_isochronal"):
            return "llama"
        return Path(path).stem

    a.key = infer(a.csv, a.key)
    if a.compare:
        a.compare_key = infer(a.compare, a.compare_key)

    new = fit_block(a.csv, a.key.upper())
    if a.compare:
        old = fit_block(a.compare, a.compare_key.upper())
        compare(new, old, a.key, a.compare_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
