#!/usr/bin/env python3
"""Exact retention priced against the resident-attention baseline (Table 3).

Reads, per model:
  stress_results/resident_baseline_<m>.csv   p0_resident and p3_dev in one
                                              randomized complete block
  stress_results/<small>_isochronal_both_modes.csv
                                              the plateau block: targets 0 and
                                              8/32/128 on both tiers

Per-cell terms are OLS slopes over targets >= 512 against total context, both
arms from the same block; the surcharge is their difference, with the two
slope standard errors combined in quadrature.

Fixed charges are the plateau (mean over targets 8/32/128) less that block's
own no-spill mean, less the resident cost of the cells a plateau run holds
beyond its reference: the full 1024-cell window plus ~135 spilled against
~591 in the reference, 568 cells priced at the resident slope. The standard
error combines the plateau mean, the reference mean and 568 x the slope SE.

The paired-block device figure is the gap between the p3_dev and p0_resident
fits at 1024 cells of context, where the retained tier is empty. It is a
cross-check on the plateau estimate, which is the only one the host tier has;
it extends the affine branch below the smallest target, where the plateau
shows the step time does not follow it.

No new runs.
"""

import csv, math, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "stress_results"
MODELS = [  # tag, name, layers, plateau block
    ("llama", "Llama 3.1 8B", 32, "llamasmall"),
    ("qwen",  "Qwen2.5 7B",   28, "qwensmall"),
    ("l3b",   "Llama 3.2 3B", 28, "l3bsmall"),
    ("l1b",   "Llama 3.2 1B", 16, "l1b"),
]
EXTRA_CELLS = 568
SMALL = ("8", "32", "128")


def ols(p, x0=0.0):
    """slope, intercept, slope SE, and the SE of the fitted value at x0"""
    xs, ys = zip(*p); n = len(p); mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in p) / sxx; a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in p) / (n - 2)
    return b, a, math.sqrt(s2 / sxx), math.sqrt(s2 * (1.0 / n + (x0 - mx) ** 2 / sxx))


def se_mean(v):
    return st.stdev(v) / math.sqrt(len(v))


def main():
    for tag, name, L, small in MODELS:
        r = [x for x in csv.DictReader((ROOT / f"resident_baseline_{tag}.csv").open())
             if x["rc"] == "0"]
        if any(x["flash_attn"] != "disabled" for x in r):
            raise SystemExit(f"STOP: flash attention on in resident_baseline_{tag}.csv")
        pts = lambda arm: [(float(x["context_mean"]), float(x["ms_mean"])) for x in r
                           if x["arm"] == arm and int(x["target"]) >= 512]
        br, ar, sbr, sr0 = ols(pts("p0_resident"), 1024)
        bd, ad, sbd, sd0 = ols(pts("p3_dev"), 1024)
        sur, se_sur = bd - br, math.hypot(sbr, sbd)
        paired, se_paired = (ad + bd * 1024) - (ar + br * 1024), math.hypot(sr0, sd0)

        s = [x for x in csv.DictReader((ROOT / f"{small}_isochronal_both_modes.csv").open())
             if x["rc"] == "0"]
        fixed, flat = {}, {}
        for dev, tier in (("1", "device"), ("0", "host")):
            ref = [float(x["ms_mean"]) for x in s if x["spill_dev"] == dev and x["target"] == "0"]
            pl = [float(x["ms_mean"]) for x in s if x["spill_dev"] == dev and x["target"] in SMALL]
            fixed[tier] = (st.mean(pl) - st.mean(ref) - br * EXTRA_CELLS,
                           math.sqrt(se_mean(pl) ** 2 + se_mean(ref) ** 2
                                     + (EXTRA_CELLS * sbr) ** 2))
            lo = [float(x["ms_mean"]) for x in s if x["spill_dev"] == dev and x["target"] == SMALL[0]]
            hi = [float(x["ms_mean"]) for x in s if x["spill_dev"] == dev and x["target"] == SMALL[-1]]
            d, se = st.mean(hi) - st.mean(lo), math.hypot(se_mean(hi), se_mean(lo))
            flat[tier] = (d, se, d / se)

        fd, fh = fixed["device"][0], fixed["host"][0]
        print(f"\n{name}  (L = {L})")
        print(f"  per cell, us:  resident {br*1e3:.3f}  device {bd*1e3:.3f}  "
              f"surcharge {sur*1e3:.3f} +/- {se_sur*1e3:.3f}")
        print(f"                 attention share of device slope {100*br/bd:.1f}%, "
              f"surcharge on attention {100*sur/br:.1f}%")
        for tier in ("device", "host"):
            f, se = fixed[tier]
            d, sed, t = flat[tier]
            print(f"  fixed {tier:6s}  {f:.2f} +/- {se:.2f} ms  ({f/L:.3f} ms per layer);  "
                  f"plateau {SMALL[0]}->{SMALL[-1]}: {d:+.3f} +/- {sed:.3f} ms, t = {t:+.2f}"
                  f"{'' if abs(t) < 2 else '  NOT flat'}")
        print(f"  device tier removes {100*(1-fd/fh):.1f}% of the host fixed charge; "
              f"surcharge reaches the device fixed charge at {fd/sur:,.0f} cells")
        dd = paired - fd; sdd = math.hypot(se_paired, fixed["device"][1])
        print(f"  paired-block device figure {paired:.2f} +/- {se_paired:.2f} ms; "
              f"minus plateau {dd:+.2f} +/- {sdd:.2f}, t = {dd/sdd:+.2f}; "
              f"reduction with it {100*(1-paired/fh):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
