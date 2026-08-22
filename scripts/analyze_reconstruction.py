#!/usr/bin/env python3
"""Reconstruction acceptance by INTERVAL, not by a point band.

Per-event cost is a difference of two noisy means divided by an event count, so a
point band on it mis-states the test: the estimator's spread can exceed the band
width, which is what made the first check unresolvable. The test is therefore:

  PASS  the block's 95% interval on per-event cost contains the published value
        at BOTH budgets
  FAIL  either budget excludes the published value WITH TIGHT BARS

and the C=2048/C=1024 per-event RATIO is judged the same way against the
published 1.961 and against the 2.000 that "the K-shift re-encodes the whole
resident cache" predicts. Section 8's attribution rests on that ratio.

Intervals by Monte Carlo over the two arm means (normal, mean and standard error
from the block), which propagates the reciprocal and the difference correctly
without a delta-method approximation.

  python3 scripts/analyze_reconstruction.py <csv> [<csv> ...]
"""

import csv, math, statistics as st, sys
import numpy as np
from pathlib import Path

GEN = 2048
EVENTS = {1024: 1535, 2048: 511}          # measured by the census, not derived
PUB_PER_EVENT = {1024: 0.923, 2048: 1.811}
PUB_RECOVERY  = {1024: 1.075, 2048: 0.695}
PUB_RATIO     = 1.961
PRED_RATIO    = 2.000
N_MC = 200000
rng = np.random.default_rng(20260821)


def cells(paths):
    out = {}
    for p in paths:
        for r in csv.DictReader(Path(p).open()):
            if r["rc"] != "0" or not r["tok_per_sec"]: continue
            if r["flash_attn"] != "disabled":
                raise SystemExit(f"STOP: flash attention on in {p}")
            out.setdefault((r["arm"], int(r["ctx"])), []).append(float(r["tok_per_sec"]))
    return out


def per_event_mc(p1, ab, ctx):
    """Monte-Carlo samples of per-event cost in ms."""
    n1, na = len(p1), len(ab)
    m1, s1 = st.mean(p1), (st.stdev(p1)/math.sqrt(n1) if n1 > 1 else 0.0)
    ma, sa = st.mean(ab), (st.stdev(ab)/math.sqrt(na) if na > 1 else 0.0)
    a = rng.normal(m1, s1, N_MC)
    b = rng.normal(ma, sa, N_MC)
    return (1000.0/a - 1000.0/b) * GEN / EVENTS[ctx], m1, ma, s1, sa


def main():
    paths = sys.argv[1:]
    if not paths: raise SystemExit(__doc__)
    c = cells(paths)
    print("Reconstruction acceptance — interval test\n")
    samples, verdicts = {}, []
    for ctx in (1024, 2048):
        k1, ka = ("p1_window", ctx), ("p1_noreencode", ctx)
        if k1 not in c or ka not in c: continue
        s, m1, ma, s1, sa = per_event_mc(c[k1], c[ka], ctx)
        samples[ctx] = s
        lo, hi = np.percentile(s, [2.5, 97.5])
        pub = PUB_PER_EVENT[ctx]
        inside = lo <= pub <= hi
        verdicts.append(inside)
        rec = ma - m1
        rse = math.hypot(s1, sa)
        print(f"C={ctx}   n={len(c[k1])}/{len(c[ka])}   events={EVENTS[ctx]}")
        print(f"  p1_window     {m1:7.3f} +/- {s1:.3f} tok/s (sd {st.stdev(c[k1]):.3f})")
        print(f"  p1_noreencode {ma:7.3f} +/- {sa:.3f} tok/s (sd {st.stdev(c[ka]):.3f})")
        print(f"  recovery      {rec:+.3f} +/- {rse:.3f} tok/s   (published {PUB_RECOVERY[ctx]:+.3f})")
        print(f"  per-event     {s.mean():.3f} ms, 95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"                published {pub:.3f} -> {'INSIDE (pass)' if inside else 'OUTSIDE (fail)'}")
        print()

    if 1024 in samples and 2048 in samples:
        ratio = samples[2048] / samples[1024]
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        print(f"PER-EVENT RATIO  C=2048 / C=1024")
        print(f"  point {np.median(ratio):.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
        for name, v in (("published 1.961", PUB_RATIO), ("whole-cache prediction 2.000", PRED_RATIO)):
            ok = lo <= v <= hi
            print(f"  {name:32s} -> {'INSIDE' if ok else 'OUTSIDE'}")
        print()
        print("  Section 8's attribution needs this ratio near 2.000: the K-shift")
        print("  re-encodes the whole resident cache, so doubling C should double the")
        print("  per-event cost. If the interval excludes 2.000 with tight bars, the")
        print("  linearity result did not replicate and that is a correction to a")
        print("  published claim, not a build problem.")

    print(f"\nOVERALL: {'PASS' if verdicts and all(verdicts) else 'FAIL / INCOMPLETE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
