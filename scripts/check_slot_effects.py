#!/usr/bin/env python3
"""The three slot checks the threats section establishes, applied to any contrast.

Written for the drain-alpha block's bad draw and reused here. Given a block CSV,
two arm names and a context, report the A-B contrast four ways:

  RAW        difference of cell means
  DETRENDED  remove cell means, regress residuals on execution slot to get a
             drift in tok/s per slot, then correct each cell mean by
             drift * (that cell's mean slot - the grand mean slot)
  LOCAL      strictly local contrasts, immune to ANY monotone drift: for each run
             of arm A, linearly interpolate arm B's value at A's slot from the
             two bracketing B runs (nearest-neighbour where A falls outside the
             bracket), and difference there
  DRIFT      the fitted drift itself, with its standard error, so the reader can
             see whether there was anything to correct

  python3 scripts/check_slot_effects.py <csv> <armA> <armB> [ctx ...]
"""

import csv, math, statistics as st, sys
from pathlib import Path


def load(path, arm, ctx):
    out = []
    for r in csv.DictReader(Path(path).open()):
        if r["arm"] == arm and int(r["ctx"]) == ctx and r["rc"] == "0" and r["tok_per_sec"]:
            out.append((int(r["order_idx"]), float(r["tok_per_sec"])))
    return sorted(out)


def all_rows(path, ctx):
    out = {}
    for r in csv.DictReader(Path(path).open()):
        if int(r["ctx"]) == ctx and r["rc"] == "0" and r["tok_per_sec"]:
            out.setdefault(r["arm"], []).append((int(r["order_idx"]), float(r["tok_per_sec"])))
    return out


def drift_fit(cells):
    """Residuals after removing cell means, regressed on slot."""
    xs, ys = [], []
    for arm, v in cells.items():
        m = st.mean([t for _, t in v])
        for s, t in v:
            xs.append(s); ys.append(t - m)
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    sd = st.stdev(resid)
    return b, sd / math.sqrt(sxx), mx


def interp(bvals, slot):
    """Arm B's value at `slot`, linearly interpolated between bracketing runs."""
    below = [(s, t) for s, t in bvals if s <= slot]
    above = [(s, t) for s, t in bvals if s >= slot]
    if below and above:
        s0, t0 = below[-1]; s1, t1 = above[0]
        if s0 == s1: return t0
        return t0 + (t1 - t0) * (slot - s0) / (s1 - s0)
    return (below[-1][1] if below else above[0][1])


def main():
    path, armA, armB = sys.argv[1], sys.argv[2], sys.argv[3]
    ctxs = [int(x) for x in sys.argv[4:]] or [1024, 2048]
    print(f"{Path(path).name}:  {armA}  -  {armB}\n")
    for ctx in ctxs:
        A, B = load(path, armA, ctx), load(path, armB, ctx)
        if not A or not B: continue
        cells = all_rows(path, ctx)
        mA, mB = st.mean([t for _, t in A]), st.mean([t for _, t in B])
        seA = st.stdev([t for _, t in A]) / math.sqrt(len(A)) if len(A) > 1 else 0.0
        seB = st.stdev([t for _, t in B]) / math.sqrt(len(B)) if len(B) > 1 else 0.0
        raw, se_raw = mA - mB, math.hypot(seA, seB)

        b, se_b, grand = drift_fit(cells)
        sA = st.mean([s for s, _ in A]); sB = st.mean([s for s, _ in B])
        det = (mA - b * (sA - grand)) - (mB - b * (sB - grand))

        loc = [t - interp(B, s) for s, t in A]
        loc2 = [interp(A, s) - t for s, t in B]
        locs = loc + loc2
        mloc = st.mean(locs)
        seloc = st.stdev(locs) / math.sqrt(len(locs)) if len(locs) > 1 else 0.0

        print(f"-- C={ctx} --")
        print(f"   {armA:14s} n={len(A)} mean {mA:7.3f}  slots {[s for s,_ in A]}")
        print(f"   {armB:14s} n={len(B)} mean {mB:7.3f}  slots {[s for s,_ in B]}")
        print(f"   DRIFT      {b:+.4f} +/- {se_b:.4f} tok/s per slot "
              f"(t = {b/se_b if se_b else float('nan'):+.2f})")
        print(f"   RAW        {raw:+.3f} +/- {se_raw:.3f} tok/s")
        print(f"   DETRENDED  {det:+.3f} tok/s   (shift {det-raw:+.3f})")
        print(f"   LOCAL      {mloc:+.3f} +/- {seloc:.3f} tok/s   (shift {mloc-raw:+.3f})")
        agree = all(abs(v - raw) <= 2 * max(se_raw, 1e-9) for v in (det, mloc))
        signs = {raw > 0, det > 0, mloc > 0}
        print(f"   -> {'AGREE within 2 SE of raw' if agree else 'DISAGREE with raw'}"
              f"; sign {'STABLE' if len(signs) == 1 else 'AT RISK'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
