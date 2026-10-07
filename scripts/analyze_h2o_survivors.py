#!/usr/bin/env python3
"""H2O's survivor set at C = 1024: the recent window, the heavy hitters, and the passkey.

Reads:
  quality_results/h2o_survivor_trace_c1024.csv   (position, cell, accumulated
                                                 attention mass) of every cell
                                                 resident after the last eviction
                                                 in the h2o_p4_c1024 arm of
                                                 run_quality_arms_unified.py,
                                                 written through UNIKV_H2O_TRACE
  quality_results/h2o_idle_check.csv             the passkey's token positions and
                                                 its line in the same prompt
                                                 (run_h2o_idle_check.py)

Reports the survivor count and its split into the most recent 512 positions and
the heavy hitters kept from before them; how many survivors come from the first
512 prompt positions; the sink's lead over the next cell; and, for the passkey,
whether every token of its line survives, the rank of each passkey token in the
whole cache, and its mass against the median of the surviving filler.

No new runs.
"""

import csv
import statistics as st
import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1] / "quality_results"
TRACE = ROOT / "h2o_survivor_trace_c1024.csv"
IDLE  = ROOT / "h2o_idle_check.csv"
RECENT = 512
PROMPT_TOKENS = 3834


def main() -> int:
    cells = [(int(r["pos"]), float(r["score"])) for r in csv.DictReader(TRACE.open())]
    idle = next(csv.DictReader(IDLE.open()))
    occ = [[int(p) for p in o.split()] for o in idle["passkey_positions"].split(";")]
    lo, hi = (int(x) for x in idle["passkey_line"].split("-"))

    pos = [p for p, _ in cells]
    last = max(pos)
    recent = [p for p in pos if p > last - RECENT]
    print(f"survivors {len(cells)}: {len(recent)} in the last {RECENT} positions, "
          f"{len(cells) - len(recent)} heavy hitters before them; "
          f"{sum(p < 512 for p in pos)} from the first 512 prompt positions")

    ranked = sorted(cells, key=lambda c: -c[1])
    rank = {p: i + 1 for i, (p, _) in enumerate(ranked)}
    mass = dict(cells)
    print(f"top cells: {[(p, round(s, 1)) for p, s in ranked[:5]]}; "
          f"the sink outweighs the next by {ranked[0][1] / ranked[1][1]:.0f}x")

    line = range(lo, hi + 1)
    missing = [p for p in line if p not in mass]
    print(f"passkey line {lo}-{hi}: {len(line) - len(missing)} of {len(line)} tokens survive")

    keys = {p for o in occ for p in o}
    filler = [s for p, s in cells
              if p < PROMPT_TOKENS and p not in line and p != 0 and p > 50]
    med = st.median(filler)
    for n, o in enumerate(occ, 1):
        for p in o:
            print(f"  mention {n}, position {p}: rank {rank.get(p, 'evicted')} of "
                  f"{len(cells)}, mass {mass.get(p, float('nan')):.1f} = "
                  f"{mass.get(p, float('nan')) / med:.1f}x the surviving filler median")
    print(f"surviving filler median {med:.1f} (positions 51..{PROMPT_TOKENS - 1} outside "
          f"the passkey line, {len(filler)} cells)")
    return 0 if not missing and keys <= mass.keys() else 1


if __name__ == "__main__":
    sys.exit(main())
