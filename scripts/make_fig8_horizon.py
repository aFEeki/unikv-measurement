#!/usr/bin/env python3
"""Figure 8 for the measurement paper — how token-identity fails, not just when.

Finding 2's central positive claim is currently two tables. Table 3 gives the
first-divergence index and nothing else, which makes the two prompts look like
the same result at different offsets. They are not, and the difference is the
reason the run was not stopped at the first mismatch:

  - on the passkey probe H2O diverges once at token 16 and never recovers. It
    falls into a repetition loop and sits at coincidence-level agreement for the
    remaining 384 tokens.
  - on prose it diverges at token 0, re-synchronises briefly, then decoheres
    gradually: 64.1% -> 37.5% -> 21.9% -> 7.8% across four 128-token windows.
    Both arms are generating the same KIND of text, so they keep colliding on
    boilerplate while the contents drift apart.

Meanwhile both exact-retention arms sit on 1.0 for all 512 tokens on both
prompts, which is the claim the paper's one recommendation rests on.

WHAT IS PLOTTED: CUMULATIVE agreement with the no-eviction reference -- the
fraction of tokens 0..i that matched -- against generated token index. Its
endpoints are exactly the totals quoted in the text (30/512 = 5.9% on passkey,
168/512 = 32.8% on prose), so the curve carries no smoothing parameter and
cannot be tuned.

A trailing-window version was built first and rejected. At a 32-token window the
prose arm spikes back above 70% around token 350, because H2O periodically
re-synchronises on markdown boilerplate. That is real, but it reads as though the
arm recovers, and it contradicts the monotone 128-token decay the text reports.
Cumulative agreement shows the sustained divergence without inviting that
reading; the local re-synchronisation is described in the text instead, where it
can be explained rather than merely seen.

WHY BOTH RETENTION ARMS ARE ONE LINE: they are token-identical to each other and
to the reference, so three curves would overplot exactly. Drawing one and saying
so is honest; drawing three suggests three measurements agreeing to within
something, and there is no "within" here.

PROVENANCE: sequences are the full sampled-ID traces written by
scripts/run_token_horizon.py, one file per (prompt, arm). Every claim in the
figure is re-derived from those files at build time and checked against the
locked values below, which are the ones printed in the paper.

Writes figures/fig8_horizon.{pdf,png} and copies the PDF to
paper/UNIKV-MEASUREMENT/.
"""

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parents[1]
SEQ_DIR   = ROOT / "artifacts" / "token_horizon" / "sequences"
FIG_DIR   = ROOT / "figures"
PAPER_DIR = ROOT / "paper" / "UNIKV-MEASUREMENT"

WINDOW  = 32
HORIZON = 512

# ---- LOCKED (from artifacts/token_horizon/token_horizon_analysis.txt) ------
# prompt -> arm -> (first divergence index or None, tokens matching reference)
LOCKED = {
    "passkey": {"p3_cpu_c1024":  (None, 512),
                "p3_dev_c1024":  (None, 512),
                "p4_h2o_c1024":  (16,    30),
                "ctl_ref_c4096": (None, 262)},
    "prose":   {"p3_cpu_c1024":  (None, 512),
                "p3_dev_c1024":  (None, 512),
                "p4_h2o_c1024":  (0,    168),
                "ctl_ref_c4096": (None, 262)},
}
# 128-token window agreement for H2O, as quoted in Section 5
LOCKED_WINDOWS = {"passkey": [14.1, 3.1, 3.1, 3.1],
                  "prose":   [64.1, 37.5, 21.9, 7.8]}

DARK    = "#222222"
GRAY    = "#888888"
C_KEEP  = "#1f4e79"   # exact retention (both tiers)
C_PASS  = "#c1272d"   # H2O, passkey probe
C_PROSE = "#e08214"   # H2O, prose


def die(msg):
    raise SystemExit(f"STOP (provenance mismatch): {msg}")


def load(prompt, arm):
    p = SEQ_DIR / f"{prompt}_{arm}.txt"
    if not p.exists():
        die(f"missing trace {p.name}")
    return p.read_text().split()


def agreement(arm_ids, ref_ids):
    n = min(len(arm_ids), len(ref_ids))
    return [arm_ids[i] == ref_ids[i] for i in range(n)]


def verify():
    """Re-derive every locked number from the traces; refuse to plot on drift."""
    out = {}
    for prompt, arms in LOCKED.items():
        ref = load(prompt, "ref_c8192")
        if len(ref) != HORIZON:
            die(f"{prompt}: reference is {len(ref)} tokens, expected {HORIZON}")
        out[prompt] = {}
        for arm, (want_div, want_match) in arms.items():
            a = agreement(load(prompt, arm), ref)
            got_div = next((i for i, x in enumerate(a) if not x), None)
            if got_div != want_div:
                die(f"{prompt}/{arm}: first divergence {got_div}, locked {want_div}")
            if sum(a) != want_match:
                die(f"{prompt}/{arm}: {sum(a)} matches, locked {want_match}")
            out[prompt][arm] = a

        # the two retention arms must be identical to each other, not merely
        # both identical to the reference -- that is what lets us draw one line
        if load(prompt, "p3_cpu_c1024") != load(prompt, "p3_dev_c1024"):
            die(f"{prompt}: the two retention tiers are not token-identical")

        # 128-token windows quoted in the text
        h = out[prompt]["p4_h2o_c1024"]
        for w, want in enumerate(LOCKED_WINDOWS[prompt]):
            seg = h[w * 128:(w + 1) * 128]
            got = round(100.0 * sum(seg) / len(seg), 1)
            if abs(got - want) > 0.05:
                die(f"{prompt}: window {w} agreement {got}%, locked {want}%")
    return out


def cumulative(flags):
    """Fraction of tokens 0..i that matched the reference, per index."""
    xs, ys, run = [], [], 0
    for i, ok in enumerate(flags):
        run += ok
        xs.append(i)
        ys.append(run / (i + 1))
    return xs, ys


def main():
    data = verify()

    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.4, 2.5))

    xs, ys = cumulative(data["passkey"]["p3_cpu_c1024"])
    ax.plot(xs, ys, color=C_KEEP, lw=2.0, zorder=4)
    ends = {}
    for prompt, colour in (("prose", C_PROSE), ("passkey", C_PASS)):
        xs, ys = cumulative(data[prompt]["p4_h2o_c1024"])
        ax.plot(xs, ys, color=colour, lw=1.4, zorder=3)
        ends[prompt] = ys[-1]

    for prompt, colour in (("passkey", C_PASS), ("prose", C_PROSE)):
        d = LOCKED[prompt]["p4_h2o_c1024"][0]
        ax.plot([d], [1.02], marker="v", ms=4, color=colour,
                clip_on=False, zorder=5)

    ax.text(HORIZON + 8, 1.0, "exact retention\nboth tiers", color=C_KEEP,
            fontsize=7, va="center", ha="left")
    ax.text(HORIZON + 8, ends["prose"], f"H2O\nprose\n{100*ends['prose']:.1f}%",
            color=C_PROSE, fontsize=7, va="center", ha="left")
    ax.text(HORIZON + 8, ends["passkey"] - 0.02,
            f"H2O\npasskey\n{100*ends['passkey']:.1f}%",
            color=C_PASS, fontsize=7, va="center", ha="left")
    ax.text(4, 1.10, "first divergence: token 0 (prose), token 16 (passkey)",
            fontsize=6.5, color=GRAY, va="bottom")

    ax.set_xlabel("generated token index")
    ax.set_ylabel("cumulative agreement\nwith no-eviction reference")
    ax.set_xlim(-8, HORIZON + 4)
    ax.set_ylim(-0.03, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xticks([0, 128, 256, 384, 512])
    ax.axhline(1.0, color=GRAY, lw=0.5, ls=":", zorder=1)

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig8_horizon.{ext}")
    if PAPER_DIR.exists():
        shutil.copy(FIG_DIR / "fig8_horizon.pdf", PAPER_DIR / "fig8_horizon.pdf")
    print("wrote figures/fig8_horizon.{pdf,png}; all locked values re-derived OK")


if __name__ == "__main__":
    main()
