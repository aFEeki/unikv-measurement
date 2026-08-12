#!/usr/bin/env python3
"""Is the pipeline-drain cost alpha-independent? The 2x2 the first block skipped.

The paper decomposes the alpha=0 -> alpha=0.25 step into a drain part and a
modeled-transfer part (the 71/29 split) and, in doing so, assumes the drain cost
is the same at every alpha. The first drain-control block never tested that: it
measured drain cost only at alpha=0, and its only alpha=1 point was UNDRAINED.

This block closes the 2x2:

                       drain OFF            drain ON
    alpha = 0.00       a0_nodrain           a0_drain
    alpha = 1.00       a1_nodrain           a1_drain     <- the missing cell

  drain cost at alpha=0  =  a0_nodrain - a0_drain
  drain cost at alpha=1  =  a1_nodrain - a1_drain

If those two agree, the decomposition is validated and the 71/29 split stands as
written. If the drain cost scales with alpha, the split is alpha-specific and the
paper has to say at which alpha it holds.

WHY ALL FOUR CELLS AND NOT JUST THE ONE MISSING ARM. The standing rule is no
splicing across blocks. a1_nodrain already exists in
alpha_results/p3_drain_control_master.csv, but subtracting a new a1_drain from
THAT mean would be a cross-block contrast of exactly the kind the fig7 provenance
note rejected -- the two blocks there agreed on level to 0.3% and disagreed on
slope at t=2.6. Both differences this block reports are therefore within-block.
Re-measuring the alpha=0 pair costs six runs and buys an independent replication
of the published drain cost, which is worth having on its own.

Protocol identical to the first drain-control block: policy 3, C=1024, 512-token
prompt, 2048 decoded tokens, -fa off, seed 123, greedy, EOS disabled,
UNINSTRUMENTED (UNIKV_LOG itself synchronizes at both ends of every decode call,
which would drain the pipeline in the arms whose whole point is that they do
not). Randomized block with cooldowns; a different shuffle seed, so this is a
fresh randomization rather than a replay of the first block's order.

UNIKV_DRAIN semantics (from llama-context.cpp):
  unset -> drain iff alpha > 0   (the published behavior)
  1     -> drain even at alpha = 0
  0     -> never drain, even at alpha > 0
So the two DRAINED conditions are reached differently -- alpha=1 drains by
default, alpha=0 needs the override -- and each condition's setting is recorded
per run so the CSV carries the evidence rather than the harness source.
"""

import csv
import datetime
import math
import os
import random
import re
import statistics
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

RESULTS_DIR = ROOT / "alpha_results"
ART_DIR     = ROOT / "artifacts" / "drain_alpha1"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTX           = 1024
PROMPT_TOKENS = 512
GEN_TOKENS    = 2048
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 512
UBATCH        = 512
SEED          = 123
SPILL_CAP     = 8192
CEILING       = 58.0
FIXED_TOKEN   = " token"
SHUFFLE_SEED  = 20260811          # fresh randomization, not the first block's

TRIALS     = int(os.environ.get("UNIKV_DA_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_DA_COOLDOWN", "200"))

# tag -> (alpha, UNIKV_DRAIN or None for the default, drains?, note)
CONDITIONS = {
    "a0_nodrain": (0.0, None, False, "alpha=0 anchor, no drain (published)"),
    "a0_drain":   (0.0, "1",  True,  "drain alone at alpha=0"),
    "a1_nodrain": (1.0, "0",  False, "alpha=1, drain suppressed"),
    "a1_drain":   (1.0, None, True,  "alpha=1 drained -- THE MISSING CELL"),
}


def count_tokens(path: Path) -> int:
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def ensure_prompt() -> Path:
    path = PROMPTS_DIR / f"prompt_{PROMPT_TOKENS}tok.txt"
    path.write_text(FIXED_TOKEN * max(PROMPT_TOKENS - 1, 0))
    got = count_tokens(path)
    if got != PROMPT_TOKENS:
        raise RuntimeError(f"prompt token mismatch: want {PROMPT_TOKENS} got {got}")
    return path


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def run_one(prompt: Path, cond: str, trial: int) -> dict:
    alpha, drain, _, _ = CONDITIONS[cond]
    e2e_csv = LOGS_DIR / f"e2e_{cond}_t{trial}.csv"
    log_txt = LOGS_DIR / f"gen_{cond}_t{trial}.txt"
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({
        "UNIKV_POLICY": "3",
        "UNIKV_ALPHA": f"{alpha:g}",
        "UNIKV_E2E_LOG": str(e2e_csv),
        "UNIKV_SPILL_CAP": str(SPILL_CAP),
    })
    env.pop("UNIKV_DRAIN", None)
    env.pop("UNIKV_LOG", None)          # uninstrumented: UNIKV_LOG would drain
    if drain is not None:
        env["UNIKV_DRAIN"] = drain

    done = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace")
    log_txt.write_text("=== STDOUT ===\n" + done.stdout +
                       "\n=== STDERR ===\n" + done.stderr[-6000:])

    dec, tps, ms = None, None, None
    if e2e_csv.exists():
        with e2e_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            dec = int(rows[-1]["decode_tokens"])
            tps = float(rows[-1]["tok_per_sec"])
            ms  = float(rows[-1]["decode_ms"])

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", done.stderr)
    return {"rc": done.returncode, "decode_tokens": dec, "tok_per_sec": tps,
            "decode_ms": ms, "flash_attn": mfa.group(1) if mfa else "UNPARSED"}


def se_of_mean(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    prompt = ensure_prompt()
    conds = list(CONDITIONS)
    plan = [(c, t) for c in conds for t in range(1, TRIALS + 1)]
    random.Random(SHUFFLE_SEED).shuffle(plan)

    print(f"drain x alpha 2x2: {len(conds)} conditions x {TRIALS} trials "
          f"= {len(plan)} runs")
    print(f"  policy 3, C={CTX}, {PROMPT_TOKENS}-token prompt, {GEN_TOKENS} decode, "
          f"-fa off, uninstrumented")
    print(f"  randomized block (seed {SHUFFLE_SEED}), cooldown {COOLDOWN_S}s "
          f"before every run including the first")
    print("  order:", " ".join(f"{c}t{t}" for c, t in plan))
    print()

    out = RESULTS_DIR / "p3_drain_alpha1_master.csv"
    out.unlink(missing_ok=True)
    with out.open("w", newline="") as fh:
        csv.writer(fh).writerow(["order_idx", "condition", "alpha", "drain_env",
                                 "drains", "trial", "rc", "decode_tokens",
                                 "decode_ms", "tok_per_sec", "flash_attn",
                                 "t_start_iso", "duration_s"])

    vals: dict[str, list[float]] = {c: [] for c in conds}
    flags = []
    for idx, (cond, trial) in enumerate(plan, 1):
        alpha, drain, drains, _ = CONDITIONS[cond]
        if COOLDOWN_S > 0:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] {cond:11s} trial={trial} ...", end="", flush=True)
        r = run_one(prompt, cond, trial)
        dur = time.time() - t0
        print(f" {r['tok_per_sec']} tok/s  dec={r['decode_tokens']} ({dur:.0f}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"{cond}t{trial}: flash_attn={r['flash_attn']}")
        if r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"{cond}t{trial}: decoded {r['decode_tokens']} != {GEN_TOKENS}")
        if r["rc"] != 0:
            flags.append(f"{cond}t{trial}: rc={r['rc']}")
        if r["tok_per_sec"] is None:
            flags.append(f"{cond}t{trial}: no tok/s")
        elif r["tok_per_sec"] >= CEILING:
            flags.append(f"{cond}t{trial}: {r['tok_per_sec']} >= {CEILING} ceiling")
        else:
            vals[cond].append(r["tok_per_sec"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, cond, alpha,
                                     drain if drain is not None else "default",
                                     int(drains), trial, r["rc"],
                                     r["decode_tokens"], r["decode_ms"],
                                     r["tok_per_sec"], r["flash_attn"],
                                     iso(t0), f"{dur:.1f}"])

    print("\n" + "=" * 78)
    print(f"{'condition':12s} {'alpha':>5s} {'drains':>7s} {'n':>2s} {'mean':>7s} "
          f"{'sd':>6s}  note")
    print("-" * 78)
    means, sds = {}, {}
    for c in conds:
        alpha, _, drains, note = CONDITIONS[c]
        v = vals[c]
        if v:
            means[c] = statistics.mean(v)
            sds[c] = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"{c:12s} {alpha:5g} {str(bool(drains)):>7s} {len(v):2d} "
                  f"{means[c]:7.3f} {sds[c]:6.3f}  {note}")
        else:
            print(f"{c:12s} {alpha:5g} {str(bool(drains)):>7s}  0 {'n/a':>7s} "
                  f"{'n/a':>6s}  {note}")
    print("=" * 78)

    if set(conds) <= means.keys():
        d0 = means["a0_nodrain"] - means["a0_drain"]
        d1 = means["a1_nodrain"] - means["a1_drain"]
        se0 = math.hypot(se_of_mean(vals["a0_nodrain"]), se_of_mean(vals["a0_drain"]))
        se1 = math.hypot(se_of_mean(vals["a1_nodrain"]), se_of_mean(vals["a1_drain"]))
        diff = d1 - d0
        se_d = math.hypot(se0, se1)
        t = diff / se_d if se_d else float("nan")

        print("\nDRAIN COST, WITHIN THIS BLOCK")
        print(f"  at alpha=0 :  {d0:+.3f} +/- {se0:.3f} tok/s")
        print(f"  at alpha=1 :  {d1:+.3f} +/- {se1:.3f} tok/s")
        print(f"  difference :  {diff:+.3f} +/- {se_d:.3f} tok/s   t = {t:+.2f}")
        print("\n  |t| < 2  -> drain cost is alpha-independent over 0..1; the "
              "decomposition\n           (and the 71/29 split) is validated as written.")
        print("  |t| >= 2 -> the drain cost depends on alpha; the split is "
              "alpha-specific and\n           the paper must state the alpha it "
              "holds at.")
        print(f"\n  VERDICT: {'alpha-INDEPENDENT' if abs(t) < 2 else 'alpha-DEPENDENT'} "
              f"(|t| = {abs(t):.2f})")

        print("\nFor reference, the alpha effect within each drain condition:")
        print(f"  undrained, alpha 0 -> 1: "
              f"{means['a1_nodrain'] - means['a0_nodrain']:+.3f} tok/s")
        print(f"  drained,   alpha 0 -> 1: "
              f"{means['a1_drain'] - means['a0_drain']:+.3f} tok/s")

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
