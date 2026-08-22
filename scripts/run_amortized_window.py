#!/usr/bin/env python3
"""Review item M1 — the shipped amortization, and a re-verification of the ablation.

M1: the fork's policy 1 discards n_tokens_all (one token per decode step), so a
2048-token generation at C=1024 pays ~1535 shift events. Upstream's
--context-shift discards n_left/2 — roughly half the window — and pays a handful.
They are not the same policy, and the re-encode cost measured in Section 8 is not
one a --context-shift user pays. UNIKV_WINDOW_DISCARD=-1 sets n_discard = C/2.

This block also RE-VERIFIES the reconstructed UNIKV_NO_REENCODE gate, which was
lost from src/ after the published ablation ran. Acceptance is semantic, not a
string match: the within-block recovery must reproduce (published +1.075 tok/s at
C=1024 and +0.695 at C=2048). Absolute rates will not match the published block —
session offset — which is why the paper only ever compares within a block.

ARMS (C in {1024, 2048}, 3 rounds):
  p1_window       per-token discard, the published policy 1
  p1_noreencode   per-token discard, K-shift pass skipped (INSTRUMENT, output invalid)
  p1_amortized    n_discard = C/2, upstream's amortization          <- M1's new arm
  p4_h2o          H2O eviction, the comparator Section 8 uses

THE PREDICTION, as a BRACKET rather than a point value. If the window's excess
over H2O is entirely PER-EVENT cost, amortizing to a handful of events should
remove essentially all of it and p1_amortized should land at H2O's level
(41.9-42.5 tok/s at C=1024). If part of the residue is per-step regardless of
event count, it lands lower (39.9-40.3). The measurement discriminates.

Randomised COMPLETE block: each round is a permutation of all 8 cells, so arm is
orthogonal to position by construction. 200 s cooldowns, uninstrumented,
512-token prompt, 2048 decode, -fa off parsed per run, greedy, seed 123.

EVENT CENSUS: shift-event counts are STRUCTURAL (deterministic, thermally
invariant), so they are counted afterwards in a short instrumented pass rather
than inside the cooled block, where UNIKV_LOG's synchronize would corrupt rates.
"""

import csv, datetime, math, os, random, re, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN            = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN / "llama-completion"
TOKENIZE_BIN   = BIN / "llama-tokenize"
MODEL_PATH     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / "amortized_window"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTXS       = [1024, 2048]
TRIALS     = int(os.environ.get("UNIKV_AW_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_AW_COOLDOWN", "200"))
PROMPT_TOKENS, GEN_TOKENS = 512, 2048
THREADS, GPU_LAYERS, BATCH, UBATCH, SEED = 10, 999, 512, 512, 123
CEILING, FIXED_TOKEN, SHUFFLE = 58.0, " token", 20260821

BANNER_RE  = "RoPE re-encode SKIPPED"
BANNER_AM  = "AMORTIZED WINDOW"
# arm -> (policy, no_reencode, window_discard, role)
ARMS = {
    "p1_window":     (1, False, None, "per-token discard (the published policy 1)"),
    "p1_noreencode": (1, True,  None, "per-token discard, K-shift skipped (INSTRUMENT)"),
    "p1_amortized":  (1, False, "-1", "n_discard = C/2 (upstream's amortization)"),
    "p4_h2o":        (4, False, None, "H2O eviction"),
}
# published within-block recovery, for the reconstruction check
PUB_RECOVERY = {1024: 1.075, 2048: 0.695}
PUB_PER_EVENT = {1024: 0.923, 2048: 1.811}
ACCEPT_PER_EVENT_1024 = (0.85, 1.00)


def count_tokens(p):
    out = subprocess.run([str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(p),
                          "--show-count", "--log-disable"], cwd=LLAMA_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m: raise RuntimeError(out)
    return int(m.group(1))


def ensure_prompt(n):
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PROMPTS_DIR / f"prompt_{n}tok.txt"
    if p.exists() and count_tokens(p) == n: return p
    reps, seen = max(n - 1, 0), {}
    for _ in range(16):
        p.write_text(FIXED_TOKEN * reps)
        got = seen[reps] = count_tokens(p)
        if got == n: return p
        nxt = reps + (n - got)
        if nxt in seen or nxt <= 0: break
        reps = nxt
    raise RuntimeError(f"prompt {n}: {seen}")


def launch(prompt, arm, ctx, tag, steplog=False):
    policy, no_re, wd, _ = ARMS[arm]
    e2e = LOGS_DIR / f"e2e_{tag}.csv"; e2e.unlink(missing_ok=True)
    step = LOGS_DIR / f"step_{tag}.csv"; step.unlink(missing_ok=True)
    args = [str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
            "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
            "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
            "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
            "--simple-io", "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0", "UNIKV_E2E_LOG": str(e2e)})
    for k in ("UNIKV_LOG", "UNIKV_H2O_TRACE", "UNIKV_NO_REENCODE",
              "UNIKV_WINDOW_DISCARD", "UNIKV_LOGIT_LOG", "UNIKV_SPILL_DEV"):
        env.pop(k, None)
    if no_re: env["UNIKV_NO_REENCODE"] = "1"
    if wd:    env["UNIKV_WINDOW_DISCARD"] = wd
    if steplog: env["UNIKV_LOG"] = str(step)

    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = round(time.time() - t0, 1)
    (LOGS_DIR / f"gen_{tag}.txt").write_text("=== STDERR ===\n" + d.stderr[-8000:])
    dec = tps = None
    if e2e.exists():
        rows = list(csv.DictReader(e2e.open()))
        if rows: dec, tps = int(rows[-1]["decode_tokens"]), float(rows[-1]["tok_per_sec"])
    ev = None
    if steplog and step.exists():
        ev = sum(int(r["shift_events"]) for r in csv.DictReader(step.open()))
    fa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    both = d.stdout + d.stderr
    return {"rc": d.returncode, "decode_tokens": dec, "tok_per_sec": tps,
            "flash_attn": fa.group(1) if fa else "UNPARSED",
            "re_skipped": BANNER_RE in both, "amortized": BANNER_AM in both,
            "shift_events": ev, "duration_s": dur}


def main():
    for p in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR): p.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt(PROMPT_TOKENS)
    # optional subsetting, for a focused re-measurement of one contrast
    only_arms = [x for x in os.environ.get("UNIKV_AW_ARMS", "").split(",") if x]
    only_ctxs = [int(x) for x in os.environ.get("UNIKV_AW_CTXS", "").split(",") if x]
    arms = [a for a in ARMS if not only_arms or a in only_arms]
    ctxs = [c for c in CTXS if not only_ctxs or c in only_ctxs]
    cells = [(a, c) for a in arms for c in ctxs]
    rng, plan = random.Random(SHUFFLE), []
    for r in range(1, TRIALS + 1):
        rnd = cells[:]; rng.shuffle(rnd)
        plan += [(a, c, r) for a, c in rnd]

    print(f"amortized-window block: {len(plan)} runs "
          f"({len(ARMS)} arms x {len(CTXS)} contexts x {TRIALS} rounds)")
    print(f"  randomised complete block (seed {SHUFFLE}), {COOLDOWN_S}s cooldowns, "
          f"uninstrumented\n")
    print(f"  reconstruction check: recovery must reproduce "
          f"{PUB_RECOVERY} tok/s; per-event at C=1024 must land in "
          f"{ACCEPT_PER_EVENT_1024} ms\n")

    out = RESULTS_DIR / os.environ.get("UNIKV_AW_OUT", "amortized_window_block.csv")
    cols = ["order_idx", "round", "arm", "ctx", "policy", "rc", "decode_tokens",
            "tok_per_sec", "flash_attn", "re_skipped", "amortized", "duration_s", "t_start"]
    with out.open("w", newline="") as fh: csv.writer(fh).writerow(cols)

    vals, flags = {}, []
    for idx, (arm, ctx, rnd) in enumerate(plan, 1):
        policy, no_re, wd, _ = ARMS[arm]
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] r{rnd} {arm:14s} C={ctx:4d} ...", end="", flush=True)
        r = launch(prompt, arm, ctx, f"{arm}_c{ctx}_r{rnd}")
        print(f" {r['tok_per_sec']} tok/s dec={r['decode_tokens']} ({r['duration_s']}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"{arm}@{ctx}r{rnd}: flash_attn={r['flash_attn']}")
        if r["re_skipped"] != no_re:
            flags.append(f"{arm}@{ctx}r{rnd}: re_skipped={r['re_skipped']} != {no_re} "
                         f"— ABLATION GATE DID NOT MATCH THE ARM")
        if r["amortized"] != (wd is not None):
            flags.append(f"{arm}@{ctx}r{rnd}: amortized={r['amortized']} != {wd is not None}")
        if r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"{arm}@{ctx}r{rnd}: decoded {r['decode_tokens']}")
        if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
            flags.append(f"{arm}@{ctx}r{rnd}: {r['tok_per_sec']} >= {CEILING}")
        elif r["tok_per_sec"]:
            vals.setdefault((arm, ctx), []).append(r["tok_per_sec"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, rnd, arm, ctx, policy, r["rc"],
                                     r["decode_tokens"], r["tok_per_sec"], r["flash_attn"],
                                     int(r["re_skipped"]), int(r["amortized"]),
                                     r["duration_s"],
                                     datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")])

    # ---------------- event census (structural, outside the cooled block) ----
    print("\n== EVENT CENSUS (instrumented; counts are structural, rates from it are not used) ==")
    census = {}
    for arm in arms:
        for ctx in ctxs:
            r = launch(prompt, arm, ctx, f"census_{arm}_c{ctx}", steplog=True)
            census[(arm, ctx)] = r["shift_events"]
            print(f"  {arm:14s} C={ctx:4d}  shift events = {r['shift_events']}")
    with (RESULTS_DIR / "amortized_window_events.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "ctx", "shift_events"])
        for (a, c), v in census.items(): w.writerow([a, c, v])

    # ---------------- results -----------------------------------------------
    print("\n" + "=" * 88)
    m = {}
    for ctx in ctxs:
        print(f"\n-- C={ctx} --")
        for arm in arms:
            v = vals.get((arm, ctx), [])
            if v:
                m[(arm, ctx)] = st.mean(v)
                se = st.stdev(v)/math.sqrt(len(v)) if len(v) > 1 else float('nan')
                print(f"  {arm:14s} n={len(v)} {st.mean(v):7.3f} +/- {se:.3f} tok/s "
                      f"(sd {st.stdev(v) if len(v)>1 else 0:.3f})  {ARMS[arm][3]}")

    print("\n" + "=" * 88)
    print("RECONSTRUCTION CHECK — does the ablation replicate the published recovery?")
    ok = True
    for ctx in ctxs:
        p1, ab = m.get(("p1_window", ctx)), m.get(("p1_noreencode", ctx))
        ev = census.get(("p1_window", ctx))
        if not (p1 and ab and ev): continue
        rec = ab - p1
        ms_p1, ms_ab = 1000/p1, 1000/ab
        per_ev = (ms_p1 - ms_ab) * GEN_TOKENS / ev
        print(f"  C={ctx}: recovery {rec:+.3f} tok/s (published {PUB_RECOVERY[ctx]:+.3f})")
        print(f"          {ev} measured shift events -> {per_ev:.3f} ms/event "
              f"(published {PUB_PER_EVENT[ctx]:.3f})")
        if ctx == 1024:
            lo, hi = ACCEPT_PER_EVENT_1024
            good = lo <= per_ev <= hi
            ok = ok and good
            print(f"          acceptance band [{lo}, {hi}] ms -> "
                  f"{'PASS' if good else 'FAIL — STOP AND REPORT'}")

    print("\nM1 — WHAT THE SHIPPED AMORTIZATION BUYS")
    for ctx in ctxs:
        p1, am, h2 = (m.get((a, ctx)) for a in ("p1_window", "p1_amortized", "p4_h2o"))
        e1, ea = census.get(("p1_window", ctx)), census.get(("p1_amortized", ctx))
        if not (p1 and am and h2): continue
        print(f"\n  C={ctx}: per-token {p1:.3f}  amortized {am:.3f}  H2O {h2:.3f} tok/s")
        print(f"     shift events: {e1} -> {ea}  ({e1/ea if ea else float('nan'):.0f}x fewer)")
        print(f"     amortized vs per-token: {100*(am/p1-1):+.2f}%")
        print(f"     amortized vs H2O:       {100*(am/h2-1):+.2f}%")
        if ctx == 1024:
            print(f"     bracket: per-event residue -> 41.9-42.5 ; "
                  f"per-step residue -> 39.9-40.3 ; measured {am:.3f}")
            verdict = ("residue is PER-EVENT — amortization removes it" if am >= 41.9 else
                       "residue is PER-STEP — amortization does not remove it" if am <= 40.3 else
                       "BETWEEN the brackets — residue is partly per-event")
            print(f"     -> {verdict}")

    print("\nposition audit:")
    with out.open(newline="") as fh: rows = list(csv.DictReader(fh))
    for arm in arms:
        s = [int(r["order_idx"]) for r in rows if r["arm"] == arm]
        print(f"  {arm:14s} slots {sorted(s)}  mean {st.mean(s):.2f}")
    if flags:
        print("\n!! FLAGS:")
        for f in flags: print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags or not ok else 0


if __name__ == "__main__":
    sys.exit(main())
