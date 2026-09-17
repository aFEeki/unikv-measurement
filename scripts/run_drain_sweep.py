#!/usr/bin/env python3
"""Block 4 — drained AND undrained across the full alpha range, in ONE block.

Section 6 names this experiment: "a block running drained points beyond
alpha = 0.25 alongside the undrained ones would settle it." Everything known
about the drain cost today comes from three coefficients measured in two
different blocks:

    alpha = 0.00   ~0        (t = 1.11 and -0.76, two blocks, never significant)
    alpha = 0.25   +0.713    (t = 10.5, block 1)
    alpha = 1.00   +0.981    (t = 3.97, block 2)

and the fig7 provenance note refuses to splice the two series because the blocks
agree on level but not on drained slope (t = -2.6). Running both series across
alpha in {0, 0.25, 0.5, 1, 2} inside ONE randomised complete block removes the
need to splice and settles two things the paper currently declines to state:

  (a) whether there is real curvature in the first quarter of the alpha range,
      which the cross-block slope disagreement hinted at but could not separate
      from a block effect;
  (b) the SHAPE of the drain cost, which is currently three points from two
      sessions rather than a curve.

UNIKV_DRAIN semantics (llama-context.cpp):
  unset -> drain iff alpha > 0   (the published behaviour)
  1     -> drain even at alpha = 0
  0     -> never drain, even at alpha > 0
So the drained series needs the override only at alpha=0, and the undrained
series needs it everywhere except alpha=0. Each run records which setting it got
so the CSV carries the evidence rather than the harness source.

Protocol identical to the drain-control blocks: policy 3, C=1024, 512-token
prompt, 2048 decoded tokens, -fa off parsed per run, greedy, seed 123, EOS
disabled, UNINSTRUMENTED (UNIKV_LOG synchronizes at both ends of every decode
call and would drain the arms whose whole point is that they do not).
Randomised COMPLETE block: each round a fresh permutation of all ten cells.
"""

import csv, datetime, math, os, random, re, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"
BIN       = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN, TOKENIZE_BIN = BIN / "llama-completion", BIN / "llama-tokenize"
MODEL_PATH = Path(os.environ.get(
    "UNIKV_MODEL", LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))

RESULTS_DIR = ROOT / "alpha_results"
ART_DIR     = ROOT / "artifacts" / "drain_sweep"
PROMPTS_DIR, LOGS_DIR = ART_DIR / "prompts", ART_DIR / "logs"

ALPHAS = [float(x) for x in os.environ.get("UNIKV_DS_ALPHAS", "0,0.25,0.5,1,2").split(",")]
TRIALS     = int(os.environ.get("UNIKV_DS_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_DS_COOLDOWN", "200"))
CTX, PROMPT_TOKENS, GEN_TOKENS = 1024, 512, 2048
THREADS, GPU_LAYERS, BATCH, UBATCH, SEED = 10, 999, 512, 512, 123
CEILING, SHUFFLE = 58.0, 20260823


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
        p.write_text(" token" * reps)
        got = seen[reps] = count_tokens(p)
        if got == n: return p
        nxt = reps + (n - got)
        if nxt in seen or nxt <= 0: break
        reps = nxt
    raise RuntimeError(f"prompt {n}: {seen}")


def drain_env(alpha, drained):
    """The UNIKV_DRAIN value needed to force `drained` at this alpha, or None
    when the default already does it."""
    default_drains = alpha > 0
    if drained == default_drains: return None
    return "1" if drained else "0"


def run_one(prompt, alpha, drained, trial):
    tag = f"a{alpha:g}_{'drain' if drained else 'nodrain'}_t{trial}"
    e2e = LOGS_DIR / f"e2e_{tag}.csv"; e2e.unlink(missing_ok=True)
    args = [str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
            "-n", str(GEN_TOKENS), "-c", str(CTX), "-b", str(BATCH), "-ub", str(UBATCH),
            "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
            "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
            "--simple-io", "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": "3", "UNIKV_ALPHA": f"{alpha:g}",
                "UNIKV_SPILL_CAP": "8192", "UNIKV_E2E_LOG": str(e2e)})
    for k in ("UNIKV_LOG", "UNIKV_DRAIN", "UNIKV_NO_REENCODE",
              "UNIKV_WINDOW_DISCARD", "UNIKV_LOGIT_LOG"): env.pop(k, None)
    de = drain_env(alpha, drained)
    if de is not None: env["UNIKV_DRAIN"] = de

    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = round(time.time() - t0, 1)
    (LOGS_DIR / f"gen_{tag}.txt").write_text(d.stderr[-6000:])
    dec = tps = None
    if e2e.exists():
        rows = list(csv.DictReader(e2e.open()))
        if rows: dec, tps = int(rows[-1]["decode_tokens"]), float(rows[-1]["tok_per_sec"])
    fa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    return {"rc": d.returncode, "decode_tokens": dec, "tok_per_sec": tps,
            "flash_attn": fa.group(1) if fa else "UNPARSED",
            "drain_env": de if de is not None else "default", "duration_s": dur}


def main():
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR): d.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt(PROMPT_TOKENS)
    cells = [(a, dr) for a in ALPHAS for dr in (False, True)]
    rng, plan = random.Random(SHUFFLE), []
    for r in range(1, TRIALS + 1):
        rnd = cells[:]; rng.shuffle(rnd)
        plan += [(a, dr, r) for a, dr in rnd]

    print(f"drained alpha sweep: {len(plan)} runs "
          f"({len(cells)} cells x {TRIALS} rounds), {COOLDOWN_S}s cooldowns")
    print(f"  alphas {ALPHAS}, both drain conditions, ONE block so the two series")
    print(f"  are never spliced. policy 3, C={CTX}, uninstrumented, seed {SHUFFLE}\n")

    out = RESULTS_DIR / "p3_drain_sweep_master.csv"
    cols = ["order_idx", "round", "alpha", "drained", "drain_env", "rc",
            "decode_tokens", "tok_per_sec", "flash_attn", "duration_s", "t_start"]
    with out.open("w", newline="") as fh: csv.writer(fh).writerow(cols)

    vals, flags, rows = {}, [], []
    for idx, (alpha, drained, rnd) in enumerate(plan, 1):
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] r{rnd} alpha={alpha:<5g} "
              f"{'drained  ' if drained else 'undrained'} ...", end="", flush=True)
        r = run_one(prompt, alpha, drained, rnd)
        print(f" {r['tok_per_sec']} tok/s dec={r['decode_tokens']} ({r['duration_s']}s)")
        if r["flash_attn"] != "disabled":
            flags.append(f"a{alpha}/{drained}/r{rnd}: flash_attn={r['flash_attn']}")
        if r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"a{alpha}/{drained}/r{rnd}: decoded {r['decode_tokens']}")
        if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
            flags.append(f"a{alpha}/{drained}/r{rnd}: {r['tok_per_sec']} >= {CEILING}")
        elif r["tok_per_sec"]:
            vals.setdefault((alpha, drained), []).append(r["tok_per_sec"])
        rows.append({"order_idx": idx, "alpha": alpha, "drained": drained, **r})
        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, rnd, alpha, int(drained), r["drain_env"],
                                     r["rc"], r["decode_tokens"], r["tok_per_sec"],
                                     r["flash_attn"], r["duration_s"],
                                     datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")])

    print("\n" + "=" * 78)
    print(f"{'alpha':>6} {'undrained':>20} {'drained':>20} {'gap':>18}")
    for a in ALPHAS:
        u, d = vals.get((a, False), []), vals.get((a, True), [])
        if u and d:
            su = st.stdev(u)/math.sqrt(len(u)) if len(u) > 1 else 0.0
            sd = st.stdev(d)/math.sqrt(len(d)) if len(d) > 1 else 0.0
            gap, se = st.mean(u) - st.mean(d), math.hypot(su, sd)
            print(f"{a:>6g} {st.mean(u):>13.3f} +/-{su:5.3f} {st.mean(d):>13.3f} +/-{sd:5.3f} "
                  f"{gap:>+10.3f} +/-{se:5.3f}  t={gap/se if se else float('nan'):+.1f}")

    print("\nslopes (tok/s per unit alpha), fitted within this one block:")
    for drained, lab in ((False, "undrained"), (True, "drained  ")):
        xs = [a for a in ALPHAS for _ in vals.get((a, drained), [])]
        ys = [v for a in ALPHAS for v in vals.get((a, drained), [])]
        if len(xs) > 2:
            mx, my = st.mean(xs), st.mean(ys); sxx = sum((x-mx)**2 for x in xs)
            b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx; a0 = my - b*mx
            s2 = sum((y-(a0+b*x))**2 for x, y in zip(xs, ys))/(len(xs)-2)
            print(f"  {lab}: {b:+.4f} +/- {math.sqrt(s2/sxx):.4f} tok/s per unit alpha "
                  f"(intercept {a0:.3f}, resid SD {math.sqrt(s2):.3f})")

    print("\nposition diagnostics:")
    xs, ys = [], []
    for (a, dr), v in vals.items():
        m = st.mean(v)
        for r in rows:
            if r["alpha"] == a and r["drained"] == dr and r["tok_per_sec"]:
                xs.append(r["order_idx"]); ys.append(r["tok_per_sec"] - m)
    if len(xs) > 2:
        mx, my = st.mean(xs), st.mean(ys); sxx = sum((x-mx)**2 for x in xs)
        b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx
        resid = [y-(my+b*(x-mx)) for x, y in zip(xs, ys)]
        print(f"  drift {b:+.4f} +/- {st.stdev(resid)/math.sqrt(sxx):.4f} tok/s per slot")
    for a in ALPHAS:
        for dr in (False, True):
            s = [r["order_idx"] for r in rows if r["alpha"] == a and r["drained"] == dr]
            if s: print(f"  alpha={a:<5g} {'drained' if dr else 'undrained':>9} "
                        f"slots {sorted(s)} mean {st.mean(s):.1f}")
    if flags:
        print("\n!! FLAGS:")
        for f in flags: print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
