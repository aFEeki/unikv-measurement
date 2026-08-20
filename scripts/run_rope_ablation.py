#!/usr/bin/env python3
"""Item 1 — the RoPE ablation. Is the K-shift re-encode really Finding 5's mechanism?

Finding 5 says the rolling window costs more than H2O because llama.cpp calls
seq_add on every overflow, which renumbers the surviving cells and forces a
rotary re-encode across the whole cache, while H2O drops cells and touches no
positions. That attribution is read off the code path; nothing measures it.

UNIKV_NO_REENCODE=1 (added in llama-kv-cache.cpp::update) skips ONLY the K-shift
graph. Eviction, the position renumbering, the head fix-up and reset_shift() all
still run, so the difference between the normal and ablated arms is the cost of
that one operation.

THE ABLATED ARM PRODUCES WRONG OUTPUT. Survivors keep positions that no longer
match their rotary encoding. Verified: its tokens are identical to the normal
arm's up to the first overflow and diverge from exactly that point. It is an
INSTRUMENT, not a policy; its text is never reported as a quality result.

THE PREDICTION, stated before the run (cooled b2 block, C=1024):
    policy 1  39.40 tok/s     H2O  42.42 tok/s     gap 3.02 (7.7%)
  If the attribution is right, p1_noreencode should recover most of that gap and
  land near H2O. If it recovers little, the re-encode is NOT the mechanism and
  Finding 5's explanation is wrong — which is the more interesting outcome and is
  reported as it lands.

DESIGN: randomised COMPLETE block. Three rounds; each round is a random
permutation of all eight (arm, ctx) cells. Every arm therefore appears exactly
once per third of the session, so arm is orthogonal to block position BY
CONSTRUCTION rather than by a lucky shuffle — the drain-alpha block relied on the
shuffle, drew a bad layout, and survived only on post-hoc checks.

Protocol otherwise identical to b2 block 2: 512-token prompt, 2048 decode,
C in {1024, 2048}, -b/-ub 512, -fa off parsed back per run, greedy, seed 123,
EOS disabled, uninstrumented, 200 s cooldowns.
"""

import csv, datetime, math, os, random, re, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN            = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN / "llama-completion"
TOKENIZE_BIN   = BIN / "llama-tokenize"
MODEL_PATH     = Path(os.environ.get(
    "UNIKV_MODEL", LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / "rope_ablation"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTXS       = [1024, 2048]
TRIALS     = int(os.environ.get("UNIKV_RA_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_RA_COOLDOWN", "200"))
PROMPT_TOKENS, GEN_TOKENS = 512, 2048
THREADS, GPU_LAYERS, BATCH, UBATCH, SEED = 10, 999, 512, 512, 123
CEILING, FIXED_TOKEN, SHUFFLE = 58.0, " token", 20260819

BANNER = "RoPE re-encode SKIPPED"
# arm -> (policy, no_reencode, completes_budget, role)
ARMS = {
    "p0_upstream":    (0, False, False, "upstream, halts at cache-full"),
    "p1_window":      (1, False, True,  "rolling window, K-shift AS SHIPPED"),
    "p1_noreencode":  (1, True,  True,  "rolling window, RoPE re-encode SKIPPED "
                                        "(INSTRUMENT — output invalid)"),
    "p4_h2o":         (4, False, True,  "H2O eviction, touches no positions"),
}
PREDICT = {"p1_window": 39.40, "p4_h2o": 42.42}   # cooled b2 block, C=1024


def count_tokens(path):
    out = subprocess.run([str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
                          "--show-count", "--log-disable"], cwd=LLAMA_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m: raise RuntimeError(f"token count parse failed:\n{out}")
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
    raise RuntimeError(f"could not build {n}-token prompt: {seen}")


def build_plan():
    """Randomised complete block: each round is a permutation of all 8 cells."""
    cells = [(a, c) for a in ARMS for c in CTXS]
    rng, plan = random.Random(SHUFFLE), []
    for r in range(1, TRIALS + 1):
        rnd = cells[:]; rng.shuffle(rnd)
        plan += [(a, c, r) for a, c in rnd]
    return plan


def run_one(prompt, arm, ctx, trial):
    policy, no_re, _, _ = ARMS[arm]
    tag = f"{arm}_c{ctx}_t{trial}"
    e2e = LOGS_DIR / f"e2e_{tag}.csv"; e2e.unlink(missing_ok=True)
    args = [str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
            "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
            "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
            "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
            "--simple-io", "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_E2E_LOG": str(e2e)})
    for k in ("UNIKV_LOG", "UNIKV_H2O_TRACE", "UNIKV_NO_REENCODE", "UNIKV_SPILL_DEV"):
        env.pop(k, None)
    if no_re: env["UNIKV_NO_REENCODE"] = "1"

    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    (LOGS_DIR / f"gen_{tag}.txt").write_text(
        "=== STDOUT ===\n" + d.stdout + "\n=== STDERR ===\n" + d.stderr[-8000:])

    dec = tps = None
    if e2e.exists():
        rows = list(csv.DictReader(e2e.open()))
        if rows: dec, tps = int(rows[-1]["decode_tokens"]), float(rows[-1]["tok_per_sec"])
    mfa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    both = d.stdout + d.stderr
    return {"rc": d.returncode, "decode_tokens": dec, "tok_per_sec": tps,
            "flash_attn": mfa.group(1) if mfa else "UNPARSED",
            "reencode_skipped": BANNER in both, "duration_s": round(dur, 1)}


def main():
    for p in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR): p.mkdir(parents=True, exist_ok=True)
    prompt, plan = ensure_prompt(PROMPT_TOKENS), build_plan()
    print(f"RoPE ablation: {len(plan)} runs "
          f"({len(ARMS)} arms x {len(CTXS)} contexts x {TRIALS} rounds)")
    print(f"  randomised COMPLETE block — every arm once per round, so arm is")
    print(f"  orthogonal to position by construction (seed {SHUFFLE})")
    print(f"  {COOLDOWN_S}s cooldowns, uninstrumented, -fa off, C={CTXS}\n")
    print(f"  PREDICTION (cooled b2, C=1024): p1 {PREDICT['p1_window']}, "
          f"h2o {PREDICT['p4_h2o']}, gap {PREDICT['p4_h2o']-PREDICT['p1_window']:.2f} tok/s\n")

    out = RESULTS_DIR / "rope_ablation_block.csv"
    cols = ["order_idx", "round", "arm", "ctx", "policy", "no_reencode", "rc",
            "decode_tokens", "tok_per_sec", "flash_attn", "reencode_skipped",
            "duration_s", "t_start"]
    with out.open("w", newline="") as fh: csv.writer(fh).writerow(cols)

    vals, flags = {}, []
    for idx, (arm, ctx, rnd) in enumerate(plan, 1):
        policy, no_re, completes, _ = ARMS[arm]
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] r{rnd} {arm:14s} C={ctx:4d} ...", end="", flush=True)
        r = run_one(prompt, arm, ctx, rnd)
        print(f" {r['tok_per_sec']} tok/s dec={r['decode_tokens']} ({r['duration_s']}s)")

        if r["flash_attn"] != "disabled":
            flags.append(f"{arm}@{ctx}r{rnd}: flash_attn={r['flash_attn']}")
        if r["reencode_skipped"] != no_re:
            flags.append(f"{arm}@{ctx}r{rnd}: reencode_skipped={r['reencode_skipped']}, "
                         f"expected {no_re} — THE ABLATION GATE DID NOT MATCH THE ARM")
        if completes and r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"{arm}@{ctx}r{rnd}: decoded {r['decode_tokens']} != {GEN_TOKENS}")
        if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
            flags.append(f"{arm}@{ctx}r{rnd}: {r['tok_per_sec']} >= {CEILING}")
        elif r["tok_per_sec"]:
            vals.setdefault((arm, ctx), []).append(r["tok_per_sec"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, rnd, arm, ctx, policy, int(no_re), r["rc"],
                                     r["decode_tokens"], r["tok_per_sec"], r["flash_attn"],
                                     int(r["reencode_skipped"]), r["duration_s"],
                                     datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")])

    print("\n" + "=" * 84)
    m = {}
    for ctx in CTXS:
        print(f"\n-- C={ctx} --")
        for arm in ARMS:
            v = vals.get((arm, ctx), [])
            if v:
                m[(arm, ctx)] = st.mean(v)
                se = st.stdev(v)/math.sqrt(len(v)) if len(v) > 1 else float('nan')
                print(f"  {arm:14s} n={len(v)} {st.mean(v):7.3f} +/- {se:.3f} tok/s "
                      f"(sd {st.stdev(v) if len(v)>1 else 0:.3f})  {ARMS[arm][3]}")

    print("\n" + "=" * 84)
    print("THE QUESTION: how much of the policy1 -> H2O gap does skipping the "
          "re-encode recover?")
    for ctx in CTXS:
        p1, ab, h2 = m.get(("p1_window", ctx)), m.get(("p1_noreencode", ctx)), m.get(("p4_h2o", ctx))
        if not (p1 and ab and h2): continue
        gap, rec = h2 - p1, ab - p1
        pct = 100*rec/gap if gap else float('nan')
        print(f"\n  C={ctx}:  p1 {p1:.3f}   p1_noreencode {ab:.3f}   h2o {h2:.3f}")
        print(f"     gap p1->h2o        = {gap:+.3f} tok/s")
        print(f"     recovered by ablation = {rec:+.3f} tok/s  = {pct:.1f}% of the gap")
        print(f"     residual to h2o    = {h2-ab:+.3f} tok/s")
        verdict = ("re-encode IS the dominant mechanism" if pct >= 70 else
                   "re-encode is PART of the mechanism" if pct >= 30 else
                   "re-encode is NOT the mechanism — Finding 5's explanation is wrong")
        print(f"     -> {verdict}")

    print("\nposition audit (randomised complete block; mean slot should be ~equal):")
    with out.open(newline="") as fh: rows = list(csv.DictReader(fh))
    for arm in ARMS:
        slots = [int(r["order_idx"]) for r in rows if r["arm"] == arm]
        print(f"  {arm:14s} slots {sorted(slots)}  mean {st.mean(slots):.2f}")

    if flags:
        print("\n!! FLAGS:")
        for f in flags: print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
