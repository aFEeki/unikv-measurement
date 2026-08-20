#!/usr/bin/env python3
"""Item 2 — the composed fused-kernel cost, in ONE block.

Section 8 gives the exactness premium (11.4% over H2O at C=1024) and Section 3
gives the fused-kernel penalty (12.4% at C=4096) from DIFFERENT blocks, and the
paper declines to compose them because that is exactly the splice Section 6
refuses on its own data. Putting all four arms in one cooled block removes the
need to decline.

ARMS (C in {1024, 2048}, 3 rounds):
  p0_fa_on    upstream WITH flash attention  <- the fused baseline
  p0_fa_off   upstream, -fa off
  p4_h2o      H2O, -fa off (needs the explicit softmax to score cells)
  p3_dev      exact retention, device-visible tier, -fa off (policy 3 refuses to
              construct with -fa on; the preflight below records that refusal as
              evidence rather than asserting it from the source)

TWO NUMBERS COME OUT, AND ONLY ONE OF THEM IS CLEAN:

  (a) THE KERNEL PENALTY, p0_fa_on vs p0_fa_off. Same policy, same halting
      behaviour, same decoded length — only the kernel differs. This is a clean
      within-block contrast and is what the brief asks be reported at these
      budgets, since the published 12.4% is from C=4096 and may not carry.

  (b) THE COMPOSED PREMIUM, p3_dev vs p0_fa_on. This is the number a reader
      wants, and it carries a caveat that must travel with it: POLICY 0 HALTS AT
      CACHE-FULL. With a 512-token prompt at C=1024 it decodes ~512 tokens and
      stops, while p3_dev and p4_h2o decode the full 2048. So p0's throughput is
      measured only over the regime where the cache is not yet full and the KV is
      small — the cheapest part of the run — whereas the retention arms are
      measured over a regime that includes a spilled tier. The comparison
      therefore FLATTERS p0 and the composed premium is an UPPER BOUND on what
      exact retention costs against a fused baseline. The decoded-token counts are
      recorded per run so this is visible in the CSV rather than buried.

      The composition via (a) — premium over H2O, both -fa off, times the kernel
      penalty on the same policy — is the defensible route and is reported too.

DESIGN: randomised COMPLETE block, three rounds, each a random permutation of all
eight (arm, ctx) cells, so arm is orthogonal to block position by construction.
200 s cooldowns, uninstrumented, seed 123, greedy, EOS disabled. flash_attn is
parsed back from every run's own log and asserted against what the arm asked for.
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
ART_DIR     = ROOT / "artifacts" / "fused_composed"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

CTXS       = [1024, 2048]
TRIALS     = int(os.environ.get("UNIKV_FC_TRIALS", "3"))
COOLDOWN_S = int(os.environ.get("UNIKV_FC_COOLDOWN", "200"))
PROMPT_TOKENS, GEN_TOKENS = 512, 2048
THREADS, GPU_LAYERS, BATCH, UBATCH, SEED = 10, 999, 512, 512, 123
CEILING, FIXED_TOKEN, SHUFFLE = 58.0, " token", 20260820

# arm -> (policy, spill_dev, fa, completes_budget, role)
ARMS = {
    "p0_fa_on":  (0, "0", "on",  False, "upstream, FUSED kernel (the baseline)"),
    "p0_fa_off": (0, "0", "off", False, "upstream, explicit softmax"),
    "p4_h2o":    (4, "0", "off", True,  "H2O eviction"),
    "p3_dev":    (3, "1", "off", True,  "exact retention, device-visible tier"),
}


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


def launch(prompt, policy, dev, fa, ctx, gen, tag, extra_env=None):
    e2e = LOGS_DIR / f"e2e_{tag}.csv"; e2e.unlink(missing_ok=True)
    args = [str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
            "-n", str(gen), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
            "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", fa, "-fit", "off",
            "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
            "--simple-io", "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": dev, "UNIKV_SPILL_CAP": "8192",
                "UNIKV_E2E_LOG": str(e2e)})
    for k in ("UNIKV_LOG", "UNIKV_H2O_TRACE", "UNIKV_NO_REENCODE"):
        env.pop(k, None)
    if extra_env: env.update(extra_env)
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
    return {"rc": d.returncode, "decode_tokens": dec, "tok_per_sec": tps,
            "flash_attn": mfa.group(1) if mfa else "UNPARSED",
            "duration_s": round(dur, 1), "stderr_tail": d.stderr[-1500:]}


def preflight(prompt):
    """Record what policies 3 and 4 actually DO with -fa on, rather than
    asserting from the source that they refuse."""
    print("PREFLIGHT — do policies 3 and 4 construct with flash attention on?")
    notes = {}
    for pol, dev, name in ((3, "1", "policy 3 device-visible"), (4, "0", "policy 4 H2O")):
        r = launch(prompt, pol, dev, "on", 1024, 8, f"preflight_p{pol}_faon")
        ok = r["rc"] == 0 and r["decode_tokens"]
        notes[name] = (f"rc={r['rc']} flash_attn={r['flash_attn']} "
                       f"decoded={r['decode_tokens']}")
        print(f"  {name:26s} {notes[name]}  -> "
              f"{'RAN' if ok else 'REFUSED / FAILED'}")
        if not ok:
            tail = [l for l in r["stderr_tail"].splitlines()
                    if re.search(r"error|abort|assert|unikv|flash", l, re.I)][-2:]
            for l in tail: print(f"      {l.strip()[:110]}")
    print()
    return notes


def main():
    for p in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR): p.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt(PROMPT_TOKENS)
    pre = preflight(prompt)

    cells = [(a, c) for a in ARMS for c in CTXS]
    rng, plan = random.Random(SHUFFLE), []
    for r in range(1, TRIALS + 1):
        rnd = cells[:]; rng.shuffle(rnd)
        plan += [(a, c, r) for a, c in rnd]

    print(f"fused-composed block: {len(plan)} runs "
          f"({len(ARMS)} arms x {len(CTXS)} contexts x {TRIALS} rounds)")
    print(f"  randomised complete block (seed {SHUFFLE}), {COOLDOWN_S}s cooldowns, "
          f"uninstrumented\n")

    out = RESULTS_DIR / "fused_composed_block.csv"
    cols = ["order_idx", "round", "arm", "ctx", "policy", "spill_dev", "fa_requested",
            "flash_attn", "rc", "decode_tokens", "tok_per_sec", "duration_s", "t_start"]
    with out.open("w", newline="") as fh: csv.writer(fh).writerow(cols)

    vals, flags = {}, []
    for idx, (arm, ctx, rnd) in enumerate(plan, 1):
        policy, dev, fa, completes, _ = ARMS[arm]
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] r{rnd} {arm:10s} C={ctx:4d} fa={fa:3s} ...",
              end="", flush=True)
        r = launch(prompt, policy, dev, fa, ctx, GEN_TOKENS, f"{arm}_c{ctx}_t{rnd}")
        print(f" {r['tok_per_sec']} tok/s dec={r['decode_tokens']} "
              f"fa={r['flash_attn']} ({r['duration_s']}s)")

        want_fa = "enabled" if fa == "on" else "disabled"
        if r["flash_attn"] != want_fa:
            flags.append(f"{arm}@{ctx}r{rnd}: flash_attn={r['flash_attn']}, wanted {want_fa}")
        if completes and r["decode_tokens"] != GEN_TOKENS:
            flags.append(f"{arm}@{ctx}r{rnd}: decoded {r['decode_tokens']} != {GEN_TOKENS}")
        if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
            flags.append(f"{arm}@{ctx}r{rnd}: {r['tok_per_sec']} >= {CEILING}")
        elif r["tok_per_sec"]:
            vals.setdefault((arm, ctx), []).append(r["tok_per_sec"])

        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([idx, rnd, arm, ctx, policy, dev, fa,
                                     r["flash_attn"], r["rc"], r["decode_tokens"],
                                     r["tok_per_sec"], r["duration_s"],
                                     datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")])

    print("\n" + "=" * 86)
    m, decs = {}, {}
    with out.open(newline="") as fh: rows = list(csv.DictReader(fh))
    for ctx in CTXS:
        print(f"\n-- C={ctx} --")
        for arm in ARMS:
            v = vals.get((arm, ctx), [])
            d = {r["decode_tokens"] for r in rows if r["arm"] == arm and int(r["ctx"]) == ctx}
            if v:
                m[(arm, ctx)] = st.mean(v); decs[(arm, ctx)] = d
                se = st.stdev(v)/math.sqrt(len(v)) if len(v) > 1 else float('nan')
                print(f"  {arm:10s} n={len(v)} {st.mean(v):7.3f} +/- {se:.3f} tok/s "
                      f"(sd {st.stdev(v) if len(v)>1 else 0:.3f})  decoded {sorted(d)}  "
                      f"{ARMS[arm][4]}")

    print("\n" + "=" * 86)
    print("(a) KERNEL PENALTY — same policy, same halting behaviour, only the kernel differs")
    for ctx in CTXS:
        on, off = m.get(("p0_fa_on", ctx)), m.get(("p0_fa_off", ctx))
        if on and off:
            print(f"  C={ctx}: fa_on {on:.3f} vs fa_off {off:.3f} -> "
                  f"fused is {100*(on/off-1):+.2f}% faster "
                  f"(published 12.4% at C=4096)")

    print("\n(b) EXACTNESS PREMIUM over H2O, both -fa off (same block)")
    for ctx in CTXS:
        h, p3 = m.get(("p4_h2o", ctx)), m.get(("p3_dev", ctx))
        if h and p3:
            print(f"  C={ctx}: p3_dev {p3:.3f} vs h2o {h:.3f} -> premium {100*(h/p3-1):+.2f}%")

    print("\n(c) COMPOSED — exact retention against the FUSED baseline, one block")
    print("    UPPER BOUND: policy 0 halts at cache-full and is timed only over the")
    print("    not-yet-full regime, so this flatters the baseline. Decoded counts above.")
    for ctx in CTXS:
        on, p3 = m.get(("p0_fa_on", ctx)), m.get(("p3_dev", ctx))
        if on and p3:
            print(f"  C={ctx}: p3_dev {p3:.3f} vs p0_fa_on {on:.3f} -> "
                  f"{100*(on/p3-1):+.2f}%")
    print("\n    via the clean route (premium over H2O x kernel penalty, same block):")
    for ctx in CTXS:
        on, off, h, p3 = (m.get((a, ctx)) for a in ("p0_fa_on", "p0_fa_off", "p4_h2o", "p3_dev"))
        if on and off and h and p3:
            comp = (h/p3) * (on/off)
            print(f"  C={ctx}: (1{100*(h/p3-1):+.2f}%) x (1{100*(on/off-1):+.2f}%) "
                  f"= {100*(comp-1):+.2f}%")

    print("\nposition audit:")
    for arm in ARMS:
        slots = [int(r["order_idx"]) for r in rows if r["arm"] == arm]
        print(f"  {arm:10s} slots {sorted(slots)}  mean {st.mean(slots):.2f}")
    print(f"\npreflight: {pre}")
    if flags:
        print("\n!! FLAGS:")
        for f in flags: print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
