#!/usr/bin/env python3
"""Cooled prefill blocks over (ubatch, context) cells. Serves review Blocks 2 and 3.

BLOCK 2 — the cooled ubatch block Section 11 concedes was never run.
  "we have not run a cooled ubatch block, so a small penalty would be invisible
  to us." The published figures are single UNCOOLED runs at C=49152:
  259.6 / 259.4 / 268.2 / 237.0 tok/s for ubatch 64/128/256/512, which the paper
  reports as a non-effect at its resolution rather than as an absence. Three
  cooled trials per cell turns that into a powered null or a resolved penalty.

    UNIKV_UB_CELLS="64:49152,128:49152,256:49152,512:49152" UNIKV_UB_OUT=... 

BLOCK 3 — the n=1 degradation Section 7 flags as "observed once and not repeated".
  One run at ubatch 128, C=81920 took 251.8 s against 62-71 s for every other
  completing configuration in the sweep. Repeat it three times, with the two
  nearest COMPLETING configurations three times each as controls so a slowdown
  has something to be slow against.

    UNIKV_UB_CELLS="128:81920,64:81920,128:65536" UNIKV_UB_OUT=...

Randomised COMPLETE block: each round is a fresh permutation of every cell, so
cell is orthogonal to execution position by construction. 200 s cooldowns,
uninstrumented, -fa off parsed back per run, greedy, seed 123. Prefill duration
is a rate, so it gets the cooled protocol.

Reports the residual-on-slot fit and the mean slot per cell, per the standing
rule that position balance be visible rather than assumed.
"""

import csv, datetime, math, os, random, re, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"
BIN       = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN / "llama-completion"
TOKENIZE_BIN   = BIN / "llama-tokenize"
MODEL_PATH = Path(os.environ.get(
    "UNIKV_MODEL", LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
TAG = os.environ.get("UNIKV_TAG", "ub")

RESULTS_DIR = ROOT / "stress_results"
ART_DIR     = ROOT / "artifacts" / f"{TAG}_cooled"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

PROMPT_TOKENS = int(os.environ.get("UNIKV_UB_PROMPT", "16384"))
GEN_TOKENS    = int(os.environ.get("UNIKV_UB_GEN", "8"))
TRIALS        = int(os.environ.get("UNIKV_UB_TRIALS", "3"))
COOLDOWN_S    = int(os.environ.get("UNIKV_UB_COOLDOWN", "200"))
CELLS = [tuple(int(x) for x in c.split(":"))
         for c in os.environ.get("UNIKV_UB_CELLS",
                                 "64:49152,128:49152,256:49152,512:49152").split(",")]
OUT_NAME  = os.environ.get("UNIKV_UB_OUT", "ubatch_cooled_block.csv")
SHUFFLE   = int(os.environ.get("UNIKV_UB_SHUFFLE", "20260822"))
THREADS, GPU_LAYERS, SEED = 10, 999, 123
TIMEOUT_S = int(os.environ.get("UNIKV_UB_TIMEOUT", "1800"))


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


def run_cell(prompt, ub, ctx, tag):
    args = [str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
            "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(max(512, ub)),
            "-ub", str(ub), "-ngl", str(GPU_LAYERS), "-t", str(THREADS),
            "-fa", "off", "-fit", "off", "--temp", "0", "--seed", str(SEED),
            "--ignore-eos", "--no-warmup", "--simple-io", "--no-display-prompt",
            "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": "0", "UNIKV_ALPHA": "0", "UNIKV_MEM_BREAKDOWN": "1"})
    for k in ("UNIKV_LOG", "UNIKV_E2E_LOG", "UNIKV_NO_REENCODE",
              "UNIKV_WINDOW_DISCARD", "UNIKV_LOGIT_LOG"): env.pop(k, None)
    t0 = time.time()
    try:
        d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, errors="replace",
                           timeout=TIMEOUT_S)
        rc, err, to = d.returncode, d.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, to = None, True
        err = e.stderr.decode("utf-8", "replace") if e.stderr else ""
    dur = time.time() - t0
    (LOGS_DIR / f"gen_{tag}.txt").write_text(err[-9000:])

    def g(pat, cast=str):
        m = re.search(pat, err)
        return cast(m.group(1)) if m else None
    oom = len(re.findall(r"kIOGPUCommandBufferCallbackErrorOutOfMemory", err))
    prefill_ms = g(r"prompt eval time =\s+([\d.]+) ms", float)
    return {"rc": rc, "timed_out": to, "duration_s": round(dur, 1),
            "prefill_ms": prefill_ms,
            "prefill_tok_s": (PROMPT_TOKENS / (prefill_ms / 1000.0)) if prefill_ms else None,
            "flash_attn": g(r"flash_attn\s*=\s*(\w+)"),
            "kv_mib": g(r"llama_kv_cache:\s+size\s+=\s+([\d.]+)\s+MiB", float),
            "scratch_mib": g(r"MTL0 compute buffer size =\s+([\d.]+)", float),
            "gpu_oom_lines": oom,
            "outcome": ("GPU_OOM" if oom else "TIMEOUT" if to else
                        "COMPLETES" if rc == 0 else f"RC_{rc}")}


def main():
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR): d.mkdir(parents=True, exist_ok=True)
    prompt = ensure_prompt(PROMPT_TOKENS)
    rng, plan = random.Random(SHUFFLE), []
    for r in range(1, TRIALS + 1):
        rnd = CELLS[:]; rng.shuffle(rnd)
        plan += [(ub, ctx, r) for ub, ctx in rnd]

    print(f"cooled prefill block: {len(plan)} runs "
          f"({len(CELLS)} cells x {TRIALS} rounds), {COOLDOWN_S}s cooldowns")
    print(f"  model {MODEL_PATH.name}, prompt {PROMPT_TOKENS} tok, -fa off, uninstrumented")
    print(f"  cells (ubatch:ctx): {', '.join(f'{u}:{c}' for u,c in CELLS)}")
    print(f"  randomised COMPLETE block, seed {SHUFFLE}\n")

    out = RESULTS_DIR / OUT_NAME
    cols = ["order_idx", "round", "n_ubatch", "ctx", "outcome", "rc", "prefill_ms",
            "prefill_tok_s", "duration_s", "kv_mib", "scratch_mib", "gpu_oom_lines",
            "flash_attn", "t_start"]
    with out.open("w", newline="") as fh: csv.writer(fh).writerow(cols)

    rows, flags = [], []
    for idx, (ub, ctx, rnd) in enumerate(plan, 1):
        if COOLDOWN_S:
            print(f"  [{idx:2d}/{len(plan)}] cooldown {COOLDOWN_S}s ...", flush=True)
            time.sleep(COOLDOWN_S)
        t0 = time.time()
        print(f"  [{idx:2d}/{len(plan)}] r{rnd} ub={ub:4d} C={ctx:6d} ...", end="", flush=True)
        r = run_cell(prompt, ub, ctx, f"ub{ub}_c{ctx}_r{rnd}")
        print(f" {r['outcome']:10s} prefill {r['prefill_tok_s'] and round(r['prefill_tok_s'],1)} tok/s "
              f"({r['duration_s']}s)")
        r.update({"order_idx": idx, "round": rnd, "n_ubatch": ub, "ctx": ctx,
                  "t_start": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")})
        rows.append(r)
        if r["flash_attn"] != "disabled":
            flags.append(f"ub{ub}/C{ctx}/r{rnd}: flash_attn={r['flash_attn']}")
        with out.open("a", newline="") as fh:
            csv.writer(fh).writerow([r.get(c) for c in cols])

    print("\n" + "=" * 80)
    by = {}
    for r in rows:
        if r["prefill_tok_s"]: by.setdefault((r["n_ubatch"], r["ctx"]), []).append(r["prefill_tok_s"])
    print(f"{'ubatch':>7} {'ctx':>7} {'n':>2} {'prefill tok/s':>14} {'sd':>7} {'se':>7} "
          f"{'duration s':>11} {'outcomes':>22}")
    for (ub, ctx) in CELLS:
        v = by.get((ub, ctx), [])
        outs = {r["outcome"] for r in rows if r["n_ubatch"] == ub and r["ctx"] == ctx}
        durs = [r["duration_s"] for r in rows if r["n_ubatch"] == ub and r["ctx"] == ctx]
        if v:
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"{ub:>7} {ctx:>7} {len(v):>2} {st.mean(v):>14.2f} {sd:>7.2f} "
                  f"{sd/math.sqrt(len(v)) if len(v)>1 else 0:>7.2f} "
                  f"{st.mean(durs):>11.1f} {','.join(sorted(outs)):>22}")
        else:
            print(f"{ub:>7} {ctx:>7} {'0':>2} {'n/a':>14} {'':>7} {'':>7} "
                  f"{st.mean(durs) if durs else 0:>11.1f} {','.join(sorted(outs)):>22}")

    # position diagnostics, per the standing rule
    print("\nposition diagnostics (residual-on-slot, and mean slot per cell):")
    xs, ys = [], []
    for (ub, ctx), v in by.items():
        m = st.mean(v)
        for r in rows:
            if r["n_ubatch"] == ub and r["ctx"] == ctx and r["prefill_tok_s"]:
                xs.append(r["order_idx"]); ys.append(r["prefill_tok_s"] - m)
    if len(xs) > 2:
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
        se = st.stdev(resid) / math.sqrt(sxx)
        print(f"  drift {b:+.4f} +/- {se:.4f} tok/s per slot  (t = {b/se:+.2f})")
    for (ub, ctx) in CELLS:
        s = [r["order_idx"] for r in rows if r["n_ubatch"] == ub and r["ctx"] == ctx]
        print(f"  ub{ub:<4} C={ctx:<7} slots {sorted(s)}  mean {st.mean(s):.2f}")

    if flags:
        print("\n!! FLAGS:")
        for f in flags: print(f"   {f}")
    print(f"\nwrote {out}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
