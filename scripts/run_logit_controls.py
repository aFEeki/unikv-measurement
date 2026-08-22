#!/usr/bin/env python3
"""Controls for M2 — is the measured delta-logit the SPILL, or just graph shape?

The main M2 run compares a C=8192 reference against C=1024 retention arms. Those
differ in TWO ways: the spill mechanism, and the size of the attention window,
which changes the shape of the reduction and therefore the float summation order.
Non-associativity alone can move a logit. A raw delta-logit from that comparison
attributes to the spill something that may just be arithmetic reordering.

Three controls, all on the passkey prompt, all -fa off, greedy, seed 123:

  A. DETERMINISM   policy 0 C=8192, run twice, identical settings.
                   Must be exactly 0. If it is not, nothing else here means
                   anything.
  B. GRAPH SHAPE   policy 0 C=8192 vs policy 0 C=4096, NO SPILL ANYWHERE, over
                   the 262 generated tokens before C=4096 fills. Any delta here
                   is pure cache-size / reduction-order effect. This is the
                   number the retention arms must be compared AGAINST, not zero.
  C. TIER          policy 3 CPU-pinned vs policy 3 device-visible, both C=1024,
                   same spilled set. Isolates the tier placement from the
                   spill itself.
"""

import os, re, subprocess, sys, array
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"
BIN       = LLAMA_DIR / "build-m4pro-metal" / "bin" / "llama-completion"
MODEL     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PROMPT    = ROOT / "artifacts" / "token_horizon" / "prompts" / "prompt_passkey.txt"
ART       = ROOT / "artifacts" / "logit_bound"
N_VOCAB   = 128256
BATCH     = 256


def run(tag, ctx, pol, dev, gen):
    lg = ART / f"ctl_{tag}.bin"; lg.unlink(missing_ok=True)
    args = [str(BIN), "-m", str(MODEL), "-f", str(PROMPT), "-n", str(gen),
            "-c", str(ctx), "-b", str(BATCH), "-ub", str(BATCH), "-ngl", "999",
            "-t", "10", "-fa", "off", "-fit", "off", "--temp", "0",
            "--seed", "123", "--ignore-eos", "--no-warmup", "--simple-io",
            "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(pol), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": dev, "UNIKV_SPILL_CAP": "8192",
                "UNIKV_LOGIT_LOG": str(lg)})
    for k in ("UNIKV_LOG", "UNIKV_NO_REENCODE", "UNIKV_WINDOW_DISCARD"): env.pop(k, None)
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    fa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    assert d.returncode == 0 and fa and fa.group(1) == "disabled", (d.returncode, d.stderr[-400:])
    return lg


def cmp(a, b, limit=None):
    A = array.array("f"); A.frombytes(a.read_bytes())
    B = array.array("f"); B.frombytes(b.read_bytes())
    n = min(len(A), len(B)) // N_VOCAB
    if limit: n = min(n, limit)
    mx, at, ident = 0.0, -1, 0
    for s in range(n):
        o = s * N_VOCAB; m = 0.0
        for i in range(o, o + N_VOCAB):
            d = A[i] - B[i]
            if d < 0: d = -d
            if d > m: m = d
        if m == 0.0: ident += 1
        if m > mx: mx, at = m, s
    return n, mx, at, ident


def main():
    ART.mkdir(parents=True, exist_ok=True)
    print("A. DETERMINISM — same settings twice, must be exactly 0")
    a1 = run("det1", 8192, 0, "0", 64)
    a2 = run("det2", 8192, 0, "0", 64)
    n, mx, at, ident = cmp(a1, a2)
    print(f"   ref vs ref: max|dlogit| = {mx:.6g} over {n} steps, "
          f"{ident}/{n} bit-identical -> {'DETERMINISTIC' if mx == 0 else 'NON-DETERMINISTIC'}")
    if mx != 0.0:
        print("   STOP: the runtime is not deterministic; no delta below is interpretable.")
        return 1

    print("\nB. GRAPH SHAPE — policy 0 at two cache sizes, NO SPILL ANYWHERE")
    b1 = run("p0_c8192", 8192, 0, "0", 262)
    b2 = run("p0_c4096", 4096, 0, "0", 262)
    n, mx, at, ident = cmp(b1, b2, 262)
    print(f"   C=8192 vs C=4096: max|dlogit| = {mx:.6g} at step {at}, "
          f"{ident}/{n} bit-identical")
    print("   ^ this is the floor: cache size alone moves the logits by this much,")
    print("     with no spill mechanism involved at all.")
    graph_floor = mx

    print("\nC. TIER — policy 3 CPU-pinned vs device-visible, both C=1024")
    c1 = run("p3cpu", 1024, 3, "0", 512)
    c2 = run("p3dev", 1024, 3, "1", 512)
    n, mx, at, ident = cmp(c1, c2)
    print(f"   cpu vs dev: max|dlogit| = {mx:.6g} at step {at}, {ident}/{n} bit-identical")

    print("\nINTERPRETATION")
    print(f"   graph-shape floor (no spill):            {graph_floor:.6g}")
    print(f"   p3_dev vs ref (from the main M2 run):    0.172573")
    print(f"   p3_cpu vs ref (from the main M2 run):    1.99196")
    print(f"   p4_h2o vs ref (from the main M2 run):    33.4892")
    return 0


if __name__ == "__main__":
    sys.exit(main())
