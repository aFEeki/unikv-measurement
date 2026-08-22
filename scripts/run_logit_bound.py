#!/usr/bin/env python3
"""Review item M2 — a quantitative bound on exactness, not a token-identity claim.

DA-1 is right that token identity under greedy argmax cannot distinguish exact
from near-exact: two arms can agree on all 512 sampled tokens while their logits
differ by an amount that would flip a token on a different prompt. This measures
the underlying quantity directly.

UNIKV_LOGIT_LOG=<path> dumps the RAW model logits for every sampled step as
float32, before the sampler touches them. Differencing the files across arms
gives max |delta logit| over the 512-token horizon.

Arms (C=1024 for the retention tiers, C=8192 for the no-eviction reference, the
same geometry as the published token-identity block), both prompts, greedy,
seed 123, -fa off. Identity is deterministic, so no cooldowns and no repeats:
this measures arithmetic, not rate.

Outcome, per the brief:
  ~1e-6  -> exact to float32 round-off; state a quantitative bound and the
            recommendation in Section 9 gets stronger.
  ~1e-3  -> the arms are near-exact rather than exact, and that is a finding in
            its own right.
"""

import csv, os, re, subprocess, sys
from pathlib import Path
import struct

ROOT      = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"
BIN       = LLAMA_DIR / "build-m4pro-metal" / "bin" / "llama-completion"
MODEL     = LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PROMPTS   = ROOT / "artifacts" / "token_horizon" / "prompts"
ART       = ROOT / "artifacts" / "logit_bound"
RESULTS   = ROOT / "quality_results"

GEN, BATCH, SEED = 512, 256, 123
REF = "ref_c8192"
ARMS = {                       # tag -> (ctx, policy, spill_dev)
    "ref_c8192":    (8192, 0, "0"),
    "p3_cpu_c1024": (1024, 3, "0"),
    "p3_dev_c1024": (1024, 3, "1"),
    "p4_h2o_c1024": (1024, 4, "0"),
}


def run(prompt, tag, pname):
    ctx, pol, dev = ARMS[tag]
    lg = ART / f"logits_{pname}_{tag}.bin"
    tk = ART / f"tokens_{pname}_{tag}.txt"
    for f in (lg, tk): f.unlink(missing_ok=True)
    args = [str(BIN), "-m", str(MODEL), "-f", str(prompt), "-n", str(GEN),
            "-c", str(ctx), "-b", str(BATCH), "-ub", str(BATCH), "-ngl", "999",
            "-t", "10", "-fa", "off", "-fit", "off", "--temp", "0",
            "--seed", str(SEED), "--ignore-eos", "--no-warmup", "--simple-io",
            "--no-display-prompt", "-no-cnv"]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(pol), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": dev, "UNIKV_SPILL_CAP": "8192",
                "UNIKV_LOGIT_LOG": str(lg), "UNIKV_TOKEN_LOG": str(tk)})
    for k in ("UNIKV_LOG", "UNIKV_NO_REENCODE", "UNIKV_WINDOW_DISCARD"): env.pop(k, None)
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    fa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    return {"rc": d.returncode, "fa": fa.group(1) if fa else "UNPARSED",
            "logits": lg, "tokens": tk}


def load(path, n_vocab):
    raw = path.read_bytes()
    n = len(raw) // 4 // n_vocab
    return raw, n


def compare(a_path, b_path, n_vocab):
    ra, na = load(a_path, n_vocab)
    rb, nb = load(b_path, n_vocab)
    n = min(na, nb)
    import array
    A = array.array("f"); A.frombytes(ra[:n*n_vocab*4])
    B = array.array("f"); B.frombytes(rb[:n*n_vocab*4])
    max_abs, at_step, n_ident = 0.0, -1, 0
    for s in range(n):
        o = s * n_vocab
        m = 0.0
        for i in range(o, o + n_vocab):
            d = A[i] - B[i]
            if d < 0: d = -d
            if d > m: m = d
        if m == 0.0: n_ident += 1
        if m > max_abs: max_abs, at_step = m, s
    return n, max_abs, at_step, n_ident


def main():
    ART.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    prompts = {"passkey": PROMPTS / "prompt_passkey.txt",
               "prose":   PROMPTS / "prompt_prose.txt"}
    for p in prompts.values():
        if not p.exists(): raise SystemExit(f"missing prompt {p}")

    # vocab size, parsed from the runtime rather than assumed
    probe = subprocess.run([str(BIN), "-m", str(MODEL), "-p", "hi", "-n", "1",
                            "-c", "256", "-ngl", "999", "--no-warmup",
                            "--simple-io", "-no-cnv"], cwd=LLAMA_DIR,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, errors="replace")
    mv = re.search(r"n_vocab\s*=\s*(\d+)", probe.stderr)
    if not mv: raise SystemExit("could not parse n_vocab")
    n_vocab = int(mv.group(1))
    print(f"n_vocab = {n_vocab} (parsed from the runtime)\n")

    rows = []
    for pname, ppath in prompts.items():
        print(f"== {pname} ==")
        got = {}
        for tag in ARMS:
            r = run(ppath, tag, pname)
            sz = r["logits"].stat().st_size if r["logits"].exists() else 0
            steps = sz // 4 // n_vocab
            print(f"  {tag:14s} rc={r['rc']} fa={r['fa']} logit steps={steps}")
            if r["fa"] != "disabled": raise SystemExit("flash attention not off")
            got[tag] = r
        for tag in ARMS:
            if tag == REF: continue
            n, mx, at, ident = compare(got[REF]["logits"], got[tag]["logits"], n_vocab)
            ta = got[REF]["tokens"].read_text().split()
            tb = got[tag]["tokens"].read_text().split()
            same = sum(1 for x, y in zip(ta, tb) if x == y)
            print(f"     vs {REF}: max|dlogit| = {mx:.6g} at step {at}; "
                  f"{ident}/{n} steps bit-identical; tokens {same}/{min(len(ta),len(tb))}")
            rows.append({"prompt": pname, "arm": tag, "steps": n,
                         "max_abs_dlogit": mx, "at_step": at,
                         "steps_bit_identical": ident,
                         "tokens_matching": same, "n_vocab": n_vocab})
        print()

    out = RESULTS / "logit_bound.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
