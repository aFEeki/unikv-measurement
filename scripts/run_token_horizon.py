#!/usr/bin/env python3
"""Token-identity horizon: how far does exact retention stay bit-identical?

The paper's only positive recommendation — that exact retention buys
reproducible decode — rested on 32 generated tokens on one prompt. This extends
the horizon to 512 tokens on two structurally different prompts and reports, per
arm per prompt, the token index at which the arm FIRST diverges from the
no-eviction reference rather than only whether it does.

Arms (one block, one batch tiling, flash attention off, greedy, seed 123):
  ref_c8192      policy 0, C=8192  -- the no-eviction reference
  p3_cpu_c1024   policy 3, C=1024, UNIKV_SPILL_DEV=0   exact retention, CPU tier
  p3_dev_c1024   policy 3, C=1024, UNIKV_SPILL_DEV=1   exact retention, GPU tier
  p4_h2o_c1024   policy 4, C=1024                      H2O fixed-budget eviction
  ctl_ref_c4096  policy 0, C=4096  -- CONTROL, see below

Why the reference is C=8192 and not C=4096. The brief specified C=4096, which
was right at the old 32-token horizon: 3834 + 32 = 3866 fits. At 512 tokens the
same prompt needs 4346 cells, so C=4096 would itself evict and could not serve
as a no-eviction reference. C=8192 holds the whole run with room to spare.

ctl_ref_c4096 exists to show that choice is harmless: over the 262 generated
tokens before C=4096 fills, it must emit exactly the same tokens as the C=8192
reference. If it does, the reference does not depend on the cache size and the
substitution is free. If it does not, that is itself a finding and every
identity claim in the paper would need re-examining.

Two prompts, matched to the same token count so the arms see identical cache
pressure and differ only in structure:
  passkey  the published probe: repetitive filler with a planted key. Low
           entropy, and therefore possibly unusually easy to reproduce.
  prose    a prefix of llama.cpp's README: natural technical English with
           markdown, lists, links and code spans. Much higher entropy, and the
           harder test of the identity claim.

The run does NOT stop at first divergence — every arm generates the full budget
and the complete token sequence is written out, because the trace after the
divergence is part of the result.

Identity under greedy decoding with a fixed seed is deterministic and does not
depend on thermal state, so this block uses no cooldowns. It also records
throughput, but those figures are UNCOOLED and single-trial and must not be
quoted as rates.
"""

import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT           = Path(__file__).resolve().parents[1]
LLAMA_DIR      = ROOT / "llama.cpp"
BIN_DIR        = LLAMA_DIR / "build-m4pro-metal" / "bin"
COMPLETION_BIN = BIN_DIR / "llama-completion"
TOKENIZE_BIN   = BIN_DIR / "llama-tokenize"
# Model and output tag are overridable so a second model runs THIS code path.
# With both unset the Llama block reproduces exactly.
MODEL_PATH     = Path(os.environ.get(
    "UNIKV_MODEL", LLAMA_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
TAG            = os.environ.get("UNIKV_TAG", "")
SUF            = f"_{TAG}" if TAG else ""
PROSE_SRC      = LLAMA_DIR / "README.md"

RESULTS_DIR = ROOT / "quality_results"
ART_DIR     = ROOT / "artifacts" / f"token_horizon{SUF}"
PROMPTS_DIR = ART_DIR / "prompts"
LOGS_DIR    = ART_DIR / "logs"

GEN_TOKENS    = int(os.environ.get("UNIKV_TH_GEN", "512"))
PROMPT_TOKENS = 3834          # matches the published probe exactly
THREADS       = 10
GPU_LAYERS    = 999
BATCH         = 256           # ONE batch tiling, as in the six-arm block
UBATCH        = 256
SEED          = 123
SPILL_CAP     = 8192
CEILING       = 58.0

PASSKEY      = "48291"
FILLER       = ("The grass is green. The sky is blue. The sun is yellow. "
                "Here we go. There and back again. ")
INTRO        = ("There is an important piece of information hidden inside the "
                "text below. Find it and remember it, because you will be asked "
                "about it at the end.\n\n")
PASSKEY_LINE = f"\nThe pass key is {PASSKEY}. Remember it. {PASSKEY} is the pass key.\n\n"
QUESTION     = "\nQuestion: What is the pass key?\nAnswer: The pass key is"
PAD_TOKEN    = " token"     # one token in both tokenizers; used only to land exactly

REFERENCE = "ref_c8192"
# tag -> (ctx, policy, spill_dev, role)
ARMS = {
    "ref_c8192":     (8192, 0, "0", "no-eviction reference"),
    "p3_cpu_c1024":  (1024, 3, "0", "exact retention, CPU-pinned tier"),
    "p3_dev_c1024":  (1024, 3, "1", "exact retention, device-visible tier"),
    "p4_h2o_c1024":  (1024, 4, "0", "H2O fixed-budget eviction"),
    "ctl_ref_c4096": (4096, 0, "0", "CONTROL: reference at C=4096 (fills at ~262)"),
}
ORDER = ["ref_c8192", "p3_cpu_c1024", "p3_dev_c1024", "p4_h2o_c1024", "ctl_ref_c4096"]


def tok_count(path: Path) -> int:
    out = subprocess.run(
        [str(TOKENIZE_BIN), "-m", str(MODEL_PATH), "-f", str(path),
         "--show-count", "--log-disable"],
        cwd=LLAMA_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=True).stdout
    m = re.search(r"Total number of tokens:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"token count parse failed:\n{out}")
    return int(m.group(1))


def build_passkey_prompt() -> Path:
    """The published probe, solved to exactly PROMPT_TOKENS under this tokenizer.

    The Llama version hard-coded 150 filler repeats and asserted the total. A
    different tokenizer gives a different count for the same text, so the repeat
    count is solved for here and the remainder padded with a single-token unit.
    The prompt therefore stays the same PROBE (same intro, same planted key, same
    question, same repetitive filler) at the same length across models, which is
    what the comparison needs; only the repeat count moves."""
    p = PROMPTS_DIR / "prompt_passkey.txt"

    def build(reps, pad):
        p.write_text(INTRO + (FILLER * 7) + PASSKEY_LINE + (FILLER * reps)
                     + (PAD_TOKEN * pad) + QUESTION)
        return tok_count(p)

    lo, hi, best = 0, 400, None      # largest repeat count that stays under budget
    while lo <= hi:
        mid = (lo + hi) // 2
        if build(mid, 0) <= PROMPT_TOKENS:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError("passkey scaffold alone exceeds the budget")
    # Pad to the exact count. The pad count is SOLVED, not assumed: inserting the
    # unit changes tokenization at the seams, so k pad units need not add k tokens.
    pad, seen = 0, {}
    for _ in range(16):
        n = seen[pad] = build(best, pad)
        if n == PROMPT_TOKENS:
            return p
        nxt = pad + (PROMPT_TOKENS - n)
        if nxt in seen or nxt < 0:
            break
        pad = nxt
    raise RuntimeError(f"passkey prompt: pad count -> token count {seen}, "
                       f"never hit {PROMPT_TOKENS}")


def build_prose_prompt() -> Path:
    """A README prefix trimmed to exactly PROMPT_TOKENS, so the two prompts differ
    in structure and entropy but not in length or cache pressure."""
    p = PROMPTS_DIR / "prompt_prose.txt"
    if p.exists() and tok_count(p) == PROMPT_TOKENS:
        return p
    src = PROSE_SRC.read_text(encoding="utf-8", errors="replace")
    lo, hi = 0, len(src)
    best = None
    while lo <= hi:                       # binary search on characters
        mid = (lo + hi) // 2
        p.write_text(src[:mid])
        n = tok_count(p)
        if n == PROMPT_TOKENS:
            best = src[:mid]
            break
        if n < PROMPT_TOKENS:
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:                      # walk to an exact hit
        p.write_text(src[:lo])
        n = tok_count(p)
        while n != PROMPT_TOKENS and 0 < lo < len(src):
            lo += 1 if n < PROMPT_TOKENS else -1
            p.write_text(src[:lo])
            n = tok_count(p)
        if n != PROMPT_TOKENS:
            raise RuntimeError(f"could not trim prose prompt to {PROMPT_TOKENS}: got {n}")
        best = src[:lo]
    p.write_text(best)
    return p


def run_arm(prompt: Path, prompt_name: str, tag: str) -> dict:
    ctx, policy, spill_dev, role = ARMS[tag]
    stem    = f"{prompt_name}_{tag}"
    tok_log = LOGS_DIR / f"tokens_{stem}.txt"
    e2e_csv = LOGS_DIR / f"e2e_{stem}.csv"
    tok_log.unlink(missing_ok=True)
    e2e_csv.unlink(missing_ok=True)

    args = [
        str(COMPLETION_BIN), "-m", str(MODEL_PATH), "-f", str(prompt),
        "-n", str(GEN_TOKENS), "-c", str(ctx), "-b", str(BATCH), "-ub", str(UBATCH),
        "-ngl", str(GPU_LAYERS), "-t", str(THREADS), "-fa", "off", "-fit", "off",
        "--temp", "0", "--seed", str(SEED), "--ignore-eos", "--no-warmup",
        "--simple-io", "--no-display-prompt", "-no-cnv",
    ]
    env = os.environ.copy()
    env.update({"UNIKV_POLICY": str(policy), "UNIKV_ALPHA": "0",
                "UNIKV_SPILL_DEV": spill_dev, "UNIKV_SPILL_CAP": str(SPILL_CAP),
                "UNIKV_TOKEN_LOG": str(tok_log), "UNIKV_E2E_LOG": str(e2e_csv)})
    env.pop("UNIKV_LOG", None)
    env.pop("UNIKV_H2O_TRACE", None)

    t0 = time.time()
    d = subprocess.run(args, cwd=LLAMA_DIR, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, errors="replace")
    dur = time.time() - t0
    (LOGS_DIR / f"gen_{stem}.txt").write_text(
        "=== STDOUT ===\n" + d.stdout + "\n=== STDERR ===\n" + d.stderr[-16000:])

    ids = []
    if tok_log.exists():
        ids = [int(x) for x in tok_log.read_text().split() if x.strip()]

    tps = None
    if e2e_csv.exists():
        rows = list(csv.DictReader(e2e_csv.open()))
        if rows:
            tps = float(rows[-1]["tok_per_sec"])

    mfa = re.search(r"flash_attn\s*=\s*(\w+)", d.stderr)
    mb  = re.search(r"n_batch\s*=\s*(\d+)", d.stderr)
    mu  = re.search(r"n_ubatch\s*=\s*(\d+)", d.stderr)
    text = d.stdout.split("UNIKV_E2E")[0].strip()

    return {
        "prompt": prompt_name, "arm": tag, "role": role, "ctx": ctx,
        "policy": policy, "spill_dev": spill_dev, "rc": d.returncode,
        "n_tokens": len(ids), "token_ids": ids,
        "flash_attn": mfa.group(1) if mfa else "UNPARSED",
        "n_batch": int(mb.group(1)) if mb else None,
        "n_ubatch": int(mu.group(1)) if mu else None,
        "tok_per_sec": tps, "duration_s": round(dur, 1),
        "passkey_found": PASSKEY in text,
        "token_md5": hashlib.md5(" ".join(map(str, ids)).encode()).hexdigest(),
        "text_head": text[:120],
    }


def first_divergence(ref, arm):
    """Index of the first differing token, or None if one is a prefix of the other."""
    for i, (a, b) in enumerate(zip(ref, arm)):
        if a != b:
            return i
    return None


def main() -> int:
    for d in (RESULTS_DIR, PROMPTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in (COMPLETION_BIN, TOKENIZE_BIN, MODEL_PATH, PROSE_SRC):
        if not p.exists():
            raise FileNotFoundError(p)

    prompts = {"passkey": build_passkey_prompt(), "prose": build_prose_prompt()}
    print(f"token-identity horizon: {GEN_TOKENS} generated tokens, "
          f"{len(prompts)} prompts x {len(ARMS)} arms")
    print(f"  model {MODEL_PATH.name}")
    for name, path in prompts.items():
        print(f"  prompt {name:8s} {tok_count(path):5d} tokens  {path.name}")
    print(f"  -b {BATCH} -ub {UBATCH}, -fa off, greedy, seed {SEED}, --ignore-eos")
    print(f"  reference = {REFERENCE}; no cooldowns (identity is deterministic)\n")

    rows, flags = [], []
    for pname, ppath in prompts.items():
        print(f"== prompt: {pname} ==")
        for tag in ORDER:
            print(f"  {tag:14s} ...", end="", flush=True)
            r = run_arm(ppath, pname, tag)
            rows.append(r)
            print(f" {r['n_tokens']:4d} tok  rc={r['rc']}  fa={r['flash_attn']}  "
                  f"{r['tok_per_sec']} tok/s  ({r['duration_s']}s)")
            if r["flash_attn"] != "disabled":
                flags.append(f"{pname}/{tag}: flash_attn={r['flash_attn']}")
            if (r["n_batch"], r["n_ubatch"]) != (BATCH, UBATCH):
                flags.append(f"{pname}/{tag}: batch tiling "
                             f"{r['n_batch']}/{r['n_ubatch']}")
            if r["tok_per_sec"] and r["tok_per_sec"] >= CEILING:
                flags.append(f"{pname}/{tag}: {r['tok_per_sec']} >= {CEILING}")
        print()

    # ---------------- divergence analysis ----------------
    print("=" * 78)
    print(f"FIRST DIVERGENCE FROM THE REFERENCE ({REFERENCE}), per prompt")
    print("=" * 78)
    analysis = []
    for pname in prompts:
        ref = next(r for r in rows if r["prompt"] == pname and r["arm"] == REFERENCE)
        print(f"\n-- {pname} --   reference emitted {ref['n_tokens']} tokens, "
              f"md5 {ref['token_md5'][:12]}")
        for tag in ORDER:
            if tag == REFERENCE:
                continue
            arm = next(r for r in rows if r["prompt"] == pname and r["arm"] == tag)
            common = min(len(ref["token_ids"]), len(arm["token_ids"]))
            idx = first_divergence(ref["token_ids"][:common], arm["token_ids"][:common])
            if idx is None:
                verdict = (f"IDENTICAL for all {common} compared tokens"
                           if common else "no tokens to compare")
            else:
                verdict = (f"diverges at token {idx} "
                           f"(ref {ref['token_ids'][idx]} vs {arm['token_ids'][idx]})")
            analysis.append({"prompt": pname, "arm": tag, "role": arm["role"],
                             "n_tokens": arm["n_tokens"], "compared": common,
                             "first_divergence": idx, "verdict": verdict,
                             "passkey_found": arm["passkey_found"]})
            print(f"   {tag:14s} {arm['n_tokens']:4d} tok  {verdict}")

    # the control must reproduce the reference over its own shorter horizon
    print("\n" + "-" * 78)
    for pname in prompts:
        c = next(a for a in analysis if a["prompt"] == pname and a["arm"] == "ctl_ref_c4096")
        ok = c["first_divergence"] is None
        print(f"CONTROL {pname:8s}: reference at C=4096 vs C=8192 over "
              f"{c['compared']} tokens -> {'AGREE' if ok else 'DISAGREE'}")
        if not ok:
            flags.append(f"{pname}: the reference depends on C — "
                         f"diverges at token {c['first_divergence']}. Every identity "
                         f"claim in the paper needs re-examining.")

    out = RESULTS_DIR / f"token_horizon{SUF}.csv"
    cols = ["prompt", "arm", "role", "ctx", "policy", "spill_dev", "rc", "n_tokens",
            "compared_tokens", "first_divergence", "identical", "flash_attn",
            "n_batch", "n_ubatch", "tok_per_sec", "passkey_found", "token_md5",
            "duration_s", "text_head"]
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            a = next((x for x in analysis
                      if x["prompt"] == r["prompt"] and x["arm"] == r["arm"]), None)
            w.writerow([r["prompt"], r["arm"], r["role"], r["ctx"], r["policy"],
                        r["spill_dev"], r["rc"], r["n_tokens"],
                        a["compared"] if a else r["n_tokens"],
                        a["first_divergence"] if a else "",
                        "" if a is None else (a["first_divergence"] is None),
                        r["flash_attn"], r["n_batch"], r["n_ubatch"],
                        r["tok_per_sec"], r["passkey_found"], r["token_md5"],
                        r["duration_s"], r["text_head"]])
    print(f"\nwrote {out}")

    seq_dir = ART_DIR / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (seq_dir / f"{r['prompt']}_{r['arm']}.txt").write_text(
            " ".join(map(str, r["token_ids"])) + "\n")
    print(f"wrote full token sequences -> {seq_dir}")

    if flags:
        print("\n!! FLAGS:")
        for f in flags:
            print(f"   {f}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
