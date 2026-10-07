#!/usr/bin/env python3
"""H2O comparator checks that need no eviction: idle identity and the passkey's position.

Section 5 of the paper checks that its H2O implementation is not a strawman.
This harness covers the parts that run_quality_arms_unified.py's block does not:

  idle identity   at C = 4096 the passkey prompt (3834 tokens) and 32 generated
                  tokens fit without eviction, so policy 4 must emit exactly the
                  tokens of the unmodified runtime (policy 0). Any difference
                  would mean the scoring machinery perturbs the computation.
  passkey place   the token positions of the passkey in the prompt, and the
                  span of its line from the newline before it to the newline
                  after it, so that analyze_h2o_survivors.py can rank them in
                  the survivor trace of the C = 1024 H2O arm.

Same prompt, settings and arm runner as run_quality_arms_unified.py: greedy,
seed 123, -fa off, -b/-ub 256, 32 generated tokens. The reference arm is also
compared with the ref_p0_c4096 sequence stored in quality_arms_unified.csv, as a
check that the current build reproduces it. Deterministic, so there are no
cooldowns and no repeats.

  python3 scripts/run_h2o_idle_check.py
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_quality_arms_unified as qa   # noqa: E402  (prompt, settings, run_arm)

qa.ARTIFACT_DIR = ROOT / "artifacts" / "h2o_idle_check"
qa.PROMPTS_DIR  = qa.ARTIFACT_DIR / "prompts"
qa.LOGS_DIR     = qa.ARTIFACT_DIR / "logs"

OUT    = ROOT / "quality_results" / "h2o_idle_check.csv"
STORED = ROOT / "quality_results" / "quality_arms_unified.csv"

# (tag, ctx, policy, sink, role)
ARMS = [
    ("ref_p0_c4096", 4096, 0, 0, "unmodified runtime, no eviction"),
    ("h2o_p4_c4096", 4096, 4, 0, "H2O, no eviction"),
]
FIELDS = ["arm", "role", "ctx", "policy", "rc", "flash_attn", "n_batch", "n_ubatch",
          "prompt_tokens", "gen_tokens", "demotion_calls", "token_md5",
          "identical_to_ref", "first_divergence", "matches_stored_ref",
          "passkey_found", "passkey_positions", "passkey_line"]


def token_ids(path: Path, bos: bool = True) -> list:
    args = [str(qa.TOKENIZE_BIN), "-m", str(qa.MODEL_PATH), "-f", str(path),
            "--ids", "--log-disable"]
    if not bos:
        args.append("--no-bos")
    out = subprocess.run(args, cwd=qa.LLAMA_DIR, capture_output=True, text=True,
                         check=True).stdout
    return [int(x) for x in out.strip().strip("[]").split(",") if x.strip()]


def token_pieces(path: Path) -> list:
    out = subprocess.run([str(qa.TOKENIZE_BIN), "-m", str(qa.MODEL_PATH), "-f", str(path),
                          "--log-disable"], cwd=qa.LLAMA_DIR, capture_output=True,
                         text=True, check=True).stdout
    return [p for _, p in re.findall(r"(\d+) -> '((?:[^']|'(?!\s*\n\s*\d+ ->))*)'", out)]


def passkey_line(prompt: Path, occ: list) -> str:
    """From the newline before the first passkey token to the newline after the last."""
    pieces = token_pieces(prompt)
    lo = max(i for i in range(occ[0][0] + 1) if "\n" in pieces[i])
    hi = min(i for i in range(occ[-1][-1], len(pieces)) if "\n" in pieces[i])
    return f"{lo}-{hi}"


def passkey_positions(prompt: Path) -> list:
    """Positions (BOS = 0, as in the survivor trace) of each passkey occurrence."""
    key_file = qa.PROMPTS_DIR / "_passkey.txt"
    key_file.write_text(qa.PASSKEY)
    key = token_ids(key_file, bos=False)
    ids = token_ids(prompt)
    hits = [i for i in range(len(ids) - len(key) + 1) if ids[i:i + len(key)] == key]
    if not hits:
        raise SystemExit(f"STOP: passkey tokens {key} not found in the prompt")
    return [list(range(i, i + len(key))) for i in hits]


def main() -> int:
    qa.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    qa.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt, n_prompt = qa.build_prompt()
    pk = passkey_positions(prompt)
    pk_str = ";".join(" ".join(str(p) for p in occ) for occ in pk)
    line = passkey_line(prompt, pk)
    print(f"prompt {n_prompt} tokens; passkey at positions {pk_str}; line {line}")

    stored = {r["arm"]: r for r in csv.DictReader(STORED.open())}
    rows, flags = {}, []
    for tag, ctx, policy, sink, role in ARMS:
        r = qa.run_arm(prompt, tag, ctx, policy, sink, role)
        rows[tag] = r
        print(f"  {tag}: rc={r['rc']} fa={r['flash_attn']} tokens={len(r['token_ids'])} "
              f"demotions={r['demotion_calls']} md5={r['token_md5']}", flush=True)
        if r["rc"] != 0:
            flags.append(f"{tag}: rc={r['rc']}")
        if r["flash_attn"] != "disabled":
            flags.append(f"{tag}: flash_attn={r['flash_attn']}")
        if r["demotion_calls"] != 0:
            flags.append(f"{tag}: {r['demotion_calls']} demotions; the check needs none")
        if len(r["token_ids"]) != qa.GEN_TOKENS:
            flags.append(f"{tag}: {len(r['token_ids'])} tokens, expected {qa.GEN_TOKENS}")

    ref = rows["ref_p0_c4096"]
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for tag, ctx, policy, _, role in ARMS:
            r = rows[tag]
            div = qa.first_divergence(r["token_ids"], ref["token_ids"])
            w.writerow([tag, role, ctx, policy, r["rc"], r["flash_attn"], r["n_batch"],
                        r["n_ubatch"], n_prompt, len(r["token_ids"]), r["demotion_calls"],
                        r["token_md5"], r["token_ids"] == ref["token_ids"], div,
                        r["token_md5"] == stored["ref_p0_c4096"]["token_md5"],
                        r["passkey_found"], pk_str, line])

    same = rows["h2o_p4_c4096"]["token_ids"] == ref["token_ids"]
    print(f"\nH2O at C=4096 identical to the unmodified runtime: {same}")
    print(f"reference identical to the stored ref_p0_c4096: "
          f"{ref['token_md5'] == stored['ref_p0_c4096']['token_md5']}")
    if not same:
        flags.append("H2O differs from the unmodified runtime with no eviction")
    print("FLAGS:", flags or "none")
    print(f"wrote {OUT}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
