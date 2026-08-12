# UniKV measurement index

> **2026-08-07 — ratio correction.** A MB/MiB unit error made every
> "x budget" ratio 4.9% low in some files (ggml prints the Metal budget in
> decimal MB; footprints are MiB; 17179.89 MB = 16384.0 MiB). MiB values,
> KiB/token figures and all outcomes are unaffected, as are Findings 1, 2, 3
> and 5. Corrected numbers and the full list:
> `artifacts/CORRECTION_2026-08-07_budget_units.txt`. **Two values in Finding
> 4's capacity table need fixing in the draft: 0.64 -> 0.67 and 1.72 -> 1.80.**

**2026-08-05 — Phase A (measurement paper, findings F1–F5).** See the section at
the bottom. Two Phase A results contradict Finding 4's "Beyond the ceiling"
paragraph and must be fixed before the preprint is posted.

---

# New measurements for the TACO major revision — 2026-08-04

Executor output for the measurement items in `paper/UNIKV-TACO/REVIEW_2026-08-03.md`.
**`main.tex` was not touched.** Prose changes are listed here as owed, not made.

Every run: seed 123, temp 0 (greedy), EOS disabled, flash attention **off** on
both arms of every pair, and the `flash_attn` line parsed back out of each run's
own log and recorded per row rather than trusted from the harness source.

---

## What was produced

| # | Review item | Result | Data | Analysis |
|---|---|---|---|---|
| 1 | M1, DA-3, P7 (Fig 3 comparator) | Policy-1 arm added; old datum was flash-attention **ON** and did not match | `stress_results/r3_policy_compare_cooled_master.csv` | `artifacts/r3_policy_compare/fig3_provenance.txt` |
| 2 | M7, P5 (recall-cost shape) | Affine: **~9.0 ms fixed + 3.29 µs/cell**; measured to n_spill = 10,239 | `stress_results/recall_cost_isochronal.csv`, `recall_cost_steps_*.csv`, `recall_cost_e2e.csv` | `artifacts/recall_cost/recall_cost_summary.txt` |
| 3 | M2, P1, C4 (capacity bound) | **Demonstrated**: 65k-token workload stock cannot run at any `-c`; UniKV completes | `stress_results/capacity_probe.csv`, `capacity_workload.csv`, `capacity_demo.csv` | `artifacts/capacity_probe/capacity_analysis.txt` |
| 4 | M5 (Table 2 provenance) | All arms one harness, one batch config, FA off; **42 demotions on both** eviction arms | `quality_results/quality_arms_unified.csv` | `artifacts/quality_unified/quality_arms_analysis.txt` |
| 5 | M4, DA-5 (α=0 drain) | Drain = **29%** of the α=0→0.25 step; delay is invisible without it | `alpha_results/p3_drain_control_master.csv` | `artifacts/drain_control/drain_control_analysis.txt` |
| 6 | M6 (uncooled headline) | Cooled randomized block, SD 0.02–0.41 tok/s | `stress_results/r3_policy_compare_cooled_master.csv` | `artifacts/r3_policy_compare/policy_compare_analysis.txt` |
| 7 | P4 (UniKV on A100) | **BLOCKED** — `tosun.sabanciuniv.edu` does not resolve without the university VPN | — | — |
| 8 | **R3 / D1 / C3 (prior-art comparator)** | **H2O in-fork as `UNIKV_POLICY=4`. It RETRIEVES the passkey and is the fastest arm** | `quality_results/quality_arms_unified.csv`, `stress_results/r3_policy_compare_cooled_master.csv` | `artifacts/h2o_comparator/h2o_analysis.txt` |
| 9–10 | P2, D6, P6 | Not attempted | — | — |

New harnesses: `scripts/run_policy_compare_cooled.py`, `run_recall_cost_sweep.py`,
`run_recall_cost_isochronal.py`, `analyze_recall_cost.py`,
`run_quality_arms_unified.py`, `run_drain_control.py`, `run_capacity_probe.py`,
`run_capacity_workload.py`, `run_capacity_demo.py`.

---

## Code changes (llama.cpp fork, inert by default)

1. **`UNIKV_DRAIN`** — decouples the pipeline drain from α. Unset = published
   behavior (drain iff α>0); `1` = drain even at α=0; `0` = never drain. Needed
   for item 5; the default path is logically identical to before.
2. **`n_spilled` column** appended to the `UNIKV_LOG` per-call CSV, so the
   recall-cost curve is measured rather than inferred from C. Additive schema
   change; existing readers use `DictReader` by name.

Both required one rebuild of `llama-completion`. Everything below was measured
on that build.

---

## Headline numbers the paper should now use

**Cooled policy comparison** (512-token prompt + 2048 decode, FA off, n=3,
8 arms in ONE randomized block — supersedes the earlier 18-run block):

| arm | C | tokens | demotions | tok/s | sd |
|---|---|---|---|---|---|
| p0 stock | 1024 | 512 (halts) | 0 | 42.27 | 0.16 |
| p1 rolling window | 1024 | 2048 | 0 | 38.52 | 0.08 |
| **p4 H2O** | 1024 | 2048 | 1535 | **41.41** | 0.09 |
| p3 UniKV | 1024 | 2048 | 1535 | 29.05 | 0.06 |
| p0 stock | 2048 | 1536 (halts) | 0 | 40.56 | 0.07 |
| p1 rolling window | 2048 | 2048 | 0 | 38.45 | 0.01 |
| **p4 H2O** | 2048 | 2048 | 511 | **39.19** | 0.01 |
| p3 UniKV | 2048 | 2048 | 511 | 34.64 | 0.14 |

**H2O retrieves the passkey and is the fastest continuation policy.** UniKV is
29.8% slower than it at C=1024 and 11.6% at C=2048. The exactness premium — not
a retrieval advantage — is what UniKV buys.

**Recall cost** (controlled, C=1024, valid for n_spill ≤ ~8k):
`t_step = 32.25 ms + 3.289 µs/cell · n_spill`, against a 23.22 ms no-spill
reference. **Do not extrapolate** — see the correction below.

**Capacity demonstration** (65,536-token prompt + 32 decode):

| arm | C | device | ×budget | outcome |
|---|---|---|---|---|
| stock | 49,152 (largest workable) | 14,017 MiB | 0.82× | rejected, prompt > context |
| stock | 73,728 (holds the prompt) | 18,673 MiB | 1.09× | **GPU out of memory** |
| UniKV | 4,096 + 8,704 MiB host tier | 5,481 MiB | 0.32× | **completed** |

Every `-c` below stock's ceiling is too small for the workload; every `-c` large
enough is above the ceiling. UniKV completes at 10.3 tok/s prefill and 1.50
tok/s decode — feasibility under a hard bound, not a usable operating point.

---

## Prose changes owed (NOT made — `main.tex` untouched)

1. **Drop "4×" as a UniKV property.** With policy 1 in Figure 3 the token counts
   tie at 2048. The measured claim is completes-vs-halts, and the difference
   between the two continuation policies is retention, not count. Affects the
   abstract, introduction, contribution list, §5.2, the Fig-3 caption, and the
   conclusion.
2. **Stop using 42.5 → 28.5 as "pure recall cost."** 42.5 is the arm that halts
   after 512 tokens. The like-for-like number is 38.64 → 28.90 (25.2%) at
   C=1024 and 38.64 → 34.98 (9.5%) at C=2048.
3. **Recall cost is affine, not proportional** — within its fitted range.
   ~9 ms/step fixed + 3.3 µs/cell; the fixed term dominates below n_spill ≈
   2,750. This is the compute term D3 asks for, and it should enter Eq. (2) as
   an activation cost plus a per-entry cost. **But it is superlinear beyond
   ~10^4 cells**: at n_spill = 61,471 the measured 666 ms/step is 184% above
   the linear prediction. P5's objection to extrapolating is correct and the
   paper should concede it rather than extrapolate.
4. **Disclose the drain.** 29% of the α=0→0.25 step is the pipeline drain. Report
   the drain-corrected 0.50 tok/s as the transfer effect at the boundary, and
   stop calling the raw offset evidence of a physical blocking transition. The
   α>0 regression slope (−1.197 ± 0.052, t = 23.1) is unaffected and stands.
5. **Report the α sweep as a regression** rather than pairwise SE overlap (M3).
   Verified independently from the existing CSV; the reviewer's numbers are
   exactly right.
6. **§5.7's controlled-comparison sentence is now true** and can be stated as
   verified: one harness, one batch config, FA off everywhere, 42 demotions on
   both eviction arms.
7. **Correct the device-memory accounting.** The per-cell KV arithmetic
   understates the device cost of enlarging C by ~50%: with flash attention off
   the attention scratch adds ~65 KiB/token on top of the 128 KiB of KV.
8. **Disclose the flash-attention asymmetry in the capacity result.** Stock's
   ~62k-token ceiling is measured at FA off, matched to UniKV (which cannot use
   FA). With FA on, stock would reach a higher ceiling. Say so.
9. **Rewrite the quality claim against prior art.** UniKV's advantage over the
   naive window and StreamingLLM is retrieval; over H2O it is **exactness only**
   (token-identical to the no-eviction reference vs merely recovering the same
   fact). The passkey probe no longer discriminates UniKV from good prior art —
   only the token-identity check does. Quote the ~30% exactness premium against
   H2O rather than any comparison against the halting baseline.
10. **Restate the A100 claim** as being about the upstream error path only, unless
   the A100 arm is run (item 7 is blocked on VPN access).

---

## Caveats a reviewer would find

- n=3 (policy comparison, drain control), n=2 (most recall-cost points). Small.
- The n_spill = 4096 recall-cost point is irreproducible (5 trials, 44.0–60.9 ms)
  where every other point repeats to <0.5 ms. Unexplained, not trimmed; the fit
  is reported with and without it (3.41 vs 3.29 µs/cell).
- The controlled recall-cost design still self-heats: bigger targets need longer
  prefills immediately before the burst, so 3.3 µs/cell is an upper bound.
- The capacity numbers are one machine, one model, one quantization, one ubatch.
  A smaller ubatch shrinks the scratch term and raises stock's ceiling; that
  sweep was not run and is a fair thing for a reviewer to ask for.


---

# Phase A — measurement paper (2026-08-05)

Findings are numbered per `paper/UNIKV-MEASUREMENT/main.tex`. No cooldowns:
outcomes are binary or coarse. All runs flash-attention-verified per row from
their own logs; anything reporting tok/s ran uninstrumented.

| item | maps to | result | data | analysis |
|---|---|---|---|---|
| A3 | **F4** | 🚩 **Fused upstream RUNS the 65k workload** at C=73,728 (0.824× budget), 15.99 tok/s. F4's headline claim is false as stated | `stress_results/f4_a3_flash_attn_capacity.csv` | `artifacts/f4_phaseA/F4_A3_flash_attention_capacity.txt` |
| A1 | **F4** | Scratch is exactly linear in ubatch; **193 KiB/token is a ubatch-512 figure**, range 136–194. Ceiling moves 1.7× (49,152 → 81,920) | `stress_results/f4_a1_ubatch_sweep.csv` | `artifacts/f4_phaseA/F4_A1_ubatch_sweep.txt` |
| A2 | **F4** | 🚩 Rolling window and H2O both complete at C=4,096 (asserted claim confirmed). **And upstream fa-off at ubatch 64, C=81,920 completes it losslessly** — second contradiction | `stress_results/f4_a2_continuation_arms.csv` | `artifacts/f4_phaseA/F4_A2_continuation_arms.txt` |

## The 65,536-token workload, complete picture

| configuration | device MiB | ×budget | decode tok/s | lossless? |
|---|---|---|---|---|
| exact retention, C=4096 + host tier | 5,481 | 0.32 | 1.50 | yes |
| rolling window, C=4096 | 5,481 | 0.32 | 27.68 | no |
| H2O, C=4096 | 5,738 | 0.33 | 31.63 | no |
| upstream fa-**off**, ubatch 64, C=81920 | 15,588 | 0.91 | 5.86 | **yes (whole prompt)** |
| upstream fa-**on**, ubatch 512, C=73728 | 14,160 | 0.82 | 15.99 | **yes (whole prompt)** |

**F4 rewrite required.** "No upstream setting runs it" is false on the fused path
(A3) and false on the non-fused path (A2), so it fails under any scoping. What
survives is a memory/throughput trade, which should be stated in both directions:

> Exact retention completes the 65,536-token workload with a 5,481 MiB device
> working set, 2.8× smaller than the smallest upstream configuration that also
> completes it losslessly, and pays 3.9×–10.7× in decode throughput for that.

Restoring a feasibility claim needs a workload above the fused ceiling (~91,000
tokens). Extrapolated spilled-tier prefill for 120k tokens is ~7 hours — a Phase B
scale decision, and worth asking whether a feasibility claim requiring a 7-hour
prefill is one the paper wants.

## Phase B — B1 done, B2 queued

**B1 = bucket (a)**, backend-assignment/buffer-type change, ~2 h. Allocate the
spilled tier from the resident cache's buffer type (Metal shared, zero-copy) and
remove **two** CPU pins — `kq_spill` and `kqv_spill`. Gating only the first makes
it *slower* (22.9 vs 30.4 tok/s): the graph half-migrates and copies a 7 MB
permuted V view across 41 CPU splits per step. Default (`UNIKV_SPILL_DEV` unset)
unchanged, so all existing numbers stand.
Analysis: `artifacts/b1_gpu_tier/B1_triage.txt`,
`artifacts/b1_gpu_tier/B1_verification_and_ceiling.txt`.

Counterbalanced verification (n=4/arm, uncooled, within-round paired):

| comparison | difference | per round |
|---|---|---|
| **H2O vs device-visible** | **+7.77% ± 2.08** | +10.1 +7.2 +8.5 +5.2 |
| device-visible vs CPU-pinned | +46.50% ± 15.41 | +25.9 +61.6 +44.5 +54.1 |

The sub-10% premium is confirmed. The device-visible speedup is *not* a point
value: the CPU-pinned arm swings 30% across a session (thermal throttling of CPU
work) against 10% for the GPU-resident arms, so it needs the cooled block.

**The ceiling — the other half of the trade:**

| mode | ceiling | device accounting |
|---|---|---|
| device-visible | **81,920–90,112 cells** (10.2–11.3 GiB tier), then GPU OOM | tier charged in full; free memory falls by exactly the cap |
| CPU-pinned | **none** — 16 GiB tier runs fine | free stays 11,311 MiB at every cap; tier invisible to Metal |

Failure is again **GPU OOM, not allocation failure** — F4's central finding
reproduces in a configuration it wasn't derived from.

**The claims cannot be combined.** At its largest working capacity the
device-visible config holds ~82,900 tokens; fused upstream reaches ~91,000 with
no spill machinery at all. So the capacity argument survives only in CPU-pinned
mode and the throughput argument only in device-visible mode.

**B2 RUN 2026-08-07 (see below).** ~~written and NOT run~~ — `scripts/run_b2_cooled.py`, both modes as arms
(the finding is the curve, not the fast endpoint). Two blocks, ~2 h, needs an
idle machine: isochronal γ/δ per mode, then a 4-arm policy block. Reserved for
the extended version; not for the workshop cut.


---

# Phase B / B2 — protocol numbers, 2026-08-07

68 runs, machine idle, 200 s cooldowns, `-fa off` verified per run (68/68), all
rc=0. Supersedes everything from B1 (smoke) and the 8-arm block.
Data: `stress_results/b2_isochronal_both_modes.csv`, `b2_policy_block.csv`.
Analysis: `artifacts/b2_cooled/B2_analysis.txt`.

**Two corrections to what B1 reported:**
1. "δ fell by about half" → it falls **30.5%** (real, t = 12.34, but smaller).
2. "premium under 10%" → **11.4% at C=1024**; under 10% only at C=2048 (1.3%).

## Block 1 — recall cost, both modes (C=1024)

| | CPU-pinned | device-visible |
|---|---|---|
| fit | 31.563 ms + **3.0990** µs/cell | 25.423 ms + **2.1531** µs/cell |
| slope SE | 0.0764 | 0.0055 |
| residual SD | 0.832 ms | 0.059 ms |
| **γ** | **9.009 ms** | **2.879 ms** (−68%) |

- **No-spill references agree** (22.554 vs 22.544, t = −0.28) — the tier is inert
  at idle, so the differences aren't an allocation artefact.
- **Slopes separable**: difference 0.9459 ± 0.0766 µs/cell, **t = 12.34**.
  δ is *not* location-independent — streaming the tier is cheaper on the GPU.
- **The n=4096 anomaly is deleted**: 43.892 ± 0.696 (n=5) vs the old 44.0–60.9
  spread. It was background activity.
- Reproducible across nights: γ 9.03 → 9.009; δ 3.289 ± 0.086 → 3.099 ± 0.076.
- Device-visible is **14× more precise**, and CPU-pinned scatter *grows* with
  n_spill (0.085 → 1.456) while device-visible stays flat (0.064 → 0.028).

## Block 2 — five arms, three trials

| arm | C=1024 | C=2048 |
|---|---|---|
| upstream (halts) | 43.39 ± 0.04 (512 tok) | 41.51 ± 0.03 (1536 tok) |
| rolling window | 39.40 ± 0.16 | 39.38 ± 0.01 |
| H2O | 42.42 ± 0.01 | 40.17 ± 0.04 |
| exact, CPU-pinned | 30.24 ± 0.15 | 35.62 ± 0.25 |
| exact, device-visible | 38.07 ± 0.01 | 39.64 ± 0.01 |

**Exactness premium** (vs H2O / vs rolling window):

| | vs H2O | vs window |
|---|---|---|
| C=1024 CPU-pinned | +40.26% | +30.27% |
| C=1024 device-visible | **+11.42%** | +3.48% |
| C=2048 CPU-pinned | +12.77% | +10.54% |
| C=2048 device-visible | **+1.33%** | **−0.67%** |

At C=2048 device-visible exact retention is within 1.3% of H2O and *marginally
faster* than the rolling window. The premium scales with how much is spilled
(1535 cells at C=1024 vs 511 at C=2048), agreeing with Block 1.

`p3_cpu` position audit: values cluster by context, not slot (C=2048 runs
35.38/35.88/35.61 at slots 2/14/22; C=1024 runs 30.07/30.28/30.37 at 10/16/28).
No thermal-position effect.

## Method note worth a sentence in the paper

The uncooled interleaved A/B disagreed with protocol in *both* directions:
it overstated the device-visible speedup (+46.5% vs +25.9%) and understated the
premium (+7.8% vs +11.4%). Every arm gains on a cool machine, but CPU-pinned
gains most (+19.9% vs +4.0% and +7.5%). **Counterbalancing guards against order
effects within a session; it does not guard against an arm being more sensitive
to the session's thermal level.** When arms load different processors, only a
cooled block gives comparable numbers.


---

# Figures for the measurement paper (2026-08-07)

| figure | finding | source | generator | in main.tex? |
|---|---|---|---|---|
| `fig5_recall_cost` | F1 | `b2_isochronal_both_modes.csv` (B2 Block 1) | `make_fig5_recall_cost.py` | yes — delete the `% FIGURE PENDING` comment at lines 400–403 |
| `fig6_capacity` | F4 | `f4_a1_ubatch_sweep.csv` (Phase A / A1) | `make_fig6_capacity.py` | no — float not added |
| `fig7_drain` | F3 | `p3_drain_control_master.csv` | `make_fig7_drain.py` | no — float not added |
| `fig4_alpha` | F3 | `p3_alpha_sweep_cooled_master.csv` | `make_figures_final.py` | no — staged, verified to match F3's reported 29.67→27.01 and −1.197 ± 0.052 |

All three new generators assert every plotted value against their CSV and die
naming the file on mismatch. All are copied into `paper/UNIKV-MEASUREMENT/`.

**fig7 provenance:** drain-control block alone, **not** spliced with the cooled
sweep. The blocks agree on level (0.30% at the shared α=0) but disagree on
drained slope (−1.991 ± 0.300 vs −1.197 ± 0.052, t = −2.6), and the figure's
claim is about slope. Details: `artifacts/drain_control/fig7_provenance.txt`.

**One wording correction it forces:** the undrained series is *not* flat.
α=0 → 0.25 is +0.019 ± 0.174 (t = 0.11, genuinely invisible), but α=0 → 1 is
−0.541 ± 0.183 (t = −2.96, real). The supported claim is "largely absorbed
unless drained", not "invisible unless drained" — absorption degrades as the
injected sleep approaches the step time, which is the expected behaviour.

**Axis conventions differ deliberately:** fig5 keeps a zero baseline because its
caption quotes percentage reductions; fig7 is truncated (with a break glyph)
because it claims a slope difference and the whole effect spans 0.7 tok/s
out of ~29.5.

---

# Post-review runs — 2026-08-11

| item | status | result | data | analysis |
|---|---|---|---|---|
| **Token-identity horizon** (BLOCKING) | done | Exact retention **identical for all 512 tokens on both prompts, both spill tiers**; H2O diverges at **token 16** (passkey) and **token 0** (prose) | `quality_results/token_horizon.csv`, traces in `artifacts/token_horizon/sequences/` | `artifacts/token_horizon/token_horizon_analysis.txt` |
| **α-independence of the drain cost** (optional) | done | **It is not α-independent.** Drain costs −0.078 ± 0.103 at α=0 and **+0.981 ± 0.247 at α=1** (interaction t = 3.96). **The 71/29 split is withdrawn.** | `alpha_results/p3_drain_alpha1_master.csv` | `artifacts/drain_alpha1/drain_alpha_interaction_analysis.txt` |
| **ubatch vs prefill parallelism** | done, no run needed | The A1 sweep already answers it and **contradicts the draft**: no measurable prefill penalty | `stress_results/f4_a1_ubatch_sweep.csv` | `artifacts/f4_phaseA/ubatch_prefill_tradeoff.txt` |

## Token identity at 512 tokens

Harness `scripts/run_token_horizon.py`. One block, `-b 256 -ub 256`, `-fa off`
parsed back as `disabled` on all ten runs, greedy, seed 123, EOS disabled, two
prompts matched to exactly 3834 tokens.

| arm | passkey | prose |
|---|---|---|
| exact retention, CPU-pinned (C=1024) | identical 512/512 | identical 512/512 |
| exact retention, device-visible (C=1024) | identical 512/512 | identical 512/512 |
| H2O (C=1024) | **diverges at token 16** | **diverges at token 0** |
| CONTROL: reference at C=4096 | identical over its 262 | identical over its 262 |

Whole-sequence, not prefix: reference and both retention arms share one md5 per
prompt. Not vacuous either — both retention arms ended with **3321 cells in the
host tier against 1024 resident** (76% off-device), matched exactly by H2O's
3321 evicted; the reference logs contain zero spill or eviction lines.

**Deviation, covered by a control.** The brief said reference at C=4096, but
3834 + 512 = 4346 > 4096, so a C=4096 arm evicts and cannot be a no-eviction
reference. Reference is C=8192; `ctl_ref_c4096` ran to exactly 262 generated
tokens (3834 + 262 = 4096, the arithmetic confirming it filled) and matched the
C=8192 reference over all of them, on both prompts. The reference is C-invariant,
so the substitution is free.

**The full trace was the result, not the divergence index.** Agreement with the
reference in 128-token windows — H2O loops on the passkey probe and decoheres
gradually on prose:

| window | passkey | prose |
|---|---|---|
| 0–127 | 14.1% | 64.1% |
| 128–255 | 3.1% | 37.5% |
| 256–383 | 3.1% | 21.9% |
| 384–511 | 3.1% | 7.8% |

**H2O is not a strawman here.** It still retrieves the passkey (48291) and opens
with it. The supported claim is only that exact retention reproduces the
reference token stream and H2O does not — *not* that H2O fails the task.

**Scope for the prose:** say "identical to 512 generated tokens on two prompts",
not "identical". Two prompts is two, one budget is one, and the divergence
indices 16 and 0 are single observations — "within the first 16 tokens on both
prompts" is what the data supports.

## The ubatch sentence is unsupported and should be replaced

The draft concedes that lowering ubatch buys capacity "at a prefill-throughput
cost". Checked before running anything, as instructed: `f4_a1_ubatch_sweep.csv`
already measured this and shows no such cost. At C=49152, the one context all
four settings run, prefill is **259.6 / 259.4 / 268.2 / 237.0 tok/s** for ubatch
64/128/256/512 — flat 64→256 and *slowest* at the largest ubatch. Capacity gain
is real and large: max workable context 81920 at ub64 vs 49152 at ub512 (1.67×),
as scratch falls 66.4 → 8.3 KiB per token of context.

These are n=1 uncooled runs, so "no measurable penalty" is supported but "ub512
is 12% slower" is **not** — do not claim the ordering. Suggested replacement:
"on this machine the capacity gain is not offset by a measurable prefill penalty
— over ubatch 64–512 at a 16k prompt, prefill throughput varies by 3.3% with no
monotone trend."

**One thing this surfaced that F4 does not say.** ub128 at C=81920 runs at 65.8
tok/s, 3.9× slower than the same ubatch at smaller C. That is not a ubatch
effect — it is a soft step in front of the OOM cliff at 0.946× of the advisory
budget, where a configuration completes at roughly a quarter speed instead of
being refused. F4 presents the boundary as clean. It is clean except for this one
observation, and saying so costs a sentence.

## The drain cost is not α-independent — the 71/29 split is withdrawn

Harness `scripts/run_drain_alpha1.py`. Second cooled block, all four cells of the
2×2 in one block (α ∈ {0, 1} × drain ∈ {off, on}), 3 trials, 200 s cooldowns,
uninstrumented, fresh randomization. All 12 runs `rc=0`, `flash_attn=disabled`,
2048 tokens.

| condition | α | drains | mean tok/s | sd |
|---|---|---|---|---|
| a0_nodrain | 0 | no | 29.642 | 0.067 |
| a0_drain | 0 | yes | 29.720 | 0.165 |
| a1_nodrain | 1 | no | 29.069 | 0.428 |
| a1_drain | 1 | yes | **28.088** | 0.018 |

Drain cost (undrained − drained), **every point within its own block**:

| α | drain cost | t | block |
|---|---|---|---|
| 0.00 | +0.197 ± 0.177 | 1.11 (n.s.) | 1 |
| 0.00 | −0.078 ± 0.103 | −0.76 (n.s.) | 2 |
| 0.25 | +0.713 ± 0.068 | 10.46 | 1 |
| 1.00 | +0.981 ± 0.247 | 3.97 | 2 |

**There is no fixed drain cost.** `synchronize()` has no intrinsic price; it
costs what it stops the pipeline absorbing, so it is free at α=0 and expensive at
α=1. Drain and modeled transfer are one mechanism seen through a switch, not two
additive terms.

**The additivity argument in block 1 was a telescoping sum.** 0.20 + 0.50 = 0.70
vs 0.69 measured together is true of *any* path across a 2×2. Additivity requires
the two paths to agree on each term; in block 1's own data the drain costs +0.197
at α=0 and +0.713 at α=0.25 — interaction **+0.516 ± 0.190, t = 2.72**, present
in the published data and never computed. This block confirms it independently at
wider spacing (+1.059, t = 3.96). And the "29%" itself (+0.197 ± 0.177, t = 1.11)
was never significant and does **not replicate** — block 2 gets the opposite sign,
while the blocks agree on levels to 0.4–0.5% on all three shared conditions.

**Suggested replacement claim:** "the pipeline drain has no measurable cost of its
own (α=0: t = 1.11 and t = −0.76 in two independent blocks); its cost is the
injected delay it stops the pipeline absorbing, and it grows with α — +0.71 tok/s
at α=0.25 and +0.98 at α=1."

**This does not rescue the α=0 offset from the reviewer's objection**, and should
not be presented as if it did. Like-with-like at drain OFF, the α=0 → 0.25 step is
+0.019 ± 0.174 — nothing. The whole step exists only in the drained comparison,
which is the one whose instrumentation changes at the treatment boundary.

**Unaffected:** fig7 (every plotted value is a block-1 cell mean; its headline is
the within-block α=0.25 gap, which stands — and the figure's two-different-slopes
picture is a *more* direct rendering of the corrected claim than of the one it was
drawn for), the recall-cost fit, capacity, token identity, and the sweep's drained
slope −1.197 ± 0.052.

**Design defect, reported and checked.** The shuffle drew α=1 early and α=0 late
(mean slots 3.00/5.00/7.33/10.67), so condition is confounded with position and
the two drain contrasts run in *opposite* directions w.r.t. position — a drift
could have manufactured this interaction. It did not: residual drift is +0.0009
tok/s per slot (t = 0.05), strictly local adjacent-slot contrasts give +1.027, and
detrended cell means give +1.064, against the full-block +1.059. A future block
should position-balance explicitly rather than rely on randomization, per the B2
brief's warning.

**Superseded:** the DECOMPOSITION section of
`artifacts/drain_control/drain_control_analysis.txt` now carries a withdrawal
header. Its RESULTS table and "the delay is invisible without the drain" section
stand.
