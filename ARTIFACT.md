# Artifact — "What Vanishing Transfer Cost Does Not Buy"

Code, harnesses, raw results and per-experiment analyses for the UniKV
measurement paper. Every quantitative claim in the paper resolves through the
table below to the harness that produced it, the file the data lives in, and an
analysis file stating the protocol, the exclusions and the arithmetic.

## What is here

| | |
|---|---|
| `scripts/` | Measurement harnesses and figure generators |
| `stress_results/`, `alpha_results/`, `quality_results/` | Raw result CSVs |
| `artifacts/` | Per-experiment analyses, run logs, prompts, token traces |
| `figures/` | Figures as published, regenerable from `scripts/make_fig*.py` |
| `llama.cpp` (submodule) | The fork the measurements run against |

Two archives, both needed:

- Harnesses and data — this repository, [10.5281/zenodo.21907114](https://doi.org/10.5281/zenodo.21907114)
- The fork — `llama.cpp-unikv`, [10.5281/zenodo.21907123](https://doi.org/10.5281/zenodo.21907123),
  pinned as a submodule at commit `7c2b27d`, branched from upstream `9725a313b`

## What is *not* here, and why

- **Model weights.** Every run uses Llama 3.1 8B Instruct at Q4_K_M, fetched
  separately. Nothing in the artifact modifies the model.
- **Build output.** The fork must be built locally; the build tree is several GB
  and machine-specific.

## Reproducing

Not every claim needs the same effort:

- **Deterministic** — token sequences, spill counts, passkey outcomes. Greedy
  decoding at a fixed seed, so these reproduce on any machine that runs the
  fork. `quality_results/token_horizon.csv` and its traces are the clearest
  case.
- **Binary** — capacity outcomes. A configuration either executes or is refused;
  no thermal protocol needed.
- **Rates** — anything in tok/s or ms/step. These require the cooled protocol
  described in the paper's methodology section: randomised block order, 200 s
  cooldowns, an otherwise idle machine, and runs left uninstrumented because the
  per-step log itself drains the GPU pipeline. Numbers produced without that
  protocol will not match, and two experiments in the artifact document exactly
  how far off they land.

Flash attention state is parsed back from each run's own stderr rather than
trusted from the harness source; several analyses fail loudly if it disagrees.

## Traceability

| Claim | Harness | Data | Analysis |
|---|---|---|---|
| Eq. (1)–(2), γ, δ, Fig. 1 | `run_b2_cooled.py`, `make_fig5_recall_cost.py` | `stress_results/b2_isochronal_both_modes.csv` | `b2_cooled/B2_analysis.txt`; fit spec — including why the fit regresses on measured `n_spill` rather than the harness target — in `b2_cooled/REGRESSION_SPEC.txt` |
| Isochronal design; confounded long-run slope | `run_recall_cost_isochronal.py` | `stress_results/recall_cost_steps_A_p3_long.csv`, `recall_cost_isochronal.csv` | `recall_cost/recall_cost_analysis.txt` — the design and the confounded slope stand, but its γ and δ are superseded by the B2 row above |
| Device-visible tier; counterbalanced A/B failure | `run_b1_ceiling.py`, `run_b1_interleaved.py` | `stress_results/b1_device_tier_ceiling.csv`, `b1_interleaved_ab.csv` | `b1_gpu_tier/B1_triage.txt`, `B1_verification_and_ceiling.txt` |
| Retrieval + 32-token continuation (Table 2) | `run_quality_arms_unified.py` | `quality_results/quality_arms_unified.csv` | `quality_unified/quality_arms_analysis.txt` |
| H2O comparator fidelity | `run_quality_probe.py` | `quality_results/quality_probe.csv` | `h2o_comparator/h2o_analysis.txt` |
| 512-token identity, two prompts (Table 3, Fig. 4) | `run_token_horizon.py`, `make_fig8_horizon.py` | `quality_results/token_horizon.csv`, traces in `token_horizon/sequences/` | `token_horizon/token_horizon_analysis.txt` |
| α sweep slope | `run_alpha_sweep_p3.py` | `alpha_results/p3_alpha_sweep_cooled_master.csv` | `r2_policy3/alpha_sweep_cooled_analysis.txt` — supersedes `alpha_sweep_analysis.txt`, which was thermally masked |
| Drain control (Table 4, Fig. 3) | `run_drain_control.py`, `make_fig7_drain.py` | `alpha_results/p3_drain_control_master.csv` | `drain_control/drain_control_analysis.txt`, `fig7_provenance.txt` |
| drain×α interaction; withdrawal of the 71/29 split (Table 5) | `run_drain_alpha1.py` | `alpha_results/p3_drain_alpha1_master.csv` | `drain_alpha1/drain_alpha_interaction_analysis.txt` — supersedes the decomposition in `drain_control/drain_control_analysis.txt` |
| Scratch term, ceilings, ubatch linearity, prefill non-effect (Table 6, Fig. 2) | `run_f4_phaseA.py`, `make_fig6_capacity.py` | `stress_results/f4_a1_ubatch_sweep.csv` | `f4_phaseA/F4_A1_ubatch_sweep.txt`, `ubatch_prefill_tradeoff.txt` |
| Upstream completes the 65k workload (Table 7) | `run_capacity_workload.py` | `stress_results/f4_a2_continuation_arms.csv`, `f4_a3_flash_attn_capacity.csv` | `f4_phaseA/F4_A2_continuation_arms.txt`, `F4_A3_flash_attention_capacity.txt` |
| Allocation vs. execution; advisory-budget ratios | `run_capacity_probe.py` | `stress_results/capacity_probe.csv` | `capacity_probe/capacity_analysis.txt`; unit correction in `CORRECTION_2026-08-07_budget_units.txt` |
| Policy comparison, exactness premium (Table 8) | `run_b2_cooled.py` | `stress_results/b2_policy_block.csv` | `b2_cooled/B2_analysis.txt` — supersedes `r3_policy_compare/policy_compare_analysis.txt` and its `r3_policy_compare_cooled_master.csv`, which lack the device-visible arm |

Where two analysis files are listed, the second corrects the first. Those entries
are kept deliberately: the 71/29 decomposition, the advisory-budget ratios and
the isochronal slope were all written down before they were right, and the
correcting step is the part worth preserving. Blocks superseded outright are
retained with their reason in `stress_results/SUPERSEDED.md`.

## License

Harnesses under `scripts/`: MIT (`LICENSE`). Result files, analyses and figures:
CC-BY-4.0 (`LICENSE-DATA`). The `llama.cpp` fork inherits upstream's MIT license
and retains its attribution.

## Not included

The paper source is deliberately not in this repository. The artifact exists to
make the measurements checkable, which the data and harnesses do on their own;
publishing draft history of the manuscript alongside them would expose
superseded prose without the reasoning that retired it, and would complicate
anonymous review. Superseded *measurements*, by contrast, are kept here on
purpose, each with a header saying what replaced it and why.
