# Artifact — "What Vanishing Transfer Cost Does Not Buy"

Harnesses and raw results for the UniKV measurement paper. Every quantitative
claim in the paper resolves through the table below to the harness that produced
it and the file the data lives in; the paper states the protocol, the exclusions
and the arithmetic for each.

## What is here

| | |
|---|---|
| `scripts/` | Measurement harnesses and figure generators |
| `stress_results/`, `alpha_results/`, `quality_results/` | Raw result CSVs |
| `quality_results/token_sequences/` | Full sampled-token-ID sequences, four models |
| `figures/` | Figures as published, regenerable from `scripts/make_fig*.py` |
| `llama.cpp` (submodule) | The fork the measurements run against |

Two archives, both needed:

- Harnesses and data — this repository, [10.5281/zenodo.21907114](https://doi.org/10.5281/zenodo.21907114)
- The fork — `llama.cpp-unikv`, [10.5281/zenodo.21907123](https://doi.org/10.5281/zenodo.21907123),
  pinned as a submodule at commit `719463a`, branched from upstream `9725a313b`

## What is *not* here, and why

- **Model weights.** Runs use Llama 3.1 8B, Qwen2.5 7B, Llama 3.2 3B and
  Llama 3.2 1B, all Instruct at Q4_K_M and all fetched separately. Nothing in
  the artifact modifies a model.
- **Build output.** The fork must be built locally; the build tree is several GB
  and machine-specific.

## Reproducing

Not every claim needs the same effort:

- **Deterministic** — token sequences, spill counts, passkey outcomes. Greedy
  decoding at a fixed seed, so these reproduce on any machine that runs the
  fork. `quality_results/token_horizon.csv` is the clearest case.
- **Binary** — capacity outcomes. A configuration either executes or is refused;
  no thermal protocol needed.
- **Rates** — anything in tok/s or ms/step. These require the cooled protocol
  described in the paper's methodology section: randomised block order, 200 s
  cooldowns, an otherwise idle machine, and runs left uninstrumented because the
  per-step log itself drains the GPU pipeline. Numbers produced without that
  protocol will not match.

Flash attention state is parsed back from each run's own stderr rather than
trusted from the harness source; several analyses fail loudly if it disagrees.

## Traceability

| Claim | Harness | Data |
|---|---|---|
| Eq. (1)–(2), γ, δ, Fig. 1 | `run_b2_cooled.py`, `make_fig5_recall_cost.py` | `stress_results/b2_isochronal_both_modes.csv` |
| Isochronal design; confounded long-run slope | `run_recall_cost_isochronal.py` | `stress_results/recall_cost_steps_A_p3_long.csv`, `recall_cost_isochronal.csv` |
| Device-visible tier; counterbalanced A/B failure | `run_b1_ceiling.py`, `run_b1_interleaved.py` | `stress_results/b1_device_tier_ceiling.csv`, `b1_interleaved_ab.csv` |
| Retrieval + 32-token continuation (Table 2) | `run_quality_arms_unified.py` | `quality_results/quality_arms_unified.csv` |
| H2O comparator fidelity | `run_quality_probe.py` | `quality_results/quality_probe.csv` |
| 512-token identity, two prompts (Table 3, Fig. 4) | `run_token_horizon.py`, `make_fig8_horizon.py` | `quality_results/token_horizon.csv`, sequences in `quality_results/token_sequences/` |
| α sweep slope | `run_alpha_sweep_p3.py` | `alpha_results/p3_alpha_sweep_cooled_master.csv` |
| Drain control (Table 4, Fig. 3) | `run_drain_control.py`, `make_fig7_drain.py` | `alpha_results/p3_drain_control_master.csv` |
| drain×α interaction; withdrawal of the 71/29 split (Table 5) | `run_drain_alpha1.py` | `alpha_results/p3_drain_alpha1_master.csv` |
| Scratch term, ceilings, ubatch linearity, prefill non-effect (Table 6, Fig. 2) | `run_f4_phaseA.py`, `make_fig6_capacity.py` | `stress_results/f4_a1_ubatch_sweep.csv` |
| Upstream completes the 65k workload (Table 7) | `run_capacity_workload.py` | `stress_results/f4_a2_continuation_arms.csv`, `f4_a3_flash_attn_capacity.csv` |
| Allocation vs. execution; advisory-budget ratios | `run_capacity_probe.py` | `stress_results/capacity_probe.csv` |
| Cross-model test (Qwen2.5 7B): Finding 1 coefficients, Finding 4 scratch prediction, Finding 2 replication | `run_b2_cooled.py`, `run_f4_phaseA.py`, `run_token_horizon.py`, `analyze_isochronal.py` | `stress_results/qwen_isochronal_both_modes.csv`, `f4_a1_ubatch_sweep_qwen.csv`, `quality_results/token_horizon_qwen.csv` |
| RoPE ablation: Finding 5's mechanism isolated (`UNIKV_NO_REENCODE`) | `run_rope_ablation.py` | `stress_results/rope_ablation_block.csv` |
| Composed fused-kernel cost; kernel penalty at C=1024 and 2048 | `run_fused_composed.py` | `stress_results/fused_composed_block.csv` |
| Constant-work thermal control; independent replication of Eq. (1)–(2) | `run_thermal_control.py`, `analyze_thermal_control.py` | `stress_results/thermal_control_block.csv` |
| Discard granularity; shift-event census; second per-event block | `run_amortized_window.py`, `analyze_reconstruction.py`, `check_slot_effects.py` | `stress_results/amortized_window_block.csv`, `amortized_window_events.csv`, `amortized_window_recheck.csv`, `amortized_window_recheck_c2048.csv` |
| Logit bound, argmax margins, at-risk steps | `run_logit_bound.py`, `run_logit_controls.py`, `analyze_logit_margins.py` | `quality_results/logit_bound.csv`, `logit_margins.csv` |
| Policy comparison, exactness premium (Table 8) | `run_b2_cooled.py` | `stress_results/b2_policy_block.csv` |

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
