# Artifact: "What Vanishing Transfer Cost Does Not Buy"

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

- Harnesses and data: this repository, [10.5281/zenodo.21907114](https://doi.org/10.5281/zenodo.21907114)
- The fork: `llama.cpp-unikv`, [10.5281/zenodo.21907123](https://doi.org/10.5281/zenodo.21907123),
  pinned as a submodule at commit `719463a`, branched from upstream `9725a313b`

## What is *not* here

- **Model weights.** Runs use Llama 3.1 8B, Qwen2.5 7B, Llama 3.2 3B and
  Llama 3.2 1B, all Instruct at Q4_K_M and all fetched separately. Nothing in
  the artifact modifies a model.
- **Build output.** The fork must be built locally; the build tree is several GB
  and machine-specific.
- **The paper source.** The data and harnesses make the measurements checkable
  on their own.

## Reproducing

Not every claim needs the same effort:

- **Deterministic**: token sequences, spill counts, passkey outcomes, logit
  dumps and graph split counts. Greedy decoding at a fixed seed, so these reproduce on any
  machine that runs the fork. `quality_results/token_horizon.csv` is the
  clearest case.
- **Binary**: capacity outcomes. A configuration either executes or is refused;
  no thermal protocol needed.
- **Rates**: anything in tok/s or ms/step. These require the cooled protocol
  described in the paper's methodology section: randomised block order, 200 s
  cooldowns, an otherwise idle machine, and runs left uninstrumented because the
  per-step log itself drains the GPU pipeline. Numbers produced without that
  protocol will not match.

Flash attention state is parsed back from each run's own stderr rather than
trusted from the harness source; several harnesses fail loudly if it disagrees.

## Traceability

Table numbers follow the paper.

| Claim | Harness | Data |
|---|---|---|
| Eq. (1)–(2), γ, δ, Fig. 1 | `run_b2_cooled.py`, `make_fig5_recall_cost.py` | `stress_results/b2_isochronal_both_modes.csv` |
| Isochronal design; the confounded long-run slope | `run_recall_cost_isochronal.py` | `stress_results/recall_cost_isochronal.csv`, `recall_cost_steps_A_p3_long.csv`, `recall_cost_steps_B_p1_ctrl.csv` |
| Device-visible tier; counterbalanced A/B failure | `run_b1_ceiling.py`, `run_b1_interleaved.py` | `stress_results/b1_device_tier_ceiling.csv`, `b1_interleaved_ab.csv` |
| Table 4: retrieval and 32-token continuation | `run_quality_arms_unified.py` | `quality_results/quality_arms_unified.csv` |
| H2O comparator fidelity | `run_quality_probe.py` | `quality_results/quality_probe.csv` |
| Table 5 and Figure 2: 512-token identity, two prompts | `run_token_horizon.py`, `make_fig8_horizon.py` | `quality_results/token_horizon.csv`, full sampled-ID sequences in `quality_results/token_sequences/` |
| Table 2, block 1 | `run_drain_control.py` | `alpha_results/p3_drain_control_master.csv` |
| Table 2, block 2: drain×α interaction | `run_drain_alpha1.py` | `alpha_results/p3_drain_alpha1_master.csv` |
| Table 6, Fig. 3: scratch term, ceilings, ubatch linearity | `run_f4_phaseA.py`, `make_fig6_capacity.py` | `stress_results/f4_a1_ubatch_sweep.csv` |
| Table 7: upstream completes the workload; the exact-retention arm's device and host footprints | `run_capacity_workload.py`, `run_capacity_demo.py` | `stress_results/f4_a2_continuation_arms.csv`, `f4_a3_flash_attn_capacity.csv`, `capacity_demo.csv` |
| Allocation-versus-execution; advisory-budget ratios | `run_capacity_probe.py` | `stress_results/capacity_probe.csv` |
| Table 3: the resident-attention baseline, paired with the device-visible arm | `run_resident_baseline.py` | `stress_results/resident_baseline_llama.csv`, `resident_baseline_qwen.csv`, `resident_baseline_l3b.csv`, `resident_baseline_l1b.csv` |
| Section 5: logit bound on the fourth model | `run_logit_bound.py`, `analyze_logit_margins.py` | `quality_results/logit_bound_l1b.csv`, `logit_margins_l1b.csv` |
| Graph-split census: 2 + 2L CPU-pinned, 2 device-visible | `run_split_census.py` | `stress_results/split_census.csv` |
| Constant-work thermal control; replication of Eq. (1) | `run_thermal_control.py`, `analyze_thermal_control.py` | `stress_results/thermal_control_block.csv` |
| RoPE ablation and the composed fused-kernel cost | `run_rope_ablation.py`, `run_fused_composed.py` | `stress_results/rope_ablation_block.csv`, `fused_composed_block.csv` |
| Cross-model tests (Qwen2.5 7B, Llama 3.2 3B): Section 4 coefficients, Section 6 scratch prediction, Section 5 replication | `run_b2_cooled.py`, `run_f4_phaseA.py`, `run_token_horizon.py` | `stress_results/qwen_isochronal_both_modes.csv`, `l3b_isochronal_both_modes.csv`, `f4_a1_ubatch_sweep_qwen.csv`, `f4_a1_ubatch_sweep_l3b.csv`, `quality_results/token_horizon_qwen.csv`, `token_horizon_l3b.csv` |
| The fourth model (Llama 3.2 1B): Section 4 coefficients, Section 6 scratch prediction, Section 5 replication | `run_b2_cooled.py`, `run_f4_phaseA.py`, `run_token_horizon.py`, `analyze_isochronal.py` | `stress_results/l1b_isochronal_both_modes.csv`, `f4_a1_ubatch_sweep_l1b.csv`, `quality_results/token_horizon_l1b.csv` |
| The plateau below 512 cells | `run_b2_cooled.py` with small targets, `analyze_gamma_small.py` | `stress_results/llamasmall_isochronal_both_modes.csv`, `qwensmall_isochronal_both_modes.csv`, `l3bsmall_isochronal_both_modes.csv`; the 1B's small targets are in `l1b_isochronal_both_modes.csv` |
| Cooled micro-batch block; the degradation repeat | `run_ubatch_cooled.py` | `stress_results/ubatch_cooled_block.csv`, `degradation_repeat_block.csv` |
| Drain cost across the range: saturation and curvature (Table 2, block 3) | `run_drain_sweep.py` | `alpha_results/p3_drain_sweep_master.csv` |
| Table 9: discard granularity, shift-event census, and the second per-event block | `run_amortized_window.py`, `analyze_reconstruction.py`, `check_slot_effects.py` | `stress_results/amortized_window_block.csv`, `amortized_window_events.csv`, `amortized_window_recheck.csv`, `amortized_window_recheck_c2048.csv` |
| Section 5: logit bound, argmax margins, at-risk steps | `run_logit_bound.py`, `run_logit_controls.py`, `analyze_logit_margins.py` | `quality_results/logit_bound.csv`, `logit_margins.csv` |
| Table 8: policy comparison, exactness premium | `run_b2_cooled.py` | `stress_results/b2_policy_block.csv` |

Result files not listed above are earlier runs that a later cooled block
replaced. They are kept rather than pruned; the paper names which block
supersedes which.

## License

Harnesses under `scripts/`: MIT (`LICENSE`). Result files and figures:
CC-BY-4.0 (`LICENSE-DATA`). The `llama.cpp` fork inherits upstream's MIT license
and retains its attribution.
