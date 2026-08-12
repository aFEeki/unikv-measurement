# Superseded data files

CSVs cannot carry a comment header without breaking `csv.DictReader`, so the
markers live here. Analysis files carry their own `*** SUPERSEDED ***` headers.

**If you are looking for a number to put in the paper, use the right-hand
column.** Last updated 2026-08-07.

| stale file | superseded by | for what |
|---|---|---|
| `recall_cost_isochronal.csv` | `b2_isochronal_both_modes.csv` | γ and δ at C=1024. B2 measured both tier modes on a quiet machine; the CPU-pinned values reproduce to within noise but quote B2's. |
| `r3_policy_compare_cooled_master.csv` | `b2_policy_block.csv` | every throughput number at C=1024 and C=2048. B2 adds the device-visible arm and position-balances the CPU-pinned one. |
| `b1_interleaved_ab.csv` | `b2_policy_block.csv` | the exactness premium and the device-visible speedup. The uncooled A/B was biased in **both** directions (see below). |
| `r3_policy_compare_quickcheck_master.csv`, `r3_policy_compare_h2osmoke_master.csv` | — | single-run smoke checks kept only as evidence that the smoke tests happened. Never quote. |

## Still current, do not assume superseded

- `b1_device_tier_ceiling.csv` — the device-visible ceiling (81,920–90,112
  cells) and the CPU-pinned tier's absence of any device-side ceiling. B2 did
  **not** re-measure these.
- `capacity_probe.csv`, `capacity_workload.csv`, `capacity_demo.csv`,
  `f4_a1_ubatch_sweep.csv`, `f4_a2_continuation_arms.csv`,
  `f4_a3_flash_attn_capacity.csv` — all of Finding 4.
- `recall_cost_steps_A_p3_long.csv`, `recall_cost_steps_B_p1_ctrl.csv`,
  `recall_cost_e2e.csv` — the long-run/thermal-control datasets and the
  superlinearity result beyond the fitted range (666 ms/step at n_spill =
  61,471). B2's fits stop at 8,192 and do not replace these.
- `p3_alpha_sweep_cooled_master.csv`, `p3_drain_control_master.csv` — Finding 3.
- `quality_arms_unified.csv` — Finding 2.

## Why the uncooled A/B was wrong in both directions

It overstated the device-visible speedup (+46.5% vs the protocol +25.9%) and
understated the exactness premium (+7.77% vs +11.42% at C=1024). Every arm is
faster on a cool machine, but the CPU-pinned arm gains most (+19.9%, against
+4.0% and +7.5%), because CPU work throttles harder than GPU work on this
device. Counterbalancing removes order effects within a session; it does not
remove an arm's sensitivity to the session's thermal level.

## Known consumer of a superseded file

`scripts/make_figures_final.py` still reads `r3_policy_compare_cooled_master.csv`
to verify `fig3_tokens` (token **counts**, which B2 did not change). That figure
belongs to the frozen TACO draft and is not used by the measurement paper. If
`fig3_tokens` is ever regenerated, re-point it at `b2_policy_block.csv` first.
