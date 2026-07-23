# ReMAP-Former：Null-First Context-Need G2

> 默认用 null context 建立前缀 HPC。只有当 read-before-write retrieval 出现强 contradiction 时才调用 dynamic context；null episode 直接复用原 HPC，dynamic episode 才重建一次。

## Frozen Design

- prefix：`72` observed steps
- contradiction：`||v-r||^2 - ||v||^2`
- aggregation：top 10% CVaR；tail count `8`
- calibration midpoint threshold：`0.19240336`
- protocol SHA256：`8e06e4894e11a0bffd79ee370d3399c40e9280ded94dcaa2bf1ef66db048e9ac`

## Separation

| split | T1 median | T2 median | T1 null | T2 dynamic | BAcc | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 0.100201 | 0.284606 | 0.9609 | 1.0000 | 0.9805 | 1.0000 |
| fresh held-out | 0.097997 | 0.282692 | 0.9609 | 1.0000 | 0.9805 | 1.0000 |

## End-to-End and Work

| task | decision | primary | secondary | mean prefix passes |
|---|---:|---:|---:|---:|
| T1 strict 4096, n=64 | null `0.9531` | final `0.9743` | HPC `0.9819` | `1.0469` |
| T2 observed, n=128 | dynamic `1.0000` | return-conflict `0.8594` | clean `0.9891` | `2.0000` |

## Gates

- `PASS` `t2_calibration_median_above_t1`
- `PASS` `heldout_t1_null_selection_rate`
- `PASS` `heldout_t2_dynamic_selection_rate`
- `PASS` `heldout_balanced_accuracy`
- `PASS` `heldout_auroc`
- `PASS` `t1_strict_final_4096`
- `PASS` `t2_return_conflict`
- `PASS` `t2_clean`
- `PASS` `t1_prefix_pass_efficiency`
- `PASS` `null_selected_replayed_zero`
- `PASS` `all_t2_return_conflicts_after_prefix`
- `PASS` `future_write_steps_zero`
- `PASS` `stateful_scan_exact`
- `PASS` `source_model_state_unchanged`
- `PASS` `one_selected_content_hpc`

## Decision

**`CONTEXT_NEED_NULL_FIRST_G2_PASS`**

全部冻结 gate 通过：null-first controller 可以替代 G1，下一步冻结多 checkpoint 确认。
