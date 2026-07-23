# ReMAP-Former：Context-Need Retrieval G1 高功效复现

> G0 已正式判 FAIL。本次只增加全新样本量；score、64-step prefix、midpoint-of-medians 阈值、决策方向、checkpoint 和 acceptance gates 全部不变。

## Protocol

- calibration：T1 `256` + T2 `256`
- held-out：T1 `256` + T2 `256`
- threshold：`0.60076547`
- protocol SHA256：`b1607fcc011e465dae27e55528a9d99a7f0bd9d8cb05e2d15a8da3e6f74f7574`

## Separation

| split | T1 median | T2 median | T1 null | T2 dynamic | balanced acc | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 0.486820 | 0.714711 | 0.9141 | 0.9844 | 0.9492 | 0.9940 |
| fresh held-out | 0.476558 | 0.734664 | 0.9375 | 0.9922 | 0.9648 | 0.9946 |

## End-to-End

| task | selection | primary | secondary |
|---|---:|---:|---:|
| T1 strict 4096, n=64 | null `0.8906` | final `0.9721` | HPC `0.9823` |
| T2 dev, n=128 | dynamic `1.0000` | return-conflict `0.8125` | clean `0.9938` |

## Gates

- `PASS` `t2_calibration_median_above_t1`
- `PASS` `heldout_t1_null_selection_rate`
- `PASS` `heldout_t2_dynamic_selection_rate`
- `PASS` `heldout_balanced_accuracy`
- `PASS` `heldout_auroc`
- `PASS` `t1_strict_final_4096`
- `PASS` `t2_return_conflict`
- `PASS` `t2_clean`
- `PASS` `all_t2_return_conflicts_after_prefix`
- `PASS` `future_write_steps_zero`
- `PASS` `source_model_state_unchanged`
- `PASS` `wrapper_base_state_unchanged`
- `PASS` `one_selected_content_hpc`

## Decision

**`CONTEXT_NEED_RETRIEVAL_G1_REPLICATION_PASS`**

全新大样本复现全部通过：检索一致性可以作为 context-need 的因果教师信号，下一步可单独冻结 neuralization。
