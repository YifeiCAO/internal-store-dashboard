# ReMAP-Former：Neural Context-Need 单 Seed Pilot

> 解析式 retrieval-consistency 只在训练时提供连续 teacher score；正式推理的选择只读冻结 PFC 的前 64 步 hidden，不调用 HPC。做出一次 hard decision 后，从第 0 步重放并建立一套 selected HPC。

## Architecture / Training

- trainable parameters：`16770`
- fixed step：`600`；checkpoint selection：无
- single loss：SmoothL1 teacher-score regression
- sensory CE：无；ST estimator：无；room/task/context label：无
- protocol SHA256：`c8364202e0b002c2425e68dbf21b4e20dd4759bdb9f05e23e6afba765a7cb4c5`

## Held-Out Context Decision

| teacher agreement | T1 null | T2 dynamic | balanced acc | score MAE(z) | score Pearson | inference HPC calls |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8047 | 0.7656 | 0.8984 | 0.8320 | 0.6887 | 0.6355 | 0 |

## End-to-End

| task | decision | primary | secondary |
|---|---:|---:|---:|
| T1 strict 4096, n=64 | null `0.7188` | final `0.9159` | HPC `0.9445` |
| T2 dev, n=128 | dynamic `0.8906` | return-conflict `0.7578` | clean `0.9969` |

## Gates

- `FAIL` `heldout_teacher_decision_agreement`
- `FAIL` `heldout_t1_null_selection_rate`
- `FAIL` `heldout_t2_dynamic_selection_rate`
- `FAIL` `heldout_balanced_accuracy`
- `FAIL` `t1_strict_final_4096`
- `PASS` `t2_return_conflict`
- `PASS` `t2_clean`
- `PASS` `all_t2_return_conflicts_after_prefix`
- `PASS` `neural_decision_hpc_calls_zero`
- `PASS` `future_write_steps_zero`
- `PASS` `source_model_state_unchanged`
- `PASS` `wrapper_base_state_unchanged`
- `PASS` `only_neural_head_trainable`
- `PASS` `one_selected_content_hpc`

## Decision

**`CONTEXT_NEED_NEURAL_SINGLE_SEED_FAIL`**

至少一个冻结 gate 未通过：停止 pilot，不在本批数据上调参。
