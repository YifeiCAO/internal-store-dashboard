# ReMAP-Former：Context-Need Retrieval G0

> 零训练、冻结 M1b。先用前 64 个已观察步的 HPC read-before-write 检索一致性做一次 episode 级选择；动态试跑状态随后丢弃，从第 0 步重放并只建立一套被选中的 HPC。

## 冻结规则

- score：`||v_t||^2 - ||v_t-r_t||^2`，64 步算术平均
- threshold：calibration 两类中位数中点 = `0.59487367`
- 大于等于 threshold 选动态 context，否则选 null context
- 无 threshold grid、无 held-out 后换分数、无参数训练
- protocol SHA256：`f2a5ffab1a2553adcfc1b4dac70c296dcf1f1d51162f5902ed3000d6ca679291`

## Signal Separation

| split | T1 median | T2 median | T1 选 null | T2 选 dynamic | balanced acc | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 0.483760 | 0.705987 | 0.9688 | 0.9688 | 0.9688 | 0.9922 |
| held-out | 0.480070 | 0.728588 | 0.8906 | 0.9688 | 0.9297 | 0.9961 |

## End-to-End

| task | selection | final / return-conflict | HPC / clean |
|---|---:|---:|---:|
| T1 strict 4096 | null `0.7500` | final `0.9313` | HPC `0.9590` |
| T2 dev | dynamic `1.0000` | return-conflict `0.8594` | clean `1.0000` |

## Gates

- `PASS` `t2_calibration_median_above_t1`
- `FAIL` `heldout_t1_null_selection_rate`
- `PASS` `heldout_t2_dynamic_selection_rate`
- `PASS` `heldout_balanced_accuracy`
- `PASS` `heldout_auroc`
- `FAIL` `t1_strict_final_4096`
- `PASS` `t2_return_conflict`
- `PASS` `t2_clean`
- `PASS` `all_t2_return_conflicts_after_prefix`
- `PASS` `future_write_steps_zero`
- `PASS` `source_model_state_unchanged`
- `PASS` `wrapper_base_state_unchanged`
- `PASS` `one_selected_content_hpc`

## 结论

**`CONTEXT_NEED_RETRIEVAL_G0_FAIL`**

至少一个冻结 gate 未通过：停止 G0，不在本批 held-out 上改阈值、前缀长度或分数公式。
