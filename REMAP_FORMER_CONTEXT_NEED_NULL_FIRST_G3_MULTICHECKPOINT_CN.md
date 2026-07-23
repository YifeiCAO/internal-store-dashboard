# ReMAP-Former：Null-First G3 八 Checkpoint 确认

> G2 的 72-step null-first score、top-10% contradiction 和状态机完全不变。八个冻结 M1b 共用 fresh 轨迹与一个 pooled-calibration 阈值。

## Frozen Design

- checkpoint：`8` 个，seed 712-719
- shared threshold：`0.18923524`
- prefix / tail count：`72 / 8`
- protocol SHA256：`a5efb35bd1ac4091495d809c29a6aa78a636b63207baa88097c53e11ad36d7ff`
- 每 checkpoint：128+128 calibration、128+128 held-out、16 条 T1 strict 4096、32 条 T2 observed

## Pooled Result

| Held-out T1 null | Held-out T2 dynamic | BAcc | AUROC | T1 4096 | T1 passes | T2 return | T2 clean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9766 | 1.0000 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8242 | 0.9969 |

## Per Checkpoint

| seed | cal T1/T2 median | held BAcc | held AUC | T1 4096 | passes | T2 return | clean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 712 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8125 | 0.9938 |
| 713 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8125 | 1.0000 |
| 714 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8438 | 0.9812 |
| 715 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8125 | 1.0000 |
| 716 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8438 | 1.0000 |
| 717 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8438 | 1.0000 |
| 718 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.8438 | 1.0000 |
| 719 | 0.0930/0.2855 | 0.9883 | 1.0000 | 0.9985 | 1.0000 | 0.7812 | 1.0000 |

## Gates

- `PASS` `checkpoint_count`
- `PASS` `all_checkpoint_calibration_directions`
- `PASS` `pooled_heldout_t1_null`
- `PASS` `pooled_heldout_t2_dynamic`
- `PASS` `pooled_heldout_balanced_accuracy`
- `PASS` `pooled_heldout_auroc`
- `PASS` `checkpoint_bacc_success_count`
- `PASS` `minimum_checkpoint_bacc`
- `PASS` `pooled_t1_strict_final`
- `PASS` `checkpoint_t1_success_count`
- `PASS` `pooled_t2_return_conflict`
- `PASS` `checkpoint_t2_success_count`
- `PASS` `pooled_t2_clean`
- `PASS` `pooled_t1_prefix_efficiency`
- `PASS` `maximum_checkpoint_t1_prefix_passes`
- `PASS` `null_selected_replayed_zero`
- `PASS` `all_t2_return_conflicts_after_prefix`
- `PASS` `future_write_steps_zero`
- `PASS` `all_stateful_scans_exact`
- `PASS` `all_source_model_states_unchanged`
- `PASS` `one_hpc_per_checkpoint`

## Decision

**`CONTEXT_NEED_NULL_FIRST_G3_MULTICHECKPOINT_PASS`**

全部冻结 gate 通过：G2 从单 checkpoint 结果升级为八 checkpoint 共享阈值结果；下一步只允许冻结 OOD task-family 测试。

## Post-result Pathway Hash Audit

G3 通过后追加了不改变任何 gate 的模块级 SHA256 审计，防止把共享权重误写成八次独立复现：

| 审计项 | 唯一 hash 数 | 结论 |
|---|---:|---|
| 完整 model state | `8` | 八个 checkpoint 文件确实不同 |
| `state_token` | `8` | dynamic context retention 随 seed 变化 |
| `context_head` | `8` | dynamic context projection 随 seed 变化 |
| `EC + place + fixed-null address + HPC` 联合通路 | `1` | null-controller 直接读取的权重是共享的 |
| calibration / held-out score vector | `1 / 1` | 八个 controller decision 不是独立重复 |

因此 G3 **支持**：同一 null-first controller 和共享阈值，在八套不同 dynamic-context 推断模块下都能正确调用，逐 checkpoint T2 return-conflict 为 `0.7812-0.8438`。

G3 **不支持**：“八套独立训练的 controller-facing EC/place/HPC 都复现了该 score”。下一项有效实验应改变 fresh task distribution，或取得真正独立的 controller-facing pathway；不能继续用相同 EC/place/HPC 的 checkpoint 数包装独立性。

机器审计：`reports/REMAP_FORMER_CONTEXT_NEED_G3_PATHWAY_HASH_AUDIT.json`。
