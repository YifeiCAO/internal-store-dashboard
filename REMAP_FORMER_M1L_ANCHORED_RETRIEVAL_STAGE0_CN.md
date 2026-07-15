# M1l Stage 0：Anchored Retrieval 跨容量零训练审计

> 全新 seed527151；K=1/4/8；每档 64 episodes、128 return-conflict probes。固定 cyclic signature、proposal 上升沿事件锚点、排除当前事件；无训练、无参数修改、无 room/context/位置输入。

## 主结果

| K | Source | Query-only candidate | Shuffled value | Context pair | Segment top-1 | Attn max | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8438 | 0.8906 | 0.3125 | 0.9531 | 0.9375 | 0.8936 | 0.1482 |
| 4 | 0.7344 | 0.8281 | 0.2656 | 0.8906 | 0.8438 | 0.8344 | 0.1903 |
| 8 | 0.3750 | 0.6406 | 0.3125 | 0.9219 | 0.8125 | 0.7810 | 0.2427 |

## 可拒绝性

- Pooled confidence AUROC：`0.6783`。
- Pair-correct / wrong：`354 / 30`。
- Correct / wrong confidence mean：`0.7185 / 0.5777`。

## 冻结门

- 实现 `source_checkpoint_digest`：`PASS`
- 实现 `every_probe_has_latest_event`：`PASS`
- 实现 `latest_event_is_final_segment`：`PASS`
- 实现 `correct_reference_event_available`：`PASS`
- 实现 `current_event_excluded`：`PASS`
- 实现 `attention_rows_normalized`：`PASS`
- 实现 `query_only_candidate_isolation`：`PASS`
- 实现 `query_only_shuffle_isolation`：`PASS`
- 实现 `expected_probe_count_every_level`：`PASS`
- 实现 `parameter_state_unchanged`：`PASS`
- 科学 `k8_context_pair`：`PASS`
- 科学 `k8_correct_segment_top1`：`PASS`
- 科学 `k8_query_only_return_absolute`：`FAIL`
- 科学 `k8_query_only_gain_vs_source`：`PASS`
- 科学 `k8_candidate_beats_shuffled`：`PASS`
- 科学 `context_pair_capacity_stable`：`PASS`
- 科学 `confidence_separates_pair_correct`：`FAIL`

## 结果边界

- 固定事件锚点把 K=8 attention max 保持在 `0.7810`，context pair 达 `0.9219`；相较 M1k 的弥散 learned Q/K，身份检索几何已经明显恢复。
- Query-only candidate 比 source 提高 `+0.2656`，并比同权重 shuffled value 高 `+0.3281`，证明增益来自正确 archival value，不是任意 context 扰动。
- 但 K=8 绝对 recall 只有 `0.6406`，低于冻结 `0.70`；context pair 很高但仍未达到 exact context/downstream recall 所需精度。
- Confidence AUROC `0.6783` 低于 `0.70`，30 个 pair-wrong 候选还不能被可靠拒绝。没有这道 abstention 门，部署式全序列调用仍可能重演 M1k 的过度调用。

冻结状态：`M1L_ANCHORED_RETRIEVAL_STAGE0_REJECTED`。

冻结分支：`STOP_M1L_WITHOUT_TRAINING_AND_RETAIN_FORMAL_M1B`。

本审计只决定是否允许另立 M1l Stage 1；不得在 seed527151 调 temperature、事件定义、confidence 或阈值。正式模型仍为 frozen M1b。
