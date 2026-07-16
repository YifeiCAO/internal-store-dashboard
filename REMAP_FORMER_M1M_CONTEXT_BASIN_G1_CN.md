# M1m G1：Current-Context Basin 零参数检索

> 三个全新 seeds；K=8；共 384 个 return-conflict probes。当前 causal base-context 作为 query，对更早 event base-context 做 hard nearest-neighbor；无训练、无新 memory、无 room/context/位置输入。

## 主结果

| Seed | Source | Signature hard | Context basin | Shuffled value | Oracle-best | Exact correct | Exact wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 727151 | 0.4375 | 0.6875 | 0.5469 | 0.3906 | 1.0000 | 1.0000 | 0.0000 |
| 737151 | 0.3594 | 0.7344 | 0.2969 | 0.2812 | 1.0000 | 1.0000 | 0.0000 |
| 747151 | 0.3906 | 0.7188 | 0.4062 | 0.1719 | 0.9688 | 0.9688 | 0.0156 |
| **aggregate** | **0.3958** | **0.7135** | **0.4167** | **0.2812** | **0.9896** | **0.9896** | **0.0052** |

## Basin 几何与置信度

- Context pair：`0.7760`。
- Correct / wrong cosine：`0.8659/0.7287`；margin `0.1372`。
- Correct-segment top-1：`0.3333`。
- Top1-top2 margin 对 pair-correct AUROC：`0.6519`。
- Pair-correct / wrong：`298/86`；confidence mean `0.0328/0.0202`。

## 冻结门

- 实现 `parent_protocol_digest`：`PASS`
- 实现 `parent_result_digest`：`PASS`
- 实现 `source_checkpoint_digest`：`PASS`
- 实现 `every_probe_has_latest_event`：`PASS`
- 实现 `latest_event_is_final_segment`：`PASS`
- 实现 `correct_reference_event_available`：`PASS`
- 实现 `current_event_excluded`：`PASS`
- 实现 `signature_top1_equals_fixed_key_argmax`：`PASS`
- 实现 `oracle_best_equals_value_cosine_argmax`：`PASS`
- 实现 `context_basin_equals_current_context_argmax`：`PASS`
- 实现 `shuffled_preserves_selected_index_and_candidate_count`：`PASS`
- 实现 `basin_record_for_every_probe`：`PASS`
- 实现 `confidence_is_finite`：`PASS`
- 实现 `signature_query_only_isolation`：`PASS`
- 实现 `basin_query_only_isolation`：`PASS`
- 实现 `shuffled_query_only_isolation`：`PASS`
- 实现 `oracle_query_only_isolation`：`PASS`
- 实现 `exact_correct_query_only_isolation`：`PASS`
- 实现 `exact_wrong_query_only_isolation`：`PASS`
- 实现 `normal_hpc_replay_matches_source`：`PASS`
- 实现 `direct_basin_matches_serial_smoke`：`PASS`
- 实现 `expected_probe_count_every_seed`：`PASS`
- 实现 `expected_total_probe_count`：`PASS`
- 实现 `parameter_state_unchanged`：`PASS`
- 科学 `exact_correct_path_healthy`：`PASS`
- 科学 `exact_wrong_path_specific`：`PASS`
- 科学 `oracle_ceiling_healthy`：`PASS`
- 科学 `basin_absolute`：`FAIL`
- 科学 `basin_gain_vs_signature`：`FAIL`
- 科学 `basin_gain_vs_source`：`FAIL`
- 科学 `basin_gain_vs_shuffled`：`FAIL`
- 科学 `basin_context_pair`：`FAIL`
- 科学 `basin_correct_segment_top1`：`FAIL`
- 科学 `basin_seed_direction`：`FAIL`
- 科学 `basin_confidence_auroc`：`FAIL`

## 结果边界

- Basin 相对 signature / source / shuffled 的增益为 `-0.2969` / `+0.0208` / `+0.1354`。
- 局部 HPC replay 与完整 query-only rollout 的 prediction 最大差为 `2.59e-06`。
- 本 G1 只检验合法零参数 event ranking 与 abstention，不直接解锁训练或 formal。

冻结状态：`CONTEXT_BASIN_RANKING_INSUFFICIENT`。

冻结分支：`PRE_REGISTER_NEURAL_EVENT_RANKER_WITH_SINGLE_SENSORY_CE_AND_FRESH_TRAIN_MONITOR_BLIND_SEEDS`。

运行耗时：`121.8s`；设备：`cuda`。
