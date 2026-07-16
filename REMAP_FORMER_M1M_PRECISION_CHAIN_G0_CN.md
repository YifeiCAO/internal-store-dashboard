# M1m G0：精确 Context 与 HPC 安装链路分解

> 三个全新 seeds；K=8；共 384 个 return-conflict probes。零训练、query-only，冻结复用 M1l soft candidate，并加入 hard top-1、oracle-best archival value、exact correct/wrong context。

## 主结果

| Seed | Source | Soft | Hard top-1 | Oracle-best value | Exact correct | Exact wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 627151 | 0.3438 | 0.7031 | 0.7188 | 1.0000 | 1.0000 | 0.0156 |
| 637151 | 0.3750 | 0.6719 | 0.7188 | 1.0000 | 1.0000 | 0.0000 |
| 647151 | 0.3438 | 0.7031 | 0.7969 | 1.0000 | 1.0000 | 0.0000 |
| **aggregate** | **0.3542** | **0.6927** | **0.7448** | **1.0000** | **1.0000** | **0.0052** |

## Context 几何

| 条件 | Pair accuracy | Correct cosine | Wrong cosine | Margin |
|---|---:|---:|---:|---:|
| soft_candidate | 0.8698 | 0.9304 | 0.7486 | 0.1818 |
| hard_top1_event | 0.8646 | 0.9126 | 0.7281 | 0.1845 |
| oracle_best_archival_value | 1.0000 | 0.9984 | 0.6995 | 0.2989 |

## Soft candidate 首个失败层

- `context_pair_failure`：`50`
- `address_pair_failure`：`0`
- `retrieval_pair_failure`：`14`
- `decoder_or_fusion_failure`：`60`
- `correct`：`260`
- 错误主导层：`decoder_or_fusion_failure`；share `0.4839`；dominant `False`。

## 冻结门与分类

- 实现 `parent_protocol_digest`：`PASS`
- 实现 `parent_result_digest`：`PASS`
- 实现 `source_checkpoint_digest`：`PASS`
- 实现 `every_probe_has_latest_event`：`PASS`
- 实现 `latest_event_is_final_segment`：`PASS`
- 实现 `correct_reference_event_available`：`PASS`
- 实现 `current_event_excluded`：`PASS`
- 实现 `soft_attention_rows_normalized`：`PASS`
- 实现 `hard_top1_equals_fixed_key_argmax`：`PASS`
- 实现 `oracle_best_equals_value_cosine_argmax`：`PASS`
- 实现 `precision_record_for_every_probe`：`PASS`
- 实现 `soft_query_only_isolation`：`PASS`
- 实现 `hard_query_only_isolation`：`PASS`
- 实现 `oracle_best_query_only_isolation`：`PASS`
- 实现 `exact_correct_query_only_isolation`：`PASS`
- 实现 `exact_wrong_query_only_isolation`：`PASS`
- 实现 `expected_probe_count_every_seed`：`PASS`
- 实现 `expected_total_probe_count`：`PASS`
- 实现 `chain_outputs_aligned`：`PASS`
- 实现 `normal_hpc_replay_matches_source`：`PASS`
- 实现 `direct_soft_matches_serial_smoke`：`PASS`
- 实现 `parameter_state_unchanged`：`PASS`
- 分类 `exact_correct_path_healthy`：`PASS`
- 分类 `exact_wrong_path_specific`：`PASS`
- 分类 `hard_absolute`：`PASS`
- 分类 `hard_gain_vs_soft`：`FAIL`
- 分类 `hard_preserves_context_pair`：`PASS`
- 分类 `hard_seed_direction`：`PASS`
- 分类 `oracle_best_absolute`：`PASS`
- 分类 `oracle_best_gain_vs_hard`：`PASS`
- 分类 `oracle_best_seed_direction`：`PASS`
- 分类 `archival_value_below_floor`：`FAIL`
- 分类 `exact_gain_vs_oracle_best`：`FAIL`

## 结果边界

- Hard top-1 在三个 seed 都高于 soft，但聚合只从 `0.6927` 升到 `0.7448`，增益 `+0.0521` 未达冻结 `+0.10`；因此不能把剩余问题归为 soft mixing。
- Oracle-best archival value 达 `1.0000`，比 hard top-1 高 `+0.2552`，且 3/3 seed 正向；正确 value 已经存在于同一 causal candidate set，主要缺口是合法 key ranking。
- Exact-correct / exact-wrong 为 `1.0000/0.0052`，oracle-best 与 exact-correct 无差，排除 archival value 不足和 context→address→HPC 下游失效。
- 局部 HPC replay 与完整 query-only rollout 的 prediction 最大差为 `1.79e-06`，低于实现门 `1.0e-05`。
- 下一步只允许研究合法的 event key ranking；不增加 slot、第二套 fast weights、room/context/位置输入，也不直接访问 formal。

冻结状态：`M1M_KEY_RANKING_DOMINANT`。

冻结分支：`PRE_REGISTER_KEY_RANKING_RESEARCH_WITH_FRESH_TRAIN_MONITOR_AND_BLIND_SEEDS`。

运行耗时：`133.9s`；设备：`cuda`。

本 G0 只定位失败层，不直接解锁训练或 formal。三个 generator seeds、hard tie break、value 定义、阈值和分类优先级均已封存。
