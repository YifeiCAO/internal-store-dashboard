# ReMAP-Former 2D-R2 Anti-Shortcut 审计

> 标签是 episode 内“当前 revisit 属于第一种还是第二种首次见到的 context”。当前 target 不进入 probe；完整历史特征只由 causal action signature 的匹配关系构成。

## OOD 协议

- train：6/7/8 段，jitter pair 0–2。
- validation：9/10/11 段，jitter pair 3–5。
- train / validation path family 完全不相交。

## Probe

| Probe | balanced accuracy |
|---|---:|
| 当前可用 observation | 0.4802 |
| 时间 | 0.5302 |
| 当前/两步 action | 0.5000 |
| 8 步 observation/action | 0.4741 |
| 当前路径签名 | 0.5288 |
| 完整历史关系 | 1.0000 |
| 完整历史解析 oracle | 1.0000 |

## 反事实与防泄漏

- 同时间桶、同局部线索、相反历史标签支持率：`0.2269`
- gauge-pair observation mismatch：`0`
- train/test family overlap：`0`
- train/test sequence length：`274–374` / `460–580`

## Gate

- current_observation_near_chance: `True`
- time_only_near_chance: `True`
- action_only_near_chance: `True`
- short_window_limited: `True`
- current_path_alone_near_chance: `True`
- full_history_relation_high: `True`
- relative_context_labels_balanced: `True`
- strong_duration_ood: `True`
- path_family_split_disjoint: `True`
- same_time_counterfactual_pairs_exist: `True`
- gauge_pair_observations_identical: `True`
- absolute_room_oracle_is_gauge_flipped: `True`

## 判断

R2 generator gate 全绿，可以进入冻结模型与新训练模型的 context-state probe。
