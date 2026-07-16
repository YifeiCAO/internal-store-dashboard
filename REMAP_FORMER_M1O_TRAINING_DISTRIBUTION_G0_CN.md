# ReMAP-Former M1o G0：训练分布机制 Gate

状态：`INVALID_G0_DISTRIBUTION`

## 聚合结果

| 分布 | 有效 token 密度 | source | fixed hard | hard 增益 | 正确 segment |
|---|---:|---:|---:|---:|---:|
| M1n 原分布 | 0.00573 | 0.6719 | 0.8203 | +0.1484 | 0.9219 |
| M1o K4/K8 四冲突 | 0.00869 | 0.5312 | 0.7422 | +0.2109 | 0.8672 |

密度倍率：`1.515x`。

## Gates

- `protocol_digest_matches`: `PASS`
- `source_checkpoint_digest_matches`: `PASS`
- `only_actions_and_sensory_are_model_inputs`: `PASS`
- `all_queries_have_strictly_earlier_candidates`: `PASS`
- `dense_memory_required_token_density`: `FAIL`
- `dense_fixed_hard_gain_vs_source`: `PASS`
- `dense_fixed_hard_absolute`: `PASS`
- `dense_fixed_hard_correct_segment`: `PASS`
- `gauge_pair_observations_identical`: `PASS`

## 决策

G0 未全绿，不允许训练 M1o。
