# ReMAP-Former M1o G0 v2：内外环训练分布 Gate

状态：`INVALID_G0_DISTRIBUTION`

v1 只达到 `1.515x` 密度并被正确拦截；v2 在每次 context re-entry 中使用 4 个内环与 4 个外环目标，仍以统一 sensory CE 训练。

| 分布 | 有效 token 密度 | source | fixed hard | hard 增益 | 正确 segment |
|---|---:|---:|---:|---:|---:|
| 原 M1n 分布 | 0.00569 | 0.6172 | 0.8438 | +0.2266 | 0.8906 |
| K4/K8 内外环 8-target | 0.01300 | 0.5273 | 0.5859 | +0.0586 | 0.7695 |

密度倍率：`2.283x`。

## Gates

- `protocol_digest_matches`: `PASS`
- `parent_v1_g0_is_frozen_rejection`: `PASS`
- `checkpoint_digests_match`: `PASS`
- `only_actions_and_sensory_are_model_inputs`: `PASS`
- `all_queries_have_strictly_earlier_candidates`: `PASS`
- `dense_memory_required_token_density`: `PASS`
- `dense_fixed_hard_gain_vs_source`: `FAIL`
- `dense_fixed_hard_absolute`: `FAIL`
- `dense_fixed_hard_correct_segment`: `PASS`
- `gauge_pair_observations_identical`: `PASS`

## 决策

G0 v2 未全绿，不允许训练。
