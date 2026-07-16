# ReMAP-Former M1o G0 v3：外环与对角目标 Gate

状态：`INVALID_G0_DISTRIBUTION`

v1 因密度不足停止；v2 因内环内容通路不健康停止。v3 保留原外环目标并加入四个对角目标，训练无 jitter；原始 K8 transfer monitor 和 blind 仍保留完整 jitter。

| 分布 | 有效 token 密度 | source | fixed hard | hard 增益 | 正确 segment |
|---|---:|---:|---:|---:|---:|
| 原 M1n 分布 | 0.00579 | 0.6641 | 0.8281 | +0.1641 | 0.9375 |
| K4/K8 外环+对角 8-target | 0.01170 | 0.2656 | 0.5156 | +0.2500 | 0.7773 |

密度倍率：`2.020x`。

## Gates

- `protocol_digest_matches`: `PASS`
- `parent_v2_g0_is_frozen_rejection`: `PASS`
- `checkpoint_digests_match`: `PASS`
- `only_actions_and_sensory_are_model_inputs`: `PASS`
- `all_queries_have_strictly_earlier_candidates`: `PASS`
- `dense_memory_required_token_density`: `PASS`
- `dense_fixed_hard_gain_vs_source`: `PASS`
- `dense_fixed_hard_absolute`: `FAIL`
- `dense_fixed_hard_correct_segment`: `PASS`
- `gauge_pair_observations_identical`: `PASS`

## 决策

G0 v3 未全绿，不允许训练。
