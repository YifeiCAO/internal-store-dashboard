# ReMAP-Former M1o G0 v4：长延迟多次 Re-entry Gate

状态：`M1O_DENSE_DISTRIBUTION_G0_V4_PASSED`

每连续 4 个 novel distractors 后插入一次原始四目标 re-entry；K8/K12 分别含 2/3 次长延迟返回。模型输入、HPC、目标位置和统一 sensory CE 均不变。

| 分布 | 有效 token 密度 | source | fixed hard | hard 增益 | 正确 room |
|---|---:|---:|---:|---:|---:|
| 原 M1n 分布 | 0.00571 | 0.6641 | 0.7812 | +0.1172 | - |
| K8/K12 dense re-entry | 0.01567 | 0.6781 | 0.8938 | +0.2156 | 0.9469 |

密度倍率：`2.743x`。

## Gates

- `protocol_digest_matches`: `PASS`
- `parent_v3_g0_is_frozen_rejection`: `PASS`
- `checkpoint_digests_match`: `PASS`
- `only_actions_and_sensory_are_model_inputs`: `PASS`
- `all_queries_have_strictly_earlier_candidates`: `PASS`
- `dense_memory_required_token_density`: `PASS`
- `dense_fixed_hard_gain_vs_source`: `PASS`
- `dense_fixed_hard_absolute`: `PASS`
- `dense_fixed_hard_correct_room`: `PASS`
- `gauge_pair_observations_identical`: `PASS`

## 决策

G0 v4 全绿，允许执行一次固定 800-step 训练。
