# ReMAP-Former 长延迟 OOD：Stage A：8-Seed，最长 836 步

> 冻结 checkpoint；validation-only；未访问 headline test；这是观察历史下的长延迟/重复干扰，不是 free rollout。

## 协议

- seeds: `[712, 713, 714, 715, 716, 717, 718, 719]`
- generator seed: `1894`
- episodes/horizon/seed: `64`
- 四个模型共享完全相同的扩展 episode；无 room/context oracle。
- `m1b_no_covariance` 与 M1b 共用 PFC、EC/place、推断 context、address 和慢权重，只替换 episode-local 写入规则。
- 序列超过 1024 时，PFC 用经过数值等价 smoke 的 32-step chunked window evaluation。

## Return-Conflict 曲线

| repeats | 长度 | Hippoformer | M-delta | M1b covariance | M1b no-cov | M1b clean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 176 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8301 ± 0.0286 | 0.2793 ± 0.0503 | 0.9918 |
| 2 | 220 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8516 ± 0.0282 | 0.3047 ± 0.0475 | 0.9935 |
| 4 | 308 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7871 ± 0.0281 | 0.2285 ± 0.0530 | 0.9941 |
| 8 | 484 | 0.0039 ± 0.0068 | 0.0000 ± 0.0000 | 0.8496 ± 0.0246 | 0.2812 ± 0.0456 | 0.9979 |
| 16 | 836 | 0.0039 ± 0.0068 | 0.0000 ± 0.0000 | 0.8496 ± 0.0246 | 0.2773 ± 0.0493 | 0.9987 |

## 最长 Horizon（repeats=16）逐 Seed

| seed | Hippoformer | M-delta | M1b covariance | M1b no-cov | M1b clean |
|---:|---:|---:|---:|---:|---:|
| 712 | 0.0000 | 0.0000 | 0.8438 | 0.2500 | 0.9984 |
| 713 | 0.0156 | 0.0000 | 0.8750 | 0.2969 | 0.9992 |
| 714 | 0.0000 | 0.0000 | 0.8594 | 0.3125 | 0.9984 |
| 715 | 0.0000 | 0.0000 | 0.8281 | 0.2344 | 0.9992 |
| 716 | 0.0000 | 0.0000 | 0.8125 | 0.3281 | 0.9977 |
| 717 | 0.0156 | 0.0000 | 0.8594 | 0.1875 | 0.9992 |
| 718 | 0.0000 | 0.0000 | 0.8906 | 0.3438 | 0.9992 |
| 719 | 0.0000 | 0.0000 | 0.8281 | 0.2656 | 0.9984 |

## Gates

- m1b_clean_each_seed_horizon_at_least_0p98: `True`
- m1b_repeat16_mean_return_at_least_0p50: `True`
- m1b_repeat1_to_repeat16_mean_drop_at_most_0p25: `True`
- m1b_beats_each_baseline_each_seed_at_repeat16: `True`

- M1b 首尾 mean return drop: `-0.0195`
- 决策：`LONG_DELAY_STAGE_A_PASS`

## 解释边界

该实验只回答：在持续给定真实 action/sensory 历史时，延迟和 episode-local 干扰写入增加后，记忆回读是否保持。它不回答无观测闭环生成能力，因此不得标成 free rollout。重复的是同一条闭合 distractor 轨迹，Delta-style 重复写入可能趋于固定点；因此本结果也不能替代持续加入新奇 distractor/content 的容量测试。
