# ReMAP-Former 长延迟 OOD：Stage B：3-Seed，最长 4356 步

> 冻结 checkpoint；validation-only；未访问 headline test；这是观察历史下的长延迟/重复干扰，不是 free rollout。

## 协议

- seeds: `[712, 715, 719]`
- generator seed: `1895`
- episodes/horizon/seed: `16`
- 四个模型共享完全相同的扩展 episode；无 room/context oracle。
- `m1b_no_covariance` 与 M1b 共用 PFC、EC/place、推断 context、address 和慢权重，只替换 episode-local 写入规则。
- 序列超过 1024 时，PFC 用经过数值等价 smoke 的 32-step chunked window evaluation。

## Return-Conflict 曲线

| repeats | 长度 | Hippoformer | M-delta | M1b covariance | M1b no-cov | M1b clean |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 836 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7500 ± 0.0000 | 0.2708 ± 0.0780 | 1.0000 |
| 32 | 1540 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7500 ± 0.0000 | 0.2708 ± 0.0780 | 1.0000 |
| 64 | 2948 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7500 ± 0.0000 | 0.2708 ± 0.0780 | 1.0000 |
| 96 | 4356 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7500 ± 0.0000 | 0.2708 ± 0.0780 | 1.0000 |

## 最长 Horizon（repeats=96）逐 Seed

| seed | Hippoformer | M-delta | M1b covariance | M1b no-cov | M1b clean |
|---:|---:|---:|---:|---:|---:|
| 712 | 0.0000 | 0.0000 | 0.7500 | 0.2500 | 1.0000 |
| 715 | 0.0000 | 0.0000 | 0.7500 | 0.3750 | 1.0000 |
| 719 | 0.0000 | 0.0000 | 0.7500 | 0.1875 | 1.0000 |

## Gates

- m1b_repeat96_mean_return_at_least_0p30: `True`
- m1b_repeat16_to_repeat96_mean_drop_at_most_0p40: `True`
- m1b_beats_each_baseline_mean_at_every_horizon: `True`
- m1b_repeat96_clean_mean_at_least_0p95: `True`

- M1b 首尾 mean return drop: `+0.0000`
- 决策：`LONG_DELAY_STAGE_B_PASS`

## 解释边界

该实验只回答：在持续给定真实 action/sensory 历史时，延迟和 episode-local 干扰写入增加后，记忆回读是否保持。它不回答无观测闭环生成能力，因此不得标成 free rollout。重复的是同一条闭合 distractor 轨迹，Delta-style 重复写入可能趋于固定点；因此本结果也不能替代持续加入新奇 distractor/content 的容量测试。
