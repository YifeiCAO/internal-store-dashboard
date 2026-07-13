# ReMAP-Former M1b：8-Seed 组件必要性消融

> 冻结 checkpoint；validation-only；所有条件共享输入；headline test 未再次访问。

## 协议

- generator seed: `1893`
- episodes/checkpoint: `256`
- return-conflict probes/condition/seed: `512`
- `no_covariance` 复用相同 PFC/place/context/address，只替换 DeltaHPC update。
- 其余条件复用同一次 full forward 的 PFC hidden、place 与 learned context，只干预被点名组件。

## 每颗 Seed：Return-Conflict

| seed | full | no cov | fixed ctx | shuffled ctx | HPC zero | wrong return ctx | correct return ctx |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 712 | 0.7578 | 0.2500 | 0.0000 | 0.4414 | 0.0234 | 0.0000 | 0.9531 |
| 713 | 0.7734 | 0.2500 | 0.0000 | 0.4258 | 0.0234 | 0.0000 | 0.9336 |
| 714 | 0.8164 | 0.3359 | 0.0000 | 0.4297 | 0.0234 | 0.0000 | 0.9648 |
| 715 | 0.8125 | 0.2539 | 0.0000 | 0.4297 | 0.0234 | 0.0039 | 0.9414 |
| 716 | 0.7695 | 0.2773 | 0.0000 | 0.4414 | 0.0234 | 0.0000 | 0.8945 |
| 717 | 0.8164 | 0.2539 | 0.0000 | 0.4297 | 0.0234 | 0.0039 | 0.9609 |
| 718 | 0.8164 | 0.2578 | 0.0000 | 0.4336 | 0.0234 | 0.0000 | 0.9531 |
| 719 | 0.7930 | 0.2617 | 0.0000 | 0.4531 | 0.0234 | 0.0000 | 0.9375 |

## 8-Seed 均值

| 条件 | return-conflict mean ± std | clean mean | other-room rate | prob margin |
|---|---:|---:|---:|---:|
| full | 0.7944 ± 0.0229 | 0.9933 | 0.1987 | +0.5162 |
| no_covariance | 0.2676 ± 0.0271 | 1.0000 | 0.7324 | -0.3807 |
| fixed_context_all_steps | 0.0000 ± 0.0000 | 1.0000 | 1.0000 | -0.9689 |
| shuffled_context_all_steps | 0.4355 ± 0.0085 | 0.9920 | 0.5610 | -0.1395 |
| hpc_read_zero | 0.0234 ± 0.0000 | 0.1539 | 0.0117 | -0.0019 |
| wrong_return_context | 0.0010 ± 0.0017 | 0.9961 | 0.9990 | -0.9619 |
| correct_return_context | 0.9424 ± 0.0208 | 0.9960 | 0.0576 | +0.7190 |

## 关键净下降

- full_minus_no_covariance: `+0.5269 ± 0.0296` (range `+0.4805` to `+0.5625`)
- full_minus_fixed_context: `+0.7944 ± 0.0229` (range `+0.7578` to `+0.8164`)
- full_minus_shuffled_context: `+0.3589 ± 0.0272` (range `+0.3164` to `+0.3867`)
- full_minus_hpc_read_zero: `+0.7710 ± 0.0229` (range `+0.7344` to `+0.7930`)
- full_minus_wrong_return_context: `+0.7935 ± 0.0220` (range `+0.7578` to `+0.8164`)
- correct_return_context_minus_full: `+0.1479 ± 0.0207` (range `+0.1250` to `+0.1953`)
- wrong_return_other_room_rate_increase: `+0.8003 ± 0.0203` (range `+0.7695` to `+0.8281`)

## Gates

- full_clean_healthy_each_seed: `True`
- full_mean_return_at_least_0p70: `True`
- full_beats_no_covariance_each_seed: `True`
- full_minus_no_covariance_mean_at_least_0p25: `True`
- full_minus_fixed_context_mean_at_least_0p25: `True`
- full_minus_shuffled_context_mean_at_least_0p10: `True`
- full_minus_hpc_read_zero_mean_at_least_0p50: `True`
- full_minus_wrong_return_context_mean_at_least_0p30: `True`
- wrong_return_other_room_rate_increase_at_least_0p30: `True`

决策：`M1B_COMPONENT_NECESSITY_CONFIRMED`

解释：只有在 covariance write、episode-matched learned context 和 HPC retrieval 同时存在时，完整 M1b 才应保留 headline 级 return-conflict 表现。wrong-return 干预用于验证 read address 的因果方向；correct-return 是诊断上界，不是模型输入。
