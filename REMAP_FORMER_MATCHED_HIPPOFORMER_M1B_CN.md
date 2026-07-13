# ReMAP-Former：预算匹配 Hippoformer 终审

> 三颗正式 seed；相同 600×batch4 训练预算；同一 validation episode；test 未使用。

## 协议

- split: `validation`
- 每颗 seed 的 validation episodes: `256`
- validation generator seed: `892`
- Hippoformer：原 baseline 架构，全参数 AdamW，只用 all-token sensory CE。
- M-delta 与 M1b：使用同训练 seed、同 600×batch4 T2 episode 的已冻结 checkpoint。
- 三个模型逐 batch 读取完全相同的 action/sensory；没有 room/context oracle。
- Hippoformer 评估使用 `torch.no_grad()`；代码硬性拒绝 `inference_mode`，保留在线 fast-weight 写入。

## 每颗 Seed

| seed | 模型 | query | clean | conflict | return-conflict | other-room | target prob margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 712 | Hippoformer | 0.8965 | 1.0000 | 0.7930 | 0.0000 | 0.9805 | -0.6330 |
| 712 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8054 |
| 712 | M1b covariance | 0.9652 | 0.9938 | 0.9367 | 0.7969 | 0.1953 | +0.5172 |
| 713 | Hippoformer | 0.8977 | 1.0000 | 0.7953 | 0.0000 | 0.9688 | -0.6069 |
| 713 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8024 |
| 713 | M1b covariance | 0.9617 | 0.9977 | 0.9258 | 0.8047 | 0.1953 | +0.5345 |
| 714 | Hippoformer | 0.8973 | 1.0000 | 0.7945 | 0.0000 | 0.9805 | -0.6139 |
| 714 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8267 |
| 714 | M1b covariance | 0.9727 | 0.9984 | 0.9469 | 0.8203 | 0.1758 | +0.5458 |

## 三 Seed 汇总

- Hippoformer return-conflict: `0.0000`
- M-delta return-conflict: `0.0000`
- M1b covariance return-conflict: `0.8073`
- M1b - Hippoformer paired gain: `+0.8073` (range `+0.7969` to `+0.8203`)
- M1b - Hippoformer clean delta: `-0.0034`

## 逐 Probe 配对

- seed 712: M1b-only `408`, Hippoformer-only `0`, both-correct `0`, exact-sign p `3.03e-123`
- seed 713: M1b-only `412`, Hippoformer-only `0`, both-correct `0`, exact-sign p `1.89e-124`
- seed 714: M1b-only `420`, Hippoformer-only `0`, both-correct `0`, exact-sign p `7.39e-127`

## Gates

- all_models_clean_healthy: `True`
- hippoformer_return_blind_each_seed: `True`
- m1b_beats_hippoformer_each_seed: `True`
- m1b_beats_mdelta_each_seed: `True`
- m1b_mean_return_at_least_0p70: `True`
- m1b_mean_gain_vs_hippoformer_at_least_0p20: `True`

决策：`M1B_ADVANTAGE_CONFIRMED_VS_MATCHED_HIPPOFORMER_AND_MDELTA`

该终审回答的是：在相同 T2 优化预算和相同验证输入下，M1b 的 context-conditioned covariance fast memory 是否优于原 Hippoformer fast-weight baseline 与无 context 的 M-delta。
