# ReMAP-Former：8-Seed 一次性 Test Headline

> 八颗预注册 seed；相同 600×batch4 训练预算；固定 test episode；结果生成前未访问 test。

## 协议

- split: `test`
- 每颗 seed 的 test episodes: `512`
- generator seed: `1892`
- paired return-conflict probes: `8192`（每模型）
- Hippoformer：原 baseline 架构，全参数 AdamW，只用 all-token sensory CE。
- M-delta 与 M1b：使用同训练 seed、同 600×batch4 T2 episode 的已冻结 checkpoint。
- 三个模型逐 batch 读取完全相同的 action/sensory；没有 room/context oracle。
- Hippoformer 评估使用 `torch.no_grad()`；代码硬性拒绝 `inference_mode`，保留在线 fast-weight 写入。

## 每颗 Seed

| seed | 模型 | query | clean | conflict | return-conflict | other-room | target prob margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 712 | Hippoformer | 0.8975 | 1.0000 | 0.7949 | 0.0000 | 0.9609 | -0.6244 |
| 712 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.7969 |
| 712 | M1b covariance | 0.9668 | 0.9922 | 0.9414 | 0.8281 | 0.1582 | +0.5643 |
| 713 | Hippoformer | 0.8979 | 1.0000 | 0.7957 | 0.0000 | 0.9512 | -0.6027 |
| 713 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.7973 |
| 713 | M1b covariance | 0.9520 | 0.9953 | 0.9086 | 0.7852 | 0.2051 | +0.5179 |
| 714 | Hippoformer | 0.8971 | 1.0000 | 0.7941 | 0.0000 | 0.9551 | -0.6111 |
| 714 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8218 |
| 714 | M1b covariance | 0.9645 | 0.9922 | 0.9367 | 0.8457 | 0.1465 | +0.5646 |
| 715 | Hippoformer | 0.8967 | 1.0000 | 0.7934 | 0.0000 | 0.9609 | -0.6125 |
| 715 | M-delta | 0.8998 | 1.0000 | 0.7996 | 0.0000 | 1.0000 | -0.7558 |
| 715 | M1b covariance | 0.9697 | 0.9949 | 0.9445 | 0.8281 | 0.1680 | +0.5617 |
| 716 | Hippoformer | 0.8977 | 1.0000 | 0.7953 | 0.0000 | 0.9512 | -0.6077 |
| 716 | M-delta | 0.8998 | 1.0000 | 0.7996 | 0.0000 | 1.0000 | -0.7635 |
| 716 | M1b covariance | 0.9477 | 0.9941 | 0.9012 | 0.8125 | 0.1855 | +0.5303 |
| 717 | Hippoformer | 0.8977 | 1.0000 | 0.7953 | 0.0000 | 0.9590 | -0.6111 |
| 717 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8003 |
| 717 | M1b covariance | 0.9609 | 0.9969 | 0.9250 | 0.8164 | 0.1816 | +0.5392 |
| 718 | Hippoformer | 0.8971 | 1.0000 | 0.7941 | 0.0000 | 0.9531 | -0.6246 |
| 718 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8136 |
| 718 | M1b covariance | 0.9686 | 0.9977 | 0.9395 | 0.8457 | 0.1523 | +0.5886 |
| 719 | Hippoformer | 0.8977 | 1.0000 | 0.7953 | 0.0000 | 0.9590 | -0.6023 |
| 719 | M-delta | 0.9000 | 1.0000 | 0.8000 | 0.0000 | 1.0000 | -0.8048 |
| 719 | M1b covariance | 0.9646 | 0.9957 | 0.9336 | 0.7988 | 0.1973 | +0.5224 |

## 8 Seed 汇总

- Hippoformer return-conflict: `0.0000`
- M-delta return-conflict: `0.0000`
- M1b covariance return-conflict: `0.8201 ± 0.0200` (range `0.7852` to `0.8457`)
- M1b - Hippoformer paired gain: `+0.8201` (range `+0.7852` to `+0.8457`)
- M1b - Hippoformer clean delta: `-0.0051`
- M1b clean: `0.9949 ± 0.0019`; Hippoformer clean: `1.0000`
- 参数量：M1b `1,398,289`；Hippoformer `1,494,447`

## 逐 Probe 配对

- seed 712: M1b-only `848`, Hippoformer-only `0`, both-correct `0`, exact-sign p `1.07e-255`
- seed 713: M1b-only `804`, Hippoformer-only `0`, both-correct `0`, exact-sign p `1.87e-242`
- seed 714: M1b-only `866`, Hippoformer-only `0`, both-correct `0`, exact-sign p `4.06e-261`
- seed 715: M1b-only `848`, Hippoformer-only `0`, both-correct `0`, exact-sign p `1.07e-255`
- seed 716: M1b-only `832`, Hippoformer-only `0`, both-correct `0`, exact-sign p `6.98e-251`
- seed 717: M1b-only `836`, Hippoformer-only `0`, both-correct `0`, exact-sign p `4.36e-252`
- seed 718: M1b-only `866`, Hippoformer-only `0`, both-correct `0`, exact-sign p `4.06e-261`
- seed 719: M1b-only `818`, Hippoformer-only `0`, both-correct `0`, exact-sign p `1.14e-246`

## Gates

- all_models_clean_healthy: `True`
- hippoformer_return_blind_each_seed: `True`
- m1b_beats_hippoformer_each_seed: `True`
- m1b_beats_mdelta_each_seed: `True`
- m1b_mean_return_at_least_0p70: `True`
- m1b_mean_gain_vs_hippoformer_at_least_0p20: `True`

决策：`M1B_ADVANTAGE_CONFIRMED_VS_MATCHED_HIPPOFORMER_AND_MDELTA`

该终审回答的是：在相同 T2 优化预算和相同 test 输入下，M1b 的 context-conditioned covariance fast memory 是否优于原 Hippoformer fast-weight baseline 与无 context 的 M-delta。

## 结论边界

该结果证明的是冻结 random-return T2 任务中的跨房间冲突记忆调用优势；它不是对通用导航、任意长 free rollout 或真实环境 SOTA 的直接声明。
