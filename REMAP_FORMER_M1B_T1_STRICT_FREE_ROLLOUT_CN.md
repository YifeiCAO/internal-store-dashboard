# ReMAP-Former M1b：T1 严格 True Free Rollout

> 冻结 checkpoint；64 步真实 context；未来只输入 action 并反馈模型预测；future memory writes 严格为 0。

## 协议

- checkpoint seeds: `[712, 713, 714, 715, 716, 717, 718, 719]`
- generator seed: `2071601`
- episodes/checkpoint: `8`
- context length: `64`
- horizons: `[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]`
- 所有模型读取同一批最大轨迹；每个 horizon 是该轨迹的配对前缀。
- 评估未使用 `torch.inference_mode()`，以保留 Hippoformer context fast-weight 写入。

## Final Accuracy

| horizon | Hippoformer | M-delta | M1b covariance | M1b-Hippoformer |
|---:|---:|---:|---:|---:|
| 2 | 0.9792 +/- 0.0361 | 1.0000 +/- 0.0000 | 0.9167 +/- 0.0722 | -0.0625 |
| 4 | 0.9615 +/- 0.0192 | 1.0000 +/- 0.0000 | 0.8942 +/- 0.0659 | -0.0673 |
| 8 | 0.9701 +/- 0.0151 | 1.0000 +/- 0.0000 | 0.8995 +/- 0.0521 | -0.0707 |
| 16 | 0.9366 +/- 0.0161 | 0.9982 +/- 0.0048 | 0.8623 +/- 0.0377 | -0.0743 |
| 32 | 0.9611 +/- 0.0098 | 0.9990 +/- 0.0027 | 0.8084 +/- 0.0298 | -0.1527 |
| 64 | 0.9420 +/- 0.0092 | 0.9989 +/- 0.0019 | 0.7288 +/- 0.0363 | -0.2132 |
| 128 | 0.9503 +/- 0.0100 | 0.9902 +/- 0.0024 | 0.7268 +/- 0.0481 | -0.2235 |
| 256 | 0.9543 +/- 0.0135 | 0.9823 +/- 0.0017 | 0.6842 +/- 0.0445 | -0.2702 |
| 512 | 0.9047 +/- 0.0711 | 0.9878 +/- 0.0021 | 0.6830 +/- 0.0346 | -0.2217 |
| 1024 | 0.7470 +/- 0.1780 | 0.9918 +/- 0.0013 | 0.6708 +/- 0.0429 | -0.0761 |
| 2048 | 0.5122 +/- 0.2421 | 0.9917 +/- 0.0016 | 0.6613 +/- 0.0373 | +0.1491 |
| 4096 | 0.3049 +/- 0.1858 | 0.9925 +/- 0.0014 | 0.6736 +/- 0.0348 | +0.3688 |

## Seed Summary

| seed | Hippo AUC | M-delta AUC | M1b AUC | Hippo max | M1b max |
|---:|---:|---:|---:|---:|---:|
| 712 | 0.9421 | 0.9956 | 0.7329 | 0.5884 | 0.6587 |
| 713 | 0.8108 | 0.9938 | 0.7679 | 0.1600 | 0.6639 |
| 714 | 0.8284 | 0.9945 | 0.7591 | 0.1869 | 0.6172 |
| 715 | 0.8961 | 0.9944 | 0.8203 | 0.3009 | 0.7502 |
| 716 | 0.9394 | 0.9917 | 0.7517 | 0.6420 | 0.6841 |
| 717 | 0.8785 | 0.9954 | 0.7481 | 0.2464 | 0.6629 |
| 718 | 0.7908 | 0.9927 | 0.7783 | 0.1550 | 0.6816 |
| 719 | 0.8098 | 0.9954 | 0.7614 | 0.1592 | 0.6703 |

## Primary Endpoints

- M1b - Hippoformer log-horizon AUC: `-0.0970`
- M1b - Hippoformer max-horizon accuracy: `+0.3688`

## Gates

- hippoformer_short_health: `False`
- hippoformer_4096_health: `False`
- m1b_short_health: `False`
- m1b_auc_noninferiority: `False`
- m1b_4096_noninferiority: `True`
- future_writes_zero: `True`
- context_write_count_exact: `True`
- all_model_states_unchanged: `True`
- shared_trajectory_hashes: `True`

决策：`M1B_T1_STRICT_FREE_ROLLOUT_FAIL`

## 解释边界

T1 只检查 frozen M1b 在单房间 action-conditioned true free rollout 中是否保持健康，不证明多房间 context inference。T2 headline 与 observed 4356-step 结果仍是不同实验，不能混写。

## 文件

- protocol: `runs/remap_former/m1b_t1_strict_rollout_protocol.json`
- result: `runs/remap_former/m1b_t1_strict_rollout_seed2071601.json`
