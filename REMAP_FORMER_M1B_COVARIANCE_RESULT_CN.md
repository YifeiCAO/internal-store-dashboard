# ReMAP-Former M1b：协方差校正 HPC 结果

> 结论：保留 frozen covariance M1b；拒绝额外 warm-start CE 微调。test split 未使用。

## 1. 为什么改 HPC

Raw M1 的 place 在正确/错误 reference 上 cosine 都为 1.0，但直接重放正确完整 address 后，raw HPC return-conflict 仍只有 0.2812，retrieval 也更像另一房间 value。瓶颈是相似 context key 的顺序 delta 写入互相覆盖，不是 EC/place 漂移或 Transformer 读出失败。

## 2. M1b 唯一结构改动

内容仍存于每个 episode 现场建立的 `[value_dim, address_dim]` fast synaptic matrix。M1b 额外维护一个 episode-local `16×16` context covariance，用其正则化逆方向构造 dual write key；读取、Transformer、EC、place、context head 和 all-token sensory CE 全部不变。ridge 固定为 0.03。

它不存在离散 slot、按 place index 查表的 content store，也不跨 episode 保存内容。

## 3. Ridge dev sweep

| ridge | full return | return clean | exact raw HPC | exact target preference |
|---:|---:|---:|---:|---:|
| 1 | 0.7188 | 1.0000 | 0.7188 | 0.7292 |
| 0.3 | 0.8646 | 1.0000 | 0.8750 | 0.8750 |
| 0.1 | 0.9167 | 1.0000 | 0.9479 | 0.9583 |
| 0.03 | 0.9271 | 1.0000 | 1.0000 | 1.0000 |

候选只在 dev seed 891 上选择；0.03 最优后立即冻结，不继续向更小 ridge 追分。

## 4. 严格配对 validation

同一颗 slow weights、同一批 validation seed892 episodes，只切换 HPC backend：

| seed | Delta return | Covariance return | 净增益 | Cov clean | exact address | exact raw HPC |
|---:|---:|---:|---:|---:|---:|---:|
| 712 | 0.3164 | 0.7969 | +0.4805 | 0.9883 | 0.9609 | 0.9727 |
| 713 | 0.2656 | 0.8047 | +0.5391 | 0.9961 | 0.9375 | 0.9531 |
| 714 | 0.3203 | 0.8203 | +0.5000 | 1.0000 | 0.9570 | 0.9688 |

三 seed 均值：Delta `0.3008`，Covariance `0.8073`，净增益 `+0.5065`；增益范围 `+0.4805` 到 `+0.5391`。

决策：`M1B_COVARIANCE_SUBSTRATE_CONFIRMED`。

## 5. Warm-start 微调审计

ridge 与 LR 均冻结后，三颗各训练 100 steps，只开放 retention gate 与 context head，step 0 参与 dev-loss checkpoint 选择：

| run | loss before→after | return before→after | return delta | clean | HPC-zero drop |
|---|---:|---:|---:|---:|---:|
| m1b_covariance_warm_seed1012_s100 | 0.5576→0.4895 | 0.8281→0.8125 | -0.0156 | 1.0000 | 0.7969 |
| m1b_covariance_warm_seed1013_s100 | 0.5352→0.4839 | 0.8750→0.8516 | -0.0234 | 1.0000 | 0.8281 |
| m1b_covariance_warm_seed1014_s100 | 0.5483→0.4809 | 0.9297→0.8984 | -0.0312 | 0.9984 | 0.8750 |

loss 三颗都下降，但 return 平均变化 `-0.0234`，且三颗全为负。因此决策为 `WARMSTART_CE_REJECTED`：不使用 warm-start checkpoint，不用总 token CE 的下降覆盖 rare re-entry 回退。

## 6. 当前正式版本

最终选择：`frozen_converted_m1b`。三颗 checkpoint：

- `runs\remap_former\m1b_covariance_ridge0p03_seed712_frozen\m1b_frozen.pt`
- `runs\remap_former\m1b_covariance_ridge0p03_seed713_frozen\m1b_frozen.pt`
- `runs\remap_former\m1b_covariance_ridge0p03_seed714_frozen\m1b_frozen.pt`

该版本已经越过原 H3 的 0.50 绝对门，但目前仍是三 seed 机制结果，不冒充 8-seed headline。

## 7. 预算匹配 Hippoformer 终审

独立 seed711 只按固定 dev all-token CE 在 `1e-4/3e-4` 中选择 Hippoformer LR `1e-4`。随后 seeds `712/713/714` 均从同一健康 checkpoint 出发，全参数训练 `600 steps × batch4`；训练 episode 与对应 M-delta/M1 seed 使用相同 `seed+step` 生成规则。validation/test 均未参与训练或 checkpoint 选择。

在每颗 seed 的 best-dev checkpoint 冻结后，三模型共同读取 validation seed892 的同一批 256 episodes：

| seed | Hippoformer return | M-delta return | M1b return | M1b-Hippo gain | M1b clean |
|---:|---:|---:|---:|---:|---:|
| 712 | 0.0000 | 0.0000 | 0.7969 | +0.7969 | 0.9938 |
| 713 | 0.0000 | 0.0000 | 0.8047 | +0.8047 | 0.9977 |
| 714 | 0.0000 | 0.0000 | 0.8203 | +0.8203 | 0.9984 |

Hippoformer clean 三颗均为 `1.0`，因此 return 失败不是基础记忆塌陷。其 other-room target rate 为 `0.9805/0.9688/0.9805`，正确目标概率 margin 均为负；M1b margin 均为正。每颗 512 个 return-conflict probes 中，M1b-only 正确 `408/412/420`，Hippoformer-only 均为 `0`。

M1b 参数量为 `1,398,289`，Hippoformer 为 `1,494,447`。结果不是靠更大的 slow-weight 参数量获得。正式决策：`M1B_ADVANTAGE_CONFIRMED_VS_MATCHED_HIPPOFORMER_AND_MDELTA`。

完整协议和机器结果见 `reports/REMAP_FORMER_MATCHED_HIPPOFORMER_M1B_CN.md` 与 `runs/remap_former/matched_hippoformer_m1b_validation892.json`。下一步冻结全部 recipe，补五颗独立 seed，并只使用一次 test split 形成 8-seed headline。
