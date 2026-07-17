# ReMAP-Former：P0-P4 最终证据汇总

> 正式模型固定为 M1b covariance；本文档由冻结 JSON 自动生成，不手抄实验数字。

## 一句话模型

32-step Transformer/PFC 从 action 与严格滞后的 sensory history 形成 hidden context；neural EC 产生 sparse place；place×context 构成 4096 维地址，调用一个 episode-local covariance-corrected fast-weight HPC。没有 room/context/position/place ID、slot bank、第二套 fast weights 或 learned write gate。

## 主张对账

| 主张 | 状态 | 证据 |
|---|---|---|
| hidden-context re-entry | 支持 | P3 M1b return-conflict 0.8376，所有健康基线 <=0.0134 |
| context / covariance / HPC read 必要 | 支持 | P2 五个干预的 paired CI 均排除 0 |
| strict free-rollout 全曲线非劣效 | 失败 | M1b-Hippo AUC -0.0970，95% CI [-0.1544,-0.0440] |
| 4096-step rollout 端点优势 | 支持 | M1b-Hippo +0.3688，95% CI [+0.2289,+0.4836] |
| 新奇事件可扩展到任意 K | 不支持 | P4 normal K50=4；K=12 降至 0.0990 |
| K=2 扩展门 | 通过 | 0.7109，95% CI [0.6693,0.7513] |

## Table 1：P3 健康基线

| 模型 | 参数 | steps | clean | conflict | return-conflict (95% CI) |
|---|---:|---:|---:|---:|---:|
| Hippoformer | 1,494,447 | 600 | 1.0000 | 0.7937 | 0.0000 [0.0000, 0.0000] |
| Hippoformer HPC branch | 1,494,447 | 600 | 0.9868 | 0.6854 | 0.0000 [0.0000, 0.0000] |
| M-delta | 1,388,257 | 600 | 1.0000 | 0.7999 | 0.0000 [0.0000, 0.0000] |
| ReMAP-Former M1b | 1,398,289 | 600 | 0.9944 | 0.9372 | 0.8376 [0.8221, 0.8527] |
| Window Transformer | 1,280,960 | 1200 | 0.8953 | 0.8011 | 0.0134 [0.0098, 0.0176] |
| Parameter-matched Transformer | 1,396,096 | 1200 | 0.8954 | 0.8012 | 0.0122 [0.0089, 0.0159] |
| Titans-MAC adaptation | 1,344,260 | 1200 | 0.9234 | 0.8001 | 0.0095 [0.0063, 0.0129] |

外部三基线使用 1200-step best-effort；旧三模型使用 600 steps，不能混称完全预算匹配。Hippoformer HPC branch 是机制消融，不是官方 mmTEM；Titans-MAC 是任务适配版，不是作者 checkpoint。

## Table 2：M1b 因果干预

| 条件 | return-conflict (95% CI) |
|---|---:|
| Full M1b | 0.7944 [0.7725, 0.8154] |
| No covariance | 0.2676 [0.2437, 0.2949] |
| Fixed context | 0.0000 [0.0000, 0.0000] |
| Shuffled context | 0.4355 [0.4148, 0.4565] |
| HPC read = 0 | 0.0234 [0.0188, 0.0281] |
| Wrong return context | 0.0010 [0.0000, 0.0027] |
| Correct context oracle | 0.9424 [0.9246, 0.9575] |

## Table 3：新奇事件容量

| 条件 | K=1 | K=2 | K=4 | K=8 | K=12 | AUC | K50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| m1b_covariance | 0.7760 | 0.7109 | 0.3750 | 0.1927 | 0.0990 | 0.4618 | 4 |
| m1b_correct_context_oracle | 0.9818 | 0.9479 | 0.6719 | 0.3880 | 0.2057 | 0.6913 | 8 |
| m1b_exact_address_ridge0p03 | 0.9818 | 0.9479 | 0.6719 | 0.3854 | 0.2031 | 0.6905 | 8 |
| m1b_exact_address_ridge0p001 | 0.9870 | 0.9714 | 0.8359 | 0.7370 | 0.4766 | 0.8436 | 12 |

## Free-rollout 边界

- log-horizon AUC delta：`-0.0970`，95% CI `[-0.1544, -0.0440]`，主 gate 失败。
- 4096 endpoint delta：`+0.3688`，95% CI `[+0.2289, +0.4836]`。
- 因此论文贡献必须限定为 observed-history hidden-context re-entry，不写成通用 autonomous rollout 已解决。

## P3 recovery 披露

- Test-2 计算访问 pass：`2`。
- 第一 pass 完成八颗 seed 的全部前向后，pooled exact-sign p 的大整数转 float 溢出；正式 JSON 尚未写出。
- 只替换数值稳定公式后，第二 pass 使用同 checkpoint、同 generator、同顺序重放；八个 M1b seed 值逐项复现。
- 模型、checkpoint、阈值和选择规则均未改变；失败记录与访问账本保留。这个偏差必须在 appendix/reproducibility statement 中披露。

## 当前项目水平

现在已经不是玩具 demo：有冻结模型、无 oracle generator gates、8-seed 主结果、逐 episode paired statistics、因果干预、健康外部基线和容量边界。作为 ICLR 投稿候选，核心机制叙事已成立。

但还不能写成稳收：任务仍是合成导航；strict free rollout 主 gate 失败；Titans/mmTEM 不是官方复现；P4 只有三组独立 cluster；P3 有透明但真实的两-pass recovery 偏差。最合理口径是‘有明确新机制与强内部证据的 ICLR 候选’，不是已经证明通用记忆架构。

## 完整性

- P3 integrity：`True`。
- P4 integrity：`True`；关键布尔门全绿。
- 根目录回归：`314 passed`。
- P0 当前 hash 全匹配：`False`。
- 唯一 P0 mismatch：`remap_former/pfc.py`；代码字符相同，113 个 LF 曾变为 CRLF；待明确授权恢复 LF。
- 在该字节哈希恢复前，release manifest 保持 NOT_READY，不通过重写 manifest 隐藏差异。

## 关键文件

- P3 结果：`runs/remap_former/p3_test2_seed2471601.json`
- P3 报告：`reports/REMAP_FORMER_P3_TEST2_CN.md`
- P3 图：`reports/figures/remap_former_p3_test2.png`
- P4 结果：`runs/remap_former/p4_novel_capacity_seed2771601_2771603.json`
- P4 报告：`reports/REMAP_FORMER_P4_NOVEL_CAPACITY_CN.md`
- P4 图：`reports/figures/remap_former_p4_novel_capacity.png`
- 论文表：`runs/remap_former/p5_paper_tables.json`
- release manifest：`runs/remap_former/p5_release_manifest.json`
