# ReMAP-Former：G8 Randomized Hidden Cutoff

> 每个 episode 从 H=32/48/60/72 的平衡随机分布中抽一个 cutoff；cutoff 不作为模型 token。部署预测完成后才运行 full-shadow audit。

## 冻结设置

- protocol SHA256：`fe3dccbb8fb5b6718714f5ff71e8b23c5c4dfd22385b9571cdc88b280c210b5d`
- fresh family seed：`5371701`
- cutoffs：`[32, 48, 60, 72]`，每个 task/family batch 精确 `25%`
- threshold：`0.076884567737579346`，recurrence cosine：`0.99999`
- full-shadow trigger 仅用于事后判断 censoring，不参与部署 state、address、retrieval 或 prediction。

## G7 预测与 G8 实现

| 指标 | G7 uniform 预测 | G8 realized | 绝对差 |
|---|---:|---:|---:|
| Dynamic coverage | 0.4583 | 0.4635 | 0.0052 |
| T2 return | 0.3385 | 0.3885 | 0.0499 |
| Online passes | 1.3864 | 1.3923 | 0.0059 |
| T1 4096 | 0.9954 | 0.9978 | N/A |
| T2 clean | 0.9993 | 0.9982 | N/A |

## Cutoff Strata

| H | T1 null | T2 dynamic | T1 4096 | T2 return | clean | online/fixed passes |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.0000 | 0.0000 | 0.9980 | 0.0000 | 1.0000 | 1.0000/1.0000 |
| 48 | 1.0000 | 0.1667 | 0.9990 | 0.0608 | 1.0000 | 1.1510/1.1667 |
| 60 | 0.9792 | 0.6875 | 0.9957 | 0.6319 | 0.9966 | 1.6160/1.6875 |
| 72 | 1.0000 | 1.0000 | 0.9986 | 0.8611 | 0.9961 | 1.8021/2.0000 |

## Family 分层

| Family | T1 null | T1 4096 | coverage | T2 return | clean | online passes |
|---|---:|---:|---:|---:|---:|---:|
| grid7 | 1.0000 | 1.0000 | 0.4375 | 0.4023 | 0.9996 | 1.3965 |
| grid11 | 1.0000 | 0.9986 | 0.3438 | 0.3105 | 0.9984 | 1.3076 |
| grid15 | 0.9688 | 0.9799 | 0.4688 | 0.4180 | 0.9980 | 1.4215 |
| query3 | 1.0000 | 1.0000 | 0.5000 | 0.4570 | 0.9953 | 1.4184 |
| query2 | 1.0000 | 0.9999 | 0.7500 | 0.5234 | 1.0000 | 1.5589 |
| sparse_conflict | 1.0000 | 0.9990 | 0.2812 | 0.2539 | 0.9984 | 1.2507 |

正式样本为 6 个 fresh families；T1 每 family 32 episodes、共 192 条 4096-step rollout；T2 为 8 个冻结 checkpoints x 6 families x 32 episodes，共 1536 条。CUDA 正式运行用时 `1484.7 s`；此前 smoke 为 20/20 implementation gates。

## 因果与完整性

- shadow audits：`54/54`
- online/fixed minimum decision agreement：`1.000000`
- maximum state/logit error：`5.245e-06`
- cutoff assignment checkpoint pairing：`True`
- cutoff frequencies exact：`True`
- coverage/return Spearman：`1.0000/1.0000`

## 冻结结论

**`CONTEXT_NEED_ONLINE_G8_HIDDEN_CUTOFF_PASS`**

全部冻结 gates 通过：G7 曲线能预测 fresh hidden-cutoff mixture，且逐 episode deployment decision 严格等于 cutoff 前可见的 first crossing。Controller 证据线冻结，不再增加变体。

边界：G8 验证的是冻结的均匀四点 cutoff 分布，不等于任意 cutoff policy；G6 formal FAIL、G7 短窗不可辨识区和 dense downstream ceiling 均继续保留。

## 冻结产物

- protocol：`runs/remap_former/context_need_online_g8_hidden_cutoff_protocol.json`，SHA256 `fe3dccbb8fb5b6718714f5ff71e8b23c5c4dfd22385b9571cdc88b280c210b5d`
- evaluator：`evaluate_remap_context_need_online_g8_hidden_cutoff.py`，SHA256 `67815034cf125fbd09cb309d79f3323265c5c71f29ea214e6eadc42195949f20`
- result：`runs/remap_former/context_need_online_g8_hidden_cutoff_seed5471701_5522001.json`，SHA256 `9ea58a909cd3e9b3976f1a93b7adc541bfeee8fe897e7b30c73e184c77bdf41e`
- smoke：`tmp/context_need_online_g8_hidden_cutoff_smoke.json`
