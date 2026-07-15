# M1j G4：Probe 级梯度信用归因

> 全新 dev seed18182；exact step-0 M1j；K=8、64 episodes、128 return-conflict probes。每个 probe 单独读取 9 维 CE 梯度；只分析同 episode、最终 return 段、query 之前的真实 transport 事件。无 M1j 训练或参数修改。

## Probe 级结果

| 子组 | 数量 | Support | Reject | Neutral | 方向均值 |
|---|---:|---:|---:|---:|---:|
| 全部 probes | 128 | 0.8750 | 0.1250 | 0.0000 | 2.6412 |
| Source 已答对 | 60 | 1.0000 | 0.0000 | 0.0000 | 3.5476 |
| Source 答错 | 68 | 0.7647 | 0.2353 | 0.0000 | 1.8414 |
| 答成另一房间目标 | 2 | 1.0000 | 0.0000 | 0.0000 | 2.8979 |
| 其他错误 | 66 | 0.7576 | 0.2424 | 0.0000 | 1.8094 |
| 当前 context pair 正确 | 94 | 0.8936 | 0.1064 | 0.0000 | 2.7865 |
| 当前 context pair 错误 | 34 | 0.8235 | 0.1765 | 0.0000 | 2.2395 |

Support 表示该 probe 的单位 CE 下降方向会提高其 causal return events 的 transport logit；Reject 表示会压低。

## 六个可用特征是否够用

这里只给 CPU 上的诊断 ridge probe 输入 transport 当时已有的 6 个 causal confidence features。target、预测 logits、room、位置、target index、PFC hidden 与 metadata 均不输入；按 generator batch 留一，标准化只用训练 fold。

| 诊断目标 | Balanced accuracy | Accuracy | 正类 / 负类 |
|---|---:|---:|---:|
| Transport support vs reject | 0.5000 | 0.8750 | 112 / 16 |
| Source incorrect vs correct | 0.5961 | 0.5938 | 68 / 60 |

## G3 效应在新 seed 的复现

- 聚合 all-token vs return gradient cosine：`-0.9894`。
- 逐 batch 负 cosine：`0.8750`；均值 `-0.7517`。
- Reference-room support-rate range：`0.1250`。
- Target-index support-rate range：`0.2143`。

## 实现门

- Final 参数数：`9`。
- Source prediction/context 最大差：`0.000e+00` / `0.000e+00`。
- 参数哈希前后一致：`True`。
- 每个 probe 梯度非零 / 都有 causal event：`True` / `True`。

## 冻结判读

- 状态：`PROBE_CREDIT_MIXED`。
- 下一独立协议：`STOP_EVENT_BALANCED_TRAINING_AND_RETURN_TO_CONTEXT_VALUE_OR_ADDRESS_GEOMETRY`。

| Gate | 结果 |
|---|---:|
| `implementation_valid` | True |
| `correct_and_incorrect_subgroups_large_enough` | True |
| `neutral_rate_small` | True |
| `incorrect_probes_support_transport` | True |
| `correct_probes_reject_transport` | False |
| `outcome_support_rate_gap` | False |
| `reference_room_and_target_index_balanced` | True |
| `six_features_predict_transport_support` | False |
| `six_features_predict_source_incorrect` | False |
| `outcome_conditional_credit` | False |

## 含义

Probe 级信用仍不能由 source 成败稳定解释；停止 event-balanced 训练路线，回到 context value/address 几何。

本结果不得用于在 seed18182 上调阈值、ridge、feature 或模型；正式模型仍为 frozen M1b。
