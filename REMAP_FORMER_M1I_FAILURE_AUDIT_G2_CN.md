# M1i G2：Call-to-Recall 只读错误分解

> 全新 dev seed10932；冻结 step-600 M1i；K=8、64 episodes、128 probes；无训练、无参数修改、formal/test 未访问。room/segment metadata 只在 forward 后做标签。

## 性能与因果上界

| 条件 | Return-conflict |
|---|---:|
| 冻结 M1f | 0.3125 |
| 冻结 M1i | 0.4688 |
| 仅当前 query 换正确历史 context | 0.8594 |

Oracle 相对 M1i：`+0.3906`；正常错误共 `68` 个。每个 oracle rollout 只改当前被测 query，不修改同 episode 的其他 return query。

## 互斥错误桶

| 类别 | 数量 | 错误占比 |
|---|---:|---:|
| 正确 query context oracle 仍答错 | 14 | 0.2059 |
| Pair 正确但 context 精度不足 | 50 | 0.7353 |
| 没有因果 proposal 历史 | 0 | 0.0000 |
| 没有正确来源 exact key | 0 | 0.0000 |
| Attention 来源歧义/错误 | 0 | 0.0000 |
| Archival value 几何错误 | 4 | 0.0588 |
| 正确 call 后又被覆盖 | 0 | 0.0000 |
| 正确历史没有装入当前 context | 0 | 0.0000 |

## Key、Value 与主 Transformer Hidden

- Correct exact history available：`0.7500`。
- Signature top-1 correct segment：`0.7500`。
- Raw pfc_hidden top-1 correct segment：`0.0625`。
- Raw pfc_hidden - signature top-1：`-0.6875`。
- Raw pfc_hidden correct-minus-competing max cosine：`-0.0929`。
- Attention mass correct / wrong-ref / current-return / distractors：`0.494 / 0.032 / 0.249 / 0.225`。
- Query / archival context pair：`0.9375 / 0.9062`。
- Last proposal call weight / argmax：`0.442 / 0.625`。

Raw `pfc_hidden` cosine 只是零训练几何对照，不是已实现或已训练的 Transformer caller。

## 冻结判读

- Dominant family：`context_precision`。
- PFC hidden geometry：`PFC_HIDDEN_GEOMETRY_NOT_ESTABLISHED`。
- 下一独立协议分支：`PRE_REGISTER_CONFIDENCE_CONDITIONED_TRANSPORT`。

本结果不得用于在 seed10932 上调 threshold、temperature、null 或模型参数；任何新模型必须另立协议并使用新 generator seed。
