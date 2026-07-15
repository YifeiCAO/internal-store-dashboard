# M1j G5：共享 Transport 的 Context/Value/Address 几何分解

> 全新 dev seed19184；exact step-0 M1j；K=8、64 episodes、128 probes。逐 probe 梯度、同 episode 配对和 query-only correct-context oracle 均为只读诊断；无训练、无参数修改。

## 总体

| 指标 | 数值 |
|---|---:|
| Source return-conflict | 0.4375 |
| Query-only correct-context oracle | 0.8594 |
| Support / Reject / Neutral probes | 88 / 40 / 0 |
| Reject probe 的 oracle correct rate | 0.8000 |
| Reject probe 的 archival pair correct rate | 0.9500 |
| Shared scalar conflict pairs | 14 / 64 |

## Reject Probe 互斥分解

| 类别 | 数量 | Reject 占比 |
|---|---:|---:|
| Source 已答对但局部梯度要求关 transport | 8 | 0.2000 |
| 正确 query context 仍答错 | 8 | 0.2000 |
| Archival context value pair 错 | 2 | 0.0500 |
| 同 episode 共享事件的 probe 梯度冲突 | 6 | 0.1500 |
| 局部标量 transport 几何不匹配 | 16 | 0.4000 |

## 同 Episode 两 Probe

- support/support：`36`。
- support/reject：`16`。
- reject/reject：`12`。
- 含 neutral：`0`。

共享冲突要求同一 episode 一正一负、9 维梯度 cosine ≤-0.5，且 causal event-time Jaccard ≥0.5；只满足标签混合不算机制证据。

## 实现门

- Source prediction/context 最大差：`0.000e+00` / `0.000e+00`。
- 参数哈希不变：`True`。
- 每 probe 非零梯度 / causal event：`True` / `True`。
- 每 episode 两 probes / query oracle isolation：`True` / `True`。

## 冻结判读

- 状态：`MIXED_CONTEXT_VALUE_ADDRESS_GEOMETRY`。
- Dominant category：`None`。
- Leader / runner-up share：`0.4000` / `0.2000`。
- 下一独立协议：`NO_SINGLE_MECHANISM_YET_AND_DO_NOT_TRAIN`。

本结果不得用于在 seed19184 调阈值、category priority 或模型；正式模型仍为 frozen M1b，event-balanced M1j 训练继续禁止。
