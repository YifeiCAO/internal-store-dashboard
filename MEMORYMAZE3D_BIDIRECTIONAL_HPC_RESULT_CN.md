# MemoryMaze3D 双向 Neural HPC 实验报告

日期：2026-07-23

## 1. 本轮问题

上一版单向 HPC 只能执行：

`grid/place/context -> sensory`

它可以根据结构地址读取视觉内容，却不能让当前观察反向召回历史结构状态。因此即使视觉 loop closure 改善了 place 地址，模型仍无法像 mm-TEM 那样用 sensory 校正 grid。

本轮实现并检验：

`sensory <-> grid` 双向 episode-local neural fast weights。

## 2. 当前模型

主干仍是 window Transformer/PFC。模型输入只有：

- 当前与过去的第一人称 RGB
- 当前与过去的六类离散动作

明确不存在 room id、绝对位置、朝向、地图、place id 或未来 RGB 输入。

每个时间步严格执行：

1. PFC 根据动作和历史 sensory 产生短期预测。
2. EC 根据动作历史产生 `g_gen`。
3. `grid -> sensory` fast weight 在观察当前帧前读出视觉记忆。
4. 当前 sensory 到达后，`sensory -> grid` fast weight 召回 `g_hat`。
5. 闭环置信度检验 `g_hat -> sensory` 是否能解释当前观察。
6. neural gate 融合 `g_gen` 与 `g_hat`，得到 `g_inf`。
7. `g_inf` 的 phase residual 从下一步开始持续写回 EC 状态。
8. 同一 observed pair 同时写入两个方向的 fast weights。

主预测在步骤 3 已经完成。步骤 4-8 只能影响未来，不能偷看当前预测目标。

内容状态每个 episode 从零初始化，不是参数表，也没有 memory slots。

## 3. 新增实现

- `remap_former/memory.py`
  - `BidirectionalCovarianceHPC`
  - episode-local `grid -> sensory` 与 `sensory -> grid` fast weights
- `remap_former/memorymaze3d.py`
  - `NeuralGridCorrection`
  - 闭环置信度
  - causal phase reset
- `remap_former/bidirectional_objective.py`
  - 只从至少 4 步以前寻找视觉匹配的 causal structure-recall objective
- `diagnose_memorymaze3d_bidirectional_recall.py`
  - true-vs-shuffled sensory-to-grid 机制门
- `render_memorymaze3d_bidirectional_visuals.py`
  - 固定连续验证样本视觉对照

## 4. 因果与工程门

完整测试：`39 passed`

新增门覆盖：

- 两束 fast weights 均能 read/write
- 当前 sensory 不影响当前及以前的主预测
- phase correction 只从下一步持续生效
- strict free rollout 对未来真值完全不变
- reverse objective 的 target 只来自严格过去
- 梯度能到达 sensory key 与 neural correction gate
- PFC 在指定训练 scope 中保持冻结

所有正式诊断：

- `future_ground_truth_reads = 0`
- `future_ground_truth_writes = 0`
- test split 未读取

## 5. 机制结果

100-step reverse-recall 预训练：

- reverse loss：`0.585 -> 0.458`
- visual-history structural cosine：`0.409 -> 0.510`
- 峰值显存：`1.92 GiB`

独立 `Val32-63`，1952 个严格因果 paired queries：

| 指标 | True query | Shuffled query | Margin |
|---|---:|---:|---:|
| 闭环置信度 | 0.5769 | 0.4660 | **+0.1108** |
| 历史结构余弦 | 0.4481 | 0.2884 | **+0.1597** |

两项冻结机制门均通过。

结论：sensory-to-grid 反向召回确实学到了内容相关的历史结构，而不是只输出一个与查询无关的大向量。

## 6. 下游因果消融

固定 checkpoint 内仅把 HPC retrieval contribution 置零，得到 PFC-only。其余权重、输入和 rollout 完全相同。

| Checkpoint / split | C20 | C32 | Pooled | 裁决 |
|---|---:|---:|---:|---|
| Recall100，Val64-95 | -3.40% | +4.32% | -0.05% | FAIL |
| Rollout100，Val96-127 | -1.73% | +2.23% | +0.16% | FAIL |
| Phase reset zero-shot，Val144-151，N8 | +4.33% | -0.97% | +2.13% | 探索样本 |
| Phase reset rollout80，Val152-167，N16 | +7.25% | +7.38% | +7.30% | 小样本候选 |
| Phase reset rollout80，Val168-199，N32 | -1.17% | +3.12% | +0.76% | **未复现** |

最终独立 Val32 的 pooled 90% paired-bootstrap 区间跨 0，C20 为负。因此：

- 不晋级 Val128/256
- 不读取 test
- 不把 Val16 的 +7.30% 当作 headline

## 7. 视觉结果

固定连续验证 episode 168-170，没有按效果挑图：

| Episode | C20->H44 HPC effect |
|---|---:|
| 168 | +1.63% |
| 169 | -1.84% |
| 170 | -9.79% |

图中同时展示 H+1/4/8/16/32/44 的：

- Ground truth
- Bidirectional HPC
- PFC-only
- 两套绝对误差放大图

视觉上，两种模型都能保持主色调与粗略墙面结构，但长期预测明显模糊，目标和转角细节容易丢失。HPC 的改善与伤害都是真实存在的，和大样本下的“不稳定”结论一致。

## 8. 当前结论

本轮第一次把关键机制真正补齐：

1. sensory 可以反向召回历史 grid。
2. 召回通过 true-vs-shuffled 独立机制门。
3. HC correction 可以持续重置后续 EC phase。
4. 在部分 split 上出现 2%-7% 的下游收益。

但论文级结论仍未成立：

> 双向 HPC 的机制已经有效，但何时把它调用进 PFC 仍不可靠。

下一轮不再改 memory 容量或增加第二套 fast weights。只研究一个问题：

> 能否用严格因果、在生成分布上校准的调用置信度，决定何时让 HPC retrieval 进入 PFC，并在低置信度时自动回退到 PFC-only？

晋级门保持不变：独立大样本上 pooled 与 C20/C32 均为正，90% paired-bootstrap 下界均高于 0，pooled 改善至少 1%。
