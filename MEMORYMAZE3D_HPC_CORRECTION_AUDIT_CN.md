# MemoryMaze3D HPC Correction Audit

更新日期：2026-07-23

## 结论

当前 variable-layout 3D world model 本身成立，但当前单向 HPC 仍未在大样本
free rollout 中证明有效。

- Stage2 world model 相对 Stage1 的 paired Val64 改善为 **69.26%**。
- 原始 HPC 在独立 Val256 上相对 PFC-only 为 **-1.43%**。
- 本轮最佳小样本结果来自 visual-corrected EC：Val32 为 **+4.73%**。
- 同一候选在完全独立的 Val256 上变为 **-0.42%**，90% bootstrap CI 为
  **[-0.190, +0.052] x1e-3**。
- 因此不进入 test，不声称 HPC contribution 成立。

这轮排除了三个看似合理、但不足以解决问题的方向：

1. episode-context InfoNCE；
2. place/context factorized covariance write；
3. 只靠视觉 loop closure 间接校正 action-only EC。

真正缺失的是原始 mm-TEM/Hippoformer 中的双向关系记忆：
当前实现只能用结构地址读 sensory，不能用 sensory 从 HPC 反向召回并修正
structural/grid code。

## 不变的合法性边界

- 模型输入只有 `egocentric RGB history + action history`。
- 没有 room id、layout id、绝对位置、方向、place id 或未来 RGB 输入。
- `agent_pos` 只用于 held-out 诊断，不进入训练 loss 或模型 forward。
- 所有本轮选择都只使用 train/val；test 未被触碰。
- HPC content state 每个 episode 从零开始，是 differentiable fast weight，不是
  记忆槽，也不是跨训练持久的 lookup table。

## A. Context InfoNCE

### 设计

同一 episode 的两个不重叠已观测历史片段为正对，batch 内其他 episode 为负对：

```text
L_ctx = 0.5 * CE(e_early e_late^T / tau, diagonal)
      + 0.5 * CE(e_late e_early^T / tau, diagonal)
```

### 结果

| 指标 | 训练前 | InfoNCE 200 steps |
|---|---:|---:|
| same-minus-cross cosine margin | 0.571 | 0.291 |
| effective rank | 5.18 | 4.34 |
| global episode top-1 | 27.3% | 22.7% |
| positive cosine | 0.716 | 0.924 |
| cross-episode cosine | 0.145 | 0.633 |

InfoNCE 把正对拉近的同时把所有 context 推进同一个窄锥体，几何反而退化。
Val64 的 HPC pooled 表面增益只有 **+1.06%**，置信区间跨零，C32 为负。

判断：**失败，停止。**

## B. Fixed Context 与 Factorized Covariance

### Fixed episode context

variable-layout 数据每个 episode 只有一个 maze，且 HPC 每个 episode 清零，因此
跨 episode context 分离本身不是 memory addressing 的必要条件。我们保持 PFC、
place、value 和 memory update 不变，只把地址中的 context 固定。

结果：Val64 pooled HPC effect 为 **-20.58%**。

解释：动态 context 虽然破坏空间几何，也在充当时间去重键；直接固定后，重叠
place write 的串扰急剧增加。

### Factorized covariance write

新写入保持同一块 dense fast-weight matrix，不添加 slot。写入前分别对 place 与
context 因子做 episode-local dual-key correction。

合成重叠 key 单测中，回访 recall MSE 从 **0.375** 降到 **0.0012**。但真实
MemoryMaze3D 经过 100-step 适配后：

- C20: **-2.52%**
- C32: **+1.79%**
- pooled: **-0.79%**

判断：**合成机制成立，真实 free rollout 不成立，停止。**

## C. Place/Address 根因诊断

在 Val128 上用 `agent_pos` 做纯评估诊断：

| 指标 | 原始 stage2 |
|---|---:|
| integrated distance vs physical distance Spearman | 0.353 |
| place cosine vs negative physical distance Spearman | 0.168 |
| conjunctive address Spearman | 0.015 |
| place top-1 near recall，条件为近邻可用 | 72.2% |
| address top-1 near recall，条件为近邻可用 | 61.0% |

MemoryMaze3D 的离散 action 最终产生带噪声、加速度和碰撞的连续运动。当前 EC
只积分 action command，因此 action path 与真实运动几何不一致。动态 context
进一步削弱了 conjunctive address 的空间结构。

同时，冻结 encoder 的视觉 latent 是合法且有效的 loop-closure teacher：

- visual cosine vs negative physical distance Spearman: **0.619**
- visual top-1 near recall: **79.8%**

## D. Causal Visual-Corrected EC

### 架构

在 action-only SE(2) integrator 上加入一个 8.7K 参数的 causal correction head：

```text
PFC hidden history
    -> LayerNorm -> MLP -> tanh
    -> [forward residual, lateral residual, turn residual]
    -> recurrent SE(2) integration
    -> grid phases -> sparse neural place
```

- correction head 只读 causal Transformer hidden；
- 不读当前/未来真实 RGB；
- 最后一层零初始化，所以初始输出与旧 EC 逐位一致；
- PFC 冻结时，梯度只能进入 EC/place/HPC 允许范围。

### Self-supervised loop closure

对每个时刻，只在至少相隔 4 步的过去帧中，用冻结 visual latent 选择最相似历史
帧，并要求 place similarity 将它排在第一：

```text
target_t = argmax_j cosine(stopgrad(x_t), stopgrad(x_j))
L_vp = CE(cosine(p_t, p_j) / tau, target_t)
```

没有位置监督。

### 分阶段训练

1. EC+place only，300 updates，`L_vp` 预训练；
2. EC+place+HPC，100 updates，teacher + rollout 回调；
3. 保留较小 `L_vp` 防止 loop closure 遗忘。

回调后 prediction MSE 恢复到 **2.686e-3**。局部几何改善：

| 指标 | 原始 | 预训练后 | 回调后 |
|---|---:|---:|---:|
| place near top-1 | 72.2% | 78.2% | 77.1% |
| address near top-1 | 61.0% | 67.3% | 66.3% |
| selected prior distance median | 0.092 | 0.067 | 0.071 |

但全局 place Spearman 没有提升，说明它学到的是局部 loop closure，而不是完整
metric map。

### 功能性检验

发现集 Val32：

| Context | HPC effect |
|---|---:|
| C20/H44 | +4.51% |
| C32/H32 | +5.07% |
| pooled | +4.73% |

独立确认集 Val256：

| Context | Full MSE x1e3 | PFC-only MSE x1e3 | HPC effect |
|---|---:|---:|---:|
| C20/H44 | 17.809 | 17.673 | -0.77% |
| C32/H32 | 13.062 | 13.068 | +0.05% |
| pooled | 15.436 | 15.371 | -0.42% |

判断：**小样本效果未复制，停止。**

## 为什么这些修补仍不够

当前 HPC 是单向映射：

```text
(place x context) -> sensory value
```

它可以尝试从结构地址召回视觉内容，但 sensory observation 不能反向查询并修正
grid/place state。visual-loop loss 只能离线地把部分 place pair 拉近，无法在在线
episode 中执行真正的 HC-to-EC correction。

原始 mm-TEM/Hippoformer 的关键闭环是：

```text
action -> generated grid
generated grid -> memory -> generated sensory
generated grid + observed sensory -> memory -> corrected grid
corrected grid -> memory -> inferred sensory
write(corrected grid, sensory)
```

我们的当前 HPC 缺少中间的 `sensory -> corrected grid` 通路。这比 cutoff、gain、
context loss 或 covariance 细节更根本。

## 下一版冻结设计

下一步不再调当前单向 HPC，而是实现最小双向 neural associative HPC：

1. `M_xg`: structural/grid key 到 sensory value 的 episode-local fast weight；
2. `M_gx`: sensory key 到 structural/grid value 的 episode-local fast weight；
3. action path 先生成 `g_gen` 和 `x_gen`；
4. 观察到当前 sensory 后，由 `M_gx` 召回 `g_hat`；
5. causal neural gate 融合 `g_gen` 与 `g_hat` 得到 `g_inf`；
6. 用同一 observed pair 更新两个方向的 fast weights；
7. free rollout 中只使用模型自己生成的 sensory，未来真值读写仍为零。

损失保持四项、但分阶段启用：

```text
L = L_gen + lambda_con L_con + lambda_inf L_inf + lambda_roll L_roll
```

- `L_gen`: action-generated grid 的视觉预测；
- `L_con`: `g_gen` 与 memory-corrected `g_inf` 的一致性；
- `L_inf`: corrected grid 的视觉重建；
- `L_roll`: 严格 free rollout。

先做三个 gate：

1. 零未来读写与因果不变性；
2. sensory-to-grid recall 在 held-out episode 内优于 shuffle；
3. Val256 full 相对 PFC-only 至少 +1%，且 C20/C32 的 90% CI 下界都大于零。

只有三项全过，才进入 test 和 Hippoformer/Titans/Transformer 正式基线。
