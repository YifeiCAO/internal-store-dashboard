# MemoryMaze3D 可变布局 HPC 因果审计

日期：2026-07-23

## 一句话结论

`maze9-variable` 数据、严格 free rollout 和 Transformer/PFC 主干都已成立，但当前外挂 HPC **尚未证明有效**。旧 action-only context 在独立 Val256 上使 full 比 PFC-only 差 1.43%；改成读取因果 `pfc_hidden` 后，小样本一度显示 +6.26%，但固定 Val256 复现反转为 -1.80%。这些候选均未进入 test。

下一版不再调 cutoff 或 memory gain。唯一合理方向是给 `pfc_hidden -> context` 增加无标签的 episode 内稳定、episode 间分离目标，再重新做 full vs PFC-only 因果门。

## 1. 不变的任务边界

- 数据：`data/memorymaze3d_variable_full_v1`
- 环境：真实 MuJoCo 第一人称 RGB，`maze9-variable`
- 数量：train 4096 / val 512 / test 512
- 布局：train 4002 个唯一布局；val/test 各 512 个唯一布局
- 跨 split layout hash overlap：全部为 0
- 模型输入：RGB 与动作
- 禁止输入：绝对位置、朝向、room id、place id、maze layout
- 评测：C20->H44 与 C32->H32 strict free rollout
- 所有本轮诊断：future ground-truth reads = 0，writes = 0

数据审计见：

`runs/remap_former/memorymaze3d_variable_full_m1b_seed1701/data_audit.json`

## 2. 可变环境 world model 本身成立

旧训练分成 teacher stage1 与 rollout stage2。Val64 配对检查显示 stage2 是正确起点：

| 条件 | Stage1 MSE x1e-3 | Stage2 MSE x1e-3 | Stage2 改善 |
|---|---:|---:|---:|
| C20->H44 | 56.833 | 16.114 | 71.65% |
| C32->H32 | 39.654 | 13.545 | 65.84% |
| pooled | 48.244 | 14.829 | 69.26% |

pooled 90% paired-bootstrap gain 为 `[28.604, 38.210]e-3`。因此旧 training summary 的 `descent_gate=false` 不能解读为 rollout 微调失败；curriculum 提高了训练 loss 的难度，但显著改善了真实 free rollout。

完整旧 test512 已经存在：

- C32->H32 test MSE：`13.856e-3`
- persistence：`31.992e-3`
- 相对 persistence 改善：56.69%
- leakage gate：PASS

这证明视觉 world model 能跑，但不证明 HPC 是成绩来源。

## 3. Open9 写入 gate 分支的最终裁决

### 3.1 局部反事实写入教师

用同一生成前缀，仅分叉当前 HPC 写入 `g_t=0/1`，比较未来 4 步像素误差。三轮 train-only G0：

- 基础 12 特征：held AUC 0.527
- 加历史熟悉度：held AUC 0.584
- 加精确 HPC update 特征：held AUC 0.466
- timer control：held AUC 0.633

结论：局部写入收益主要是时间效应，不能训练成可信神经 gate。

### 3.2 episode 级 cutoff 上界

Train64 中，前 48 条选出的全局 cutoff 是 K20；后 16 条：

- 全局 K20：`21.732e-3`
- episode oracle：`18.740e-3`
- 事后上界：13.77%

但合法 router 失败：

- context/internal history router：比 K20 更差
- history + 完整 action plan：`22.442e-3`
- K20：`21.732e-3`
- shuffle one-sided p：0.644

因此 13.77% 是看完未来 target 后才能取得的不可部署上界。Open9 cutoff/gate 分支到此封存。

## 4. 可变布局组件审计

发现集 Val64、C20：

| 条件 | MSE x1e-3 | 相对 full |
|---|---:|---:|
| full PFC+HPC | 16.114 | 基准 |
| PFC-only | 16.471 | +2.22% penalty |
| HPC-only | 54.223 | +236.50% penalty |
| observed-only writes | 16.253 | +0.86% penalty |
| frozen rollout context | 19.541 | +21.27% penalty |
| observed-only + frozen context | 15.402 | -4.41% |

`observed-only + frozen context` 的漂亮结果没有复现。独立 `val[64:128]`：

- pooled full：`12.660e-3`
- pooled stable retrieval：`12.886e-3`
- 相对变化：-1.79%
- 90% paired-bootstrap gain：`[-0.645, 0.155]e-3`

结论：稳定检索策略封存，不进入 test。

## 5. 旧 HPC 的高功效因果确认

旧模型在全新 `val[128:384]`、N256 上：

| 条件 | C20 MSE | C32 MSE | Pooled MSE |
|---|---:|---:|---:|
| full | 17.919 | 13.263 | 15.591 |
| PFC-only | 17.673 | 13.068 | 15.371 |

pooled full 相对 PFC-only **差 1.43%**，HPC gain 为 `-0.220e-3`，90% paired-bootstrap 为 `[-0.329, -0.111]e-3`。

连续 memory-call 标定使用剩余 `val[384:512]`：

- 扫描有效 gain：0 / 0.5 / 1 / 2 / 3 / 4
- 数值最低点：gain1
- gain1 相对 gain0：仅 +0.17%
- 90% paired-bootstrap：`[-0.010, 0.053]e-3`
- 预注册决定：选择 gain0，旧 HPC 需要重训

## 6. 根因：context 没有读取环境

旧代码的 context 路径是：

```python
pfc_prediction, pfc_hidden = self.pfc(actions, sensory, return_hidden=True)
action_tokens = self.pfc.action_encoder(actions)
state = self.state_token(action_tokens, actions=actions)
context = self.context_head(state)
```

所以相同动作历史在不同布局中产生相同 context。固定 Open9 尚可掩盖这个问题；可变布局下，HPC 无法根据视觉历史 remap。

本轮已实现兼容修复：

```python
if self.config.context_input == "pfc_hidden":
    context_tokens = pfc_hidden
else:
    context_tokens = self.pfc.action_encoder(actions)
state = self.state_token(context_tokens, actions=actions)
context = self.context_head(state)
```

- 旧 checkpoint 默认 `context_input=action_tokens`，行为不变
- 新实验显式使用 `context_input=pfc_hidden`
- 新增 `m1b_context_hpc` scope，只训练 state/context/HPC/memory projection
- 因果测试证明：观测只会在进入历史后改变 context，未来观测不能影响过去 context

## 7. pfc_hidden context 试验

训练：

- 从旧 stage2 warmstart
- teacher adaptation：750 updates
- rollout alignment：100 updates，curriculum H1/2/4/8
- trainable：543,137 参数
- effective memory-call gain：1

Val64 初筛：

- full：`14.159e-3`
- PFC-only：`15.104e-3`
- pooled 表观提升：6.26%
- pooled 90% interval：`[0.124, 1.949]e-3`

固定 N256 复现：

- full：`15.648e-3`
- PFC-only：`15.371e-3`
- 相对变化：-1.80%
- 90% interval：`[-0.611, 0.064]e-3`

结论：`pfc_hidden` 是必要的接口修复，但仅靠预测 loss 不足以学出稳定、可泛化的 context。该 checkpoint 不进入 test。

## 8. 当前可信结论

1. 可变布局数据与 strict free-rollout 管线可信。
2. rollout stage2 对 world-model 能力有大幅、可复现的帮助。
3. 当前可变环境成绩主要来自 Transformer/PFC。
4. 旧 HPC 在高功效审计中显著伤害结果。
5. 降低固定 memory gain 不能救回 HPC。
6. `pfc_hidden` 修正了 context 的输入边界，但没有自动产生稳定 context。
7. 所有漂亮的小样本 memory 结果都经过独立复现；未复现者已明确封存。
8. 新候选没有读取 test，因此 test 仍可用于下一版最终确认。

## 9. 下一版唯一建议：无标签 context 对比目标

### 架构

保持：

- Transformer/PFC 主干
- action-only EC/grid
- neural sparse place
- conjunctive `place x context` address
- episode-local covariance-corrected fast-weight HPC

修改：

- context 输入固定为因果 `pfc_hidden`
- 在同一 episode 的两个已观测历史段形成正对
- batch 内其他 episode 形成负对

### 目标

对每条 episode，从观测前缀中取两个不重叠历史段，得到单位 context `e_a` 与 `e_b`：

```text
L_ctx = 0.5 * CE(e_a e_b^T / tau, diagonal)
      + 0.5 * CE(e_b e_a^T / tau, diagonal)
```

约束：

- 不使用 room id
- 不使用位置、朝向、地图或 place id
- 不使用 future RGB
- 正对只来自同一 episode 的因果历史

### 必过 gates

1. 实现 gate：未来观测扰动不能改变过去 context。
2. 几何 gate：held episodes 上 same-episode context 相似度显著高于 cross-episode 和 shuffled。
3. 禁止 collapse：有效 context rank 与 batch separation 同时通过。
4. 机制 gate：N256 上 full 相对 PFC-only 至少改善 1%，pooled 与 C20/C32 的 90% paired-bootstrap 下界均大于 0。
5. 绝对性能 gate：full 同时不差于旧 stage2。
6. 只有 1-5 全过，才打开 test512 和 Transformer/Titans/Hippoformer 正式基线。

## 10. 当前项目判断

现在还不能把 3D 可变环境写成“外挂海马提高 Transformer”的正结果。可以诚实主张的是：

- 任务、数据、严格 rollout、PFC 主干已经跑通；
- 我们建立了一套能阻止小样本假阳性的因果审计流程；
- 审计定位了 context 输入边界这一真实架构缺陷；
- 修接口仍不足，下一科学问题已收敛为“怎样无标签地学习稳定 remapping context”。

这比继续调 cutoff 或堆 memory slots 更接近一篇可信论文，但核心正结果仍需下一版训练拿下。
