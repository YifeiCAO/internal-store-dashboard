# MemoryMaze3D Baseline Rollout Catch-up 加件

## 晋级名单

Stage H 和 MM-TEM v2 已经冻结。只有以下模型通过统一的 dev teacher
prediction MSE `<= 0.012`：

- Transformer：`0.004709`
- Titans：`0.004683`
- Hippoformer：`0.004316`

纯 MM-TEM 的低 LR 救援全程有限，但 1600 步 dev MSE 为 `0.014724`，
因此不进入 rollout 网格。Hippoformer 内部的 MM-TEM 支路不能替代这条
被排除的独立 baseline。

## R0：严格 rollout 起点

在同一批前 32 个 adaptive dev episodes 上固定：

- 总长度 64
- 可见 context 20
- 严格 free rollout 44
- 输入只有 RGB 和 action
- 未来真实图像读写均为 0
- 位置和朝向只用于事后 visible/non-visible 评分

同时评估当前 M1b checkpoint，但它只作项目参考。M1b 有更长的 checkpoint
祖先训练链，不能与本轮 baseline 写成完全预算匹配。

## R1：固定四格训练

每个晋级 baseline 都从自己的 Stage H checkpoint 做 `warmstart`，重置
optimizer，仅训练 backbone，视觉壳保持冻结。固定网格为：

- LR：`5e-5`、`1e-4`
- rollout weight：`0.5`、`1.0`
- 每格 1200 updates
- rollout curriculum：`2, 4, 8, 16`
- context：20
- teacher weight：`0.5`

每个模型恰好四格，不允许看结果后追加第五格。

## R2：统一选择

每格重新跑同一套 32-episode `C20-H44` 严格评估，主选择指标是 free
rollout pixel MSE；并列时依次看 action advantage、更低 LR、更低 rollout
weight。所有结果必须通过 finite、descent、视觉哈希、future perturbation
和无 oracle 门。

整个 R0-R2 都属于 adaptive development，禁止访问 test。机器可读协议见
`protocols/memorymaze3d_baseline_rollout_catchup_v1_addendum.json`。
