# ReMAP-Former 8-Seed Headline 冻结协议

> 状态：`FROZEN_BEFORE_ADDITIONAL_TRAINING`。冻结日期：2026-07-13。test split 尚未使用。

## 1. 目的

三 seed validation 已确认 M1b covariance 在相同训练预算下优于 Hippoformer 与 M-delta。Headline 阶段只扩大训练 seed 数并切换到从未查看的 test family pool，不再改变模型、损失、学习率或 checkpoint 选择规则。

训练 seeds 固定为 `712–719`；已有 `712–714`，新增 `715–719`。任何 seed 都不得因结果差而排除。

## 2. 统一训练预算

所有需要训练的模型均使用 `600 steps × batch4 = 2400 episode sequences`、grid size 11、weight decay `1e-4`、global gradient clip `1.0`。第 `s` 颗训练 seed 在第 `t` 步读取 generator seed `s+t`。训练和选点只允许使用 train/dev split。

| 模型 | 初始化 | LR | 可训练范围 | 目标与选点 |
|---|---|---:|---|---|
| Hippoformer | 健康 seq256 checkpoint | `1e-4` | 全参数 | 单一 all-token sensory CE；固定 dev CE 最低点 |
| M-delta | frozen M-delta substrate | `3e-4` | 全参数 | 单一 all-token sensory CE；固定 dev CE 最低点 |
| Raw M1 | 同 frozen M-delta substrate | `1e-3` | retention gate + context projection | 单一 all-token sensory CE；**raw** 固定 dev CE 最低点 |
| M1b | Raw M1 checkpoint | 无训练 | 无 | ridge `0.03` 的 delta→covariance backend 转换；slow weights 不变 |

Warm-start CE 已被三 seed 因果审计拒绝，Headline 阶段禁止恢复。EMA 可以作为离线候选记录，但 M1b 必须从预注册的 raw-dev-CE checkpoint 转换。

## 3. 一次性 Test

只有 `8 seeds × 3 models = 24` 个 checkpoint 全部冻结并通过格式/recipe 审计后，才允许运行一次：

- split：`test`
- generator seed：`1892`
- batch：`16 × 32`
- 每颗训练 seed：512 episodes，预计每模型 1024 个 return-conflict probes
- 三模型逐 batch 使用完全相同的 action/sensory 输入
- 无 room/context/position oracle
- Hippoformer 使用 `torch.no_grad()`，硬性禁止 `torch.inference_mode()`
- 固定输出：`runs/remap_former/headline8_test_seed1892.json`

输出文件一旦存在，evaluator 必须拒绝再次运行，防止反复看 test。

## 4. 预注册指标与 Gates

主指标：8 seed mean return-conflict accuracy。主比较：M1b - Hippoformer，按训练 seed 与 probe 配对。

- 所有模型每颗 seed clean accuracy `>=0.98`
- M1b mean return-conflict `>=0.70`
- M1b mean gain vs Hippoformer `>=0.20`
- M1b 每颗 seed 均优于 Hippoformer
- M1b 每颗 seed 均优于 M-delta

同时报告 query/conflict、other-room target rate、target probability margin、参数量、逐 seed 差值与逐 probe exact-sign p。无论 gate 是否通过，都完整保留全部 8 seed。

## 5. 冻结后禁止事项

不得修改结构、LR、loss、训练步数或 seed；不得依据 test 选择 checkpoint、淘汰 seed 或回头调参。若工程故障导致某次训练中断，只能从已保存的同 recipe 状态恢复或从头复现该 seed。
