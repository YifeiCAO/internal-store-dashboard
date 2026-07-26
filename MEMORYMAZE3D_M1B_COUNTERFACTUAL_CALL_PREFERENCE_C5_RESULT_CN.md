# M1b C5 反事实调用偏好实验结果

## 一句话结论

C5 的 class-balanced counterfactual preference 成功把 gate 从近似全开校准到可调用范围，并显著收窄灾难性负尾；但 teacher-forced 分支偏好没有稳定转化为 free-rollout 平均增益。hard 和 soft 两个部署方式都未通过预注册主统计门，因此停止继续调 router。

## 实现与冻结

C5 从 C4 checkpoint 出发，仅训练原有 `NeuralMemoryCallGate`：

- source SHA256：`5EE8B0B70B58EEDEA8A999DB09DB390DD4005989FE8E1916C8953F52009161B0`
- C5 SHA256：`C9E98D99B63A33066C040E05110D396DE6059E87320F35563134B0DEEEA1B7EB`
- trainable parameters：`26,315`
- 发生变化的 tensor：`10`
- 所有变化均属于 `backbone.memory_call_gate.*`
- 最大非 gate 权重差：`0`

没有新参数、新记忆状态、新槽或第二套 fast weights。

训练目标只使用两个冻结分支的 detached pixel error 生成标签：

1. PFC-only；
2. C3 static conditional recall。

目标 RGB 只在训练期生成分支偏好标签，不进入 gate feature、PFC、HPC、地址或 fast-weight 写入；推理时完全不存在。

## 测试与 smoke

- MemoryMaze3D regression：`71 passed`
- smoke：8 steps
- 每个 smoke batch 同时存在 memory-preferred 与 PFC-preferred token
- 有效 token 比例：约 `85%-92%`
- loss、gradient：finite
- 非 gate 模块：无梯度

## 训练

- seed：`2191`
- updates：`400`
- batch：`4`
- sequence：`64`
- learning rate：`2e-4`
- teacher weight：`0`
- rollout weight：`0`
- preference weight：`1`
- elapsed：`190.21 s`
- peak VRAM：约 `0.489 GiB`

训练结果：

- first preference loss：`1.5843`
- final preference loss：`0.6878`
- early mean：`1.1051`
- late mean：`0.6854`
- final batch balanced accuracy：`0.5753`
- validation gate mean：`0.5008`

因此 objective 确实完成了概率校准，没有靠全开或全关投机。

## 预注册盲测

slice：`Val[416:480]`，64 episode，`C20-H44`。

| 分支 | MSE | 相对 paired PFC | call rate |
|---|---:|---:|---:|
| C5 soft | `0.01183695` | `+1.7138%` | soft |
| C5 hard 0.5 | `0.01198463` | `+0.4875%` | `0.2244` |
| PFC-only | `0.01204335` | 基准 | `0` |
| C4 soft | `0.01205352` | `-0.0845%` | soft |
| C3 static | `0.01206611` | `-0.1890%` | `1` |

### C5 hard 主分支

- gain：`+0.058714 × 10^-3`
- relative：`+0.4875%`
- 90% CI：约 `[-0.0304, +0.1512] × 10^-3`
- episode win rate：`48.44%`
- gain p10：`-0.000307`

主门要求 `>=2%` 且 CI 下界大于 0，因此 hard 正式失败。

### C5 soft

- gain：`+0.206398 × 10^-3`
- relative：`+1.7138%`
- 90% CI：约 `[-0.0538, +0.4668] × 10^-3`
- episode win rate：`71.88%`
- gain p10：`-0.000986`

soft 也没有达到 `2%`，且 CI 跨 0，所以不能进入协议的 soft-only 成功分支。

## 相对旧 gate

C5 soft 相对 C4 soft：

- relative：`+1.7968%`
- gain：`+0.216572 × 10^-3`
- 90% CI：`[-0.0553, +0.5039] × 10^-3`
- CI 跨 0

C5 soft 相对 C3 static：

- relative：`+1.8992%`
- gain：`+0.229165 × 10^-3`
- 90% CI：`[-0.0818, +0.5514] × 10^-3`
- CI 跨 0

点估计有改善，但统计证据不足。

## 负尾修复

同 slice 相对 PFC 的 episode gain p10：

| 模型 | p10 |
|---|---:|
| C3 static | `-0.002597` |
| C4 soft | `-0.002387` |
| C5 soft | `-0.000986` |
| C5 hard 0.5 | `-0.000307` |

C5 soft 收回约 `62%` 的 C3 负尾，C5 hard 收回约 `88%`。因此反事实偏好确实学到了 stop-loss 行为。

问题是 hard 调用同时丢失过多正收益，soft 的均值又未达到统计门。

## 正式判断

1. **训练机制有效**：class-balanced preference 把 gate 从 `~0.95` 校准到 `~0.50`。
2. **尾部明显改善**：soft 和 hard 都大幅降低 p10 负收益。
3. **free-rollout 主结果未建立**：hard `+0.49%`，soft `+1.71%`，两个 CI 都跨 0。
4. **teacher-forced 标签存在分布错配**：它能判断当前真实历史下哪个分支更好，但无法充分预测递归想象轨迹中的长期分支价值。
5. **按协议停止 router 调参**：不再在已使用的 val slice 上改 threshold、margin 或 loss weight。
6. **当前保留模型**：C3 128D static-HPC 是最干净的核心模型；C4 soft gate 可作为小幅稳定化消融，C5 作为负尾控制的机制结果，不升级为 headline。

## 下一步研究方向

下一次真正值得做的模型实验不是再调 teacher preference，而是预先冻结一个 rollout-counterfactual teacher：

- 在训练 split 上并行运行 PFC 与 static-HPC 因果 rollout；
- 用未来若干步的累计 detached regret 标注当前调用；
- gate 推理输入仍保持完全因果；
- 在新的数据 seed / 新地图 split 上一次性验证。

该实验成本更高，应作为独立协议，不在本轮继续临时追加。

## 机器产物

- checkpoint：`runs/remap_former/memorymaze3d_m1b_counterfactual_call_preference_c5_pilot400_seed2191/checkpoint_final.pt`
- training summary：`runs/remap_former/memorymaze3d_m1b_counterfactual_call_preference_c5_pilot400_seed2191/training_summary.json`
- decisive evaluation：`reports/memorymaze3d_m1b_counterfactual_call_preference_c5_val416_479`
- C4 same-slice control：`reports/memorymaze3d_m1b_c4_soft_full_val416_479`
- C3 same-slice control：`reports/memorymaze3d_m1b_c3_static_full_val416_479`
