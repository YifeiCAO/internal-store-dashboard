# M1b C4/C4b 因果调用门实验结果

## 一句话结论

C4 neural call gate 学到了方向正确但幅度不足的 retrieval 排序，且相对 C3 static call 有一个很小、跨 slice 一致的改善；但它没有使 HPC 相对 PFC 的主结果跨 slice 稳定。C4b 固定硬阈值显著收窄负尾，却误杀有用调用并降低平均收益。两轮都按预注册门判定失败，不再继续调 threshold。

## C4 训练

- source：C3 128D checkpoint
- source SHA256：`8D17844FFC291BD16503A7A1A938C26ABEC7D0EB60282908473102A43BED1706`
- seed：`2181`
- updates：`400`
- trainable scope：仅 `m1b_read_gate`
- trainable parameters：`26,315`
- learning rate：`2e-4`
- rollout：full causal self-write
- initial call probability：`0.9`
- peak VRAM：`4.103 GiB`
- elapsed：`1,614.13 s`
- final SHA256：`5EE8B0B70B58EEDEA8A999DB09DB390DD4005989FE8E1916C8953F52009161B0`

冻结审计：

- C3 与 C4 共有 tensor：`120`
- 发生变化的共有 tensor：`0`
- 最大冻结权重差：`0`
- 新增 gate tensor：`10`

因此 C4 的差异只来自调用门。

训练健康：

- finite gate：通过
- H16 early total：`0.019309`
- H16 late total：`0.016400`
- H16 early rollout：`0.009245`
- H16 late rollout：`0.006759`
- final-horizon descent gate：通过

## C4 预注册盲测

slice：`Val[288:352]`，64 episode，`C20-H44`。

| 分支 | MSE |
|---|---:|
| PFC-only | `0.01826520` |
| C3 static full | `0.01825522` |
| C4 soft call | `0.01819481` |

C4 soft 相对 PFC-only：

- relative：`+0.3854%`
- gain：`+0.070392 × 10^-3`
- 90% CI：`[-0.141370, +0.275256] × 10^-3`
- episode win rate：`64.06%`

主门要求至少 `2%` 且 CI 下界大于 0，因此 **C4 正式失败**。

## gate 学到了什么

gate 分布：

- mean：`0.9500`
- std：`0.02194`
- min / max：`0.8722 / 0.9751`
- Spearman(gate, step gain)：`+0.1426`

| gate 四分位 | HPC gain | step win rate |
|---|---:|---:|
| Q1，最低 | `-0.83 × 10^-3` | `60%` |
| Q2 | `-0.47 × 10^-3` | `58%` |
| Q3 | `+0.70 × 10^-3` | `59%` |
| Q4，最高 | `+0.89 × 10^-3` | `69%` |

排序方向是对的，但所有 alpha 都太接近 1，低分 retrieval 没有真正关闭。

同 slice 上，C4 soft 相对 C3 static：

- relative：`+0.3309%`
- gain：`+0.060406 × 10^-3`
- 90% CI：`[+0.019607, +0.106426] × 10^-3`

gate 确实做了小修正，但远不足以建立主结果。

## C4b 固定硬调用

开发结果后固定 threshold `0.95`，在全新 `Val[352:416]` 一次性评估：

| 分支 | MSE | call rate |
|---|---:|---:|
| C4 soft | `0.01680271` | `1.000` |
| C4b threshold 0.95 | `0.01711773` | `0.631` |
| PFC-only | `0.01764199` | `0.000` |
| C3 static full | `0.01683781` | `1.000` |

C4b hard 相对 PFC：

- relative：`+2.9717%`
- gain：`+0.524261 × 10^-3`
- 90% CI：约 `[-0.0713, +1.1095] × 10^-3`
- CI 下界未过 0。

C4b hard 相对 C4 soft：

- relative：`-1.8748%`
- gain：`-0.315019 × 10^-3`
- episode win rate：`29.69%`

负尾确实改善：

- soft 相对 PFC 的 episode gain p10：`-0.004454`
- hard 相对 PFC 的 episode gain p10：`-0.000808`
- 负尾收回约 `81.9%`

但平均值明显变差，所以 C4b 也正式失败。固定 threshold 把“少数坏调用”连同大量有用的边界调用一起切掉了。

## slice 异质性

`Val[352:416]` 上：

- C3 static 相对 PFC：`+4.5583%`
- 90% CI：约 `[+0.0046, +1.6153] × 10^-3`
- C4 soft 相对 PFC：`+4.7573%`
- 90% CI：约 `[+0.0837, +1.6220] × 10^-3`
- C4 soft 相对 C3 static：仅 `+0.2085%`，CI 跨 0

所以这个 slice 的漂亮主结果主要来自 C3 HPC，而不是 gate。

合并两个相邻 slice，共 128 episode：

- C3 static 相对 PFC：`+2.2674%`，CI 仍跨 0
- C4 soft 相对 PFC：`+2.5334%`，CI 仍跨 0
- C4 soft 相对 C3 static：`+0.2722%`
- C4-vs-C3 paired 90% CI：`[+0.0104, +0.0896] × 10^-3`

结论：

> gate 的小修正是真实且一致的，但 HPC 相对 PFC 的收益存在更大的 episode 级异质性，当前 soft MSE objective 没有把调用概率校准到足以处理灾难性尾部。

## 下一步

停止继续扫阈值。C5 只改 gate 训练目标：

1. PFC、C3 static-HPC、codec、adapter 全部冻结；
2. 对每个 teacher-forced token 计算冻结 PFC 分支和冻结 static-HPC 分支的 detached pixel error；
3. 标签只表示哪个分支更好；
4. 用一个 class-balanced counterfactual preference BCE 训练同一个因果 gate；
5. 推理时 gate 仍只读原有因果特征，不读目标或 oracle。

这是一个新 loss，但不是新记忆、不是新输入，也不是第二套 fast weights。

## 机器产物

- C4 checkpoint：`runs/remap_former/memorymaze3d_m1b_causal_call_gate_c4_pilot400_seed2181/checkpoint_final.pt`
- C4 training summary：`runs/remap_former/memorymaze3d_m1b_causal_call_gate_c4_pilot400_seed2181/training_summary.json`
- C4 diagnostic：`reports/memorymaze3d_m1b_causal_call_gate_c4_diagnostic_val288_351`
- C4b evaluation：`reports/memorymaze3d_m1b_hard_invocation_c4b_val352_415`
- C3 static control A：`reports/memorymaze3d_m1b_c3_static_full_val288_351`
- C3 static control B：`reports/memorymaze3d_m1b_c3_static_full_val352_415`
