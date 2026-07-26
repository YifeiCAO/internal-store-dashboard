# M1b C6 Rollout-Counterfactual 调用实验结果

## 一句话结论

C6 证明了“未来调用收益”可以在小规模 train-only pilot 中被当前因果内部状态部分预测，但该信号没有在正式的 96/32 episode 拆分上泛化。正式 held AUC 为 `0.5131`、BAcc 为 `0.4980`，均低于预注册门；horizon-only 控制反而达到 `0.5974` AUC。按协议在打开全新地图盲测前停止，C3 128D static-HPC 继续作为当前最干净的 3D 核心。

## C6 改了什么

C6 不增加推理模型，只增加训练期的反事实标签生成：

1. 从冻结 C5 的 soft strict free rollout 获得当前因果状态；
2. 从完全相同的生成历史分别强制 `alpha_t=0` 与 `alpha_t=1`；
3. 两个分支随后恢复冻结 C5 soft policy；
4. 用当前及未来最多 4 步的累计 RGB MSE 打分；
5. 只训练现有 `memory_call_gate.head`。

保持不变：

- 新参数：`0`；
- 新记忆状态：`0`；
- memory slot：`0`；
- 第二套 fast weights：不存在；
- gate 推理输入：不变；
- room、位置、朝向、place ID、未来观测输入：不存在；
- future ground-truth read/write：`0/0`。

目标 RGB 只在两个因果分支完成预测后计算 detached regret，不会进入任何模型输入、HPC 写入、地址或 gate feature。

## 实现审计

- 回归测试：`76 passed`；
- 强制 `alpha=0`：严格等价 PFC-only；
- 强制 `alpha=1`：调用同一条 C3 HPC conditional recall；
- 改写未来 target：gate feature 与参考 probability 逐位不变；
- 发生变化的正式 checkpoint tensor：4 个；
- 全部属于 `backbone.memory_call_gate.head.*`；
- 最大非 head 权重差：`0`；
- 正式 rejected checkpoint SHA256：`F70FCB2EB881DE433B9C1C3EC813C62A148DCA34AF89639E5B77BE46F9CCE6B1`。

## G0 可预测性 Pilot

固定 train episode `[2240,2264)`，16 fit + 8 held。

| 模型/控制 | Held AUC | BAcc@0.5 |
|---|---:|---:|
| 冻结 C5 head | `0.4753` | `0.5648` |
| Horizon-only | `0.4588` | `0.5278` |
| Shuffled-label | `0.5086` | `0.4537` |
| **C6 rollout teacher** | **`0.6159`** | **`0.6019`** |

G0 的全部预注册门通过，因此进入正式训练。该结果说明小样本中确实存在不只由 rollout age 解释的调用价值信号。

## 正式训练

固定 train episode `[2304,2432)`：

- 96 fit；
- 32 held；
- 11 个候选时刻：`0,4,...,40`；
- lookahead：4；
- discount：0.9；
- teacher branch target comparisons：`11,264`；
- fixed 800 steps；
- 不按 held 指标选 checkpoint；
- elapsed：`1,556.36 s`；
- peak VRAM：`0.324 GiB`。

训练 loss：

- first：`0.62464`；
- final：`0.07378`；
- early mean：`0.55387`；
- late mean：`0.08032`。

fit 指标升到：

- AUC：`0.9986`；
- BAcc：`0.9804`。

因此优化本身成功，但泛化失败。

## 正式 Held 结果

| 模型/控制 | Held AUC | BAcc@0.5 | Mean probability |
|---|---:|---:|---:|
| 冻结 C5 head | `0.5730` | `0.5701` | `0.4837` |
| Horizon-only | **`0.5974`** | **`0.5647`** | `0.4430` |
| Shuffled-label | `0.4385` | `0.4553` | `0.5491` |
| **正式 C6** | `0.5131` | `0.4980` | `0.5679` |

失败的预注册门：

- held AUC `>=0.60`：失败；
- held BAcc `>=0.55`：失败；
- AUC 比 C5 高 `>=0.02`：失败；
- AUC 比 horizon-only 高 `>=0.02`：失败。

实现、标签类别、shuffled control、权重冻结和无泄漏门全部通过。

## 为什么会失败

正式数据中的平均调用收益具有明显时间结构：

| Candidate step | Mean benefit x1e-3 | HPC-preferred rate |
|---:|---:|---:|
| 0 | `+0.0452` | `0.602` |
| 4 | `+0.0690` | `0.648` |
| 8 | `+0.0403` | `0.648` |
| 12 | `+0.0592` | `0.648` |
| 16 | `-0.0754` | `0.547` |
| 20 | `-0.0399` | `0.594` |
| 24 | `-0.1595` | `0.523` |
| 28 | `-0.1217` | `0.555` |
| 32 | `-0.1406` | `0.484` |
| 36 | `-0.1057` | `0.500` |
| 40 | `-0.0410` | `0.578` |

早期调用平均有利，后期平均有害，这与之前 K16 写入结果的“早写晚停”方向一致。然而现有瞬时 call features 没有稳定区分同一 horizon 内哪些 episode 真正受益；高容量 head 可以记住 fit episode，却不能迁移到 held episode。正式结果因此更接近：

> 当前调用价值主要受 rollout phase 与未建模的长期不确定性支配，瞬时 PFC/HPC 对齐特征不足以形成稳定的 episode-specific caller。

这不是 HPC 内容无效：C3 已经通过内容容量 gate，并在多数 horizon 上提供正收益。失败点是 learned invocation 的跨 episode 泛化。

## 正式决定

1. C6 formal 决定：`STOP_BEFORE_FRESH_BLIND`。
2. 不生成 C6 新地图，不访问 fresh blind。
3. rejected checkpoint 只保留作复现，不作为候选模型。
4. C3 128D static-HPC 继续作为 3D 核心。
5. C4/C5/C6 只作为调用稳定性和负结果链。
6. 当前论文不再继续扫读取 gate 的 threshold、loss、margin 或 hidden size。

## 对未来的含义

下一次重新打开 caller 线，至少需要：

- 显式 recurrent uncertainty state，而不是只看逐步瞬时特征；
- 多个完全独立 teacher episode pools；
- 训练前冻结正则化、容量和 early-stop 规则；
- 与 horizon-only、shuffled 和 C5 source 同时比较；
- 仍禁止 age/K token、位置、room/context 标签与第二套记忆。

在这些条件未满足前，论文主线应转向已经更扎实的证据：2D hidden-context re-entry、3D content/address 容量、严格 rollout 的 source-aware write protection，以及公平 baseline 训练。

## 产物

- 协议：`protocols/memorymaze3d_m1b_rollout_counterfactual_call_c6_v1.json`
- G0：`runs/remap_former/memorymaze3d_m1b_rollout_call_c6_g0_seed2201/`
- Formal：`runs/remap_former/memorymaze3d_m1b_rollout_call_c6_formal_seed2202/`
- Teacher 数据：各目录下 `teacher_dataset.npz`
- 训练总结：各目录下 `training_summary.json`
- 正式可视化：`runs/remap_former/memorymaze3d_m1b_rollout_call_c6_formal_seed2202/rollout_call_teacher.png`
