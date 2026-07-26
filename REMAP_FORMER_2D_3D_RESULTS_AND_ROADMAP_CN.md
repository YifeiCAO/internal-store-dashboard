# ReMAP-Former 当前结果与下一阶段路线图

**冻结日期：2026-07-25**

**范围：2D hidden-context re-entry、3D MemoryMaze 视觉 strict free rollout、C3-C6 机制链**

## 摘要

当前最强、最干净的结果来自 2D hidden-context re-entry：在不输入 room ID、context 标签、绝对位置、place ID 或当前目标的条件下，ReMAP-Former M1b 的 8-seed return-conflict 准确率为 `0.8201 +/- 0.0200`，而 Hippoformer 与 M-delta 均为 `0`。单次冻结的 P3 Test-2 中，M1b 达到 `0.8376 [0.8221, 0.8527]`，clean recall 为 `0.9944`。组件消融显示 context、covariance correction 和 HPC retrieval 都是必要的。

3D 结果已经证明同一类 episode-local neural HPC 可以在视觉 strict free rollout 中保存并检索有用内容，但目前还不是一个统计稳定的 headline winner。冻结 C3 在 128D variable-content gate 上通过，validation 平均 pixel MSE 从 PFC-only 的 `0.016447` 降至 `0.015938`，相对改善 `3.10%`，但 90% paired bootstrap CI 跨零。C4-C6 进一步研究调用与负尾：C6 的 rollout-counterfactual teacher 在小 pilot 上有信号，但正式 held AUC 只有 `0.513`，因此按预注册协议在 fresh blind 前停止。

当前论文定位应是：

> **Transformer/PFC 从历史推断隐情境，EC 提供关系地址，episode-local neural HPC 以 fast weights 保存内容；2D 已有强因果证据，3D 已完成机制可行性与边界刻画，下一阶段需要公平预算、全新地图盲测和视觉隐情境桥接。**

## 1. 冻结模型

### 1.1 共同合同

ReMAP-Former 的主干是 causal window Transformer，也就是 PFC。模型只接收动作和滞后一拍的感觉输入；当前目标在预测后才可用于训练期写入。模型不接收：

- room ID 或 context 标签；
- context lookup table；
- switch flag、segment ID 或固定时间表；
- 绝对位置、局部位置、朝向或 place ID；
- 当前或未来 target；
- memory slot bank；
- 第二套 fast weights。

### 1.2 EC、地址与 HPC

EC 是动作驱动的 neural grid/path-integration 网络。grid code 经 sparse neural place projection 得到 place code，再与 PFC 的 latent context 做乘性绑定，形成 conjunctive address。

HPC 不是持久参数表，而是每个 episode 从零开始、episode 结束即丢弃的一张可微 fast-weight matrix。写入采用 covariance-corrected delta update；读取先于当前真值写入。HPC retrieval 经投影后回到 Transformer，参与当前感觉预测。

2D 与 3D 使用同一机制原则，但不是同一个 checkpoint：

- **2D M1b**：离散感觉预测、隐情境 A-B-A re-entry、256-unit neural place、16D context；
- **3D C3**：RGB latent/value prediction、128D content bottleneck、同一 episode-local HPC、strict free rollout。

## 2. 2D 核心结果

### 2.1 冻结 P3 Test-2

所有模型在共享 episode 上评估，checkpoint 冻结，无 context oracle。

| 模型 | 参数量 | 训练步 | Strict T1@4096 | Clean | Return-conflict |
|---|---:|---:|---:|---:|---:|
| Hippoformer | 1,494,447 | 600 | 0.3049 | 1.0000 | 0.0000 |
| Hippoformer HPC branch | - | - | - | 0.9868 | 0.0000 |
| M-delta | 1,388,257 | 600 | 0.9925 | 1.0000 | 0.0000 |
| Window Transformer | 1,280,960 | 1,200 | - | 0.8953 | 0.0134 |
| Parameter-matched Transformer | 1,396,096 | 1,200 | - | 0.8954 | 0.0122 |
| Titans-MAC adaptation | 1,344,260 | 1,200 | - | 0.9234 | 0.0095 |
| **ReMAP-Former M1b** | **1,398,289** | **600** | **0.6736** | **0.9944** | **0.8376** |

区间：

- Hippoformer Strict T1@4096：`0.3049 [0.1874, 0.4439]`；
- M-delta Strict T1@4096：`0.9925 [0.9880, 0.9964]`；
- M1b Return-conflict：`0.8376 [0.8221, 0.8527]`。

这里有两个不同问题：

- M-delta 擅长单一当前映射的长 rollout，却不会在隐情境返回时恢复旧内容；
- M1b 的优势是 hidden-context re-entry，而不是所有 strict rollout horizon 上都优于所有模型。

### 2.2 8-seed headline 稳定性

| 指标 | ReMAP-Former M1b | Hippoformer | M-delta |
|---|---:|---:|---:|
| Return-conflict mean | **0.8201 +/- 0.0200** | 0.0000 | 0.0000 |
| Seed range | 0.7852-0.8457 | 0.0000 | 0.0000 |
| Clean mean | 0.9949 +/- 0.0019 | 1.0000 | 1.0000 |

全部预注册稳定性门通过。该结果支持“历史推断出的 context 与 HPC 内容共同解决 A-B-A re-entry”，但不等于解决通用视觉世界模型。

### 2.3 必要组件

8-seed component ablation：

| 条件 | Return-conflict mean | 结论 |
|---|---:|---|
| **Full M1b** | **0.7944** | 完整机制 |
| No covariance correction | 0.2676 | 去串扰更新必要 |
| Fixed context | 0.0000 | 历史条件 context 必要 |
| Shuffled context | 0.4355 | 正确 context 对齐必要 |
| HPC zero | 0.0234 | 外挂记忆内容必要 |
| Forced wrong return context | 0.0010 | 错误重映射摧毁回忆 |
| Correct oracle context | 0.9424 | 剩余上限主要来自 context inference |

这组消融把“只是更大的 Transformer”“只是多一个输出分支”和“只是固定 fast weights”三个解释排除。

### 2.4 长延迟与容量边界

Observed-history 长延迟不是 strict free rollout，必须单独报告：

- 8 seeds，长度 `176-836`：M1b 在 836 步 return 为 `0.8496 +/- 0.0246`；
- 3 seeds，长度 `4356`：M1b return 为 `0.7500`；
- 同条件 Hippoformer 接近 `0`，M-delta 为 `0`，no-covariance 约 `0.27`。

Novel-event capacity：

| 条件 | K1 | K2 | K4 | K8 | K12 | AUC | K50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1b normal | 0.776 | 0.711 | 0.375 | 0.193 | 0.099 | 0.462 | 4 |
| Correct-context oracle | 0.982 | 0.948 | 0.672 | 0.388 | 0.206 | 0.691 | 8 |
| Exact-address diagnostic | 0.987 | 0.971 | 0.836 | 0.737 | 0.477 | 0.844 | 12 |

这些 oracle 只用于定位瓶颈，不能进入正式模型 claim。它们说明容量损失同时来自 context inference 与有限地址分离。

### 2.5 2D 证据边界

目前不能声称：

- strict free rollout 的完整 horizon 曲线全面获胜；
- 官方 mm-TEM 已在本仓库被忠实复现；
- Titans-MAC adaptation 等于官方 Titans；
- 4356 步 observed-history 结果等于 4356 步 free rollout；
- 旧 P5 release archive 已满足发布完整性。旧审计发现 `remap_former/pfc.py` hash 不匹配，必须先重建 canonical release。

## 3. 3D 核心结果

### 3.1 任务与指标

3D 线使用 variable-layout RGB MemoryMaze。模型根据动作和上一帧，在 strict free rollout 中递归生成未来视觉；GT 未来帧不会回灌模型。主要指标为 pixel MSE，越低越好。

当前开发阶段仍使用 adaptive dev/validation。除明确标注的 fresh blind 外，不能把这些数字写成最终 test 结果。

### 3.2 外部 baseline catch-up

在 C20-H44 variable-environment dev 上：

| 模型 | Best rollout MSE | 状态 |
|---|---:|---|
| Transformer | 0.022994 | 1,200 rollout steps |
| Titans adaptation | 0.021891 | 1,200 rollout steps |
| Hippoformer adaptation | 0.023302 | 1,200 rollout steps |
| M1b reference | **0.020073** | 约 12,180 ancestor updates |

所有外部 baseline 都在 rollout training 后改善，但 M1b 的累计训练预算远高于 1,200 steps，因此这张表不能支持 budget-matched superiority。Hippoformer 目前没有增量 rollout cache，速度也不公平。

### 3.3 写入策略

Open-9 development：

- full write：`0.029933`；
- fixed K16 write：`0.027157`，降低 `9.28%`；
- future GT read/write：`0/0`。

3-seed neural write gate：

- hard calibrated mean：`0.027839 +/- 0.000682`；
- 相对 full write 改善约 `7%`；
- 比 fixed K16 差 `2.51%`。

结论是“早写晚停”是稳定现象，但 learned write gate 尚未击败简单 K16。

### 3.4 C3：当前冻结 3D 核心

128D variable-content gate：

- direct reconstruction error ratio：`0.183588`；
- actual retrieval error ratio：`0.242422`；
- retrieval cosine：`0.918`。

Validation episodes `[224,288)`：

| 条件 | Mean pixel MSE | 相对 PFC-only |
|---|---:|---:|
| PFC-only | 0.01644746 | - |
| **C3 full** | **0.01593798** | **+3.0976%** |

90% paired bootstrap CI 为 `[-0.4999,+1.4387] x 1e-3`，跨零；44 个 horizon 中 37 个为正，但 p10 episode 仍有明显负尾。因此 C3 是“最干净、最可复现的机制核心”，还不是统计显著的最终胜者。

### 3.5 C4-C6：调用与负尾

| 版本 | 关键结果 | 决策 |
|---|---|---|
| C4 | 相对 C3 仅约 `+0.272%` | 不升级 |
| C5 soft | Val `[416,480)`：`0.01183695` vs PFC `0.01204335`，改善 `1.7138%`，CI 跨零 | 不升级 |
| C5 hard | 改善 `0.4875%`；p10 repair `62-88%` | 作为负尾诊断 |
| C6 G0 | held AUC `0.6159`，BAcc `0.6019` | 允许进入 formal |
| **C6 formal** | held AUC `0.5131`，BAcc `0.4980` | **fresh blind 前停止** |

C6 formal 使用冻结 C5 rollout，在相同生成历史上强制候选步 `alpha=0` 与 `alpha=1`，用当前及未来最多 4 步 RGB MSE 差生成 detached teacher。只训练原有 gate head，非 head 权重最大变化为 `0`。

正式训练集 AUC 达到 `0.9986`，但 held AUC 只有 `0.5131`；horizon-only 控制为 `0.5974`。这说明优化成功但 episode-specific 泛化失败。调用收益具有明显时间结构：早期平均有利，后期平均有害；现有瞬时 features 没有捕获稳定的长期不确定性。

按预注册协议：

- 不生成 C6 fresh maps；
- 不访问 C6 fresh blind；
- rejected checkpoint 只保留作复现；
- 不继续扫 threshold、loss、margin 或 hidden size；
- C3 保持为 3D 核心。

## 4. 当前证据等级

| 命题 | 当前等级 | 依据 |
|---|---|---|
| 2D hidden-context re-entry 可行 | **强** | 冻结测试、8 seeds、因果消融 |
| 2D 长 observed-history 保持 | **中强** | 8 seeds 到 836，3 seeds 到 4356 |
| 2D 全程 strict free rollout 胜出 | **未成立** | 完整曲线/AUC 不支持全面胜出 |
| 3D HPC content 可存可取 | **强机制证据** | C3 content/retrieval gate |
| 3D full 优于 PFC-only | **趋势** | +3.10%，CI 跨零 |
| 3D learned write gate 优于 K16 | **未成立** | 仍差 2.51% |
| 3D learned call gate 可泛化 | **否定** | C6 formal held AUC 0.513 |
| 3D 相对外部 baseline 公平胜出 | **未检验完成** | 训练预算与实现速度不匹配 |

## 5. 面向 ICLR 的论文主线

### 5.1 最小可辩护贡献

1. 提出一个无 oracle 的 PFC-EC-HPC 合同：Transformer 是 PFC，EC 只从动作构造关系地址，HPC 是 episode-local differentiable fast weights。
2. 在 2D hidden-context A-B-A re-entry 上展示跨 seed 的高准确率和必要组件。
3. 在 3D variable-layout visual rollout 上证明同一记忆原则可运行，并用 C3-C6 给出内容容量、调用收益和泛化失败的完整边界。
4. 提供严格 read-before-write、target-independence、metadata-independence、checkpoint-freeze 与 pre-blind stop 审计。

### 5.2 当前不应使用的标题级措辞

- “解决了长期 free rollout”；
- “全面击败 Hippoformer/mm-TEM/Titans”；
- “learned caller 自动决定何时调用记忆”；
- “3D 结果统计显著”；
- “具有生物学真实性”。

更稳妥的标题方向是：

> **ReMAP-Former: Hidden-Context Re-entry with an Episode-Local Neural Hippocampus**

## 6. 下一阶段实验计划

### P0：发布完整性冻结

目标：先让现有结果可被一键复现。

- 修复旧 archive 中 `remap_former/pfc.py` hash mismatch；
- 从当前 source、协议、checkpoint、summary、figure 重建 canonical release；
- 固定环境、命令、seed、数据 split hash；
- 全量测试必须保持 `76/76`；
- 对 2D/3D 每个 headline 数字建立 machine-readable evidence index。

**Go gate：** clean checkout 能重跑评估并逐项匹配冻结 summary；任何 hash mismatch 都停止后续 sealed test。

### P1：2D strict free-rollout headline

目标：把目前最强 re-entry 结果放到完整 rollout 轴上。

- horizons：`2,4,8,16,32,64,128,256,512,1024,2048,4096`；
- 8 seeds，共享冻结 episode；
- 同时报 accuracy、log-horizon AUC、4096 endpoint、clean、return-conflict；
- 区分 observed-history 与 strict free rollout；
- Hippoformer、M-delta、window Transformer 使用相同训练/eval budget；
- mm-TEM 只在忠实复现官方 64-step 设置后进入正文，否则标注为未复现。

**Go gate：** M1b 的 return-conflict 优势保持，且 strict curve 的预注册主指标至少一个显著优于 Hippoformer；否则论文 headline 只保留 re-entry，不宣称 rollout superiority。

### P2：3D 公平预算与 fresh blind

目标：把 C3 从“趋势”升级成可发表证据。

- 冻结 C3，不再加入 gate；
- 为 Hippoformer 实现 incremental cache，并做逐步等价测试；
- Transformer、Titans adaptation、Hippoformer、C3 按 optimizer steps、seen frames、参数量和 wall-clock 分层报告；
- 至少 3 seeds development，最终 5-8 seeds；
- 新地图 layout hashes 在训练与 dev manifests 中严格排除；
- sealed fresh blind 只打开一次；
- 主指标：paired mean MSE、horizon AUC、p10 tail、C3-PFC paired bootstrap CI。

**Go gate：** C3 对 PFC-only 的 95% CI 不跨零，且相对至少一个 strongest external baseline 在公平预算下保持优势。未通过则 3D 定位为机制扩展，不做 headline。

### P3：2D 到 3D 的隐情境桥

目标：让 3D 真正测试本课题的核心，而不只是视觉预测。

- 同一几何位置在不同视觉 context 下绑定不同内容；
- A-B-A route re-entry，入口与路径族跨 split；
- 不输入 room、context、pose 或 switch；
- PFC 只从动作与视觉历史形成 latent context；
- 外部 probe 只评估，不反向传播；
- 比较 correct context、wrong context、HPC zero、no covariance。

**Go gate：** 轨迹历史 room probe 高、cue-only 接近 chance、标签歧义率充分；full 在 re-entry 上显著优于 PFC-only 和 fixed-context，同时 clean 不下降超过 2 pp。

### P4：容量与尺度律

目标：解释何时 neural HPC 值得使用。

- content dim：`64/128/256`；
- address dim/place sparsity 分层；
- novel events：`K=1,2,4,8,12,16`；
- maze size、rollout horizon、context 数量正交变化；
- 报告 retrieval cosine、direct/retrieval error ratio、K50、计算量和显存。

**Go gate：** 至少出现一个跨 seed 稳定的容量规律，而不是只在单个 checkpoint 上改善。

### P5：caller 线的重开条件

C6 之后不继续调瞬时 gate。只有满足以下条件才重开：

- 显式 recurrent uncertainty state；
- 多个完全独立 teacher episode pools；
- 预先冻结容量、正则化和 early-stop；
- 同时比较 horizon-only、shuffled-label 和 C5 source；
- 继续禁止 age/K token、位置、room/context 标签和第二套记忆。

**Go gate：** held AUC `>=0.60`、BAcc `>=0.55`，且 AUC 同时比 source 与 horizon-only 高 `>=0.02`。未通过就保持 C3 static invocation。

## 7. 建议执行顺序

1. **P0 发布完整性**：一到两个工作日，先把证据链封死。
2. **P1 2D strict curve**：优先级最高，最接近论文 headline。
3. **P2 3D 公平预算**：并行仅保留 baseline cache 工程，不同时开新模型变体。
4. **P3 3D hidden-context bridge**：在 P2 pipeline 稳定后开始。
5. **P4 scale law**：租卡阶段批量运行。
6. **P5 caller**：暂不执行。

## 8. 产物索引

- 2D 总证据：`reports/REMAP_FORMER_PAPER_EVIDENCE_G8_CN.md`
- 2D 8-seed headline：`reports/REMAP_FORMER_8SEED_HEADLINE_TEST_CN.md`
- 2D components：`reports/REMAP_FORMER_M1B_COMPONENT_ABLATION_8SEED_CN.md`
- 2D long delay：`reports/REMAP_FORMER_LONG_DELAY_STAGE_A_8SEED_CN.md`
- 2D capacity：`reports/REMAP_FORMER_P4_NOVEL_CAPACITY_CN.md`
- 3D baseline catch-up：`MEMORYMAZE3D_BASELINE_ROLLOUT_CATCHUP_RESULT_CN.md`
- 3D C3：`MEMORYMAZE3D_M1B_NESTED_VALUE_CAPACITY_C3_RESULT_CN.md`
- 3D C6：`MEMORYMAZE3D_M1B_ROLLOUT_COUNTERFACTUAL_CALL_C6_RESULT_CN.md`
- C6 protocol：`protocols/memorymaze3d_m1b_rollout_counterfactual_call_c6_v1.json`
- 在线中文看板：`reports/internal_store_dashboard.html`

