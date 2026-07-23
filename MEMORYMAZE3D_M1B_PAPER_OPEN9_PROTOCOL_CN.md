# M1b × Hippoformer 原文 3D 协议

## 这次只回答什么

把我们的 M1b 放到 Hippoformer Figure 6 的 3D 任务范式里，先回答：在原文的空平面、随机纹理、64 步视觉预测任务上，M1b 能否完成一步预测和严格多步想象。

多房间、每轮重建几何的 `maze9-variable` 是更难的扩展实验，不参与这次主结果。

## 原文已明确的设置

- 环境：MemoryMaze3D，`9×9` 空二维平面，无内部房间墙。
- 每个 trial：重新采样环境纹理和第一视角轨迹。
- 空平面保留 1 个位于地板下的 MemoryMaze 占位 target；它不会出现在 RGB，也不进入模型输入。生成器对上游的单 target 重选死循环加了显式 guard，仅修任务 bookkeeping，不改变视觉或动力学。
- 输入：`64×64` egocentric RGB 与六种离散动作；运动结果连续、带噪声和加速度。
- 序列：64 个 action step，65 帧（初始帧加 64 个目标帧）。
- 训练：next-frame prediction，batch size 16，learning rate `5e-4`。
- 优化：Adam；StepLR 每 500 step 乘 `0.9`；最多 20,000 update。
- 评估：像素 MSE；位置 `9×9`、方向 12 bins，仅用于 visible / non-visible 分组，绝不进模型。
- 多步想象：context 后只继续提供 action，不再提供未来真值 RGB。

## 本次 M1b

- PFC：一层 window Transformer，window 32。
- EC：128 维可学习 SE(2) path integration，只读 action。
- place：256 个 soft sparse place units。
- context：由 Transformer 的历史状态推断，不输入 room id、绝对位置或 place id。
- HPC：episode 内 covariance-corrected differentiable fast-weight memory，value dim 64。
- 调用增益：`memory_call_gain=4`。
- 模型输入只有 `egocentric_rgb_64x64 + six_way_action`。

## 训练边界

本次没有 rollout curriculum，也没有 rollout loss。总损失仍保留 M1b 视觉外壳需要的三项：

```text
L = L_next_pixel + 0.05 L_next_latent + 0.25 L_reconstruction
```

M1b 在此任务上的 auxiliary 项为零。这个设计让训练目标仍然是一步预测，不通过 free-rollout 真值给模型额外训练信息。

## 明确不是原文直接给出的量

- 训练/验证/测试 trial 数：原文未报告；本地正式版固定为 `4096 / 512 / 512`。
- 数值精度：原文未报告；3090 上使用 bf16。
- 多步 context：原文没有写出 `T1`。Figure 6 第一张想象图标为 Step 21，因此主评估采用 `context=20, rollout=44`；这是有依据的推断，不冒充原文显式参数。
- 被测模型是 M1b，不是原文 Hippoformer；任务协议对齐不等于模型复现。

## 开跑前 gate

1. 每个 `maze_layout` 必须是 `9×9` 且 81/81 全部可行，无内部墙。
2. 生成器直接读取 MemoryMaze3D 的 `_current_wall_texture`，每个 trial 保存真实纹理分配和 SHA256；每个 split 的唯一纹理率至少 95%。
3. 每个 episode 必须恰好 64 actions、65 RGB frames，且移动帧比例至少 80%。
4. 模型字段固定为 `images, actions`；`agent_pos, agent_dir` 只能用于评估；oracle 字段必须为空。
5. train / val / test 使用互不相交的 seed 段。空平面的 layout hash 跨 split 相同是正确现象，不再误判为数据泄漏。

## 正式运行

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_memorymaze3d_m1b_paper_open9.ps1
```

主要产物：

- `data/memorymaze3d_paper_open9_full_v1/`
- `runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/protocol.json`
- `runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/data_audit.json`
- `runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/next_frame/checkpoint_final.pt`
- `runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/eval_test_c20_h44/evaluation_summary.json`

## 正式结果与写入诊断

20,000 update 的冻结 M1b 在 512 条 test episode 上得到：

- teacher-forced one-step pixel MSE：`1.610e-3`。
- strict 44-step free-rollout MSE：`29.933e-3`。
- persistence：`32.186e-3`，原始 M1b 相对提升 `7.00%`。
- counterfactual action MSE：`49.628e-3`，actual-action advantage 为 `19.694e-3`。
- `future_ground_truth_reads = 0`，leakage gate 通过。

随后在 validation 上只干预 imagined latent 的 HPC 写入时长，`K=16` 最优；冻结该选择后，完整 test 的 MSE 为 `27.157e-3`，相对原始全程写入下降 `9.28%`，相对 persistence 提升 `15.62%`。这说明早期 predicted write 有益，但晚期继续写会把 rollout 误差固化进 fast weights。

完整数字、边界和下一步 neural write gate 方案见 `MEMORYMAZE3D_M1B_WRITE_POLICY_RESULT_CN.md`。

Neural write gate follow-up 已完成：冻结 M1b，仅训练 `833` 参数因果 gate。冻结选择的 hard gate 在完整 test512 上得到 `27.174e-3`，相对全程写入改善 `9.22%`，与固定 K16 的 `27.157e-3` 基本持平；future ground-truth reads/writes 均为 0。soft gate 的 validation 收益未迁移到 test，hard gate 跨 seed calibration 方差仍高，完整结果见 `MEMORYMAZE3D_M1B_NEURAL_WRITE_GATE_RESULT_CN.md`。
