# M1b 3D 想象写入策略：验证扫描与冻结测试结果

## 一句话结论

同一个 next-frame-only M1b checkpoint 在严格 44-step free rollout 中，允许前 16 个 imagined latent 写入 HPC、随后保持只读，比全程写入将完整 512-episode test MSE 从 `29.933e-3` 降到 `27.157e-3`，相对下降 `9.28%`。这确认了长期误差的一项明确来源：早期 imagined write 有益，晚期继续固化模型自己的预测会污染 episode-local fast weights。

## 固定对象

- 数据：MemoryMaze3D paper-aligned open `9x9`，`4096 / 512 / 512` train/val/test。
- checkpoint：`runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/next_frame/checkpoint_final.pt`。
- 训练：20,000 update，只有 next-frame 目标；本实验不重训、不改权重。
- 评估：20 步真实 context，随后 44 步只给 action，预测 latent 自回灌。
- HPC read 始终开启；真实 context 始终写入。`K` 只表示 context 后最多允许多少个 imagined latent 写入 HPC。
- 模型输入仍只有 egocentric RGB 与 action；未来 RGB、位置、朝向、room id 均不进入模型。

## Validation 选型

先在固定的 64 条 validation episode 上扫描 `K in {0,4,8,16,24,32,44}`，再冻结唯一候选 `K=16`。数值均为 pixel MSE x `1e-3`。

| Future writes K | 44-step mean | h32 | h44 | 相对 persistence |
|---:|---:|---:|---:|---:|
| 0 | 32.578 | 40.040 | 50.415 | -2.86% |
| 4 | 30.411 | 40.680 | 51.156 | +3.98% |
| 8 | 28.865 | 41.869 | 51.798 | +8.86% |
| **16** | **26.167** | **39.275** | **49.158** | **+17.38%** |
| 24 | 27.622 | 40.626 | 57.993 | +12.79% |
| 32 | 29.943 | 50.425 | 60.316 | +5.46% |
| 44（全程写） | 31.078 | 50.425 | 68.373 | +1.87% |

补充诊断：全程写入从 horizon 29 开始持续劣于完全停止 future write。`K=0` 又不如 `K=16`，所以结论不是“海马没用”，而是“预测来源的内容不能无限期固化”。

## 一次性 Test

选定 `K=16` 后，只在完整 test 上运行这一个候选，不在 test 上继续扫描 cutoff。

| 条件 | 44-step mean MSE x1e-3 | 相对 persistence | 说明 |
|---|---:|---:|---|
| Persistence | 32.186 | 0% | 最后一帧复制 |
| 原始 M1b，全程 predicted-write | 29.933 | +7.00% | 正式 checkpoint 默认 rollout |
| **M1b，future-write K=16** | **27.157** | **+15.62%** | validation 预选，test 只跑一次 |

`K=16` 相对原始 M1b 的平均 MSE 下降 `2.776e-3`，即 `9.28%`。其 test h32 为 `39.816e-3`，h44 为 `44.404e-3`；对应 persistence 为 `40.367e-3` 与 `45.175e-3`。

## 完整性审计

- test episode：`512`。
- `future_ground_truth_reads = 0`。
- 新旧评测得到的 persistence 均为 `32.185730460351664e-3`，差值严格为 `0`。
- 同一 checkpoint、数据顺序、context 和 rollout horizon；唯一干预是 imagined-state write gate。
- 原始模型、checkpoint 和默认 forward 行为未被覆盖；`K=16` 仍是诊断/控制条件。

## 机制判断

1. HPC 在 rollout 早期有效：完全禁止 imagined write 的 `K=0` 明显不如 `K=16`。
2. HPC 不能把低置信度预测无限固化：全程写入在中后段累积内容污染。
3. PFC window 不是唯一故障点：污染反转约从 h29 开始，早于 window=32 的硬边界；越过边界后恶化更明显。
4. 当前结果支持 source-aware consolidation，而不支持新增 memory slots、第二套 fast weights 或更大的查表结构。

## 下一步：Neural Write Gate

固定 `K=16` 是因果上界和工程控制，不是最终方法。下一版只在现有 HPC 更新上加入一个标量 `g_write_t in [0,1]`：

```text
M_t = M_{t-1} + g_write_t * DeltaM_t
```

候选 gate 只读模型内部量：PFC hidden、PFC prediction 与当前 latent 的自洽残差、context/state、place/address 统计和距最后真实观察的 age。它不读未来真值、pose 或 room/context 标签。先冻结原 M1b，只训练小 gate head；主损失使用随机 context/horizon 的多步 rollout loss，并只加一个轻量写入预算项防止全开/全关。

验收不能只复现固定 16：gate 必须在不同 context 长度和 rollout horizon 下稳定，并报告 full / K=0 / K=16 / learned-gate 四条件消融。

## 产物

- Validation 扫描：`runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/diagnostics_future_write_cutoff_val64/`
- Test 确认：`runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/diagnostics_future_write_cutoff16_test512/`
- Test 状态：`runs/remap_former/memorymaze3d_paper_open9_m1b_seed1701/future_write_cutoff16_test_status.json`
- 可复现入口：`run_memorymaze3d_future_write_cutoff16_test512.ps1`

