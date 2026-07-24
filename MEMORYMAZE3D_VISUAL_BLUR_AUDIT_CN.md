# MemoryMaze3D 视觉模糊来源审计

## 结论

当前图像模糊不是单一问题。固定验证集 `Val168-199` 的误差阶梯表明：

| 阶段 | Pixel MSE x1e3 | PSNR | 相对 GT 梯度能量 |
|---|---:|---:|---:|
| Autoencoder 重建 | 1.614 | 27.92 dB | 104.6% |
| Teacher-forced 一步预测 | 3.261 | 24.87 dB | 91.6% |
| 严格 C20→H44 free rollout | 17.538 | 17.56 dB | 80.3% |

从最终 free-rollout MSE 的数值阶梯看：

- decoder / latent 压缩底噪：`1.614`，占 `9.2%`
- teacher-forced 相对重建新增：`1.647`，占 `9.4%`
- free rollout 相对 teacher-forced 新增：`14.277`，占 `81.4%`

因此，**最大瓶颈是闭环 rollout 的状态漂移**。Decoder 也确实会把边缘和小物体抹平，但它不是本轮 `17.538` 总误差的主要增量来源。

## 审计设置

- checkpoint：`memorymaze3d_bidirectional_phaseprop_rollout80_seed2085`
- 数据：`memorymaze3d_variable_full_v1`
- split：validation
- 固定连续 episode：`168-199`，共 32 条
- context：20 steps
- free rollout：44 steps
- 展示样本：固定连续 `168-170`
- test：未读取
- 模型输入：egocentric RGB + action
- free rollout future truth read/write：均为 0

## 四层对照是什么

1. `Ground truth`
   - 环境真实未来帧。
2. `AE reconstruction`
   - 把当前真实帧经过四层卷积压到 `4×4×128=2048` 个特征，再映射到 4096 维 latent，并立即解码。
   - 它会读取目标帧，只用于测 encoder/decoder 上限，不是预测结果。
3. `Teacher-forced one-step`
   - 每一步可以读取此前真实帧，但不能读取当前目标帧。
   - 它测一步视觉动力学误差。
4. `Strict free rollout`
   - C20 后只给 action，把模型自己的 latent 预测喂回下一步。
   - future truth read/write 均为 0。

## 为什么 AE 梯度能量会超过 100%

`104.6%` 不代表 AE 比真值更清晰。当前 decoder 使用四层 `ConvTranspose2d`，会产生轻微高频纹理或 ringing；简单梯度能量会把这些伪高频也计作“边缘”。因此它只能辅助判断高频是否衰减，不能替代肉眼质量、MSE 或感知指标。

## 训练边界

产生这些图的最后一轮 `phaseprop80` 使用：

```text
trainable_scope = m1b_bidirectional_hpc
```

只更新：

- place
- bidirectional HPC
- grid correction
- memory projection

被冻结：

- visual encoder
- pixel decoder
- Transformer/PFC
- 其余 world dynamics

所以这轮训练从机制上就不可能修复 decoder 模糊或 PFC rollout 漂移。

## 下一步

下一轮做受控的全模型 rollout 微调：

1. 从当前双向 HPC checkpoint warm-start。
2. `trainable_scope=all`，让 PFC、视觉编码器和 decoder 能收到 rollout 梯度。
3. 先用短到长 curriculum，避免直接 H44 训练不稳定。
4. 同时保留 reconstruction 和 teacher-forced loss，防止只追长程 MSE 导致一步质量退化。
5. 只在新的 validation slice 选择 checkpoint，不读 test。

晋级条件必须同时满足：

- AE reconstruction MSE 不变差。
- teacher-forced MSE 不变差。
- C20→H44 free-rollout MSE 明显下降。
- HPC-on 相对 PFC-only 不被全模型微调抹掉或反转。

## 后续 Pilot 结果

按照上述方案运行了一个低学习率全模型 pilot：

- warm-start：`memorymaze3d_bidirectional_phaseprop_rollout80_seed2085`
- seed：2088
- updates：50
- learning rate：`1e-5`
- trainable scope：`all`
- curriculum：`2,4,8,16,32`
- 独立门控：固定 `Val200-215`，C20→H44

| 模型 | AE MSE x1e3 | Teacher MSE x1e3 | Rollout MSE x1e3 |
|---|---:|---:|---:|
| 原 checkpoint | 1.510 | 3.025 | 25.414 |
| Update 40（训到 H16） | 1.520（+0.62%） | 3.023（-0.09%） | 25.713（+1.18%） |
| Update 50（训到 H32） | 1.522（+0.80%） | 3.012（-0.45%） | 26.225（+3.19%） |

结论：

- 一步预测只有极小改善。
- AE 重建轻微退化。
- H44 rollout 在 update 40 和 update 50 均退化。
- 最后 H32 阶段进一步放大退化，但不是失败的唯一来源。
- 该 pilot 不晋级，当前最佳 checkpoint 保持不变。

下一步不继续盲加 update。应先诊断 predicted latent 是否在闭环过程中逐步离开视觉 encoder 的 latent manifold，再决定使用 latent projection、denoising dynamics 或多 horizon 稳定性目标。

## Latent-Manifold 因果门

在新的固定 `Val216-231` 上执行无需训练的 causal projection：

```text
predicted latent
→ pixel decoder
→ visual encoder
→ projected latent
→ 只作为下一步 feedback
```

该操作不读取未来 RGB、位置、方向、房间 ID 或地图。

### Cycle 距离

| Latent 来源 | Decode→encode cycle MSE | Cycle cosine |
|---|---:|---:|
| 真实 target latent | 0.00273 | 0.9981 |
| Teacher-forced prediction | 0.01301 | 0.9910 |
| Strict rollout prediction | 0.01744 | 0.9889 |

预测 latent 的确比真实 latent 更偏离视觉 autoencoder manifold，并且 strict rollout 比 teacher-forced 更严重。

### 硬投影结果

| Feedback | C20→H44 pixel MSE x1e3 |
|---|---:|
| 原 strict feedback | 12.591 |
| Decode→encode projected feedback | 16.305 |

相对变化：`-29.5%`，即投影后明显变差。分 horizon 只有 H8 小幅改善 `5.1%`，H32 恶化 `41.9%`。

结论：

- off-manifold 漂移是真实存在的诊断信号。
- 推理时硬投影不是解法，因为当前模型依赖一些 AE round-trip 不能无损保留的预测 latent 坐标。
- 不采用 hard projection。
- 下一候选进入训练期独立门控：分别测试 soft cycle consistency 与 latent denoising，不能因为它们改善单步质量就直接并入主模型。

## 维度更正

当前正式 checkpoint 的 `dim_latent` 是 **4096**，不是 128。此前“128 维 latent”的表述混淆了 decoder 内部的 128 个卷积通道，现已更正。

当前视觉路径为：

```text
RGB 3×64×64
→ CNN 128×4×4（2048 个卷积特征）
→ Linear + LayerNorm
→ 4096 维 visual latent
→ Linear
→ 128×4×4
→ 四层 ConvTranspose2d
→ RGB 3×64×64
```

因此当前没有“visual latent 只有 128 维、容量明显太小”的证据。后续容量门固定 latent=4096，比较卷积宽度 `64/128/256`，测试真正的空间视觉编码与解码容量。

## 训练期稳定性目标门控

在固定 `Val232-239` 上，以当前最佳 checkpoint 为基线，分别进行 100 update 的低学习率全模型微调。两项机制独立运行，避免归因混杂：

- latent denoising：对真实 visual latent 添加相对 RMS 高斯扰动，要求恢复干净 latent 与图像。
- soft cycle：将 rollout 预测 latent 经 `decoder→encoder` 往返，仅用 cycle loss 拉向真实 target latent；推理时不做硬替换。

| 模型 | AE MSE x1e3 | Teacher MSE x1e3 | C20→H44 MSE x1e3 | H44 相对基线 |
|---|---:|---:|---:|---:|
| 原 checkpoint | 1.483 | 3.108 | 15.011 | - |
| Latent denoising 100 | 1.443 | 3.050 | 16.175 | **+7.75% 变差** |
| Soft cycle 100 | 1.467 | 3.092 | 17.011 | **+13.32% 变差** |

结论：

- 两种目标都略微改善 AE 或 teacher-forced 一步质量。
- 两种目标都明显恶化严格 H44 rollout。
- “更贴近视觉 manifold”不等于“闭环动力学更稳定”；它可能抹掉 PFC rollout 依赖的预测坐标。
- 两项机制均不晋级，也不把两个失败机制组合后再赌一次。
- 当前最佳 checkpoint 仍是 `memorymaze3d_bidirectional_phaseprop_rollout80_seed2085`。

## 视觉编解码器容量门控

为了隔离视觉容量，新增独立 autoencoder 训练路径。它只读取 `egocentric_rgb`，完全绕开 Transformer/PFC、HPC、action dynamics 和位置等 oracle 输入。

公平设置：

- latent 固定为 4096。
- visual width 比较 `64/128/256`。
- 三档使用相同 seed、数据顺序、batch、sequence 长度、学习率日程和 2000 updates。
- 固定审计切片为 `Val240-255`，test 未读取。

| Visual width | 参数量 | Pixel MSE x1e3 | 相对 GT 梯度能量 |
|---:|---:|---:|---:|
| 64 | 8.62M | 2.623 | 159.9% |
| **128** | **17.65M** | **2.092** | 122.5% |
| 256 | 36.99M | 2.552 | 121.6% |

结论：

- 64 宽度存在容量不足，MSE 比 128 高 `25.4%`。
- 256 参数量约为 128 的 2.10 倍，但 MSE 反而高 `22.0%`。
- 当前 128 宽度是同预算下的甜点位，盲目加宽 decoder 不能解决模糊。
- 即使是只做当前帧重建的 autoencoder，输出仍有颜色平均和平滑；视觉编解码器贡献了底噪。
- 但原正式 checkpoint 上，teacher→rollout 阶段贡献了总 MSE 增量的 `81.4%`，所以首要瓶颈仍是闭环 world dynamics，而不是卷积宽度。
- 梯度能量高于 100% 包含 `ConvTranspose2d` 的伪高频纹理，不代表感知质量优于真值。

## 当前决策

1. 保留正式模型的 4096 维 latent 与 visual width 128。
2. 不采用 hard projection、latent denoising 100 或 soft cycle 100。
3. 不把未通过独立 gate 的机制组合进主模型。
4. 下一轮优先改变闭环训练信号，而不是继续堆视觉参数：用多步 latent dynamics 目标直接约束 rollout transition，同时保留严格的 AE、teacher 与 H44 三门验收。
5. test split 继续保持未读取。
