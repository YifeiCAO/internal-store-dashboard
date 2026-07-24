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
   - 把当前真实帧编码到 128 维 latent，再立即解码。
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
- 下一候选是训练期的 soft cycle consistency 或 latent denoising，使动力学逐步学会留在稳定区域，而不是推理时强制替换状态。
