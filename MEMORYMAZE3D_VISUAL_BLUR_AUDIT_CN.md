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
