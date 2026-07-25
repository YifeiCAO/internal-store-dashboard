# M1b HPC 内容通路与条件式回忆适配器报告

## 结论先行

当前最可靠的主干仍是 **window Transformer PFC-only**。现有 HPC 的 EC/context/address 几何没有塌，但它存储和读出的 neural content 不能稳定提供增量。

本轮完成了两层排查：

1. **残差内容失败原因**：PFC residual 在同一地点不稳定，而且 `4096→64→4096` 内容瓶颈没有学会重构 residual。
2. **条件 recall adapter 结果**：adapter 能把旧 HPC 的大幅伤害压到接近零，但在全新验证切片上仍比 paired PFC-only 差 `0.1207%`，置信区间跨零，预注册门失败。

因此下一轮只应改 **稳定 sensory content 的神经编码器**，不应继续调 context、地址、调用阈值或固定增益。

## 当前模型

```text
过去 RGB + action
        │
        ▼
window Transformer PFC ───────────────► PFC-only 预测
        │
        ├── PFC history context v6
        │
action ─► EC grid ─► sparse neural place
                          │
context ──────────────────┤
                          ▼
                 conjunctive address
                          │
                          ▼
      episode-local covariance fast-weight HPC
              read before current write
                          │
                          ▼
                stable sensory retrieval
                          │
                          ▼
              conditional recall adapter
                          │
                          ▼
                 PFC hidden correction
```

HPC 不是持久参数表，也没有离散 memory slots。每个 sequence 都创建一个新的 dense value-by-address fast-weight state，episode 结束即丢弃。

## 内容路径审计

审计 checkpoint：

- sensory 版本：`memorymaze3d_m1b_pfc_history_v6_pilot400_seed2121`
- residual 版本：`memorymaze3d_m1b_residual_content_hpc_pilot400_seed2141`
- 数据：`memorymaze3d_variable_full_v1`
- 切片：`val[96:128]`
- 位置和方向只用于离线审计，没有进入模型。

### 地址与数值条件

| 指标 | 结果 |
|---|---:|
| 当前读地址与最佳历史写地址余弦 | `0.8025` |
| 重复姿态比例 | `84.62%` |
| 最佳历史地址命中相同姿态 | `83.84%` |
| 双钥匙分母均值 | `1.00000005` |
| covariance condition number 均值 | `27.16` |
| update RMS p99 | `0.01939` |

判断：地址不是完美的，但没有塌；双钥匙在推理期也没有出现足以解释失败的数值放大。

### Residual content

| 路径 | latent MSE | 相对空内容 |
|---|---:|---:|
| 空内容 | `0.022758` | `1.000` |
| 当前真实 residual 直接过 value AE | `0.044270` | `1.945` |
| 实际地址读出 | `0.038911` | `1.710` |
| sensory-corrected 地址读出 | `0.039645` | `1.742` |

像素层：

- PFC-only：`0.00288455`
- 精确 residual oracle：`0.00139842`，理论改善 `51.52%`
- 经过 64 维 value AE 的 residual oracle：`0.00295372`，反而差 `2.40%`

精确 residual 本身有预测价值，但当前 neural content 表征完全没有保住它。

更关键的是重复姿态稳定性：

- sensory content 相对重复误差：`0.03482`
- residual content 相对重复误差：`0.98757`
- residual 比 sensory 不稳定：`28.36×`

同一地点的 residual 随 PFC 历史变化，因此它不是合适的 place-addressed memory content。

### Stable sensory content

原 sensory content 在相同切片上：

- 64 维 direct autoencoder 相对误差：`0.5435`
- direct autoencoder cosine：`0.7744`
- 实际关联读出相对误差：`0.4821`

它比 residual 稳定，也保留了一部分视觉结构，但旧融合固定放大 `4×`：

- `4×` actual retrieval：相对 PFC 差 `13.65%`
- 无训练 gain sweep 的最佳 actual gain 为 `0.5×`
- 最佳 actual gain 只改善 `0.3913%`

结论：缩小固定增益可以止损，但达不到预注册的 `1%` 增量门。

## 条件 Recall Adapter

### 设计

```text
delta_h = A(PFC_h, projected_retrieval) - A(PFC_h, zero_retrieval)
prediction = decoder(PFC_only_h + delta_h)
```

关键性质：

- trainable 参数：`140,928`
- 没有新 memory state
- 没有新 slots
- 没有第二套 fast weights
- 没有新 loss
- 输入只有 causal PFC hidden 和 read-before-write retrieval
- retrieval 为零时，`delta_h` 精确为零
- adapter 不能靠偏置退化成额外 PFC

实现测试：`66 passed`。

### 训练

- checkpoint：`memorymaze3d_m1b_conditional_recall_adapter_pilot400_seed2151`
- SHA256：`A9D6D6014A5EFC02FA18851A16BE077318E170C2B4EED0DE9E7EDD21C1E14AFB`
- 400 updates，batch 2，sequence 64
- rollout curriculum：`2,4,8,16`
- observed-only HPC writes
- 用时：`1463.22 s`，约 `24.4 min`
- peak VRAM：`2.30 GiB`
- finite gate：通过
- rollout descent gate：失败
- teacher validation prediction MSE：`0.00278956`

teacher validation 变好，但预注册规则不允许用它替代 strict free rollout。

## 新鲜切片验证

切片：`val[128:160]`，在模型和 adapter 设计时未使用。

| 变体 | strict C20-H44 MSE | 相对 paired PFC |
|---|---:|---:|
| PFC-only | `0.01416568` | `0%` |
| observed-only HPC + adapter | `0.01418278` | `-0.1207%` |
| all-write HPC + adapter | `0.01440108` | `-1.6618%` |

Observed-only HPC 相对 PFC 的 episode 配对结果：

- mean gain：`-0.01709 × 10⁻³`
- bootstrap 90% CI：`[-0.61697, +0.52701] × 10⁻³`
- episode win rate：`62.5%`

虽然多数 episode 小幅获胜，但少数负例抵消了收益，均值为负且区间跨零。预注册的 `>=1%` 且 CI 下界大于零门失败。

## 已排除与未解决

已排除：

- context 表征整体崩溃；
- place/address 完全失效；
- 双钥匙推理期数值爆炸；
- 只需继续调固定 gain；
- 只需继续调调用 gate；
- residual 直接作为稳定 place content。

尚未解决：

- 当前 64 维 sensory value code 是为旧 `4×` 线性融合训练的；
- adapter 冻结该 code 后，能止损但提取不到可靠预测增量；
- imagined writes 仍会累积污染，因此正式路线应继续 observed-only，直到内容读出先证明有用。

## 下一步

下一轮采用单一改动：

1. 保持 sensory content、PFC、context、EC、place、地址和 fast-weight 规则不变；
2. 联合训练 `HPC value_encoder + value_decoder + memory_projection + recall adapter`；
3. value encoder 仍只读 sensory，因此 content 保持 place-stable；
4. 继续使用空回忆中心化，保证 PFC endpoint 不被额外容量污染；
5. 在全新 `val[160:192]` 上重复 paired observed-only 验证；
6. 若仍未达到 `1% + CI>0`，再比较 64/128/256 value 维度或非线性 value encoder，不再回头调 gate。

这一步回答的是一个干净问题：**HPC 的稳定 neural content code 联合对齐到 PFC 后，是否终于能产生可复现的增量？**
