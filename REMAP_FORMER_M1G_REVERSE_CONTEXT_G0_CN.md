# M1g G0：反向 HPC Context 信号门

> 单 seed、全新 K=8 dev episodes；冻结 M1f 与 content HPC；无训练、无 checkpoint 选择；未访问 stress、formal validation 或 test。

## Return-Conflict Context 身份

| 条件 | Pair accuracy | Correct cosine | Wrong cosine | Margin | Evidence strength |
|---|---:|---:|---:|---:|---:|
| Normal M1f context | 0.9375 | +0.8568 | +0.6480 | +0.2087 | 1.0000 |
| Reverse covariance causal | 0.6094 | +0.8835 | +0.8513 | +0.0322 | 2.7664 |
| Reverse raw causal | 0.5469 | +0.1761 | +0.1382 | +0.0379 | 51.4024 |
| Same-step leaky ceiling | 0.6250 | +0.8878 | +0.8458 | +0.0421 | 2.6547 |
| Shuffled-value control | 0.5781 | +0.4388 | +0.4213 | +0.0175 | 0.2481 |

## 冻结 Gates

- Causal - normal pair accuracy：`-0.3281`
- Causal - shuffled-value：`+0.0312`
- Same-step leaky - causal：`+0.0156`
- HPC replay max abs error：`0.000e+00`

冻结分类：`REVERSE_CONTEXT_SIGNAL_INSUFFICIENT`。

主 causal context 在 t 时刻只包含 x_<t；same-step arm 会使用 x_t，因此只作为泄漏上限，不是候选模型。Shuffled-value 只打乱反向查询，正常 content 写入和地址保持不变。
