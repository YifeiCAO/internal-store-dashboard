# M1g G0b：连续 Context 最小二乘逆推断

> G0 transpose 被拒绝后的全新单 seed K=8 dev episodes；冻结 M1f/content HPC；无训练；未访问 stress、formal validation 或 test。

| 条件 | Pair accuracy | Correct cosine | Wrong cosine | Margin | Strength |
|---|---:|---:|---:|---:|---:|
| Normal M1f context | 0.9062 | +0.8672 | +0.6612 | +0.2061 | 1.0000 |
| Least-squares causal | 0.4844 | +0.8723 | +0.8376 | +0.0347 | 55.3195 |
| Same-step leaky ceiling | 0.6406 | +0.9085 | +0.8074 | +0.1011 | 54.3470 |
| Shuffled-value control | 0.5625 | +0.4890 | +0.4346 | +0.0544 | 9.9903 |

## 直接功能上限

- 仅在 audit return probes 替换为 causal least-squares context 后的 fused return-conflict：`0.5469`。
- Causal correct cosine - normal：`+0.0050`。
- Causal - shuffled pair accuracy：`-0.0781`。
- Same-step leaky - causal：`+0.1562`。
- Replay max abs error：`0.000e+00`。

冻结分类：`LEAST_SQUARES_CONTEXT_SIGNAL_INSUFFICIENT`。

Return-probe override 使用 metadata 选择干预位置，因此只判断连续 context 解是否达到地址精度，不是可部署模型。主 causal 解仍严格只使用 x_<t。
