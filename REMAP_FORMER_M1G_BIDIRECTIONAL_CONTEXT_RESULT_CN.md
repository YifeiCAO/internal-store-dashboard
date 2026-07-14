# M1g 双向 HPC Context 恢复：冻结结论

> 2026-07-14；单 seed、两组互不重叠的全新 K=8 dev episodes；冻结 M1f、EC、place、content HPC 与融合层；无训练、无 checkpoint 选择；未访问 stress、formal validation 或 test。

## 问题

K=8 oracle 分解已经证明：M1f 正常 return-conflict 为 `0.4010`，换成正确历史 context 后为 `0.9844`，精确地址的 raw HPC 为 `0.9896`。因此本轮只检验一个问题：能否不增加 slot、context bank 或第二套记忆，直接从现有 episode-local content HPC 反向恢复 context？

现有地址为 `k_t = vec(p_t c_t^T)`，内容记忆为 `M_t in R^(d_value x d_address)`。所有反向 arm 都严格复用原 covariance-corrected HPC 的先读后写轨迹；`prior_context[t]` 只允许使用 `x_<t`，当前 `x_t` 只能进入诊断用 posterior。

## G0：转置相关回投

第一种方法把当前已揭示 value 投回地址空间：

`a_t^rev = M_t^T v_t`

再按当前 neural place 收缩出 context，并可选用已有 context covariance 做预条件。它是双向联想的最小实现，没有新增可学习参数。

| 条件 | Context pair acc | Correct cosine | Wrong cosine | Margin |
|---|---:|---:|---:|---:|
| 正常 M1f | 0.9375 | 0.8568 | 0.6480 | 0.2087 |
| Covariance causal reverse | 0.6094 | 0.8835 | 0.8513 | 0.0322 |
| Raw causal reverse | 0.5469 | 0.1761 | 0.1382 | 0.0379 |
| Same-step leaky | 0.6250 | 0.8878 | 0.8458 | 0.0421 |
| Shuffled-value | 0.5781 | 0.4388 | 0.4213 | 0.0175 |

冻结分类为 `REVERSE_CONTEXT_SIGNAL_INSUFFICIENT`。转置回投同时对正确和错误历史 context 给出很高相似度，得到的是相关混合方向，不是可辨识的逆映射。回放最大误差为 `0`，排除实现偏差。

## G0b：滚动最小二乘反解

第二种方法显式构造当前 pre-write memory 在给定 place 下从 context 到 value 的线性算子 `R_t`，并在最近 12 个已揭示观测上求解：

`c_hat = argmin_c sum ||R_t c - v_t||^2 + lambda ||c||^2`

该方法仍只使用原 content fast weights 与 covariance，不增加 context memory 或可学习参数。G0b 使用不同的新 dev seed，避免在 G0 episodes 上追结果。

| 条件 | Context pair acc | Correct cosine | Wrong cosine | Margin |
|---|---:|---:|---:|---:|
| 正常 M1f | 0.9063 | 0.8672 | 0.6612 | 0.2061 |
| LS causal | 0.4844 | 0.8723 | 0.8376 | 0.0347 |
| LS same-step leaky | 0.6406 | 0.9085 | 0.8074 | 0.1011 |
| LS shuffled-value | 0.5625 | 0.4890 | 0.4346 | 0.0544 |

- Causal 相对 shuffled pair accuracy：`-0.0781`。
- Same-step leaky 相对 causal：`+0.1563`，说明一部分可辨识证据只在当前 target 揭示后出现，不能用于当前预测。
- 仅在 audit return probe 换入 causal LS context 后，fused return-conflict 为 `0.5469 < 0.80`。
- Forward retrieval replay 最大误差仍为 `0`。

冻结分类为 `LEAST_SQUARES_CONTEXT_SIGNAL_INSUFFICIENT`，八个 gate 仅 exact replay 通过。按照预注册规则，不解锁 deployable M1g，也不铺多 seed。

## 结论

1. 现有 covariance HPC 是健康的内容 fast-weight memory，但它不是可逆的 context 编码器。不同房间写入共享 value 子空间后，`M^T v` 与最小二乘都会恢复出两个 context 的共同成分，而非足够尖锐的历史身份。
2. 失败不否定“Transformer 主干 + 外挂 HPC”。它精确定位出职责边界：HPC 已能在给定正确地址时读出内容；当前缺的是 PFC 对自身历史 context 的因果调用。
3. 不再继续调 reverse window、ridge 或阈值。G0 与 G0b 已覆盖相关回投和显式线性反演两类零参数方案；继续扫超参数会变成在单 seed dev 上追噪声。
4. 下一候选必须属于 Transformer/PFC 本身：用可观察动作历史形成 query，对过去 PFC token/context 做因果 attention，并把被调用的历史向量用于同一个 conjunctive address。不得增加固定 context slots、room embedding、第二套 fast-weight memory 或 metadata 输入。

## 可复现资产

- `remap_former/reverse_context.py`
- `test_remap_former_reverse_context.py`
- `evaluate_remap_m1g_reverse_context_g0.py`
- `evaluate_remap_m1g_least_squares_g0b.py`
- `runs/remap_former/m1g_reverse_context_g0_protocol.json`
- `runs/remap_former/m1g_reverse_context_g0/summary.json`
- `runs/remap_former/m1g_least_squares_g0b_protocol.json`
- `runs/remap_former/m1g_least_squares_g0b/summary.json`

完整 ReMAP 回归：`98 passed`。
