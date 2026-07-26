# ReMAP-Former 2D-R3 地址形式单种子 Stage Gate

> 状态：**STOP**。这是 development pilot，不是 fresh sealed test。基础 PFC/EC checkpoint 全冻结；只训练地址适配器，且不做 checkpoint selection。

## 公平合同

- 五种形式统一输出 `128D` key。
- 每个 episode 的动态状态为 `8192` 个 content fast-weight 标量加 `256` 个 context-covariance 标量，共 `8448` 个。
- 全部使用同一个 latent-context covariance dual-write memory，ridge=`0.03`；每种地址都以自己的 `f(place, dual_context)` 形成写 key，没有私有 memory backend。
- 所有条件严格 read-before-write，episode 开始时 memory 从零初始化。
- place 来自动作驱动 EC；context 来自动作历史 PFC。normal 条件不输入 room/context/位置/place/segment ID。
- correct/wrong context 只属于评价后诊断，用于把 context inference 与地址几何分开。

## Return-Conflict

表内曲线顺序均为 `K=[1, 2, 4, 8]`；`K` 是中间新奇 context 数量。

| 地址形式 | normal K 曲线 | correct-context K 曲线 | correct-context log-K AUC | 可学参数 |
|---|---|---|---:|---:|
| Place-only | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 ± 0.000 | 4,096 |
| Additive | 0.000 / 0.000 / 0.000 / 0.000 | 0.109 / 0.062 / 0.016 / 0.016 | 0.047 ± 0.000 | 6,144 |
| Concatenation | 0.094 / 0.000 / 0.031 / 0.016 | 0.094 / 0.000 / 0.000 / 0.031 | 0.021 ± 0.000 | 6,144 |
| Learned MLP | 0.031 / 0.000 / 0.016 / 0.000 | 0.062 / 0.016 / 0.000 / 0.016 | 0.018 ± 0.000 | 22,528 |
| Multiplicative | 0.109 / 0.031 / 0.047 / 0.047 | 0.141 / 0.094 / 0.094 / 0.031 | 0.091 ± 0.000 | 65,536 |

## K=1 地址几何

| 地址形式 | same-place / cross-context cosine | same-context / cross-place cosine | Gram |offdiag| | effective rank |
|---|---:|---:|---:|---:|
| Place-only | 1.000 | -0.002 | 0.215 | 4.00 |
| Additive | 0.910 | 0.190 | 0.302 | 4.71 |
| Concatenation | 0.916 | 0.173 | 0.342 | 4.40 |
| Learned MLP | 0.939 | 0.123 | 0.321 | 4.38 |
| Multiplicative | 0.843 | -0.003 | 0.311 | 5.39 |

## Multiplicative 成对 AUC 差值

| contrast | mean ± SD | 最差种子 |
|---|---:|---:|
| multiplicative_minus_place_only_correct_context_auc | 0.091 ± 0.000 | 0.091 |
| multiplicative_minus_additive_correct_context_auc | 0.044 ± 0.000 | 0.044 |
| multiplicative_minus_concatenation_correct_context_auc | 0.070 ± 0.000 | 0.070 |
| multiplicative_minus_learned_mlp_correct_context_auc | 0.073 ± 0.000 | 0.073 |

## Gates

- place_only_correct_context_k1_at_most_060: `True`
- multiplicative_correct_context_k1_at_least_070: `False`
- multiplicative_auc_beats_place_only_by_015: `False`
- multiplicative_clean_min_at_least_090: `False`
- causal_audits_all_green: `True`

## 解释规则

- 若 learned MLP 与 multiplicative 持平：只支持 **context-dependent conjunctive addressing**，不宣称 outer product 唯一必要。
- 若 additive/concatenation 也持平：应进一步降级为一般 context conditioning。
- additive 与 concatenation 在无 bias 的线性实现中函数类等价；保留两行是为了暴露这一点，而不是制造两个独立理论。
- 本 stage gate 仍把原始 `512D` place×context 外积压缩到统一 `128D` key；共享 covariance 后 health gate 仍失败，因此按协议停止该代理比较并转向 full-model address intervention。这个结果不能否定 outer-product。
