# M1j G3：梯度信用归因只读审计

> 全新 dev seed17181；step-0 source-equivalent M1j；K=8、64 episodes、128 return-conflict probes。只用 `torch.autograd.grad` 读取梯度，无 optimizer、无 backward 累积、无参数修改；metadata 只在 forward 后切 loss。

## 审计对象

M1j 的 transport 最后一层在 step 0 精确为零，因此 77 个可训练参数中，只有最后 `Linear(8,1)` 的 9 个参数有一阶梯度。下面比较的是同一次 causal forward 上，三种 sensory CE 对这同一 9 维首次更新方向的影响。正的 event direction 表示一次单位梯度下降会提高该事件的 transport logit shift，负值表示会压低调用。

## 聚合梯度

| Loss 视图 | Token 数 | CE | 梯度范数 | Bias 的单位下降方向 | Return 段正向率 | Return 段负向率 | Return 段均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全 token CE | 36400 | 1.7394 | 0.348837 | -0.8078 | 0.0000 | 1.0000 | -3.6743 |
| clean-query CE | 2560 | 0.1690 | 0.082317 | -0.5686 | 0.0000 | 1.0000 | -3.2715 |
| return-conflict CE | 128 | 1.9885 | 1.591386 | 0.7883 | 1.0000 | 0.0000 | 3.5055 |

## 梯度夹角

| 对比 | 聚合 cosine | 16 batch 平均 | 16 batch 中负 cosine 比例 |
|---|---:|---:|---:|
| all-token vs return-conflict | -0.9579 | -0.1277 | 0.5625 |
| clean-query vs return-conflict | -0.8623 | 0.4262 | 0.3125 |
| all-token vs clean-query | 0.9124 | 0.1539 | 0.4375 |
| all-token vs return（仅 8 个条件权重，不含 bias） | -0.8854 | -0.0749 | 0.5000 |

## 实现完整性

- 77 参数 / 9 维审计门：`True` / `True`。
- Step-0 prediction/context 与 source M1i 最大差：`0.000e+00` / `0.000e+00`。
- 其他 68 参数最大梯度：`0.000e+00`。
- 参数状态哈希前后一致：`True`。
- History events / final-return-segment events：`3718` / `286`。

## 冻结判读

- 状态：`MIXED_GRADIENT_CREDIT`。
- 下一独立协议：`DO_NOT_TRAIN_YET_AND_DESIGN_A_HIGHER_RESOLUTION_CAUSAL_CREDIT_AUDIT`。

| 冻结判断门 | 结果 |
|---|---:|
| `implementation_valid` | True |
| `return_gradient_nonzero` | True |
| `all_token_gradient_nonzero` | True |
| `return_ce_increases_return_phase_transport` | True |
| `all_token_ce_suppresses_return_phase_transport` | True |
| `all_vs_return_gradients_stably_antialigned` | False |
| `return_ce_itself_suppresses_transport` | False |
| `uniform_ce_credit_conflict` | False |

## 含义

梯度关系没有达到预注册的稳定冲突或 transport-rejection 判据。暂不训练新 loss，先设计更高分辨率的因果事件审计。

本结果不允许在 seed17181 上调阈值、换 mask 或训练；正式模型仍是已冻结通过的 M1b。
