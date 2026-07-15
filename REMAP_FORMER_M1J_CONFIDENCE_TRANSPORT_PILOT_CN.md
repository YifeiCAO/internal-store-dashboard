# M1j：Confidence-conditioned Context Transport K=8 盲测

> 固定 step 600；全新 generator seed16180；M1i caller、Transformer/PFC、EC/place、唯一 covariance content-HPC 全冻结；只训练 77 个 transport 参数，目标仍只有全 token sensory CE。

| 条件 | Return-conflict | Clean | Context pair | Context margin |
|---|---:|---:|---:|---:|
| 冻结 M1f | 0.3750 | 0.9305 | 0.7969 | +0.1574 |
| 来源 M1i | 0.7344 | 0.9664 | 0.8906 | +0.1403 |
| M1j 置信度条件 Transport | 0.0781 | 0.9477 | 0.5469 | +0.0023 |
| M1j 禁用 Transport residual | 0.7344 | 0.9664 | 0.8906 | +0.1403 |

## 冻结七门

- M1j return-conflict：`0.0781`。
- M1j - source M1i：`-0.6562`。
- M1j - disabled：`-0.6562`。
- M1j clean drop：`+0.0188`。
- M1j context pair：`0.5469`。
- Disabled prediction/context max diff：`0 / 0`。

- `m1j_return_absolute`：`FAIL`
- `m1j_gain_vs_source`：`FAIL`
- `transport_functionally_necessary`：`FAIL`
- `m1j_clean_preserved`：`PASS`
- `m1j_context_identity`：`FAIL`
- `disabled_prediction_exact`：`PASS`
- `disabled_context_exact`：`PASS`

冻结分类：`M1J_CONFIDENCE_TRANSPORT_PILOT_REJECTED`。

## Transport 动态

- 来源 call probability / transport strength：`0.442 / 0.089`。
- Logit shift mean / abs mean：`-2.533 / 2.533`。
- Strength absolute delta：`0.352`。
- Positive / negative shift rate：`0.000 / 1.000`。

本 pilot 不读取 pfc_hidden 或 native Transformer KV，不增加 slot、第二套 fast weights、hard threshold 或辅助 loss。任一门失败即停止，不在 seed16180 上调参。
