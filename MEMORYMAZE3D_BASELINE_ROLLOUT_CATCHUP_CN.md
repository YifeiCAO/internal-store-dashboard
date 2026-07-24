# MemoryMaze3D Baseline Rollout Catch-up

- 协议：`frozen_before_rollout_results`
- gate 勘误：`frozen_after_transformer_r1_before_titans_and_hippoformer_r1`
- R1/R2 完成：`12/12`
- test access：`forbidden`

## R0 严格起点

| 模型 | Teacher MSE | C20-H44 MSE | 相对 persistence | 动作优势 |
|---|---:|---:|---:|---:|
| transformer | 0.005250 | 0.055807 | -38.9% | 0.022127 |
| titans | 0.005235 | 0.044593 | -11.0% | 0.034023 |
| hippoformer | 0.004967 | 0.042423 | -5.6% | 0.016264 |
| m1b | 0.003236 | 0.020073 | +50.1% | 0.027491 |

## R1/R2 候选

| 模型 | LR | rollout w | H16 gate | C20-H44 MSE | 相对 persistence | 动作优势 |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 5.0e-05 | 0.5 | PASS | 0.024015 | +40.2% | 0.004194 |
| transformer | 5.0e-05 | 1.0 | PASS | 0.022994 | +42.8% | 0.004210 |
| transformer | 1.0e-04 | 0.5 | PASS | 0.023427 | +41.7% | 0.002492 |
| transformer | 1.0e-04 | 1.0 | PASS | 0.024683 | +38.6% | 0.000693 |
| titans | 5.0e-05 | 0.5 | PASS | 0.023252 | +42.1% | 0.019240 |
| titans | 5.0e-05 | 1.0 | PASS | 0.021891 | +45.5% | 0.019408 |
| titans | 1.0e-04 | 0.5 | PASS | 0.024817 | +38.2% | 0.027359 |
| titans | 1.0e-04 | 1.0 | PASS | 0.024680 | +38.6% | 0.008576 |
| hippoformer | 5.0e-05 | 0.5 | PASS | 0.026347 | +34.4% | 0.004288 |
| hippoformer | 5.0e-05 | 1.0 | PASS | 0.023302 | +42.0% | 0.008754 |
| hippoformer | 1.0e-04 | 0.5 | PASS | 0.025863 | +35.6% | 0.005412 |
| hippoformer | 1.0e-04 | 1.0 | PASS | 0.029198 | +27.3% | -0.001264 |

## 当前选择

- `transformer`：LR `5.0e-05`，rollout weight `1.0`，C20-H44 MSE `0.022994`，动作优势 `0.004210`。
- `titans`：LR `5.0e-05`，rollout weight `1.0`，C20-H44 MSE `0.021891`，动作优势 `0.019408`。
- `hippoformer`：LR `5.0e-05`，rollout weight `1.0`，C20-H44 MSE `0.023302`，动作优势 `0.008754`。

M1b 只作当前项目参考：其累计祖先训练预算不匹配。
以上全是 adaptive development；test 未访问。
