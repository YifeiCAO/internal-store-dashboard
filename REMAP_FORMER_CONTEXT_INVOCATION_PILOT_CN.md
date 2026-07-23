# ReMAP-Former：Context Invocation 单 Seed Pilot

> 架构只增加 345 个 action-only 二值调用参数；正式 M1b、EC/place、单一 covariance HPC 与 decoder 全冻结。训练只有一个 mixed T1/T2 all-token sensory CE。

## Pilot 结果

| 条件 | T1 strict 4096 | T1 invocation | T2 return-conflict | T2 clean |
|---|---:|---:|---:|---:|
| 初始 M1b 等价 | - | 1.0000 | 1.0000 | 1.0000 |
| 固定 step600 | 0.6838 | 1.0000 | 0.7969 | 1.0000 |

决策：`CONTEXT_INVOCATION_SINGLE_SEED_FAIL`。T2 保持 source dynamic，但 gate 学成 all-on，T1 未恢复。base state hash 不变，future writes 为 0。

## 8-Seed 信用方向审计

| seed | T1 dynamic CE | T1 fixed CE | T1 bias grad | 端点/梯度反向 | T2 dynamic CE | T2 fixed CE |
|---:|---:|---:|---:|---:|---:|---:|
| 2472102 | 1.5694 | 1.2681 | -0.001178 | True | 0.6562 | 0.4912 |
| 2472103 | 1.6148 | 1.4103 | -0.000535 | True | 0.5195 | 0.5319 |
| 2472104 | 1.8643 | 1.7172 | -0.000531 | True | 0.4667 | 0.5664 |
| 2472105 | 1.7290 | 1.4642 | -0.001049 | True | 0.6135 | 0.6119 |
| 2472106 | 1.4841 | 1.1132 | -0.001950 | True | 0.6054 | 0.6305 |
| 2472107 | 1.4587 | 1.2094 | -0.001155 | True | 0.5498 | 0.5494 |
| 2472108 | 1.9502 | 1.6487 | -0.000781 | True | 0.5887 | 0.5321 |
| 2472109 | 1.9052 | 1.5408 | -0.001023 | True | 0.4821 | 0.6111 |

## Aggregate

- T1 fixed endpoint 更优比例：`1.000`
- T1 fixed 更优但 ST 梯度仍推向 invocation 的比例：`1.000`
- T2 dynamic endpoint 更优比例：`0.500`
- 初始 T1/T2 gate-gradient cosine：`+0.7218 +/- 0.2311`

## Gates

- initial_t1_fixed_endpoint_better: `True`
- initial_t1_endpoint_gradient_mismatch: `True`
- initial_t2_dynamic_endpoint_better: `False`
- all_model_state_hashes_unchanged: `True`

## 结论

预设信用错位门未全部通过，不能据此更换 objective。

决策：`CONTEXT_INVOCATION_ST_SURROGATE_MISMATCH_NOT_CONFIRMED`
