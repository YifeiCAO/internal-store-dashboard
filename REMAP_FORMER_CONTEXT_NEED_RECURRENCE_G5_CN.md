# ReMAP-Former：Recurrence-Gated Context-Need G5

> 只在 neural place/address 与严格更早地址几乎完全重合时，retrieval contradiction 才进入调用分数；单次 sparse 冲突不再被固定 top-k 稀释。

## Frozen Design

- prefix：`72`
- recurrence cosine：`>= 0.99999`
- aggregation：熟悉重访 contradiction 的 maximum
- calibration midpoint threshold：`0.07688457`
- protocol SHA256：`548bf54ce5a1fb6a7e001af84db672797e31c435a1d5c13c2e23f313c1e2fddc`

## Pooled

| BAcc | AUROC | T1 4096 | T1 passes | T2 return | T2 clean |
|---:|---:|---:|---:|---:|---:|
| 0.9941 | 1.0000 | 0.9935 | 1.0125 | 0.8080 | 0.9953 |

## Held-out Families

| family | T1 null | T2 dynamic | BAcc | AUC | recurrence T1/T2 | T1 4096 | passes | T2 return | clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 46.17/58.28 | 1.0000 | 1.0000 | 0.7891 | 0.9953 |
| grid15 | 0.9297 | 1.0000 | 0.9648 | 1.0000 | 36.52/58.67 | 0.9388 | 1.0625 | 0.7578 | 0.9984 |
| query3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 39.91/59.33 | 1.0000 | 1.0000 | 0.9062 | 0.9859 |
| query2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40.37/58.97 | 0.9963 | 1.0000 | 0.8594 | 1.0000 |
| sparse_conflict | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 39.99/58.30 | 1.0000 | 1.0000 | 0.7969 | 0.9979 |
| dense_selection_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40.27/58.31 | N/A | N/A | N/A | N/A |

## Gates

- `PASS` `calibration_direction`
- `PASS` `grid7.generator`
- `PASS` `grid7.t1_null`
- `PASS` `grid7.t2_dynamic`
- `PASS` `grid7.balanced_accuracy`
- `PASS` `grid7.auroc`
- `PASS` `grid7.t1_final`
- `PASS` `grid7.t1_efficiency`
- `PASS` `grid7.t2_return`
- `PASS` `grid7.t2_clean`
- `PASS` `grid15.generator`
- `PASS` `grid15.t1_null`
- `PASS` `grid15.t2_dynamic`
- `PASS` `grid15.balanced_accuracy`
- `PASS` `grid15.auroc`
- `PASS` `grid15.t1_final`
- `PASS` `grid15.t1_efficiency`
- `PASS` `grid15.t2_return`
- `PASS` `grid15.t2_clean`
- `PASS` `query3.generator`
- `PASS` `query3.t1_null`
- `PASS` `query3.t2_dynamic`
- `PASS` `query3.balanced_accuracy`
- `PASS` `query3.auroc`
- `PASS` `query3.t1_final`
- `PASS` `query3.t1_efficiency`
- `PASS` `query3.t2_return`
- `PASS` `query3.t2_clean`
- `PASS` `query2.generator`
- `PASS` `query2.t1_null`
- `PASS` `query2.t2_dynamic`
- `PASS` `query2.balanced_accuracy`
- `PASS` `query2.auroc`
- `PASS` `query2.t1_final`
- `PASS` `query2.t1_efficiency`
- `PASS` `query2.t2_return`
- `PASS` `query2.t2_clean`
- `PASS` `sparse_conflict.generator`
- `PASS` `sparse_conflict.t1_null`
- `PASS` `sparse_conflict.t2_dynamic`
- `PASS` `sparse_conflict.balanced_accuracy`
- `PASS` `sparse_conflict.auroc`
- `PASS` `sparse_conflict.t1_final`
- `PASS` `sparse_conflict.t1_efficiency`
- `PASS` `sparse_conflict.t2_return`
- `PASS` `sparse_conflict.t2_clean`
- `PASS` `dense_selection_only.generator`
- `PASS` `dense_selection_only.t1_null`
- `PASS` `dense_selection_only.t2_dynamic`
- `PASS` `dense_selection_only.balanced_accuracy`
- `PASS` `dense_selection_only.auroc`
- `PASS` `pooled.balanced_accuracy`
- `PASS` `pooled.auroc`
- `PASS` `pooled.t1_final`
- `PASS` `pooled.t1_efficiency`
- `PASS` `pooled.t2_return`
- `PASS` `pooled.t2_clean`
- `PASS` `null_selected_replayed_zero`
- `PASS` `future_write_steps_zero`
- `PASS` `all_source_states_unchanged`
- `PASS` `one_hpc_per_checkpoint`

## Decision

**`CONTEXT_NEED_RECURRENCE_G5_PASS`**

全部冻结 gate 通过：熟悉地址 contradiction 可以冻结，下一步研究 online decision timing。

## Interpretation and Boundary

- G5 不是在 G4 数据上移动阈值。机制、`0.99999` recurrence gate、maximum aggregation、fresh calibration bank 和零重叠 test bank 均在看 G5 分数前冻结。
- 它修复了 G4 的两类 controller 错误：大网格陌生地址噪声不再主导；`query2/sparse` 的单次同址错取不再被 top-8 平均稀释。
- `grid15` 仍是最弱 family：T1 null `0.9297`，4096 `0.9388`。因此不能声称所有 T1 都接近 1.0。
- dense family 只用于 selection gate，dynamic selection 为 `1.0000`；其已知 forced-dynamic return ceiling `0.5469` 是下游容量问题，不混入 controller 成败。
- 当前 decision time 仍固定在 step72。下一步应固定 G5 score/threshold，测试因果在线触发，而不是继续修改 controller 公式。
