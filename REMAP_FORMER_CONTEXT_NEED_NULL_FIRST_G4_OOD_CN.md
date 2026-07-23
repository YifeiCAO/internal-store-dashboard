# ReMAP-Former：Null-First G4 OOD Task Families

> G3 score、72-step prefix、top-10% tail 和数值阈值全部冻结；G4 不做 calibration，逐 family 检验 task-distribution transfer。

## Frozen Design

- fixed G3 threshold：`0.18923524`
- families：`7`
- protocol SHA256：`d107faca60e1f6f4a2b4fc04f91507c0d3b827de9b49306be86cde79ab5102b6`
- controller score/T1 只跑共享 controller-facing pathway；T2 dynamic branch 跑全部 8 套 context checkpoint

## Pooled

| BAcc | AUROC | T1 4096 | T1 passes | T2 return | T2 clean |
|---:|---:|---:|---:|---:|---:|
| 0.8231 | 0.9351 | 0.9514 | 1.1429 | 0.6317 | 0.9984 |

## Family-wise

| family | T1 null | T2 dynamic | BAcc | AUC | T1 4096 | passes | T2 return | clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_paths | 0.9297 | 1.0000 | 0.9648 | 1.0000 | 1.0000 | 1.0000 | 0.7734 | 0.9969 |
| new_path_bank | 0.9766 | 1.0000 | 0.9883 | 1.0000 | 1.0000 | 1.0000 | 0.8047 | 0.9984 |
| grid9 | 0.9688 | 1.0000 | 0.9844 | 1.0000 | 0.9975 | 1.0000 | 0.8281 | 1.0000 |
| grid13 | 0.4297 | 1.0000 | 0.7148 | 0.7618 | 0.7648 | 1.6250 | 0.7969 | 0.9953 |
| query2 | 0.9453 | 0.3281 | 0.6367 | 0.9214 | 0.9174 | 1.1875 | 0.2500 | 1.0000 |
| sparse_conflict | 0.9688 | 0.0156 | 0.4922 | 0.9198 | 0.9688 | 1.0625 | 0.0000 | 1.0000 |
| dense_conflict | 0.9609 | 1.0000 | 0.9805 | 1.0000 | 0.9505 | 1.1250 | 0.5469 | N/A |

## Gates

- `PASS` `test_paths.generator`
- `PASS` `test_paths.t1_null`
- `PASS` `test_paths.t2_dynamic`
- `PASS` `test_paths.balanced_accuracy`
- `PASS` `test_paths.auroc`
- `PASS` `test_paths.t1_final`
- `PASS` `test_paths.t1_efficiency`
- `PASS` `test_paths.t2_return`
- `PASS` `test_paths.t2_clean`
- `PASS` `new_path_bank.generator`
- `PASS` `new_path_bank.t1_null`
- `PASS` `new_path_bank.t2_dynamic`
- `PASS` `new_path_bank.balanced_accuracy`
- `PASS` `new_path_bank.auroc`
- `PASS` `new_path_bank.t1_final`
- `PASS` `new_path_bank.t1_efficiency`
- `PASS` `new_path_bank.t2_return`
- `PASS` `new_path_bank.t2_clean`
- `PASS` `grid9.generator`
- `PASS` `grid9.t1_null`
- `PASS` `grid9.t2_dynamic`
- `PASS` `grid9.balanced_accuracy`
- `PASS` `grid9.auroc`
- `PASS` `grid9.t1_final`
- `PASS` `grid9.t1_efficiency`
- `PASS` `grid9.t2_return`
- `PASS` `grid9.t2_clean`
- `PASS` `grid13.generator`
- `FAIL` `grid13.t1_null`
- `PASS` `grid13.t2_dynamic`
- `FAIL` `grid13.balanced_accuracy`
- `FAIL` `grid13.auroc`
- `FAIL` `grid13.t1_final`
- `FAIL` `grid13.t1_efficiency`
- `PASS` `grid13.t2_return`
- `PASS` `grid13.t2_clean`
- `PASS` `query2.generator`
- `PASS` `query2.t1_null`
- `FAIL` `query2.t2_dynamic`
- `FAIL` `query2.balanced_accuracy`
- `FAIL` `query2.auroc`
- `PASS` `query2.t1_final`
- `PASS` `query2.t1_efficiency`
- `FAIL` `query2.t2_return`
- `PASS` `query2.t2_clean`
- `PASS` `sparse_conflict.generator`
- `PASS` `sparse_conflict.t1_null`
- `FAIL` `sparse_conflict.t2_dynamic`
- `FAIL` `sparse_conflict.balanced_accuracy`
- `FAIL` `sparse_conflict.auroc`
- `PASS` `sparse_conflict.t1_final`
- `PASS` `sparse_conflict.t1_efficiency`
- `FAIL` `sparse_conflict.t2_return`
- `PASS` `sparse_conflict.t2_clean`
- `PASS` `dense_conflict.generator`
- `PASS` `dense_conflict.t1_null`
- `PASS` `dense_conflict.t2_dynamic`
- `PASS` `dense_conflict.balanced_accuracy`
- `PASS` `dense_conflict.auroc`
- `PASS` `dense_conflict.t1_final`
- `PASS` `dense_conflict.t1_efficiency`
- `FAIL` `dense_conflict.t2_return`
- `PASS` `dense_conflict.t2_clean`
- `FAIL` `pooled.balanced_accuracy`
- `FAIL` `pooled.auroc`
- `PASS` `pooled.t1_final`
- `FAIL` `pooled.t1_efficiency`
- `FAIL` `pooled.t2_return`
- `PASS` `pooled.t2_clean`
- `PASS` `null_selected_replayed_zero`
- `PASS` `future_write_steps_zero`
- `PASS` `all_source_model_states_unchanged`
- `PASS` `one_hpc_per_checkpoint`

## Decision

**`CONTEXT_NEED_NULL_FIRST_G4_OOD_FAIL`**

冻结 OOD 失败；保留失败 family，不删除 family，也不修改 threshold、prefix、tail、score、checkpoint 或 seeds。失败项：dense_conflict, grid13, query2, sparse_conflict

## Frozen Endpoint Attribution

失败后另冻协议，在完全相同轨迹上强制 fixed-null / dynamic；没有调整 G4：

| family | selected | correct endpoint | endpoint result | attribution |
|---|---:|---|---:|---|
| grid13 T1 | `0.7648` | fixed null | `0.9793` | controller false-positive |
| query2 T2 return | `0.2500` | dynamic | `0.7500` | controller false-negative |
| sparse T2 return | `0.0000` | dynamic | `0.8281` | controller false-negative |
| dense T2 return | `0.5469`，call=1.0 | dynamic | `0.5469` | downstream capacity confirmed |

结论：raw top-10% contradiction 同时受网格重访密度和冲突事件数影响；dense 的失败则不属于 controller。端点审计：`reports/REMAP_FORMER_CONTEXT_NEED_G4_FAILURE_ENDPOINT_AUDIT_CN.md`。
