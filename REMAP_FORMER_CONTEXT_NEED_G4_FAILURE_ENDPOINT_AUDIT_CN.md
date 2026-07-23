# ReMAP-Former：G4 Failure Endpoint Audit

> 在 G4 原轨迹上强制 null/dynamic，只做失败归因；不重调 threshold、score、prefix、tail 或 family。

## Grid13 T1

| selected | fixed null | forced dynamic | null restoration | classification |
|---:|---:|---:|---:|---|
| 0.7648 | 0.9793 | 0.6551 | +0.2145 | CONTROLLER_FALSE_POSITIVE |

## T2 Failed Families

| family | selected call | selected return | fixed-null return | forced-dynamic return | dynamic gain | classification |
|---|---:|---:|---:|---:|---:|---|
| query2 | 0.3281 | 0.2500 | 0.0000 | 0.7500 | +0.5000 | CONTROLLER_FALSE_NEGATIVE |
| sparse_conflict | 0.0156 | 0.0000 | 0.0000 | 0.8281 | +0.8281 | CONTROLLER_FALSE_NEGATIVE |
| dense_conflict | 1.0000 | 0.5469 | 0.0000 | 0.5469 | +0.0000 | DOWNSTREAM_CONFIRMED |

## Implementation Gates

- `PASS` `fixed_null_dynamic_rate_zero`
- `PASS` `forced_dynamic_rate_one`
- `PASS` `grid13_future_write_steps_zero`
- `PASS` `dense_forced_dynamic_exact_selected`
- `PASS` `all_source_model_states_unchanged`
- `PASS` `one_hpc_per_checkpoint`

## Decision

**`CONTEXT_NEED_G4_FAILURE_ENDPOINT_AUDIT_COMPLETE`**

grid13=CONTROLLER_FALSE_POSITIVE; query2=CONTROLLER_FALSE_NEGATIVE; sparse_conflict=CONTROLLER_FALSE_NEGATIVE; dense_conflict=DOWNSTREAM_CONFIRMED. G4 仍保持 FAIL，不从端点结果回调 controller。
