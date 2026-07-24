# MemoryMaze3D Baseline Catch-up Stage H

- 协议状态：`frozen_before_results`
- 完成：`5/8`
- 已终止：`8/8`
- 全部健康门：`FAIL`
- test access：`forbidden`
- shared visual tensor SHA256：`2625367f0ebcb3c86ffe25fe0253059158e27cf7dfbf72ce54fe75a315bbb8a8`

| 模型 | LR | Dev teacher MSE | 可训练参数 | 秒 | Visual 相同 | 健康门 |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 2.0e-04 | 0.005064 | 2,714,240 | 45.5 | 是 | PASS |
| transformer | 5.0e-04 | 0.004709 | 2,714,240 | 42.1 | 是 | PASS |
| titans | 2.0e-04 | 0.005067 | 3,035,588 | 320.9 | 是 | PASS |
| titans | 5.0e-04 | 0.004683 | 3,035,588 | 297.5 | 是 | PASS |
| mmtem | 2.0e-04 | failed_nonfinite@191 | - | - | - | FAIL |
| mmtem | 5.0e-04 | failed_nonfinite@80 | - | - | - | FAIL |
| hippoformer | 2.0e-04 | 0.004316 | 9,855,561 | 1176.7 | 是 | PASS |
| hippoformer | 5.0e-04 | failed_nonfinite@90 | - | - | - | FAIL |

## 当前选择

- `transformer`：LR `5.0e-04`，dev teacher MSE `0.004709`，健康门 `PASS`
- `titans`：LR `5.0e-04`，dev teacher MSE `0.004683`，健康门 `PASS`
- `mmtem`：尚无完整结果
- `hippoformer`：LR `2.0e-04`，dev teacher MSE `0.004316`，健康门 `PASS`

本表属于 adaptive development，不是正式 validation/test headline。
