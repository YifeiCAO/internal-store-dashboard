# ReMAP-Former：G7 Variable Observation Horizon

> G5 score/threshold 与 G6 first-crossing 状态机全部冻结；四个窗口使用同一批 paired trajectories。G6 的 92/93 FAIL 继续保留。

## 冻结设置

- horizons：`[32, 48, 60, 72]`
- recurrence cosine：`0.99999`
- numeric threshold：`0.076884567737579346`
- protocol SHA256：`c24a489cda3f5c6b449d6e0dc9732ae99872a0f15892c787f1e637d08a9be255`
- fresh family seed：`4771701`
- 未在 cutoff 前 crossing 的 episode 保持 null；不使用 cutoff 后证据，不做未来倒灌。

## Smoke 修正披露

首次 CUDA smoke 为 `22/24 FAIL`，原始文件与 SHA256 均保留：`tmp/context_need_online_g7_variable_horizon_smoke.json`，`ff5885572876d1f42a8b9d88d4fc5d863f073568d26cd831df8e44d2d53db974`。失败项只有 `causal.h32/h48`：整段计算与截断计算的地址最大差为 `5.07e-7`，而 trigger、decision、memory、covariance 全部逐位一致。

正式运行前唯一修正是把地址 bitwise-zero 门明确为 `max abs <= 1e-6`；没有修改 horizon、score、similarity、threshold、family、checkpoint、seed 或性能门。修正后的 smoke 为 `24/24 PASS`，见 `tmp/context_need_online_g7_variable_horizon_smoke_v2.json`。

## Pooled Horizon 曲线

| H | T1 null | T2 dynamic | censored | BAcc/AUC | T1 4096 | T2 return | clean | online/fixed passes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.9987 | 0.0000 | 1.0000 | 0.4993/0.3436 | 0.9978 | 0.0000 | 1.0000 | 1.0000/1.0000 |
| 48 | 0.9961 | 0.1667 | 0.8333 | 0.5814/0.3295 | 0.9972 | 0.0938 | 1.0000 | 1.1510/1.1667 |
| 60 | 0.9948 | 0.6667 | 0.3333 | 0.8021/0.6718 | 0.9947 | 0.4948 | 0.9992 | 1.5958/1.6667 |
| 72 | 0.9896 | 1.0000 | 0.0000 | 0.9948/0.9987 | 0.9919 | 0.7656 | 0.9979 | 1.7986/2.0000 |

这条曲线不能被读成“短窗口模型退化”。H32 的 T1/clean 仍为 `0.9978/1.0000`，但所有 T2 都尚未看到合法 crossing，因此 dynamic coverage 为 `0`、return 为 `0`。H48 只覆盖最早出现证据的 query2 family；到 H60 覆盖 `2/3`，H72 才覆盖全部 T2。短窗 BAcc/AUC 低，明确说明 controller 不能在证据出现前预知未来 room conflict，这正是因果任务的可辨识边界。

在线成本也随证据出现自然增长：H32/H48/H60/H72 分别为 `1.0000/1.1510/1.5958/1.7986` passes；同窗口 fixed endpoint 为 `1.0000/1.1667/1.6667/2.0000`。

## 趋势与等价审计

- dynamic coverage range：`1.0000`
- return H72-H32：`+0.7656`
- horizon/return Spearman：`1.0000`
- minimum online/fixed decision agreement：`1.000000`
- maximum state/logit error：`4.768e-06`
- all nested decisions monotonic：`True`
- trajectory pairing exact：`True`

## H72 Fresh Family

| family | dynamic | T1 4096 | T2 return | clean | passes |
|---|---:|---:|---:|---:|---:|
| grid7 | 1.0000 | 1.0000 | 0.6641 | 0.9984 | 1.8333 |
| grid11 | 1.0000 | 0.9976 | 0.8281 | 0.9938 | 1.8611 |
| grid15 | 1.0000 | 0.9342 | 0.7500 | 1.0000 | 1.8333 |
| query3 | 1.0000 | 1.0000 | 0.7344 | 0.9969 | 1.7639 |
| query2 | 1.0000 | 0.9977 | 0.8438 | 1.0000 | 1.6042 |
| sparse_conflict | 1.0000 | 0.9921 | 0.8281 | 0.9990 | 1.8958 |

## 冻结结论

**`CONTEXT_NEED_ONLINE_G7_VARIABLE_HORIZON_PASS`**

全部冻结 gates 通过：first-crossing controller 在四个因果 cutoff 上形成可解释的 censoring/coverage/return 曲线，并保持逐窗口 fixed endpoint 等价。

边界：G7 测的是预注册的四个 cutoff，不等于任意未知 horizon；它不覆盖 G6 的 sparse formal FAIL，也不处理 dense downstream ceiling。正式全门通过基线仍按冻结账本单独报告。
