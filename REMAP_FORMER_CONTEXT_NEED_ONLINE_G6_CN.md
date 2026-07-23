# ReMAP-Former：G6 因果在线 Context 调用

> G5 score、0.99999 recurrence gate、数值阈值和 72-step observation horizon 全部冻结；唯一变化是第一次越阈值时立即调用。

## 冻结设置

- G5 threshold：`0.076884567737579346`
- protocol SHA256：`ce7df2158b8e078e1be04813f5a28f1f245f63048abae30480f64b5e05126abc`
- fresh family seed：`4071701`
- 调用：当前 sensory 已观察、null read-before-write 矛盾越阈值后，回放 0..t；从 t+1 起使用 dynamic HPC。
- 无 room/task/position/mask 输入；无新参数、slot 或第二套 fast weights。

## Pooled 结果

| BAcc | AUROC | T1 4096 | T2 return | T2 clean | online passes | fixed G5 passes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9941 | 1.0000 | 0.9972 | 0.8420 | 0.9977 | 1.7963 | 2.0000 |

## Fresh family

| family | T1 null | T2 dynamic | trigger median | online/fixed passes | T1 4096 | T2 return | clean | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| grid7 | 1.0000 | 1.0000 | 61.0 | 1.8264/2.0000 | 1.0000 | 0.8047 | 1.0000 | PASS |
| grid11 | 1.0000 | 1.0000 | 61.0 | 1.8403/2.0000 | 1.0000 | 0.9453 | 0.9984 | PASS |
| grid15 | 0.9297 | 1.0000 | 61.0 | 1.8542/2.0000 | 0.9736 | 0.8438 | 0.9922 | PASS |
| query3 | 1.0000 | 1.0000 | 53.0 | 1.7569/2.0000 | 0.9998 | 0.8906 | 0.9953 | PASS |
| query2 | 1.0000 | 1.0000 | 41.0 | 1.6042/2.0000 | 0.9981 | 0.8281 | 1.0000 | PASS |
| sparse_conflict | 1.0000 | 1.0000 | 65.0 | 1.8958/2.0000 | 0.9993 | 0.6719 | 1.0000 | FAIL |

## 因果与等价审计

- online / fixed G5 decision agreement：`1.000000`
- final memory max abs：`1.788e-06`
- final covariance max abs：`3.815e-06`
- post-prefix logits max abs：`5.722e-06`
- T2 trigger median / range：`57.0` / `[41.0, 69.0]`
- T2 mean prefix-pass saving：`0.2037`
- future writes：`0`

## 冻结结论

**`CONTEXT_NEED_ONLINE_G6_FAIL`**

至少一个冻结 gate 未通过：保留 G5 fixed-step controller，不在 G6 数据上修改 score、similarity、threshold、family、checkpoint 或 seeds。失败项：sparse_conflict.t2_return

边界：G6 仍使用最长 72 步 observation budget；它证明可在证据首次出现时提前调用，不等于已经证明任意未知 horizon 下都能稳定运行。dense downstream ceiling 也未在本实验处理。

失败 gates：`sparse_conflict.t2_return`

## 冻结失败后的 G6b 稳定性归因

G6 判 FAIL 后另冻四个 canonical overlap 为 0 的 sparse banks；没有修改 controller、threshold、horizon、checkpoint 或 G6 artifact。8 checkpoints x 4 banks x 32 episodes 共给出 1024 个 return-conflict probes：pooled `828/1024 = 0.8086`，Wilson 95% CI `[0.7834,0.8315]`，四个 bank 分别为 `0.8047/0.7656/0.7734/0.8906`，29/29 stability gates 全绿。

预注册分类为 `CONTEXT_NEED_ONLINE_G6B_SPARSE_STABLE`：G6 唯一红门属于 single-bank tail，而非 online/fixed 状态差异。**G6 的正式状态仍是 92/93 FAIL，不能被 G6b 覆盖。**完整归因见 `reports/REMAP_FORMER_CONTEXT_NEED_ONLINE_G6B_SPARSE_STABILITY_CN.md`。
