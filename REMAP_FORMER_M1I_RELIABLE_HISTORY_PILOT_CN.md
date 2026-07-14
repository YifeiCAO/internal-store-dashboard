# M1i：可靠 PFC 历史调用 K=8 盲测

> 固定 step 600；全新 generator seed 9931；固定结构 key、不可变 base-context value、连续 null；主干与 content-HPC 全冻结；训练只有全 token sensory CE。

| 条件 | Return-conflict | Clean | Context pair | Context margin |
|---|---:|---:|---:|---:|
| 冻结 M1f | 0.3906 | 0.9594 | 0.9062 | +0.2031 |
| M1i 可靠历史调用 | 0.6406 | 0.9750 | 0.8750 | +0.1857 |
| M1i 禁用历史调用 | 0.2812 | 0.9688 | 0.7812 | +0.1063 |
| M1i 强制历史调用（诊断） | 0.3125 | 0.8914 | 0.9531 | +0.2662 |

## 冻结五门

- M1i return-conflict：`0.6406`。
- M1i - M1f：`+0.2500`。
- M1i - no-call：`+0.3594`。
- M1i clean drop：`+0.0000`。
- M1i context pair：`0.8750`。

- `reliable_return_absolute`：`FAIL`
- `reliable_gain_vs_m1f`：`PASS`
- `history_call_necessary`：`PASS`
- `reliable_clean_preserved`：`PASS`
- `reliable_context_identity`：`PASS`

冻结分类：`M1I_RELIABLE_HISTORY_PILOT_REJECTED`。

## 调用动态

- Proposal 平均 hold/refresh/call：`0.420 / 0.157 / 0.423`。
- Call 为 argmax：`0.552`。
- Attention max / entropy：`0.406 / 0.501`。
- Evidence / learned null：`0.884 / 0.927`。

Context pair 对全部 return-conflict query 无权重统计；force-call 仅作机制诊断，不参与五门判定。任一门失败即停止，不在 seed 9931 上调参。
