# M1h：神经 PFC 历史调用 K=8 盲测

> 固定 step 600；全新 generator seed 8928；冻结 PFC/EC/place/content-HPC/fusion/decoder；训练只有全 token sensory CE；未使用 room/context/位置监督。

| 条件 | Return-conflict | Clean | Context pair | Context margin |
|---|---:|---:|---:|---:|
| 冻结 M1f | 0.4062 | 0.9539 | 0.8281 | +0.1736 |
| M1h 神经历史调用 | 0.1719 | 0.9563 | 0.6875 | +0.0661 |
| M1h 禁用 history call | 0.3906 | 0.9508 | 0.8125 | +0.1583 |

## 冻结五门

- Neural return-conflict：`0.1719`。
- Neural - M1f：`-0.2344`。
- Neural - no-call：`-0.2188`。
- Neural clean drop：`+0.0000`。
- Neural context pair：`0.6875`。

- `neural_return_absolute`：`FAIL`
- `neural_gain_vs_m1f`：`FAIL`
- `history_call_necessary`：`FAIL`
- `neural_clean_preserved`：`PASS`
- `neural_context_identity`：`FAIL`

冻结分类：`NEURAL_HISTORY_CALL_PILOT_REJECTED`。

## 学到的调用动态

- Neural proposal 平均权重 hold/refresh/call：`0.043 / 0.172 / 0.785`。
- Neural call 为 argmax 的比例：`0.983`。
- History attention max / entropy：`0.344 / 0.612`。

这是一次预注册式单 seed pilot。若任一门失败，seed 8928 不再用于调参；若全过，才允许进入三训练种子扩展。
