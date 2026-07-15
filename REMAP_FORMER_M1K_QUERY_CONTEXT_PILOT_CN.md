# M1k：Query-conditioned Archival Context Readout 单 seed Pilot

> 固定 step 600；全新 K=8 blind seed427151；模型只接收因果 actions、过去 sensory 和同 episode 历史激活。没有 room/context/位置/segment/target 输入，没有新增记忆槽或第二套 fast weights。

## 模型差异

M1f 在 proposal 时更新一次 context，随后整段共用。M1k 保留这条 source context，同时让每个 token 的 Transformer hidden 独立查询更早 proposal 的 archival base context，并用 16 维逐通道 gate 形成当前地址。训练仍只有一个均匀 all-token sensory CE。

| 条件 | Return-conflict | Clean | Context pair | Context margin |
|---|---:|---:|---:|---:|
| 冻结 M1f | 0.3594 | 0.9398 | 0.8438 | +0.1739 |
| M1k token 级历史读出 | 0.0938 | 0.9391 | 0.5938 | +0.0606 |
| M1k 禁用历史读出 | 0.3594 | 0.9398 | 0.8438 | +0.1739 |

## 冻结五门

- M1k return-conflict：`0.0938`。
- M1k - M1f：`-0.2656`。
- M1k - disabled：`-0.2656`。
- M1k clean drop：`+0.0008`。
- M1k context pair：`0.5938`。

- `m1k_return_absolute`：`FAIL`
- `m1k_gain_vs_m1f`：`FAIL`
- `query_readout_necessary`：`FAIL`
- `m1k_clean_preserved`：`PASS`
- `m1k_context_identity`：`FAIL`

## 实现门

- `disabled_prediction_equivalence`：`FAIL`
- `disabled_context_equivalence`：`PASS`
- `attention_rows_normalized`：`PASS`
- `expected_return_conflict_probes`：`PASS`
- `finite_outputs`：`PASS`
- `source_checkpoint_digest`：`PASS`

冻结分类：`INVALID_IMPLEMENTATION`。

## 读出动态

- 全 history token / return-conflict gate：`0.8393` / `0.8517`。
- Attention max / normalized entropy：`0.3328` / `0.6394`。
- 同 episode 两个 conflict probes 的 gate 平均绝对差：`0.0107`。

## 失败判读

- K=8 下 attention max 只有 `0.3328`、normalized entropy 为 `0.6394`，说明候选历史变多后读出重新变得弥散。
- Return-conflict gate 仍高达 `0.8517`，同 episode 两个 probes 的 gate 差仅 `0.0107`；token-conditioned 架构实际退化成了近似全局高调用。
- Context pair 从 M1f 的 `0.8438` 降到 `0.5938`，因此失败发生在历史身份选择，不是 clean recall 崩溃。
- Disabled context 最大差仅 `5.960e-08`，但 covariance-HPC 长序列把它放大为 prediction 差 `7.677e-05`，超过预注册 `1e-6`，所以状态必须记为 `INVALID_IMPLEMENTATION`。即使忽略这条数值门，五个科学门也只过 clean，结论仍不允许扩 seed。

本结果只决定是否允许 M1k 进入多 seed；不改变 frozen M1b 的正式地位，也不重开 M1j。blind seed427151 不得用于调参。
