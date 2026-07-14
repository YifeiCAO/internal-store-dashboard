# M1h：Transformer/PFC 因果历史调用完整结果

> 结论日期：2026-07-14。硬调用上界成立；当前 3,467 参数的神经自调用版在冻结 K=8 新种子盲测中被拒绝。正式模型仍是 frozen converted M1b，formal validation/test 未访问。

## 1. 这一版在检验什么

问题不是再造一套海马，而是让 **Transformer/PFC 调用自己的普通因果历史**，恢复当前应使用的 context，再由既有 EC/place/content-HPC 完成内容回忆。

这一版严格没有：

- room ID、context 标签、switch flag、绝对/局部位置或 place ID 输入；
- 显式 context slots、第二套 fast-weight matrix 或跨 episode 内容存储；
- context classification、matching、contrastive、hard-caller distillation、entropy 或 query-weighted loss。

唯一 object/value 记忆仍是原 M1b 的 episode-local covariance fast-weight HPC。PFC history 只是窗口内已经存在的 token KV，不是另一张可写内容表。

## 2. 神经架构

在 M1f 的 action-only PFC state、base context 和 structural proposal 上增加：

1. 每个 proposal token 的特征为 `u_t = [PFC state_t; cyclic relation signature_t]`。
2. `q_t = normalize(W_q u_t)`，`k_j = normalize(W_k u_j)`；只允许读取 `j<t` 的历史 proposal token。
3. attention value 是历史 proposal 当时由模型发出的 context，不存 object/value。
4. 三路 softmax controller 在 `hold current / refresh from base / call history` 中连续混合。
5. 非 proposal token 只保持当前 context；context 乘性绑定 sparse place 后形成 HPC address。
6. EC、place、covariance content-HPC、PFC-HPC fusion 与 sensory decoder 全部冻结。

因果顺序为：动作到达 → PFC/EC 与历史 attention 形成 context/address → HPC 读出并预测 `x_t` → 环境才揭示 `x_t` → 原 content-HPC 写入。当前 target 不能进入当前预测或 context 调用。

## 3. 先做硬可行性门

| 实验 | M1f | 正确历史调用 | Shuffled value | 结论 |
|---|---:|---:|---:|---|
| G1：调用后允许被后续 proposal 覆盖 | 0.3438 | 0.6406 | 0.1484 | 正确信号强，但 active coverage 只有 0.8281，冻结为未通过 |
| G1b：按现有 32-step PFC window 持续保持 | 0.4688 | 0.8438 | 0.1016 | 六门全过，`PERSISTENT_HISTORY_CALL_CONFIRMED` |

G1 定位显示 128/128 个 return-conflict probe 之前都曾成功调用；其中 22 个随后恰好被一次 unmatched proposal 覆盖。G1b 证明“正确历史调用 + 稳定保持”足以闭合下游任务，但它是 hard diagnostic ceiling，不是论文模型。

## 4. 神经版训练协议

- 来源：seed 712 的冻结 M1f/M1b 权重，content-HPC ridge `0.001`。
- 新增且可训练：`history_query`、`history_key`、controller LayerNorm/MLP，共 `3,467` 参数。
- 训练 seed：`1712`；`600 × batch 4`；AdamW，LR `1e-3`，weight decay `1e-4`，clip `1.0`。
- 目标：唯一、未加权、全 token sensory cross-entropy。
- checkpoint：不选择；固定 step 600 是唯一候选。
- 训练内 dev：return-conflict `0.8438 → 0.9688`，clean `0.9813 → 1.0000`。
- controller：`hold/refresh/call = 0.049/0.197/0.754`；训练耗时约 `507 s`（CUDA）。

## 5. 冻结 K=8 新种子盲测

generator seed `8928`，64 episodes，128 个 return-conflict probes；同批配对比较：

| 条件 | Return-conflict | Clean | Context pair | Context margin |
|---|---:|---:|---:|---:|
| 冻结 M1f | 0.4062 | 0.9539 | 0.8281 | +0.1736 |
| M1h neural attention | 0.1719 | 0.9563 | 0.6875 | +0.0661 |
| 同一 M1h，禁用 history call | 0.3906 | 0.9508 | 0.8125 | +0.1583 |

冻结五门只有 clean preservation 通过：

- neural absolute return：失败，`0.1719 < 0.75`；
- neural - M1f：失败，`-0.2344 < +0.20`；
- neural - no-call：失败，`-0.2188 < +0.15`；
- clean drop：通过，`0`；
- context pair：失败，`0.6875 < 0.80`。

冻结分类：`NEURAL_HISTORY_CALL_PILOT_REJECTED`。不铺三训练种子，不用 seed 8928 回调超参数。

## 6. 失败定位

复用同一批已揭盲 episodes 做只读审计：

- 100% probe 的最近 proposal 都有因果历史；84.4% 存在 exact cyclic-signature 候选。
- 原始 structural signature top-1 在候选存在时命中 100%；训练后的 Q/K 仅命中 59.3%，且与 raw top-1 只一致 37.5%。
- exact 候选仅获得 39.8% attention mass；最近 proposal 的 attention max 仅 0.292。
- 尽管证据分散，controller 的 call weight 为 0.804，128/128 次都把 call 选为 argmax；`attention max < 0.5` 时仍调用的比例为 1.0。
- learned top history value 的 context pair 也只有 0.6875；top-1 exact 与非 exact 时准确率分别为 0.1563/0.1875，说明错误递归调用已污染后续 value。

因此失败不是历史里没有信息，而是三个耦合问题：**Q/K 从有效结构 key 漂移、缺少不确定时的拒绝调用、递归 emitted-context value 被错误调用污染**。训练内高分来自较容易的训练分布，不能外推到 K=8。

## 7. 下一条合法路线

下一版若另立新 seed 协议，应只改 PFC 历史调用器，不动健康的 content-HPC：

1. 用固定 structural signature 或受限 residual 参数化锚定 soft attention key，防止 sensory CE 把已验证的关系结构拉歪。
2. 在 attention 中加入显式 `null/abstain` 选项；低置信度 proposal 必须能够保持当前 context，而不是从任意历史强行选一个。
3. history value 使用该 token 的不可递归 archival/base context 表征，读取结果只影响当前 active context，不回写污染历史 value。
4. 训练仍只用 sensory CE，但训练分布必须覆盖独立的 K=1..8 capacity curriculum；另用从未见过的新 dev seed 做一次冻结门。

这仍是 `Transformer/PFC + 普通因果 token history + 单一 content-HPC`，不是记忆槽，也不是第二套 fast weights。M1h 当前实现保留为清晰的负结果和消融资产，不升级为正式模型。

## 8. 复现入口

- 模型：`remap_former/m1h.py`
- 训练：`train_remap_m1h_history_attention.py`
- 盲测：`evaluate_remap_m1h_neural_attention_pilot.py`
- 失败审计：`diagnose_remap_m1h_neural_attention_failure.py`
- 冻结协议：`runs/remap_former/m1h_neural_attention_pilot_protocol.json`
- 训练结果：`runs/remap_former/m1h_neural_attention_seed1712_s600/summary.json`
- 盲测结果：`runs/remap_former/m1h_neural_attention_pilot/summary.json`
- 审计结果：`runs/remap_former/m1h_neural_attention_failure_audit/summary.json`
- 回归：`113 passed`。
