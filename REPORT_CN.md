# MemoryMaze3D Full-context Transformer Fresh Sealed Baseline

**状态：`SEALED_BASELINE_3SEED_COMPLETE`**

三个 checkpoint 在 development 完成选择与冻结后，才进行唯一一次
sealed 评估。该 Transformer 能直接读取完整 384 步因果历史，并获得
与 ReMAP-Former 相同的外部 write event。

| Seed | Conflict | Target cosine | Clean | Paired output cosine |
|---:|---:|---:|---:|---:|
| `67101` | 0.5312 | 0.9157 | 0.9282 | 0.9818 |
| `67102` | 0.5312 | 0.8607 | 0.8828 | 1.0000 |
| `67103` | 0.5260 | 0.9089 | 0.9198 | 1.0000 |

## 三种子聚合

- Full Transformer conflict：**`0.5295 +/- 0.0030`**
- ReMAP-Former conflict：**`0.9774 +/- 0.0060`**
- ReMAP minus Transformer conflict gap：**`0.4479`**
- Full Transformer target cosine：`0.8951`
- Full Transformer clean cosine：`0.9103`
- Full Transformer paired counterfactual output cosine：**`0.9939`**

## 结论

这个 baseline 不是被 window 卡死的：最早 query 可以直接 attend 第 0 步，
而且它还拿到了 matched write flag。它在 sealed 上仍只有约 `0.53` 的
context-conflict 选择率，同时能输出 cosine 较高的物体 feature。

因此失败模式不是“看不到历史”或“完全不会重建内容”，而是没有稳定学会
context-conditioned binding：A-B-A 与 B-A-B 的 paired output cosine
仍接近 1，模型对两种反事实历史给出了几乎相同的答案。

## 运行完整性

- Sealed optimizer steps：`0`
- Sealed checkpoint reselection：`false`
- Future ground-truth read/write：`0/0`
- Sealed summary SHA256：`8b6327a5206a0da184bfe8ec1795ddce6ef036ca1a194d5b07ca6ce46fa70269`

## 报告勘误

sealed 指标已先成功写入独占 summary；随后原 evaluator 仅在生成中文
Markdown 时把旧 ReMAP 字段 `sample_std` 误写为 `sample_sd` 而退出。
本报告由只读 renderer 从该已冻结 summary 生成，没有重跑模型、改写
summary、重选 checkpoint 或执行优化。

## 边界

- 这是同一 frozen task 上的一次性 matched baseline 结果。
- 它不是 learned-policy free rollout，也不是 RGB generation。
