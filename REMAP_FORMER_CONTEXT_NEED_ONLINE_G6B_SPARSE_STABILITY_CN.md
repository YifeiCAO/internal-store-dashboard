# ReMAP-Former：G6b Sparse Path-Bank 稳定性审计

> G6 正式结果仍是 92/93 FAIL。本审计不改 online controller，只判断 sparse endpoint 的单-bank 尾部是否可复现。

## 冻结设置

- protocol SHA256：`003c0336486c141c62b03ecbca3260070f3bd06ffe9ec7b420c0f2655a2c362b`
- 四个 fresh sparse banks，彼此及 G5/G6/original train canonical overlap 全为 0。
- 每 bank：8 checkpoints × 32 episodes；总 return-conflict probes = 1024。
- threshold `0.07688456773757935`，recurrence cosine `0.99999`，observation horizon `72`。

## 结果

| bank | family seed | selection dynamic | return-conflict | clean | online/fixed passes | checkpoint range |
|---|---:|---:|---:|---:|---:|---:|
| sparse_bank_1 | 4271701 | 1.0000 | 0.8047 | 0.9990 | 1.8785/2.0000 | 0.7500–0.9375 |
| sparse_bank_2 | 4371701 | 1.0000 | 0.7656 | 0.9974 | 1.8785/2.0000 | 0.6250–0.8750 |
| sparse_bank_3 | 4471701 | 1.0000 | 0.7734 | 0.9984 | 1.9028/2.0000 | 0.6875–0.8750 |
| sparse_bank_4 | 4671701 | 1.0000 | 0.8906 | 0.9990 | 1.9097/2.0000 | 0.6875–1.0000 |

## Pooled

- return-conflict：`0.8086` (`828/1024`)
- Wilson 95% CI：`[0.7834, 0.8315]`
- bank median：`0.7891`
- banks ≥ 0.70：`4/4`
- online/fixed decision agreement：`1.0000`
- final state/logit max abs：`7.629e-06`

## 冻结结论

**`CONTEXT_NEED_ONLINE_G6B_SPARSE_STABLE`**

四个预注册 fresh banks 满足稳定性门：G6 的 0.671875 被分类为单-bank 尾部事件；G6 原始 FAIL 保留，后续可在明确披露 sparse 方差的前提下研究 variable horizon。

边界：无论本审计分类为何，G6 原始 92/93 FAIL 不覆盖；这里估计的是 frozen dynamic downstream 对 sparse path bank 的稳定性，不是重新选择阈值或模型。
