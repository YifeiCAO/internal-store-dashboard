# M1f K=8 Oracle 归因诊断

> 与 K=8 timing confirmation 完全相同的 384 个 dev-only return-conflict probes；冻结模型；无训练、无 checkpoint 选择；未访问 stress、formal validation 或 test。

## 逐 Seed

| Model / Episode seed | Normal | Correct ctx | Wrong ctx | Exact fused | Exact raw HPC |
|---|---:|---:|---:|---:|---:|
| 712 / 3921 | 0.4219 | 0.9844 | 0.0000 | 0.9844 | 1.0000 |
| 713 / 3922 | 0.4531 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 714 / 3923 | 0.3281 | 0.9688 | 0.0000 | 0.9688 | 0.9688 |

## 聚合

| 条件 | Return-conflict | Clean | Target margin |
|---|---:|---:|---:|
| Normal fused | 0.4010 | 0.9464 | +0.2561 |
| Normal raw HPC | 0.3906 | 0.9451 | +0.0087 |
| Correct context | 0.9844 | 0.9534 | +0.7761 |
| Wrong context | 0.0000 | 0.9497 | -0.7754 |
| Exact address fused | 0.9844 | 0.9534 | +0.7761 |
| Exact address raw HPC | 0.9896 | 0.9505 | +0.0183 |

## 冻结判读

- Correct context - normal：`+0.5833`
- Correct context - wrong context：`+0.9844`
- Exact address fused - normal：`+0.5833`
- Exact address raw - fused：`+0.0052`
- Context geometry：normal 对正确/错误历史的 cosine margin `+0.1787`，更接近正确历史的 probe 比例 `0.7448`。
- Address geometry：normal 对正确/错误历史的 cosine margin `+0.1787`，更接近正确历史的 probe 比例 `0.7448`。

冻结分类：`CONTEXT_IDENTITY_RETRIEVAL_DOMINANT`。

正确历史 context 已恢复绝对性能、相对 normal 的增益和相对错误历史的方向性；下一版只应解决历史 context 身份检索，不应重做内容 HPC。

所有 oracle 都只在冻结模型正常 forward 后，借助 audit metadata 改 return probe；它们是诊断上限，不是可部署模型，也不是训练输入。
