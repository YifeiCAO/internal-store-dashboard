# ReMAP-Former P3：封存 Test-2

> 24 个外部基线检查点全部冻结并通过训练门后评估。第一计算 pass 在全部前向完成后发生纯后处理溢出；
> 第二 pass 使用同 checkpoint、同 generator、同顺序确定性重放，访问偏差完整记录，不作模型选择或调参。

## 核心结果

| 条件 | 参数量 | clean | conflict | return-conflict（95% CI） |
|---|---:|---:|---:|---:|
| Hippoformer | 1,494,447 | 1.0000 | 0.7937 | 0.0000 [0.0000, 0.0000] |
| Hippoformer HPC 分支 | 1,494,447 | 0.9868 | 0.6854 | 0.0000 [0.0000, 0.0000] |
| M-delta | 1,388,257 | 1.0000 | 0.7999 | 0.0000 [0.0000, 0.0000] |
| ReMAP-Former M1b | 1,398,289 | 0.9944 | 0.9372 | 0.8376 [0.8221, 0.8527] |
| 短窗 Transformer | 1,280,960 | 0.8953 | 0.8011 | 0.0134 [0.0098, 0.0176] |
| 参数匹配 Transformer | 1,396,096 | 0.8954 | 0.8012 | 0.0122 [0.0089, 0.0159] |
| Titans-MAC 适配版 | 1,344,260 | 0.9234 | 0.8001 | 0.0095 [0.0063, 0.0129] |

## 配对比较

| baseline | M1b - baseline | 95% CI | 稳健优势 |
|---|---:|---:|---|
| Hippoformer | +0.8376 | [+0.8221, +0.8527] | 是 |
| M-delta | +0.8376 | [+0.8221, +0.8527] | 是 |
| 短窗 Transformer | +0.8242 | [+0.8081, +0.8397] | 是 |
| 参数匹配 Transformer | +0.8254 | [+0.8096, +0.8406] | 是 |
| Titans-MAC 适配版 | +0.8281 | [+0.8121, +0.8434] | 是 |

## 完整性审计

- Test-2 generator seed：`2471601`。
- training seeds / episodes：`8` / `512`。
- 所有 checkpoint 哈希匹配：`True`。
- 所有模型评估前后慢权重不变：`True`。
- episode-row 数量精确：`True`。
- Test-2 计算访问 pass：`2`；访问账本状态：`completed`。
- 确定性恢复重放：`True`；原因：`first pass postprocessing overflow`。

## 解释边界

- Hippoformer HPC 分支只是同一实现的机制消融，不冒充官方 mmTEM 复现。
- Titans-MAC 是按论文机制做的导航适配版，不是作者官方代码或 checkpoint。
- 外部三基线使用 1200 optimizer steps；旧 Hippoformer、M-delta 与 M1b 使用冻结的 600-step checkpoint，训练预算分别报告，不混称完全预算匹配。
