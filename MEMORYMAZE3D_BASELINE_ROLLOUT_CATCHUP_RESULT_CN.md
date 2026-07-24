# MemoryMaze3D 公平基线 Rollout 补训：最终结果

日期：2026-07-24  
实验角色：adaptive development，不是 test headline  
状态：12/12 候选完成，test 未访问

## 1. 这轮到底在回答什么

这轮不直接证明哪一种记忆架构最好，而是先排除一个关键混杂：

> 旧基线的长程自由 rollout 差，是模型架构不行，还是因为它们只学过
> teacher-forced 下一步预测，没有接受足够的 free-rollout 训练？

任务统一为 MemoryMaze3D 变量环境中的严格 C20→H44：

- 先观察 20 个 action step 和对应第一人称 RGB；
- 随后只给未来 action，自由生成 44 步；
- rollout 阶段不读取或写入未来真实观察；
- 模型输入不含绝对位置、朝向、room id、place id 或 store 状态。

## 2. 公平边界

三个正式基线共用同一个冻结视觉 encoder 和 pixel decoder，只训练各自
backbone。共享视觉 tensor SHA256 为：

`2625367f0ebcb3c86ffe25fe0253059158e27cf7dfbf72ce54fe75a315bbb8a8`

正式候选：

| 模型 | Backbone 含义 |
|---|---|
| Transformer | 3 层、64-step causal window Transformer |
| Titans | 同输入输出壳上的 Titans MAC 类长期神经记忆 |
| Hippoformer | short-term Transformer + MM-TEM 双向 neural fast-weight memory + learned combiner |

M1b 只作为当前项目参考：它是 Window Transformer=PFC，加 SE(2) EC、
sparse place address 和 covariance-corrected HPC。M1b 累计祖先训练约
12,180 updates，与本轮基线预算不匹配，因此不能用于预算匹配优越性结论。

## 3. 冻结训练方案

Stage H 先把 teacher-health 补到可比较水平，再进入 Stage R：

- 每个模型固定 `LR ∈ {5e-5, 1e-4}`；
- `rollout weight ∈ {0.5, 1.0}`；
- 每格 1,200 updates，batch 4，sequence 64；
- curriculum 为 H2→H4→H8→H16；
- teacher weight 0.5，rollout latent weight 0.05；
- seed 2111；
- 只训练 backbone，视觉壳冻结；
- 每格用同一 32-episode adaptive-dev 做严格 C20→H44 评估。

旧 descent gate 错误地比较了 H2 开头与 H16 结尾。勘误后只比较最终
H16 的前 60 和后 60 updates，并要求 total loss 与 rollout pixel MSE
同时下降。勘误在 Transformer R1 后、Titans 和 Hippoformer R1 前冻结；
所有 12 格都按修正规则重新审计并通过。

## 4. 完整结果

Persistence pixel MSE 为 `0.040188`。

| 模型 | LR | rollout w | H44 MSE | 相对 persistence | 动作优势 |
|---|---:|---:|---:|---:|---:|
| Transformer | 5e-5 | 0.5 | 0.024015 | +40.2% | 0.004194 |
| Transformer | **5e-5** | **1.0** | **0.022994** | **+42.8%** | 0.004210 |
| Transformer | 1e-4 | 0.5 | 0.023427 | +41.7% | 0.002492 |
| Transformer | 1e-4 | 1.0 | 0.024683 | +38.6% | 0.000693 |
| Titans | 5e-5 | 0.5 | 0.023252 | +42.1% | 0.019240 |
| Titans | **5e-5** | **1.0** | **0.021891** | **+45.5%** | **0.019408** |
| Titans | 1e-4 | 0.5 | 0.024817 | +38.2% | 0.027359 |
| Titans | 1e-4 | 1.0 | 0.024680 | +38.6% | 0.008576 |
| Hippoformer | 5e-5 | 0.5 | 0.026347 | +34.4% | 0.004288 |
| Hippoformer | **5e-5** | **1.0** | **0.023302** | **+42.0%** | 0.008754 |
| Hippoformer | 1e-4 | 0.5 | 0.025863 | +35.6% | 0.005412 |
| Hippoformer | 1e-4 | 1.0 | 0.029198 | +27.3% | -0.001264 |

三个模型的冻结选择都是 `LR=5e-5, rollout weight=1.0`。

训练前后的最佳 H44：

| 模型 | R0 | Stage R 最佳 | 绝对下降 |
|---|---:|---:|---:|
| Transformer | 0.055807 | 0.022994 | 0.032813 |
| Titans | 0.044593 | 0.021891 | 0.022702 |
| Hippoformer | 0.042423 | 0.023302 | 0.019122 |
| M1b 非预算匹配参考 | 0.020073 | 0.020073 | - |

## 5. 配对统计

20,000 次 paired bootstrap，seed 2112，同一 32 个 episode：

| 比较 | 改善 | 95% CI | 判断 |
|---|---:|---:|---|
| Transformer Stage R vs R0 | +0.032813 | [+0.022430, +0.043091] | 稳定改善 |
| Titans Stage R vs R0 | +0.022702 | [+0.014156, +0.031539] | 稳定改善 |
| Hippoformer Stage R vs R0 | +0.019122 | [+0.010532, +0.027726] | 稳定改善 |
| Titans vs Transformer | +0.001103 | [-0.001361, +0.003681] | 无可靠差异 |
| Hippoformer vs Transformer | -0.000308 | [-0.007860, +0.006407] | 无可靠差异 |
| Hippoformer vs Titans | -0.001410 | [-0.009116, +0.005160] | 无可靠差异 |

正改善表示前者 MSE 更低。当前 32-episode adaptive-dev 足以确认
rollout objective 有效，但不足以确认三种 backbone 的可靠排序。

## 6. 结论

### 已经能说

1. 旧基线的主要问题不是 teacher-health，而是缺少 rollout objective。
2. H2→H4→H8→H16 补训让三个 backbone 都从差于 persistence 变成明显
   优于 persistence。
3. Titans 数值最好，而且对正确 action 的依赖最强；这是值得复现的信号。
4. Hippoformer 也被同一训练方案显著救活，但当前没有显示出相对
   Transformer 或 Titans 的可靠优势。

### 还不能说

1. 不能说 Titans 已经统计显著优于另外两个模型。
2. 不能用 M1b 的 `0.020073` 宣称项目模型优于预算匹配基线。
3. 不能把本轮 adaptive-dev 当作 test headline。
4. 不能从单 seed 推断跨初始化稳定性。

## 7. 工程发现

Hippoformer 四格累计训练约 6.68 candidate-hours，而 Transformer 四格只约
0.14 candidate-hours。主要慢点位于
`remap_former/memorymaze3d.py::MemoryMazeVisualWorldModel.free_rollout`：
每生成一步都会在增长中的完整历史上重新执行 backbone；Hippoformer 又在
每次执行中重建并逐 token 更新 MM-TEM neural fast weights。

本轮没有中途更换实现，以保证 12 格计算路径一致。下一轮扩 seed 前，应增加
严格等价的增量 state/cache API，并用逐步输出、loss、梯度的数值等价测试
证明它没有改变模型语义。

## 8. 下一步

1. 冻结三个模型各自的 `5e-5 / 1.0` checkpoint。
2. 先实现 Hippoformer/MM-TEM 增量 rollout cache；通过等价测试后再用于新实验。
3. 在 fresh adaptive-dev seeds 上做 3-seed paired confirmation，仍不看 test。
4. 如果三模型排序稳定，再冻结一次 sealed test；如果不稳定，论文结论写成
   “rollout objective 的机制结果”，不强行宣称 backbone winner。
5. 基线地板稳定后，再回到论文主线：变量环境和 context/remapping 条件下，
   检验 Transformer=PFC + EC/HPC 外挂记忆是否在长延迟、重访和环境切换上
   提供独立收益。

## 9. 可复核产物

- 冻结方案：`MEMORYMAZE3D_BASELINE_ROLLOUT_CATCHUP_ADDENDUM_CN.md`
- Gate 勘误：`MEMORYMAZE3D_BASELINE_ROLLOUT_GATE_ERRATUM_CN.md`
- 动态结果：`reports/memorymaze3d_baseline_rollout_catchup/REPORT_CN.md`
- 机器审计：`reports/memorymaze3d_baseline_rollout_catchup/summary.json`
- 配对统计：`reports/memorymaze3d_baseline_rollout_catchup_statistics/REPORT_CN.md`
- 结果图：`reports/figures/memorymaze3d_baseline_rollout_catchup.png`

复核命令：

```powershell
python audit_memorymaze3d_baseline_rollout_catchup.py
python analyze_memorymaze3d_baseline_rollout_statistics.py
python plot_memorymaze3d_baseline_rollout_catchup.py
python -m pytest -q test_memorymaze3d.py
```

最终验证：`50 passed`。整个 Stage R 的 test access 保持 `forbidden`。
