# MemoryMaze3D rollout descent gate 勘误

## 问题

旧 `descent_gate` 比较了 update 1 的 H2 综合 loss 与训练末尾的 H16
综合 loss。课程难度已经改变，这两个数不能直接比较，因此会把正常学习的
Transformer 四个候选全部误报为 `false`。

## 修正

Stage R 统一改为只比较最终 H16 阶段：

- H16 一共有 300 updates。
- 取该阶段前 20%，即前 60 updates 的均值。
- 取该阶段后 20%，即后 60 updates 的均值。
- 综合 loss 和 rollout pixel MSE 都下降才通过。

代码方法名为 `final_horizon_early_vs_late_mean_v2`。

## 边界

- 不改数据、模型、loss、优化器、训练权重或 R2 选择指标。
- 不重跑或改写已经产生的 checkpoint。
- 对 Transformer、Titans、Hippoformer 所有候选使用同一个修正。
- 旧的跨课程判据保留为 `legacy_cross_curriculum_descent_gate`。
- 勘误在 Transformer R1 之后、Titans 与 Hippoformer R1 之前冻结。
- test 仍禁止访问。

