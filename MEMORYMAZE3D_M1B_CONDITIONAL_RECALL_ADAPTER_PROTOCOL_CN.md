# M1b 条件式回忆适配器 Pilot 协议

## 这次只修什么

当前问题不在 EC/context 地址。`val[96:128]` 的机制审计显示：

- 过去地址最大余弦均值为 `0.80`；
- 重复姿态下，最佳历史地址命中正确姿态的比例为 `83.9%`；
- 双钥匙写入分母稳定在 `1.0`；
- 原始 sensory content 在重复姿态下稳定，相对变化误差只有 `0.0348`；
- PFC residual content 的相对变化误差为 `0.9876`，比 sensory content 不稳定 `28.36` 倍；
- 旧线性融合的最佳真实读出固定增益为 `0.5`，但只比 PFC 改善 `0.39%`。

因此本轮不改 PFC、EC、place、context 和 HPC fast-weight 本体，只改 **HPC 被调用以后如何影响 PFC**。

## 新读出

HPC 继续存每个 episode 内的稳定视觉 latent。读出后：

```text
PFC hidden ───────────────────────────────┐
                                          ├─ 条件 recall adapter ─ Δh ─┐
HPC retrieval ─ projection ───────────────┘                            │
                                                                       ├─ decoder ─ 预测
PFC-only hidden ───────────────────────────────────────────────────────┘
```

适配器使用严格的空回忆中心化：

```text
Δh = A(PFC_h, retrieval_h) - A(PFC_h, 0)
```

所以：

- HPC 没读到内容时，`Δh = 0`，模型严格退化为原 PFC；
- 适配器不能靠自己的偏置偷偷变成第二个 PFC；
- 没有新增记忆槽；
- 没有第二套 fast weights；
- 没有 room id、位置、place id 或未来真实图像输入；
- 当前时刻先读后写，读出的内容只来自过去。

## 训练

- 起点：`memorymaze3d_m1b_pfc_history_v6_pilot400_seed2121`
- seed：`2151`
- 更新：`400`
- batch：`2`
- 序列：`64`
- 训练参数：`memory_projection + conditional recall adapter`
- 冻结：视觉壳、PFC、context、EC、place、HPC value 编解码器
- rollout：`C20`，课程 `2,4,8,16`
- rollout 中仅观察段允许写 HPC
- 不新增 loss，沿用预测 loss

## 预注册判断

只在未参与本次设计的 `val[128:160]` 上判断：

1. 全部测试通过，训练和梯度有限；
2. 空回忆及 `memory_scale=0` 精确等于 PFC-only；
3. strict observed-only free rollout 至少相对改善 `1%`；
4. 配对 bootstrap 90% 区间下界大于 `0`；
5. 动作优势为正，未来真值读写和扰动泄漏均为 `0`。

任一关键门失败，就停止继续调融合系数，转向重新设计稳定的神经内容表征。
