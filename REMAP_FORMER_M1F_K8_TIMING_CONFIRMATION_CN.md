# M1f K=8 独立确认：旧下降未复现，结构重要但精确时刻不特异

> 日期：2026-07-13  
> 范围：全新 dev episodes；冻结 M1f；无训练或 checkpoint 选择；未访问 stress、formal validation 或 test。  
> 冻结分类：`K8_CONFIRMED_GENERIC_HOLD_REFRESH`。

## 1. 为什么补这个实验

上一轮 M1f 三 seed dev gate 总体通过，但 K=8 从 M1b `0.3333` 降到 M1f `0.2917`，差 `-4.17 pp`。该切片只有 96 个 probes，相当于 M1f 少对 4 道题。

本轮只回答两个问题：

1. 用 4 倍独立 K=8 样本，旧下降是否复现？
2. M1f 成功依赖 balanced-window 的结构时机，还是任意低频刷新都可以？

## 2. 看结果前冻结的协议

协议：`runs/remap_former/m1f_k8_timing_protocol.json`。

- M1b model seeds：`712/713/714`；
- 全新 generator seeds：`3921/3922/3923`；
- K 固定为 8；
- 每颗 64 episodes、128 return-conflict probes；
- 聚合 192 episodes、384 probes；
- 所有条件逐 token 使用完全相同的 episodes；
- M1f slow weights 与 M1b 逐 tensor 相同；
- 不训练、不选 checkpoint、不改 ridge。

五个条件：

1. M1b ridge=0.001；
2. M1f 原 balanced-window refresh；
3. M1f no refresh；
4. 同一 proposal schedule 因果延迟 6 步；
5. 每个 episode 保持 proposal 数完全相同，但在合法时间位置随机置换。

最后一个 random control 使用整段 proposal 个数，只是冻结后的诊断负对照，不是可部署的 causal 模型。

## 3. 逐 Seed 结果

| Model / Episode seed | M1b | Balanced | No refresh | Delay +6 | Random matched | Balanced-M1b |
|---|---:|---:|---:|---:|---:|---:|
| 712 / 3921 | 0.2969 | **0.4219** | 0.0156 | 0.3438 | 0.2188 | **+0.1250** |
| 713 / 3922 | 0.3750 | **0.4531** | 0.0156 | 0.3594 | 0.3594 | **+0.0781** |
| 714 / 3923 | 0.2812 | **0.3281** | 0.0156 | 0.4062 | 0.2578 | **+0.0469** |

三颗 M1f 都高于 M1b。上一轮 K=8 的两颗负增益没有在独立扩大样本中复现。

## 4. 聚合 384-Probe 结果

| 条件 | Return-conflict | Clean | Target margin |
|---|---:|---:|---:|
| M1b | 0.3177 | 0.8422 | +0.1605 |
| **M1f balanced** | **0.4010** | **0.9464** | **+0.2561** |
| No refresh | 0.0156 | 0.9500 | -0.0001 |
| Delay +6 | 0.3698 | 0.8794 | +0.2399 |
| Rate-matched random | 0.2786 | 0.6828 | +0.1465 |

冻结差值：

- Balanced - M1b：`+0.0833`；
- Balanced - no refresh：`+0.3854`；
- Balanced - delay+6：`+0.0313`；
- Balanced - rate-matched random：`+0.1224`。

## 5. 同 Probe 方向性

Balanced vs M1b：

- 两者都对：72；
- 仅 balanced 对：82；
- 仅 M1b 对：50；
- 两者都错：180。

净救回 `32/384=+8.33 pp`。

Balanced vs random：

- 两者都对：59；
- 仅 balanced 对：95；
- 仅 random 对：48；
- 两者都错：182。

净增益 `47/384=+12.24 pp`。因此“任意等次数刷新都可以”被否定。

Balanced vs delay+6：

- 两者都对：92；
- 仅 balanced 对：62；
- 仅 delay 对：50；
- 两者都错：180。

净增益只有 `12/384=+3.13 pp`，未达到冻结的 `+5 pp` timing-specificity 门。

## 6. Refresh 次数严格受控

| Schedule | Proposals | 每 episode | Token rate |
|---|---:|---:|---:|
| Balanced | 11,904 | 62.0 | 0.1082 |
| No refresh | 0 | 0.0 | 0.0000 |
| Delay +6 | 11,904 | 62.0 | 0.1082 |
| Rate-matched random | 11,904 | 62.0 | 0.1082 |

Balanced、delay 和 random 的调用次数完全一致。性能差异不能归因于“某条件刷新更多”。

## 7. 冻结 Gates

核心 K=8 gates：

- Balanced 不劣于 M1b：通过，`+8.33 pp`；
- 正增益 model seeds 至少 2/3：通过，实际 `3/3`；
- Balanced 相对 no-refresh 至少 +10 pp：通过，`+38.54 pp`；
- 最大 clean drop 不超过 2 pp：通过，实际最差仍提升 `7.34 pp`。

时机特异性 gates：

- Balanced 相对 random 至少 +5 pp：通过，`+12.24 pp`；
- Balanced 相对 delay+6 至少 +5 pp：失败，`+3.13 pp`。

按看结果前冻结的分类规则，状态为 `K8_CONFIRMED_GENERIC_HOLD_REFRESH`。

## 8. 人话解释

这个冻结状态比较保守，不能被误读成“时机完全没用”。数据实际支持三层结论：

1. **保持后再刷新是必要的**：不刷新只有 `0.0156`；
2. **结构选择是有信息的**：balanced 比等次数随机高 `12.24 pp`，random 还严重伤害 clean；
3. **不要求精确到某一个 token**：整体延迟 6 步仍接近原 schedule，只低 `3.13 pp`。

所以当前最准确的机制描述是：

> M1f 利用动作结构限定一个较宽的 context-refresh 时间区域；收益来自结构约束下的稳定 context，而不是精确 room boundary 分类，也不是任意频率的平滑。

## 9. 对 M1f 晋级状态的影响

- 上一轮 K=8 的 `-4.17 pp` 更像小样本波动，独立扩大样本中变为 `+8.33 pp`，且 3/3 seeds 正向；
- M1f 保持 `PROMOTE_M1F_TO_FORMAL_PROTOCOL_DESIGN`；
- 正式 claim 必须从“精确结构时机”降为“结构约束、宽时间容忍的 hold/refresh”；
- 不再需要修改 detector 或专门修 K=8；
- formal split 仍未访问。下一步是冻结正式多 seed 协议，而不是继续在 dev 上调模型。

## 10. 可复现入口

- 诊断接口：`remap_former/m1f.py`
- 控制 schedule：`evaluate_remap_m1f_k8_timing_confirmation.py`
- 专项测试：`test_remap_former_m1f_timing_controls.py`
- 冻结协议：`runs/remap_former/m1f_k8_timing_protocol.json`
- 结果：`runs/remap_former/m1f_k8_timing_confirmation/summary.json`
- 精简报告：`runs/remap_former/m1f_k8_timing_confirmation/report.md`
- 完整相关回归：`90 passed`。
