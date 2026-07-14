# M1h G1b：PFC 历史 Context 持续调用门

> G1 覆盖失效定位后的独立新 seed K=8 dev gate；冻结 M1f/content HPC；32 步取自现有 Transformer window；无训练；未访问 stress、formal validation 或 test。

| 条件 | Return-conflict | Clean | Context pair | Context margin |
|---|---:|---:|---:|---:|
| Normal M1f | 0.4688 | 0.9406 | 0.8750 | +0.1593 |
| Persistent history call | 0.8438 | 0.9422 | 0.9375 | +0.2497 |
| Persistent shuffled value | 0.1016 | 0.8957 | 0.5156 | +0.0366 |

## 冻结判读

- Caller - normal return：`+0.3750`。
- Caller - shuffled return：`+0.7422`。
- Caller clean drop：`+0.0000`。
- Return-conflict persistent recall coverage：`1.0000`。
- Mean suppressed unmatched proposals / episode：`16.5938`。

冻结分类：`PERSISTENT_HISTORY_CALL_CONFIRMED`。

32-step hard persistence 只是检验历史调用加稳定保持是否能闭合下游任务，不能作为最终论文模型。最终候选必须由 PFC recurrent attention 学会 self-versus-history 选择，且仍只训练 sensory CE。
