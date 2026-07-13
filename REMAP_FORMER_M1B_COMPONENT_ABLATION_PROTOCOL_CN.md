# ReMAP-Former M1b：8-Seed 组件必要性消融协议

> 本协议冻结于正式 validation 前。主模型、8-seed headline test 与 checkpoint 全部保持封存；本实验不再访问 test。

## 目标

回答四个机制问题：

1. M1b 的提升是否需要 covariance-corrected write，而不只是同一套慢权重？
2. inferred context 是否真的参与整段 episode 的写入与读取？
3. HPC retrieval 是否真的被 PFC 调用，而不只是旁路存在？
4. return 时把 read context 换成错误 reference，是否会因果性地把输出拉向另一房间？

## 冻结评估

- checkpoint seeds：`712–719`
- split：`validation`
- generator seed：`1893`
- 每颗 checkpoint：`16×16 = 256 episodes`
- 每条件、每 seed：预计 `512` 个 return-conflict probes
- 所有条件逐 batch 读取完全相同的 action/sensory
- 不训练、不改 checkpoint、不访问 test

正式评估前只允许一次 `seed712 / dev1993 / 4×2` 的形状与管线 smoke；smoke 不用于改门槛。

## 七个条件

- `full`：冻结 M1b 原样运行。
- `no_covariance`：同一套 slow weights、同一 learned context，只把 HPC backend 换回 Delta。
- `fixed_context_all_steps`：整段 episode 的 context 替换为同一个归一化常量。
- `shuffled_context_all_steps`：整条 context trajectory 换成另一个独立 gauge pair 的 donor。
- `hpc_read_zero`：保留写入，但 retrieval 在进入 PFC refinement 前清零。
- `wrong_return_context`：仅诊断，在 return probe 当步重放错误 reference 的 context。
- `correct_return_context`：仅诊断，在 return probe 当步重放正确 reference 的 context。

后两项使用 batch metadata 构造干预，但 metadata 只在冻结模型先产生 context 后用于诊断，不进入训练或正常模型输入。

## 预注册 Gates

- 每颗 `full clean >= 0.98`
- `full` 平均 return-conflict `>= 0.70`
- `full` 每颗都胜过 `no_covariance`
- `full - no_covariance` 平均 `>= 0.25`
- `full - fixed_context_all_steps` 平均 `>= 0.25`
- `full - shuffled_context_all_steps` 平均 `>= 0.10`
- `full - hpc_read_zero` 平均 `>= 0.50`
- `full - wrong_return_context` 平均 `>= 0.30`
- wrong-return 的 other-room rate 相对 full 平均增加 `>= 0.30`

正式输出固定为：

- `runs/remap_former/m1b_component_ablation_validation1893.json`
- `reports/REMAP_FORMER_M1B_COMPONENT_ABLATION_8SEED_CN.md`
