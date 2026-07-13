# ReMAP-Former：长延迟 / 干扰 OOD 协议

> 本实验冻结主模型，不再访问 headline test。它测试观察历史下的长延迟与重复干扰，不是 free rollout。

## OOD 构造

基础 random-return episode 有四段，每段 44 token，总长 176：reference 0、reference 1、distractor、final return。

只重复第三段 distractor，再接原始 final return。第三段是闭合路径，重复后仍回到中心；所有模型读取相同扩展后的 action/sensory。最终 return probe 不复制，probe 数保持不变。序列长度为：

`L = 176 + 44 × (distractor_repeats - 1)`

这同时增加：

- reference 到 final return 的时间距离；
- 中间 episode-local fast-weight 写入次数；
- 超出 Transformer 32-step window 的长期依赖。

## 模型

- matched Hippoformer
- M-delta
- frozen M1b covariance
- M1b 同 slow weights、同 inferred context、只关闭 covariance correction

无 room/context oracle，无训练，无 checkpoint 变化。

## Stage A：8 Seed，最长 836

- seeds：`712–719`
- validation generator：`1894`
- distractor repeats：`1, 2, 4, 8, 16`
- sequence lengths：`176, 220, 308, 484, 836`
- 每 horizon/seed：`8×8 = 64 episodes`，预计 128 个 return-conflict probes

Gates：

- M1b 每 seed/horizon clean `>= 0.98`
- repeat16 的 M1b mean return `>= 0.50`
- repeat1 到 repeat16 的 mean drop `<= 0.25`
- repeat16 上 M1b 每 seed 都胜过三个 baseline

## Stage B：预选 3 Seed，最长 4356

只有 Stage A 全 gate 通过才运行。seed 在 Stage A 结果前按 first/middle/last 固定为 `712, 715, 719`。

- validation generator：`1895`
- distractor repeats：`16, 32, 64, 96`
- sequence lengths：`836, 1540, 2948, 4356`
- 每 horizon/seed：`2×8 = 16 episodes`，预计 32 个 return-conflict probes
- 超过 1024 token 时用与 32-step window Transformer 数学等价的逐步 PFC evaluation，避免构造平方级 attention mask

Gates：

- repeat96 的 M1b mean return `>= 0.30`
- repeat16 到 repeat96 的 mean drop `<= 0.40`
- 每个 horizon 上 M1b mean 都胜过三个 baseline
- repeat96 的 M1b clean mean `>= 0.95`

Stage B 样本量用于极长序列趋势，不冒充最终高精度 headline。

## 固定输出

- Stage A：`runs/remap_former/long_delay_stage_a_validation1894.json`
- Stage B：`runs/remap_former/long_delay_stage_b_validation1895.json`
- 中文报告写入 `reports/` 对应文件。
