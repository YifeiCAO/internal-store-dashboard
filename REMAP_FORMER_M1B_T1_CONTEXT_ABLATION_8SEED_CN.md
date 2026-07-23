# ReMAP-Former M1b：T1 Context 因果消融（8 Seed）

> 冻结同一批 M1b checkpoint；同一批单房间轨迹；未来 memory writes 为 0。固定 null context 是严格 free rollout；teacher feedback 仅用于误差分解。

## 协议

- checkpoint seeds: `[712, 713, 714, 715, 716, 717, 718, 719]`
- context / max rollout: `64 / 4096`
- episodes/checkpoint: `8`
- protocol SHA256: `f17c79dcc7c5a7e8b0367049f060fa9b17b235ca4e4a83764e2b301ab54124f9`
- seed 712 pilot 已在协议冻结前披露；本实验用于检查该因果效应能否跨 8 个冻结 checkpoint 稳定复现。

## Accuracy Curve

| horizon | 动态 context final | 动态 HPC | 固定 null final | 固定 null HPC | teacher final* | 固定-动态 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9167 | 0.9167 | 1.0000 | 1.0000 | 0.9167 | +0.0833 |
| 4 | 0.8942 | 0.8990 | 1.0000 | 1.0000 | 0.8942 | +0.1058 |
| 8 | 0.8995 | 0.8995 | 1.0000 | 1.0000 | 0.8995 | +0.1005 |
| 16 | 0.8623 | 0.8768 | 1.0000 | 1.0000 | 0.8641 | +0.1377 |
| 32 | 0.8084 | 0.8484 | 1.0000 | 1.0000 | 0.8227 | +0.1916 |
| 64 | 0.7288 | 0.7997 | 1.0000 | 1.0000 | 0.7790 | +0.2712 |
| 128 | 0.7268 | 0.8082 | 1.0000 | 1.0000 | 0.7948 | +0.2732 |
| 256 | 0.6842 | 0.7865 | 1.0000 | 1.0000 | 0.7640 | +0.3158 |
| 512 | 0.6830 | 0.7838 | 1.0000 | 1.0000 | 0.7629 | +0.3170 |
| 1024 | 0.6708 | 0.7782 | 0.9987 | 1.0000 | 0.7564 | +0.3278 |
| 2048 | 0.6613 | 0.7736 | 0.9993 | 1.0000 | 0.7533 | +0.3380 |
| 4096 | 0.6736 | 0.7790 | 0.9997 | 1.0000 | 0.7597 | +0.3260 |

注：teacher feedback 看到了未来真实 sensory，只是 exposure-bias 诊断上界，不是合法模型成绩。

## 4096-Step Paired Results

| seed | 动态 final | 动态 HPC | 固定 final | 固定 HPC | teacher final |
|---:|---:|---:|---:|---:|---:|
| 712 | 0.6587 | 0.7640 | 0.9997 | 1.0000 | 0.7414 |
| 713 | 0.6639 | 0.7774 | 0.9997 | 1.0000 | 0.7562 |
| 714 | 0.6172 | 0.7360 | 0.9997 | 1.0000 | 0.7134 |
| 715 | 0.7502 | 0.8283 | 0.9997 | 1.0000 | 0.8164 |
| 716 | 0.6841 | 0.7866 | 0.9997 | 1.0000 | 0.7707 |
| 717 | 0.6629 | 0.7746 | 0.9997 | 1.0000 | 0.7522 |
| 718 | 0.6816 | 0.7864 | 0.9997 | 1.0000 | 0.7647 |
| 719 | 0.6703 | 0.7786 | 0.9997 | 1.0000 | 0.7628 |

## Causal Endpoints

- 固定 null context 的 4096 final: `0.9997`
- 固定 null context 的 4096 HPC: `1.0000`
- 固定 context 带来的配对提升: `+0.3260`
- teacher feedback 带来的配对提升: `+0.0861`
- 动态 context 下 HPC-only 减 final（融合损失）: `+0.1054`

## Gates

- dynamic_reference_exactly_reproduced: `True`
- fixed_null_final_4096: `True`
- fixed_null_hpc_4096: `True`
- fixed_null_gain_4096: `True`
- future_writes_zero: `True`
- context_write_count_exact: `True`
- all_model_states_unchanged: `True`
- shared_trajectory_hashes: `True`

## 结论

固定唯一 context 在不改权重、不增加未来写入的条件下恢复近满分长程回忆，因此单房间 T1 的主要失败源被定位为动态 context 对 place 地址的碎片化。teacher feedback 与 HPC/final 差距分别刻画次要的 exposure bias 和融合损失。

决策：`M1B_T1_CONTEXT_FRAGMENTATION_CONFIRMED`

## 文件

- protocol: `runs/remap_former/m1b_t1_context_ablation_protocol.json`
- result: `runs/remap_former/m1b_t1_context_ablation_seed2071601.json`
