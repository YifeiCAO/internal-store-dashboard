# Hippoformer T2 预算匹配：学习率 Gate

## 结论

正式训练使用 **`1e-4`**。

选择依据是预先锁定的固定 dev 集 all-token sensory cross-entropy，未查看 validation 或 test。`1e-4` 的最佳 dev loss 为 `0.607174`，`3e-4` 为 `0.614831`，绝对优势 `0.007657`。

## 公平协议

- 独立调参 seed：`711`，不占用正式 seeds `712/713/714`。
- 两个候选：`1e-4`、`3e-4`。
- 每支均训练 `60 steps × batch 4 = 240 episode sequences`。
- 两支使用完全相同的训练 episode，batch generator seed 固定为 `711 + step`。
- 全参数 AdamW；唯一损失为全 token sensory CE；BF16；梯度裁剪 `1.0`。
- 每 20 步在同一组 `8 × batch 4 = 32` 个 dev episode 上评估。
- checkpoint 只按最低 dev loss 选择，validation/test 均未使用。

## 结果

| LR | step 20 dev | step 40 dev | step 60 dev | 最佳 step | clean | 状态 |
|---:|---:|---:|---:|---:|---:|---|
| `1e-4` | 0.641079 | 0.614102 | **0.607174** | 60 | 1.0000 | 有限、通过 |
| `3e-4` | 0.629763 | 0.633374 | 0.614831 | 60 | 0.9938 | 有限、通过 |

`3e-4` 在 step 20 更快，但 step 40 发生反弹，最终仍落后。两支均无 NaN/OOM，因此 `1e-4` 是按收敛质量胜出，不是因为另一支运行失败。

## 工程 Gate

新增的 `train_remap_hippoformer_matched.py` 已通过：

- 2-step GPU 前向、反向与 checkpoint smoke；
- 强制超时模拟中断，状态停在 step 3；
- 使用 `--resume` 从 step 3 接续到 step 20；
- 恢复模型、优化器、AMP、Python/CPU/CUDA RNG；
- 评估使用 `torch.no_grad()`，没有使用会禁用 Hippoformer 在线写入的 `torch.inference_mode()`；
- dev 评估前后恢复 RNG，不改变正式训练随机轨迹。

完整 LR 数据见 `runs/remap_former/hippoformer_t2_lr_gate_seed711_summary.json`。

## 下一步

固定 `LR=1e-4`，顺序训练 seeds `712/713/714`，每个 `600 steps × batch 4`。全部 checkpoint 冻结后，只做一次共享 validation paired evaluation，与 M1b covariance 和 M-delta 比较。
