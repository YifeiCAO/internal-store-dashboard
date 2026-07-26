# ReMAP-Former P0 v2 发布完整性报告

> 本版本不篡改旧 P0/G8 的冻结预期；它把当前代码、协议、checkpoint 和证据复制到独立 Git snapshot，从工程上消除“冻结 manifest 指向可变工作树”的问题。

## 结论

- 状态：`P0_V2_READY`
- snapshot commit：`b951334bcfc7174078d9a21cadd584fa3a71459b`
- snapshot clean：`True`
- checkpoint：`24` 个，hash 全匹配 `True`
- 3D 回归：`76 passed`
- 2D 当前回归：`353 passed`
- 严格因果合同：`4 passed`，future GT read/write = `0/0`

## 旧 P0 状态

旧 mutable-path manifest 全匹配：`False`。
- `remap_former/memory.py`：旧 `0f1a74a6cd12`，当前 `19cd46f3e707`
- `remap_former/context.py`：旧 `d778d31c6468`，当前 `eee633aaa417`
- `REMAP_FORMER_CONSOLIDATED_NEXT_PLAN_CN.md`：旧 `e6747a2d7353`，当前 `fd24bdb97fc0`

这些 mismatch 不被悄悄改绿；旧证据继续保留为 legacy。P0 v2 使用独立 source repo 和新的 commit/hash 作为后续实验唯一代码身份。

## 环境

- Python：`3.13.9`
- PyTorch / CUDA：`2.11.0+cu128` / `12.8`
- GPU：`NVIDIA GeForce RTX 3090`
- NumPy：`2.3.5`
- Windows OpenMP 合同：固定 `numpy_before_torch` 导入顺序。

## Gate

- isolated_git_snapshot: `True`
- snapshot_git_clean: `True`
- checkpoint_hashes_match: `True`
- dataset_manifests_present: `True`
- path_family_splits_disjoint: `True`
- memorymaze3d_76_of_76: `True`
- current_2d_regression_green: `True`
- strict_rollout_future_gt_zero_zero: `True`
- legacy_hash_drift_disclosed: `True`
- snapshot_contract_tests_pass: `True`

## 一键验证

```powershell
python verify_remap_next_p0_release.py --run-tests
```
