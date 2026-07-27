# MemoryMaze3D 多位置 Visual A-B-A 三种子结果

> 状态：**PASS_WITH_SECONDARY_STABILITY_NOTE**。核心机制 gate 为 `3/3`；严格的
> correct-history 与 full 近同分 gate 为 `2/3`。这是看过 development
> validation 后得到的 3D visual bridge，不是 sealed test，也不是完整
> simulator-coupled free rollout。

## 核心结论

- held-out variable-layout DINO 内容上，Full conflict 为
  `0.9960 ± 0.0069`，
  最差种子 `0.9880`。
- clean cosine 为
  `0.9663 ± 0.0529`，
  latent context re-entry 为
  `1.0000 ± 0.0000`。
- HPC-zero 固定为 `0.5000`；fixed-context 为 `0.0000`；wrong-history
  将读出推向另一 context，平均正确率仅
  `0.0715`。
- correct-history 恢复到
  `0.9840 ± 0.0277`，
  orthogonal context oracle 为 `1.0000 ± 0.0000`。

## 三种子 Validation

种子：`[65001, 65002, 65003]`。每个种子独立训练 100 steps；训练布局种子
`620000–620007`，验证布局种子 `630000–630003`；route 与 object-assignment
family hash-disjoint。

| 条件 | conflict mean ± SD | 最差种子 | clean cosine mean ± SD | context re-entry | other-context |
|---|---:|---:|---:|---:|---:|
| Full | 0.9960 ± 0.0069 | 0.9880 | 0.9663 ± 0.0529 | 1.0000 | 0.0040 |
| HPC-zero | 0.5000 ± 0.0000 | 0.5000 | 0.0000 ± 0.0000 | 1.0000 | 0.5000 |
| Fixed context | 0.0000 ± 0.0000 | 0.0000 | 1.0000 ± 0.0000 | 0.5000 | 1.0000 |
| Wrong history | 0.0715 ± 0.1238 | 0.0000 | 0.9999 ± 0.0001 | 0.0723 | 0.9285 |
| Correct history | 0.9840 ± 0.0277 | 0.9520 | 0.9910 ± 0.0130 | 1.0000 | 0.0160 |
| Context oracle | 1.0000 ± 0.0000 | 1.0000 | 0.9999 ± 0.0001 | 1.0000 | 0.0000 |

## 成对机制差值

| contrast | mean ± SD | min | max |
|---|---:|---:|---:|
| full_minus_hpc_zero | 0.4960 ± 0.0069 | 0.4880 | 0.5000 |
| full_minus_fixed_context | 0.9960 ± 0.0069 | 0.9880 | 1.0000 |
| correct_minus_wrong_history | 0.9125 ± 0.1126 | 0.7855 | 1.0000 |
| oracle_minus_full | 0.0040 ± 0.0069 | 0.0000 | 0.0120 |

## 架构合同

- all_transformer_is_pfc: `True`
- all_frozen_ec_grid_place_scaffold: `True`
- all_single_episode_local_hpc: `True`
- all_frozen_identity_value_codec: `True`
- all_pfc_direct_output_zero: `True`
- all_no_slots_or_second_memory: `True`
- all_no_oracle_metadata_inputs: `True`
- all_future_gt_zero_zero: `True`

模型每个 episode 恰好写 8 次、query 4 次；HPC 严格 read-before-write；
future GT read/write 为 `0/0`。输入只有官方六类 action 与因果滞后的/中性
DINO 特征，不含 room/context/pose/place/route/assignment ID。Transformer
不能直接输出答案，只能通过 latent context 调用一张 episode-local HPC。

## 残余缺口

`seed=65001` 中 Full 为 `0.9880`，首次 context 锚点的 correct-history
intervention 为 `0.9520`，差 `0.0360`，超过预设 `0.02` 容差。因此不把
“correct-history 与 full 完全等价”写成 3/3 结论；更稳妥的结论是：
correct-history 绝对性能三种子均高，且相对 wrong-history 的恢复强。

## 适用边界

- 视觉内容来自真实 MemoryMaze3D 9×9 可变布局的 224×224 渲染，并用冻结
  DINOv2 spatial ROI 表征。
- official action 驱动 action-only periodic SE(2) EC；但事件内容被放入受控
  A-B-A 序列，尚未与 simulator 每一步 RGB 轨迹端到端绑定。
- 这是多位置视觉记忆桥，不是完整 3D world-model 预测，也不是 free rollout。
- 下一步应冻结本版本，生成从未查看的 test layout/object 数据；随后再做
  simulator-coupled trajectory 版本与 matched Transformer/Hippoformer/Titans
  基线。
