# MemoryMaze3D Visual A-B-A 三种子 Pilot 汇总

> 状态：**PASS**。这是机制开发阶段的 validation OOD pilot，不是 fresh sealed test；开发过程中看过 validation，不能把本表冒充盲测。

## 结论

三个训练种子都复现了同一个结构化图案：完整模型恢复旧 context 内容；PFC-only、固定 context 和 HPC-zero 均明显失败；强制错误历史会把读出几乎完全推向另一 context 的内容。因而本轮支持的是“历史推断的 context 通过 conjunctive address 调用 episode-local HPC”，而不是普通视觉平均或单纯容量增益。

## Validation OOD

种子：`[64001, 64002, 64003]`。训练 segment：`6/7/8`；验证 segment：`9/10/11`，路径族不重叠。

| 条件 | return conflict mean ± SD | 最差种子 | clean mean ± SD | query cosine | other-context rate |
|---|---:|---:|---:|---:|---:|
| Full | 0.9975 ± 0.0014 | 0.9962 | 1.0000 ± 0.0000 | 0.9319 | 0.0025 |
| PFC-only | 0.3338 ± 0.0046 | 0.3284 | 0.3410 ± 0.0240 | 0.0496 | 0.3284 |
| Fixed context | 0.4993 ± 0.0010 | 0.4982 | 0.9960 ± 0.0069 | 0.9057 | 0.4991 |
| HPC zero | 0.3042 ± 0.0102 | 0.2953 | 0.2291 ± 0.0167 | 0.0838 | 0.3046 |
| Wrong history | 0.0064 ± 0.0053 | 0.0021 | 1.0000 ± 0.0000 | 0.7684 | 0.9936 |
| Correct history | 0.9933 ± 0.0043 | 0.9890 | 1.0000 ± 0.0000 | 0.9339 | 0.0067 |
| Context oracle | 1.0000 ± 0.0000 | 1.0000 | 1.0000 ± 0.0000 | 0.9229 | 0.0000 |

## 成对差值

以下差值均在同一种子内先相减，再跨种子汇总；`n=3` 只报告描述统计，不伪装成高功效显著性检验。

| contrast | mean ± SD | min | max |
|---|---:|---:|---:|
| full_minus_pfc_only | 0.6637 ± 0.0036 | 0.6609 | 0.6678 |
| full_minus_fixed_context | 0.4981 ± 0.0017 | 0.4962 | 0.4992 |
| full_minus_hpc_zero | 0.6933 ± 0.0110 | 0.6809 | 0.7019 |
| correct_minus_wrong_history | 0.9869 ± 0.0096 | 0.9767 | 0.9956 |
| oracle_minus_full | 0.0025 ± 0.0014 | 0.0010 | 0.0038 |

## Pilot Gates

- all_source_runs_pass: `True`
- full_conflict_min_at_least_075: `True`
- full_minus_pfc_min_at_least_015: `True`
- full_minus_fixed_min_at_least_015: `True`
- wrong_history_other_rate_min_at_least_090: `True`
- clean_min_at_least_090: `True`
- causal_contract_all_true: `True`

## 因果合同

- all_transformer_is_pfc: `True`
- all_single_episode_local_fast_weight_hpc: `True`
- all_no_memory_slot_bank: `True`
- all_no_second_fast_weight_system: `True`
- all_no_oracle_metadata_inputs: `True`
- all_query_visual_target_hidden: `True`
- all_future_ground_truth_read_write_zero_zero: `True`

每个 episode 恰好两次可见内容写入；query target 不可见；query 阶段 future GT read/write 均为 `0/0`。模型输入不含 room ID、context ID、绝对位置或 place ID。trajectory signature 是由可观察 action history 计算的摘要。

## 边界

- 这是“两 context、一个 conflict place”的最小机制桥，不等于完整 3D world model。
- 当前 place 是共享 conflict-place code；完整 EC/grid/place 视觉路线尚未接入。
- validation 已用于 V1→V2 机制修正；下一步应先冻结协议，再扩到 5–8 seeds 与从未看过的 held-out layout/route/object split。
- V1 失败结果保留：仅靠 latent PFC state 加 cosine loss 时，完整模型约为 chance；V2 只加入可观察轨迹摘要投影和 target-vs-other 对比内容损失。
