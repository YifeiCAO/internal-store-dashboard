# M1b C3 嵌套式 128D Value Capacity 实验结果

## 一句话结论

C3 成功解决了主要的 content codec 容量问题，但没有通过严格 free-rollout 主门。128D HPC 在 `full` 自生成写入模式下相对 paired PFC-only 达到 **`+3.0976%`**，优势集中在 rollout 后半段；然而少数 episode 的灾难性负收益使 90% bootstrap 区间仍跨 0。下一步应冻结 128D 内容通路，只训练一个因果 neural call gate 来抑制这条负尾，而不是继续加宽或增加记忆槽。

## 本轮只改变了什么

C3 相对 C2 只把 HPC sensory value 从 64 维扩到 128 维：

- 旧 64D encoder 行、decoder 列和 projection 列原样复制；
- 新 decoder/projection 列初始化为 0；
- sensory-to-grid reverse content key 仍为 64D；
- Transformer PFC、EC、place、context、地址、fast-weight 写入和 recall adapter 不变；
- 没有 memory slot、第二套 fast weights、持久查找表、oracle context、room id、绝对位置或未来真实图像。

因此这是一个嵌套容量消融，不是更换记忆机制。

## 训练记录

- source checkpoint：`memorymaze3d_m1b_conditional_recall_adapter_pilot400_seed2151`
- source SHA256：`A9D6D6014A5EFC02FA18851A16BE077318E170C2B4EED0DE9E7EDD21C1E14AFB`
- seed：`2161`
- updates：`400`
- batch size：`2`
- sequence length：`64`
- rollout curriculum：`2,4,8,16`
- context：`20`
- future write：observed-only
- learning rate：`5e-5`
- peak VRAM：`4.746 GiB`
- elapsed：约 `1,860 s`
- finite gate：通过
- final checkpoint SHA256：`8D17844FFC291BD16503A7A1A938C26ABEC7D0EB60282908473102A43BED1706`

训练的 final-horizon descent gate 未通过：

- H16 early total mean：`0.0172225`
- H16 late total mean：`0.0182027`
- H16 early rollout MSE：`0.0075874`
- H16 late rollout MSE：`0.0081099`

因此 C3 数值稳定，但不能宣称 H16 optimization 已收敛。

## Gate A：内容容量

评估使用未参与此前判断的 `Val[192:224]`。

| 指标 | C3 | 预注册门槛 | 判定 |
|---|---:|---:|---:|
| direct AE / zero-content MSE ratio | `0.183588` | `<= 0.36` | 通过 |
| direct code cosine | `0.923694` | 记录项 | 良好 |
| actual retrieval / zero-content MSE ratio | `0.242422` | `<= 0.39` | 通过 |
| actual retrieval cosine | `0.917585` | 记录项 | 良好 |
| corrected retrieval ratio | `0.227950` | 记录项 | 良好 |
| conditioning | stable | stable | 通过 |

地址侧也保持健康：

- best-prior address cosine：`0.80313`
- repeated-pose hit rate：`0.85012`

结论：

> **Gate A 强通过。128D 已经把主要 content codec 容量瓶颈排除，下一步不再继续加宽。**

teacher-forced 像素结果仍显示 fusion 不是自动成立的：

- actual retrieval 相对 PFC-only：`+0.8422%`
- oracle current content 相对 PFC-only：`-52.951%`
- corrected retrieval 相对 PFC-only：`-33.718%`

高质量 content code 已经存在，但如何安全调用它仍是瓶颈。

## Gate B：严格 paired free rollout

评估使用与 Gate A 不重叠的 `Val[224:288]`：

- episode：`64`
- context：`20`
- free rollout：`44`
- bootstrap：`5,000`
- future ground-truth reads：`0`
- future ground-truth writes：`0`

| 分支 | MSE | 相对 paired PFC-only |
|---|---:|---:|
| PFC-only | `0.01644746` | 基准 |
| observed-only HPC | `0.01624937` | **`+1.2044%`** |
| full / self-generated writes | `0.01593798` | **`+3.0976%`** |

observed-only HPC：

- 绝对增益：`+0.198086 × 10^-3`
- 90% bootstrap CI：`[-0.775890, +1.064588] × 10^-3`
- episode win rate：`65.625%`

full / self-generated writes：

- 绝对增益：`+0.509480 × 10^-3`
- 90% bootstrap CI：`[-0.499917, +1.438735] × 10^-3`
- episode win rate：`62.5%`

预注册主分支 observed-only 没有达到 `2%`，且 bootstrap 下界没有大于 0。按照冻结协议，C3 **未通过 Gate B**，也不在同一 slice 上追加 C2 比较。

## 尾部归因

full 模式不是偶然只在少数时间点获益：

| rollout 区域 | full 相对 PFC 绝对增益 |
|---|---:|
| 前 11 步 | `+0.140309 × 10^-3` |
| 中间 22 步 | `+0.394084 × 10^-3` |
| 后 11 步 | `+1.109443 × 10^-3` |

- full 在 `37/44` 个 rollout 步上优于 PFC；
- observed-only 在 `32/44` 个 rollout 步上优于 PFC；
- full 的 episode 中位增益为 `+0.0002950`；
- full 的最差 episode 增益为 `-0.0256571`；
- full 的第 10 百分位增益为 `-0.0021541`；
- full 的最大正增益为 `+0.0139962`。

这说明平均证据被一条少数但很重的负收益尾巴破坏。海马在多数时间步和多数 episode 上已有正贡献，问题是当前 static call 对不可靠 retrieval 仍然无条件应用。

## 正式判断

1. **容量问题已解决**：direct ratio 从 C2 的 `0.3939` 降到 `0.1836`，actual retrieval ratio 从 `0.4166` 降到 `0.2424`。
2. **主统计门未过**：observed-only 只有 `+1.2044%`，CI 跨 0。
3. **full 模式是重要正信号**：无未来真值泄漏、点增益 `+3.0976%`，且优势随 rollout 变长而增大。
4. **当前瓶颈是调用稳定性**：少数 episode 的负尾，而不是地址、content capacity 或记忆长度。
5. **停止加宽和加槽**：保留单一 episode-local fast-weight HPC。
6. **下一步 C4**：从 C3 checkpoint warm-start，冻结全部旧模块，只训练一个读取 PFC、retrieval、alignment、reverse confidence 和 correction confidence 的因果 neural call gate。`gate=0` 严格回退 PFC，`gate=1` 保留完整 C3。

## 机器产物

- checkpoint：`runs/remap_former/memorymaze3d_m1b_nested_value_capacity_c3_pilot400_seed2161/checkpoint_final.pt`
- training summary：`runs/remap_former/memorymaze3d_m1b_nested_value_capacity_c3_pilot400_seed2161/training_summary.json`
- Gate A：`reports/memorymaze3d_m1b_nested_value_capacity_c3_content_path_val192_223/summary.json`
- Gate B：`reports/memorymaze3d_m1b_nested_value_capacity_c3_component_val224_287/summary.json`
- paired errors：`reports/memorymaze3d_m1b_nested_value_capacity_c3_component_val224_287/paired_episode_errors.npz`
- paired figure：`reports/memorymaze3d_m1b_nested_value_capacity_c3_component_val224_287/variable_component_audit.png`
