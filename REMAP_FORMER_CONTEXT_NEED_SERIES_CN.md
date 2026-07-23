# ReMAP-Former：Context-Need 完整证据链

> 2026-07-18。正式 M1b checkpoint 始终冻结。本系列回答一个单独问题：在没有 room/task/position ID、没有第二套 fast weights、没有 memory slot 的条件下，模型能否判断当前 episode 是否真的需要 context-conditioned addressing。

## 一句话结论

**可以判断。当前在线工作版本让 HPC 先按 null context 工作，再由它自己的 retrieval contradiction 首次 crossing 调用 dynamic context；PFC hidden 单独蒸馏仍然失败。**

- 冻结的解析式 controller 在全新大样本 G1 上 13/13 gates 全绿：T1 strict 4096 `0.9721`，T2 return-conflict `0.8125`。
- 只读 PFC prefix hidden 的 16,770 参数 neural head 在固定 600 步后失败：T1 strict 4096 `0.9159`，held-out balanced accuracy `0.8320`。
- G2 null-first 在冻结 fresh seeds 上 15/15 gates 全绿：held-out balanced accuracy/AUROC `0.9805/1.0000`，T1 strict 4096 `0.9743`，T2 return-conflict `0.8594`。
- G2 的 64 条 T1 中 61 条直接复用首遍 HPC，平均 prefix pass 从 G1 的 `2.0000` 降到 `1.0469`；仍只有一套被选中的 episode-local content HPC。
- G3 用一个共享阈值跨 8 个冻结 checkpoint 通过 21/21 gates：pooled T1 4096 `0.9985`、T2 return-conflict `0.8242`、T1 passes `1.0000`。
- 但模块哈希审计显示八个模型只在 `state_token/context_head` 上独立，controller-facing `EC/place/fixed-null address/HPC` 完全共享。因此 G3 是八套 dynamic-context 下游确认，不是八次独立 controller 复现。
- G4 固定 G3 数值阈值跨 7 个 OOD family 正式失败：grid13 假阳性，query2/sparse 假阴性；dense 是独立下游容量失败。
- G5 改为“熟悉 neural place 重访上的最大 contradiction”，fresh calibration/test path banks 零重叠，61/61 gates 全绿：BAcc/AUROC `0.9941/1.0000`，T1 4096/passes `0.9935/1.0125`，T2 return `0.8080`。
- G6 冻结 G5 score/threshold，只把调用改成第一次因果越阈值时触发。正式结果为 92/93 FAIL：BAcc/AUROC `0.9941/1.0000`，T1 `0.9972`，T2 return `0.8420`，T2 prefix passes 从固定 G5 的 `2.0000` 降到 `1.7963`；唯一红门是 fresh sparse bank 的 `0.6719 < 0.70`。
- G6 online 与 fixed G5 的 decision agreement 为 `1.0000`，最终 memory/covariance/post-prefix logits 最大误差不超过 `7.63e-6`。因此唯一红门不是提前调用造成的状态偏差。
- G6b 在四个再全新的 sparse banks、1024 个 return-conflict probes 上 29/29 stability gates 全绿：pooled `0.8086`，Wilson 95% CI `[0.7834,0.8315]`，四个 bank 全部至少 `0.70`。它把 G6 红门分类为单-bank 尾部，但不覆盖 G6 的正式 FAIL。
- G7 在再全新的零重叠 path bank 上预注册 H=`32/48/60/72`，148/148 gates 全绿。dynamic coverage 为 `0/0.1667/0.6667/1.0`，T2 return 为 `0/0.0938/0.4948/0.7656`，Spearman `1.0`；每个窗口的 online/fixed decision agreement 都是 `1.0`。
- G8 把同一四点分布改成逐 episode 平衡随机、对模型隐藏的 cutoff，在第三套零重叠 fresh bank 上 59/59 gates 全绿。G7 预注册的 uniform-mixture coverage/return/passes 为 `0.4583/0.3385/1.3864`，G8 实现值为 `0.4635/0.3885/1.3923`，绝对差仅 `0.0052/0.0499/0.0059`；54/54 shadow audits 证明 H 后 crossing 没有进入部署预测。
- 因此 G5 仍是没有历史红门的 fixed-step anchor，G7 给出 paired variable-horizon 因果曲线，G8 是当前通过的 hidden-cutoff streaming confirmation；G6 的 92/93 FAIL 仍必须并列披露。失败的 PFC-only head 不追认为正式模型，正式 M1b checkpoint 也未被训练或改写。

## 当前工作版本

```text
流式 observed steps，每个 episode 的隐藏环境 cutoff H 属于 {32,48,60,72}
  action + shifted sensory
      -> frozen WindowTransformer PFC

  action -> neural EC -> sparse neural place
  place x null context
      -> episode-local covariance-HPC
      -> read-before-write retrieval r_t

  当前 neural place 与严格更早 place codes
      -> familiarity_t = max cosine(place_t, place_<t)
      -> recurrent_t = familiarity_t >= 0.99999

  observed value v_t 与 r_t
      -> contradiction_t = ||v_t-r_t||^2 - ||v_t||^2
      -> 若 recurrent_t 且 contradiction_t 首次跨过冻结阈值
           立即锁定调用 inferred dynamic context
      -> 直到当前 cutoff H 都未跨阈值：保留 null context 和现有 HPC

null decision
  -> 原 HPC 直接续接，不重放 prefix

dynamic trigger at step t
  -> 当前 sensory_t 只用于 read-before-write 矛盾判断
  -> 用已观察的 step 0..t dynamic address 重建一次
  -> 从 t+1 起持续使用 dynamic HPC，不可逆、不再切换

两条分支最终都只保留一套 selected episode-local HPC
  -> T1 strict rollout 在 prefix 后 future writes = 0
```

这里没有两个并行持久 memory。null 分支直接把在线建立的 HPC 变成后续唯一 memory；dynamic 分支在首次触发时，用截至当时的已观察历史替换为一套 dynamic HPC。G7 显示证据是逐步出现的：H32 尚无 T2 crossing，H48 覆盖 `1/6`，H60 覆盖 `2/3`，H72 覆盖全部；H72 trigger 中位数/范围仍为 `57/41-69`。G8 再把 H 逐 episode 随机隐藏，部署混合平均为 `0.4635` coverage、`0.3885` return 和 `1.3923` prefix passes。

## G0：方向正确，小样本冻结门失败

协议先冻结 score、64-step prefix、两类 calibration 中位数中点和全套 gates，再看结果。

| 指标 | 结果 |
|---|---:|
| calibration T1 / T2 median | `0.4838 / 0.7060` |
| held-out T1 null selection | `0.8906` |
| held-out T2 dynamic selection | `0.9688` |
| held-out balanced accuracy | `0.9297` |
| held-out AUROC | `0.9961` |
| T1 strict 4096 final，8 episodes | `0.9313` |
| T2 return-conflict / clean，64 episodes | `0.8594 / 1.0000` |

G0 因两个预注册 gate 未过而正式判 `CONTEXT_NEED_RETRIEVAL_G0_FAIL`。没有在这批 held-out 上改 threshold、prefix 或 score。失败集中在 T1 高尾部：8 个 strict episode 中 2 个被选成 dynamic，放大了小样本 4096 指标。

## G1：同规则、全新种子、高功效复现通过

G1 没有改机制，只把 calibration/held-out 从每类 64 扩到 256，T1 strict 从 8 扩到 64，T2 end-to-end 扩到 128；所有 generator seed 全新。

| 指标 | 结果 |
|---|---:|
| frozen threshold | `0.60076547` |
| calibration T1 null / T2 dynamic | `0.9141 / 0.9844` |
| fresh held-out T1 null / T2 dynamic | `0.9375 / 0.9922` |
| fresh held-out balanced accuracy | `0.9648` |
| fresh held-out AUROC | `0.9946` |
| T1 strict 4096 final / HPC，64 episodes | `0.9721 / 0.9823` |
| T2 return-conflict / clean，128 episodes | `0.8125 / 0.9938` |
| decision future writes / content HPC count | `0 / 1` |

13/13 gates 全绿，状态为 `CONTEXT_NEED_RETRIEVAL_G1_REPLICATION_PASS`。这支持的结论是：**prefix HPC retrieval consistency 是可复现的 context-need 因果信号。**它不等于 room classifier；模型输入没有 room/task/position/query/return/conflict metadata。

## Neural Pilot：PFC-only 蒸馏失败

为去掉 inference-time transient HPC scoring，我们冻结 M1b，只训练一个 attention-pooling head：

- 输入：前 64 步 frozen PFC hidden；
- 参数：`16,770`；
- loss：唯一 SmoothL1，回归 G1 teacher 的 standardized score；
- 无 sensory CE、无 ST estimator、无 room/context label；
- fixed step600，无 checkpoint selection；
- neural decision 期间实测 HPC forward 调用为 `0`。

| 指标 | 结果 |
|---|---:|
| held-out teacher decision agreement | `0.8047` |
| held-out T1 null / T2 dynamic | `0.7656 / 0.8984` |
| held-out balanced accuracy | `0.8320` |
| teacher-score Pearson / MAE(z) | `0.6355 / 0.6887` |
| T1 strict 4096 final / HPC | `0.9159 / 0.9445` |
| T2 return-conflict / clean | `0.7578 / 0.9969` |

固定 600 步结果为 `CONTEXT_NEED_NEURAL_SINGLE_SEED_FAIL`，9/14 gates 通过。失败不是接线或泄漏：base hash 不变、future writes 为 0、决策期 HPC calls 为 0、selected replay 仍只有一个 HPC。失败发生在信息压缩：teacher 的关键证据是 `v_t` 与 HPC retrieval `r_t` 的匹配，而 frozen PFC hidden 没有接收 `r_t`；预测分数相关性只有 `0.6355`，并同时产生过多 T1 dynamic 和过少 T2 dynamic。

## G2：Null-first 单状态控制通过

G2 没有训练新参数。它把 G1 的“先建 dynamic scoring HPC，再总是 replay”改成更接近真实调用的状态机：默认先以 null context 建 HPC，只有 read-before-write retrieval 比零检索更糟时，才调用 action-history inferred context。为避免少量冲突被时间均值稀释，冻结聚合为前 72 步最大 10% contradiction 的均值；阈值仍只由 calibration 两类中位数的中点确定。

| 指标 | 结果 |
|---|---:|
| frozen threshold | `0.19240336` |
| fresh held-out T1 null / T2 dynamic | `0.9609 / 1.0000` |
| fresh held-out balanced accuracy / AUROC | `0.9805 / 1.0000` |
| T1 strict 4096 final / HPC，64 episodes | `0.9743 / 0.9819` |
| T1 mean prefix passes / null reuse | `1.0469 / 61 of 64` |
| T2 return-conflict / clean，128 episodes | `0.8594 / 0.9891` |
| T2 dynamic selection / mean prefix passes | `1.0000 / 2.0000` |
| decision future writes / selected content HPC count | `0 / 1` |

状态为 `CONTEXT_NEED_NULL_FIRST_G2_PASS`，15/15 gates 全绿。实现审计还证明：stateful covariance scan 与正式 HPC 完全等价；null-selected replay 恰为 `0`；T2 return-conflict 全部发生在 72-step decision 之后；source model 前后哈希一致。G2 因此替代 G1 成为当前首选 context-need controller。

## G3：共享阈值跨 Context Checkpoint 通过

G3 保持 G2 的 prefix、score、tail fraction 和状态机不变。seed 712-719 共用 fresh calibration/held-out/end-to-end 轨迹，并由全部 calibration score 只计算一个 pooled midpoint threshold `0.18923524`。

| pooled 指标 | 结果 |
|---|---:|
| held-out T1 null / T2 dynamic | `0.9766 / 1.0000` |
| held-out balanced accuracy / AUROC | `0.9883 / 1.0000` |
| T1 strict 4096 final / mean passes，128 episodes | `0.9985 / 1.0000` |
| T2 return-conflict / clean，256 episodes | `0.8242 / 0.9969` |
| 逐 checkpoint T2 return-conflict range | `0.7812-0.8438` |

状态为 `CONTEXT_NEED_NULL_FIRST_G3_MULTICHECKPOINT_PASS`，21/21 gates 全绿；所有 stateful scan 精确、source hash 不变、null replay 为 0、future writes 为 0。

通过后做的 pathway hash audit 给出关键限制：完整 state 有 8 个唯一 hash，`state_token` 和 `context_head` 各有 8 个；但 `EC/place/address/HPC` 联合只有 1 个唯一 hash，calibration/held-out score vector 也各只有 1 个。因此最严谨的结论是：**同一个 null-first controller 在八套不同 dynamic-context 推断下游上稳定工作。**不能把它写成八套 controller-facing 权重的独立复现。

## G4：固定 Raw Score 的 OOD 边界

G4 不重新校准 G3 threshold，预先冻结 7 个 path/grid/query/conflict families。generator gauge、prefix evidence 和 return-after-decision gates 全部健康，但正式状态为 `CONTEXT_NEED_NULL_FIRST_G4_OOD_FAIL`。

| family | selection / endpoint 失败 | forced endpoint 归因 |
|---|---:|---|
| grid13 | T1 null `0.4297`；4096 `0.7648` | fixed-null `0.9793`，controller false-positive |
| query2 | T2 dynamic `0.3281`；return `0.2500` | forced-dynamic `0.7500`，controller false-negative |
| sparse | T2 dynamic `0.0156`；return `0.0000` | forced-dynamic `0.8281`，controller false-negative |
| dense | T2 dynamic `1.0000`；return `0.5469` | forced-dynamic 仍 `0.5469`，downstream confirmed |

G4 证明 G2/G3 raw top-10% score 不是 task-invariant：大网格较少重访时，陌生地址 retrieval 噪声抬高 T1；冲突稀疏或 episode 变短时，固定 top-8 又稀释 T2 真证据。dense 则是另一个容量问题，不能靠 controller 修。

## G5：熟悉地址矛盾通过 Fresh OOD

G5 只接受 neural place/address 的因果信息：当前 place 与任一严格更早 place 的 cosine 至少 `0.99999` 时，该步才算熟悉重访；episode score 是这些步上最大的 retrieval contradiction。没有 room、位置、mask 或 metadata，也没有训练参数。

G5 calibration 使用 family seed3571701 的 train bank；held-out 使用零重叠的 seed3671701 test bank。机制、similarity、aggregation、task list 和 gates 都在看 G5 score 前冻结。

| family | T1 null | T2 dynamic | T1 4096 | passes | T2 return |
|---|---:|---:|---:|---:|---:|
| grid7 | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.7891` |
| grid15 | `0.9297` | `1.0000` | `0.9388` | `1.0625` | `0.7578` |
| query3 | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9062` |
| query2 | `1.0000` | `1.0000` | `0.9963` | `1.0000` | `0.8594` |
| sparse | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.7969` |
| dense selection-only | `1.0000` | `1.0000` | N/A | N/A | N/A |

Pooled BAcc/AUROC 为 `0.9941/1.0000`，T1 strict 4096/passes 为 `0.9935/1.0125`，T2 return-conflict/clean 为 `0.8080/0.9953`；61/61 gates 全绿，状态 `CONTEXT_NEED_RECURRENCE_G5_PASS`。grid15 是明确弱项，dense downstream ceiling 仍未解决。

## G6：首次越阈值的因果在线调用

G6 不重新校准任何东西。observation horizon 仍为 72，recurrence cosine 仍为 `0.99999`，numeric threshold 原样固定为 `0.07688456773757935`。默认 null-HPC 逐步 read-before-write；第一次熟悉地址 contradiction 越阈值后，只回放 `0..t` 的已观察历史，从 `t+1` 起锁定 dynamic HPC。未触发 episode 从不回放。

| 指标 | G6 formal |
|---|---:|
| gates | `92/93`，FAIL |
| pooled BAcc / AUROC | `0.9941 / 1.0000` |
| T1 strict 4096 | `0.9972` |
| T2 return-conflict / clean | `0.8420 / 0.9977` |
| T2 online / fixed prefix passes | `1.7963 / 2.0000` |
| trigger median / range | `57 / 41-69` |
| online / fixed decision agreement | `1.0000` |
| final memory / covariance / post-prefix logits max abs | `1.79e-6 / 3.81e-6 / 5.72e-6` |

唯一失败为 `sparse_conflict.t2_return = 0.671875 < 0.70`，相差 4/128 probes。该 family 的 selection dynamic 为 `1.0000`，八个 checkpoint 上 online 与 fixed G5 的 decision、state 和 post-prefix logits 仍全部等价。因此 G6 按协议保持 `CONTEXT_NEED_ONLINE_G6_FAIL`，但失败不归因于 online timing 实现。

## G6b：Sparse Path-Bank 稳定性归因

G6 失败后另冻四个 canonical overlap 为 0 的 sparse test banks；每 bank 使用 8 checkpoints x 32 episodes，总计 1024 个 return-conflict probes。controller、threshold、horizon 和 checkpoint 完全不变。

| fresh bank | Return-conflict | Clean | Online passes |
|---|---:|---:|---:|
| seed4271701 | `0.8047` | `0.9990` | `1.8785` |
| seed4371701 | `0.7656` | `0.9974` | `1.8785` |
| seed4471701 | `0.7734` | `0.9984` | `1.9028` |
| seed4671701 | `0.8906` | `0.9990` | `1.9097` |

Pooled 为 `828/1024 = 0.8086`，Wilson 95% CI `[0.7834,0.8315]`，bank median `0.7891`，四个 bank 全部至少 `0.70`；29/29 stability gates 全绿，状态 `CONTEXT_NEED_ONLINE_G6B_SPARSE_STABLE`。冻结解释是：G6 的 sparse 红门属于单-bank 尾部，而不是 online/fixed 状态差异。**这条审计不把 G6 formal FAIL 改写成 PASS。**

## G7：Variable Observation Horizon

G7 冻结 G5 numeric threshold、`0.99999` recurrence gate 和 G6 first-crossing 状态机，只改变因果 cutoff。四个 horizon 使用同一批 action/sensory/object/path/checkpoint/episode 前缀；未在 H 前 crossing 的 episode 保持 null，绝不使用 H 后证据。fresh family seed `4771701` 与 G5/G6/G6b/original train canonical overlap 全为 0。

首次 smoke 因 prefix/full GPU 地址差 `5.07e-7` 触发两个 bitwise-zero 门，原始 `22/24 FAIL` 保留；正式前只把地址未来不变性改为显式 `<=1e-6` 数值门，其他协议字段和性能门不变。v2 smoke 为 24/24。

| H | T1 null | T2 dynamic | censored | BAcc/AUC | T1 4096 | T2 return | clean | online/fixed passes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | `0.9987` | `0.0000` | `1.0000` | `0.4993/0.3436` | `0.9978` | `0.0000` | `1.0000` | `1.0000/1.0000` |
| 48 | `0.9961` | `0.1667` | `0.8333` | `0.5814/0.3295` | `0.9972` | `0.0938` | `1.0000` | `1.1510/1.1667` |
| 60 | `0.9948` | `0.6667` | `0.3333` | `0.8021/0.6718` | `0.9947` | `0.4948` | `0.9992` | `1.5958/1.6667` |
| 72 | `0.9896` | `1.0000` | `0.0000` | `0.9948/0.9987` | `0.9919` | `0.7656` | `0.9979` | `1.7986/2.0000` |

正式结果为 `CONTEXT_NEED_ONLINE_G7_VARIABLE_HORIZON_PASS`，148/148 gates。coverage range `1.0`，return H72-H32 `+0.7656`，horizon/return Spearman `1.0`，online/fixed minimum decision agreement `1.0`，state/logit 最大误差 `4.77e-6`。H32/H48 的低 AUC 不隐藏：它说明证据出现前无法合法预测未来 conflict；与此同时 T1/clean 保持接近 1，排除了通用 rollout 崩坏。

## G8：Randomized Hidden-Cutoff Streaming

G8 不再让四个 horizon 共享同一个 episode 做曲线，而是给每个 episode 从 `32/48/60/72` 的平衡分布中随机分配一个 cutoff。分配结果不作为模型 token，环境只在对应 H 停止 evidence accumulation；部署 prediction 封存后才计算完整 H72 shadow trace，用来审计 crossing 是否被 cutoff 合法截断。fresh family seed `5371701` 与 G5/G6/G6b/G7/original train 的 canonical overlap 均为 0。

| 指标 | G7 uniform 预注册预测 | G8 hidden realized | 绝对差 |
|---|---:|---:|---:|
| Dynamic coverage | `0.4583` | `0.4635` | `0.0052` |
| T2 return-conflict | `0.3385` | `0.3885` | `0.0499` |
| Online prefix passes | `1.3864` | `1.3923` | `0.0059` |
| T1 strict 4096 | `0.9954` | `0.9978` | 未设误差门 |
| T2 clean | `0.9993` | `0.9982` | 未设误差门 |

| Hidden H | T1 null | T2 dynamic | T1 4096 | T2 return | clean | online/fixed passes |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | `1.0000` | `0.0000` | `0.9980` | `0.0000` | `1.0000` | `1.0000/1.0000` |
| 48 | `1.0000` | `0.1667` | `0.9990` | `0.0608` | `1.0000` | `1.1510/1.1667` |
| 60 | `0.9792` | `0.6875` | `0.9957` | `0.6319` | `0.9966` | `1.6160/1.6875` |
| 72 | `1.0000` | `1.0000` | `0.9986` | `0.8611` | `0.9961` | `1.8021/2.0000` |

正式结果为 `CONTEXT_NEED_ONLINE_G8_HIDDEN_CUTOFF_PASS`，59/59 gates。coverage/return 的 cutoff Spearman 均为 `1.0`；同 cutoff online/fixed minimum decision agreement 为 `1.0`，最大 state/logit error 为 `5.25e-6`。54/54 shadow batches 全过，共有 826 次 full-shadow crossing 因发生在隐藏 cutoff 之后而被 deployment 保持为 null。该结果确认 G7 曲线能预测 fresh hidden-cutoff mixture，但只覆盖冻结的四点均匀分布，不外推到任意 cutoff policy。

## 当前科学判断

1. **T1 问题已因果定位。**不必要的动态 context 会碎片化 place address；fixed-null 的 8-seed 4096 成绩为 `0.9997`。
2. **T2 仍需要动态 context。**null endpoint 的 return-conflict 为 `0`，不能直接删掉 context。
3. **统一 all-token CE 的 ST gate 不行。**它学成 all-on，关键 return utility 被全 token 平均稀释。
4. **HPC retrieval consistency 可以做 metacontrol。**G1 在 fresh high-power replication 上稳定跨过 T1/T2 两端。
5. **不能强迫 PFC 独自猜 HPC 是否可信。**PFC-only neural distillation 没有复现 teacher；下一版若神经化，controller 必须合法读取 HPC confidence/retrieval evidence。
6. **按需调用可以省掉大部分无谓重建。**G2 默认 null 并保留首遍状态，使 T1 61/64 episode 只建一次 prefix HPC，同时没有牺牲 4096-step 或 T2 return-conflict。
7. **跨 checkpoint 下游稳定，但 controller 独立性尚未验证。**G3 的八套 context 模块都通过；controller-facing 权重共享，下一关必须改变任务分布或真正独立训练该通路。
8. **raw contradiction 不是 task-invariant。**G4 冻结失败显示它受重访密度和冲突数共同影响。
9. **地址熟悉度是必要的 metacognitive confidence。**G5 只在几乎完全同址重访时信任 contradiction，fresh OOD 同时修复大网格假阳性和 sparse/query2 假阴性。
10. **调用正确不等于下游容量无限。**dense 已 100% 调用，forced dynamic 仍只有 `0.5469`；该问题必须与 controller 分开。
11. **在线调用在状态机层面成立。**G6 首次 crossing 触发与 fixed G5 decision 完全一致，72 后 endpoint 数值等价，同时将 T2 prefix work 降低约 `10.19%`。
12. **单次 formal 尾部不能事后删除。**G6 仍是 92/93 FAIL；G6b 的 1024-probe 稳定性结果只把失败归因为 path-bank sampling tail，提供后续证据而非覆盖原判。
13. **context need 不是在 episode 开头就可知。**G7 的 H32/H48 score 不能可靠区分未来是否冲突；合法 evidence 在不同 family 的 step `41-69` 才出现。
14. **一旦 evidence 出现，在线状态机按预算平滑恢复。**dynamic coverage 和 return 随 H 单调增长，T1/clean 始终健康，且逐窗口与 fixed endpoint 等价。
15. **paired 曲线可以预测真实隐藏截止流。**G8 在 fresh bank 上复现 G7 的均匀混合预测，且 shadow audit 证明模型没有利用 cutoff token 或 H 后证据。

## 下一条合法路线

G8 已完成，按预注册 post-result rule，**controller 证据线到此冻结，不再新增 G9 或继续挑 cutoff/controller 变体**。第一批论文证据整理已经完成：

- 已生成 G7 paired curve + G8 hidden-mixture 主图，明确标出 H32/H48 的 censoring 和 G8 realized/expected 误差；
- 已生成正式 M1b/online 状态机图，标明 PFC、EC/place、episode-local covariance HPC、first crossing 与 deployment-before-shadow 边界；
- 已把 M1b、Transformer、Hippoformer、Titans/M-delta 等冻结基线统一到参数量、训练预算、T1 free rollout、T2 return/clean 机器表，未评测项保持 N/A；
- 已建立 15 个冻结输入与全部 package outputs 的 SHA256 manifest，并生成已有正式端点的 95% CI；
- G6 的 92/93 FAIL、G6b sparse 方差、G7/G8 short-cutoff 不可辨识区、strict rollout H1=False 和旧 P5 release block 均继续分开保留。

package 状态为 `PAPER_EVIDENCE_G8_READY_LEGACY_P0_BLOCK_DISCLOSED`，7/7 gates；manifest SHA256 为 `92f2fec37913a8fe5514b8ce326f16ac1279bc5eb2bc16c0087f6d8be19515f3`。下一步是 LaTeX 主文接入与完整复现索引，不是继续改 controller。当前没有历史红门的 fixed-step anchor 仍是 G5；G7 是 paired variable-horizon 证据，G8 是 hidden-cutoff streaming confirmation。整个 controller 仍是外挂 metacontroller，不追改正式 M1b headline。

## 冻结产物

- G0 protocol：`runs/remap_former/context_need_retrieval_g0_protocol.json`
- G0 result：`runs/remap_former/context_need_retrieval_g0_seed2571701_2572201.json`
- G0 report：`reports/REMAP_FORMER_CONTEXT_NEED_RETRIEVAL_G0_CN.md`
- G1 protocol：`runs/remap_former/context_need_retrieval_g1_replication_protocol.json`
- G1 result：`runs/remap_former/context_need_retrieval_g1_seed2671701_2672201.json`
- G1 report：`reports/REMAP_FORMER_CONTEXT_NEED_RETRIEVAL_G1_REPLICATION_CN.md`
- Neural protocol：`runs/remap_former/context_need_neural_pilot_protocol.json`
- Neural result：`runs/remap_former/context_need_neural_seed2871701_s600/summary.json`
- Neural report：`runs/remap_former/context_need_neural_seed2871701_s600/report.md`
- G2 protocol：`runs/remap_former/context_need_null_first_g2_protocol.json`
- G2 result：`runs/remap_former/context_need_null_first_g2_seed2971701_2972201.json`
- G2 report：`reports/REMAP_FORMER_CONTEXT_NEED_NULL_FIRST_G2_CN.md`
- G3 protocol：`runs/remap_former/context_need_null_first_g3_multicheckpoint_protocol.json`
- G3 result：`runs/remap_former/context_need_null_first_g3_multicheckpoint_seed3071601_3072201.json`
- G3 report：`reports/REMAP_FORMER_CONTEXT_NEED_NULL_FIRST_G3_MULTICHECKPOINT_CN.md`
- G3 pathway audit：`reports/REMAP_FORMER_CONTEXT_NEED_G3_PATHWAY_HASH_AUDIT.json`
- G4 protocol：`runs/remap_former/context_need_null_first_g4_ood_protocol.json`
- G4 result：`runs/remap_former/context_need_null_first_g4_ood_seed3371601_3432001.json`
- G4 report：`reports/REMAP_FORMER_CONTEXT_NEED_NULL_FIRST_G4_OOD_CN.md`
- G4 endpoint result：`runs/remap_former/context_need_g4_failure_endpoint_audit_seed3401901_3432001.json`
- G4 endpoint report：`reports/REMAP_FORMER_CONTEXT_NEED_G4_FAILURE_ENDPOINT_AUDIT_CN.md`
- G5 protocol：`runs/remap_former/context_need_recurrence_g5_protocol.json`
- G5 result：`runs/remap_former/context_need_recurrence_g5_seed3871701_4022001.json`
- G5 report：`reports/REMAP_FORMER_CONTEXT_NEED_RECURRENCE_G5_CN.md`
- G6 protocol：`runs/remap_former/context_need_online_g6_protocol.json`
- G6 result：`runs/remap_former/context_need_online_g6_seed4171701_4222001.json`
- G6 report：`reports/REMAP_FORMER_CONTEXT_NEED_ONLINE_G6_CN.md`
- G6b protocol：`runs/remap_former/context_need_online_g6b_sparse_stability_protocol.json`
- G6b result：`runs/remap_former/context_need_online_g6b_sparse_stability_seed4271701_4672001.json`
- G6b report：`reports/REMAP_FORMER_CONTEXT_NEED_ONLINE_G6B_SPARSE_STABILITY_CN.md`
- G7 protocol：`runs/remap_former/context_need_online_g7_variable_horizon_protocol.json`
- G7 initial smoke：`tmp/context_need_online_g7_variable_horizon_smoke.json`
- G7 fixed smoke：`tmp/context_need_online_g7_variable_horizon_smoke_v2.json`
- G7 result：`runs/remap_former/context_need_online_g7_variable_horizon_seed4871701_4922001.json`
- G7 report：`reports/REMAP_FORMER_CONTEXT_NEED_ONLINE_G7_VARIABLE_HORIZON_CN.md`
- G8 protocol：`runs/remap_former/context_need_online_g8_hidden_cutoff_protocol.json`
- G8 smoke：`tmp/context_need_online_g8_hidden_cutoff_smoke.json`
- G8 result：`runs/remap_former/context_need_online_g8_hidden_cutoff_seed5471701_5522001.json`
- G8 report：`reports/REMAP_FORMER_CONTEXT_NEED_ONLINE_G8_HIDDEN_CUTOFF_CN.md`
- Paper package report：`reports/REMAP_FORMER_PAPER_EVIDENCE_G8_CN.md`
- Paper package tables：`runs/remap_former/paper_evidence_g8_tables.json`
- Paper package manifest：`runs/remap_former/paper_evidence_g8_manifest.json`
- Figure 1：`reports/figures/remap_former_m1b_architecture_g8.png/.pdf`
- Figure 2：`reports/figures/remap_former_g7_g8_hidden_cutoff.png/.pdf`
- Builder/verify：`build_remap_former_g8_paper_package.py --verify`
