# M1h 神经历史调用失败定位

> 复用已拒绝的 seed 8928 K=8 pilot，只读分析固定 step 600；不训练、不改参数、不产生新的模型选择资格。

- Return-conflict probes：`128`。
- 最近 proposal 具有因果历史：`1.0000`。
- 历史中存在 exact cyclic-signature match：`0.8438`。
- Learned Q/K top-1 命中 exact match：`0.5926`。
- 原始 signature top-1 命中 exact match：`1.0000`。
- Learned Q/K 与 raw top-1 一致：`0.3750`。
- Exact-match attention mass：`0.3980`。
- Attention max / entropy：`0.2915 / 0.5357`。
- 最近 proposal 的 call weight / argmax rate：`0.8041 / 1.0000`。
- Attention max < 0.5 仍选择 call：`1.0000`。

## 下游对应

- Neural / no-call return-conflict：`0.1719 / 0.3906`。
- Neural 当前 context pair：`0.6875`。
- Learned top-1 历史 value 自身的 context pair：`0.6875`。
- Top-1 exact / 非 exact 时 neural accuracy：`0.1562 / 0.1875`。

该审计只用于区分 key 匹配、拒绝调用与递归 value 污染三类失败，不能据此回调 seed 8928 或改写冻结 pilot 结论。
