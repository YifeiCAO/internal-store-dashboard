# MemoryMaze3D Rollout Paired Bootstrap

- bootstrap samples：`20,000`
- bootstrap seed：`2112`
- episodes：`32`
- 这是 adaptive-dev 描述性统计，不参与选模，不是 test headline。

| 候选 | 对照 | 候选 MSE | 对照 MSE | 改善 | 95% CI | P(改善>0) | Episode 胜率 |
|---|---|---:|---:|---:|---:|---:|---:|
| transformer Stage R | transformer R0 | 0.022994 | 0.055807 | +0.032813 | [+0.022430, +0.043091] | 1.000 | 0.906 |
| transformer Stage R | persistence | 0.022994 | 0.040188 | +0.017194 | [+0.007433, +0.027867] | 1.000 | 0.812 |
| transformer Stage R | M1b reference | 0.022994 | 0.020073 | -0.002921 | [-0.009843, +0.004698] | 0.208 | 0.188 |
| titans Stage R | titans R0 | 0.021891 | 0.044593 | +0.022702 | [+0.014156, +0.031539] | 1.000 | 0.875 |
| titans Stage R | persistence | 0.021891 | 0.040188 | +0.018296 | [+0.008795, +0.028759] | 1.000 | 0.844 |
| titans Stage R | M1b reference | 0.021891 | 0.020073 | -0.001818 | [-0.008154, +0.005587] | 0.288 | 0.219 |
| hippoformer Stage R | hippoformer R0 | 0.023302 | 0.042423 | +0.019122 | [+0.010532, +0.027726] | 1.000 | 0.812 |
| hippoformer Stage R | persistence | 0.023302 | 0.040188 | +0.016886 | [+0.009690, +0.025377] | 1.000 | 0.875 |
| hippoformer Stage R | M1b reference | 0.023302 | 0.020073 | -0.003229 | [-0.006650, +0.000946] | 0.059 | 0.125 |
| titans Stage R | transformer Stage R | 0.021891 | 0.022994 | +0.001103 | [-0.001361, +0.003681] | 0.799 | 0.531 |
| hippoformer Stage R | transformer Stage R | 0.023302 | 0.022994 | -0.000308 | [-0.007860, +0.006407] | 0.489 | 0.500 |
| hippoformer Stage R | titans Stage R | 0.023302 | 0.021891 | -0.001410 | [-0.009116, +0.005160] | 0.356 | 0.469 |

正的改善表示候选 MSE 更低。M1b 比较只描述当前同 episode 差异；训练预算仍不匹配。
