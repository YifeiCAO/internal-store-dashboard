# ReMAP-Former P3 Test-2：训练冻结门

> 24 个外部基线 checkpoint 全部训练结束后生成；此阶段未访问 Test-2。

## 家族汇总

| family | runs | clean mean +/- SD | clean min | return-conflict dev mean | total GPU-process seconds | gate |
|---|---:|---:|---:|---:|---:|---|
| short_window_transformer | 8 | 0.9018 +/- 0.0055 | 0.8891 | 0.0049 | 248.1 | PASS |
| parameter_matched_transformer | 8 | 0.9055 +/- 0.0062 | 0.8969 | 0.0039 | 245.8 | PASS |
| titans_mac_adaptation | 8 | 0.9266 +/- 0.0111 | 0.9125 | 0.0029 | 10060.6 | PASS |

## 决策

- 24/24 runs 完成且通过：`True`。
- recipe/checkpoint hash manifest 完整：`True`。
- Test-2 仍未访问：`True`。
- 允许一次性 Test-2：`True`。

## 文件

- protocol: `runs/remap_former/p3_test2_protocol.json`
- manifest: `runs/remap_former/p3_test2_checkpoint_manifest.json`
