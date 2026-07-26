# MemoryMaze3D 3D-D0 DINOv2 Target Audit

> Frozen `dinov2_vits14_reg`，使用 `16×16×384` spatial patch feature；CLS-only 不进入任何主指标。

内容指标在 evaluation-only conflict-object mask 覆盖的 DINO patch 上计算；mask 不进入 encoder、model 或 context probe。whole-ROI 结果保留为诊断，不冒充 conflict-patch 指标。

## 内容可分性

- object linear probe：`1.0000`
- nearest prototype：`1.0000`
- within-object cross-view distance：`0.0918`
- between-object same-environment distance：`0.1451`
- between / within ratio：`1.58`
- broad-ROI nearest prototype（诊断）：`0.4722`
- broad-ROI between / within（诊断）：`1.16`

## Context 泄漏

- current corridor frame probe：`0.5000`
- current neutral cue probe：`0.5000`
- full visual history probe：`1.0000`
- paired current/cue pixel mismatch：`0 / 0`

## Gate

- object_identity_linear_probe_high: `True`
- object_identity_prototype_high: `True`
- between_object_distance_exceeds_within: `True`
- current_frame_context_at_chance: `True`
- current_cue_context_at_chance: `True`
- full_history_context_high: `True`
- spatial_patch_target_nonblank: `True`
- current_and_cue_pixels_exactly_paired: `True`

## 判断

D0 gate 全绿：可以进入 DINO feature world model 与最小 visual A-B-A。

当前 object identity 是真实 MemoryMaze `TargetSphere` 的红/绿/蓝 identity pilot。它足够决定 DINO target 是否可用，但正式 A-B-A 仍应补 shape/category conflict，避免把论文主张缩成颜色分类。
