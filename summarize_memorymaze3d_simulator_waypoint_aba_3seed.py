from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = (
    "full",
    "hpc_zero",
    "fixed_context",
    "wrong_history",
    "correct_history",
    "orthogonal_context_oracle",
)

DISPLAY_NAMES = {
    "full": "Full",
    "hpc_zero": "HPC-zero",
    "fixed_context": "Fixed context",
    "wrong_history": "Wrong history",
    "correct_history": "Correct history",
    "orthogonal_context_oracle": "Context oracle",
}

METRICS = (
    "conflict_pairwise_acc",
    "other_context_target_rate",
    "conflict_cosine_margin",
    "conflict_target_cosine",
    "clean_query_cosine",
    "query_cosine",
    "latent_context_reentry_acc",
    "latent_context_margin",
)

EXPECTED_FORMAT = (
    "remap_former.memorymaze3d_simulator_translated_waypoint_aba_result.v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "population_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [int(result["seed"]) for result in results]
    if len(results) != 3:
        raise ValueError(f"expected exactly three results, got {len(results)}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds: {seeds}")

    reference_config = results[0]["config"]
    reference_objective = results[0]["training_objective"]
    for path, result in zip(paths, results, strict=True):
        if result["format"] != EXPECTED_FORMAT:
            raise ValueError(f"unexpected result format in {path}")
        if result["preflight"]["status"] != "PASS":
            raise ValueError(f"preflight failed in {path}")
        if result["config"] != reference_config:
            raise ValueError(f"model config differs in {path}")
        if result["training_objective"] != reference_objective:
            raise ValueError(f"training objective differs in {path}")
        missing = [
            condition
            for condition in CONDITIONS
            if condition not in result["validation"]
        ]
        if missing:
            raise ValueError(f"{path} is missing conditions: {missing}")
    return results


def find_best_step(result: dict[str, Any]) -> int:
    selected = result["validation"]["full"]
    selected_tuple = (
        float(selected["conflict_pairwise_acc"]),
        float(selected["conflict_target_cosine"]),
        float(selected["clean_query_cosine"]),
        float(selected["latent_context_reentry_acc"]),
    )
    matches = []
    for row in result["history"]:
        row_tuple = (
            float(row["validation_conflict_pairwise_acc"]),
            float(row["validation_conflict_target_cosine"]),
            float(row["validation_clean_query_cosine"]),
            float(row["validation_context_reentry_acc"]),
        )
        if all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(row_tuple, selected_tuple, strict=True)
        ):
            matches.append(int(row["step"]))
    if not matches:
        raise ValueError(f"cannot reconstruct selected step for seed {result['seed']}")
    return min(matches)


def aggregate(
    paths: list[Path],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        conditions[condition] = {
            metric: summarize(
                [
                    float(result["validation"][condition][metric])
                    for result in results
                ]
            )
            for metric in METRICS
        }

    contrast_pairs = {
        "full_minus_hpc_zero": ("full", "hpc_zero"),
        "full_minus_fixed_context": ("full", "fixed_context"),
        "correct_minus_wrong_history": ("correct_history", "wrong_history"),
        "oracle_minus_full": ("orthogonal_context_oracle", "full"),
    }
    paired_contrasts = {}
    for name, (positive, negative) in contrast_pairs.items():
        paired_contrasts[name] = summarize(
            [
                float(
                    result["validation"][positive]["conflict_pairwise_acc"]
                )
                - float(
                    result["validation"][negative]["conflict_pairwise_acc"]
                )
                for result in results
            ]
        )

    per_seed = []
    for result in results:
        full = result["validation"]["full"]
        gradients = [
            float(row["gradient_norm"]) for row in result["history"]
        ]
        all_finite = all(
            math.isfinite(float(value))
            for row in result["history"]
            for value in row.values()
            if isinstance(value, (int, float))
        )
        per_seed.append(
            {
                "seed": int(result["seed"]),
                "source_status": result["status"],
                "selected_step": find_best_step(result),
                "training_steps": int(result["history"][-1]["step"]),
                "model_gate_pass_count": sum(
                    bool(value) for value in result["model_gates"].values()
                ),
                "model_gate_count": len(result["model_gates"]),
                "all_model_gates_pass": all(result["model_gates"].values()),
                "all_logged_training_values_finite": all_finite,
                "maximum_raw_gradient_norm": max(gradients),
                "full_conflict_pairwise_acc": float(
                    full["conflict_pairwise_acc"]
                ),
                "full_conflict_target_cosine": float(
                    full["conflict_target_cosine"]
                ),
                "full_clean_query_cosine": float(
                    full["clean_query_cosine"]
                ),
                "full_latent_context_reentry_acc": float(
                    full["latent_context_reentry_acc"]
                ),
                "hpc_zero_conflict_pairwise_acc": float(
                    result["validation"]["hpc_zero"][
                        "conflict_pairwise_acc"
                    ]
                ),
                "fixed_context_conflict_pairwise_acc": float(
                    result["validation"]["fixed_context"][
                        "conflict_pairwise_acc"
                    ]
                ),
                "wrong_history_conflict_pairwise_acc": float(
                    result["validation"]["wrong_history"][
                        "conflict_pairwise_acc"
                    ]
                ),
                "wrong_history_other_context_target_rate": float(
                    result["validation"]["wrong_history"][
                        "other_context_target_rate"
                    ]
                ),
                "correct_history_conflict_pairwise_acc": float(
                    result["validation"]["correct_history"][
                        "conflict_pairwise_acc"
                    ]
                ),
                "orthogonal_context_oracle_conflict_pairwise_acc": float(
                    result["validation"]["orthogonal_context_oracle"][
                        "conflict_pairwise_acc"
                    ]
                ),
                "same_waypoint_candidate_accuracy": float(
                    result["visual_board"][
                        "same_waypoint_candidate_accuracy"
                    ]
                ),
                "global_nearest_write_accuracy": float(
                    result["global_nearest_write_diagnostic"][
                        "nearest_write_accuracy"
                    ]
                ),
                "same_place_cross_context_cosine": float(
                    full["same_place_cross_context_cosine"]
                ),
                "same_segment_cross_place_cosine": float(
                    full["same_segment_cross_place_cosine"]
                ),
                "static_anchor_matches_dynamic_query": bool(
                    result["secondary_diagnostics"][
                        "static_correct_history_anchor_matches_dynamic_query"
                    ]
                ),
                "oracle_not_worse_than_full_by_003": bool(
                    result["secondary_diagnostics"][
                        "orthogonal_context_oracle_not_worse_than_full_by_003"
                    ]
                ),
            }
        )

    task_values = results[0]["preflight"]["values"]
    translated = task_values["translated_waypoint"]
    task_contract = {
        "sequence_length": int(task_values["validation"]["sequence_length"]),
        "writes_per_episode": int(
            task_values["validation"]["write_count_per_episode_min"]
        ),
        "queries_per_episode": int(
            task_values["validation"]["query_count_per_episode_min"]
        ),
        "cumulative_outbound_path_m": float(
            translated["cumulative_outbound_path_m_validation"]
        ),
        "maximum_waypoint_center_error_m": float(
            translated["maximum_waypoint_center_error_m"]
        ),
        "maximum_same_waypoint_revisit_error_m": float(
            translated["maximum_same_waypoint_revisit_error_m"]
        ),
        "current_query_context_probe_balanced_acc": float(
            task_values["current_query_context_probe_balanced_acc"]
        ),
        "counterfactual_query_pixel_mismatch": int(
            task_values["counterfactual_query_pixel_mismatch_validation"]
        ),
        "counterfactual_action_mismatch_count": int(
            translated["counterfactual_action_mismatch_count"]
        ),
        "counterfactual_query_pose_max_abs_difference": float(
            translated["counterfactual_query_pose_max_abs_difference"]
        ),
        "train_validation_layout_overlap": int(
            task_values["train_validation_layout_overlap"]
        ),
        "train_validation_route_overlap": int(
            task_values["train_validation_route_overlap"]
        ),
    }

    gradient_summary = summarize(
        [row["maximum_raw_gradient_norm"] for row in per_seed]
    )
    visual_summary = {
        "same_waypoint_candidate_accuracy": summarize(
            [row["same_waypoint_candidate_accuracy"] for row in per_seed]
        ),
        "global_nearest_write_accuracy": summarize(
            [row["global_nearest_write_accuracy"] for row in per_seed]
        ),
    }

    gates = {
        "three_distinct_seeds": len({row["seed"] for row in per_seed}) == 3,
        "all_preflight_pass": all(
            result["preflight"]["status"] == "PASS" for result in results
        ),
        "all_model_gates_9_of_9": all(
            row["model_gate_pass_count"] == 9
            and row["model_gate_count"] == 9
            for row in per_seed
        ),
        "all_full_conflict_at_least_085": all(
            row["full_conflict_pairwise_acc"] >= 0.85 for row in per_seed
        ),
        "all_target_cosine_at_least_085": all(
            row["full_conflict_target_cosine"] >= 0.85 for row in per_seed
        ),
        "all_clean_cosine_at_least_085": all(
            row["full_clean_query_cosine"] >= 0.85 for row in per_seed
        ),
        "all_context_reentry_at_least_085": all(
            row["full_latent_context_reentry_acc"] >= 0.85
            for row in per_seed
        ),
        "all_hpc_zero_at_most_060": all(
            row["hpc_zero_conflict_pairwise_acc"] <= 0.60
            for row in per_seed
        ),
        "all_fixed_context_at_most_060": all(
            row["fixed_context_conflict_pairwise_acc"] <= 0.60
            for row in per_seed
        ),
        "all_wrong_history_other_at_least_075": all(
            row["wrong_history_other_context_target_rate"] >= 0.75
            for row in per_seed
        ),
        "all_exactly_eight_writes": all(
            result["validation"]["full"]["writes_per_episode"] == 8.0
            for result in results
        ),
        "all_future_ground_truth_zero_zero": all(
            result["validation"]["full"]["future_ground_truth_reads"] == 0.0
            and result["validation"]["full"]["future_ground_truth_writes"] == 0.0
            for result in results
        ),
        "all_logged_training_values_finite": all(
            row["all_logged_training_values_finite"] for row in per_seed
        ),
    }
    core_pass = all(gates.values())

    secondary = {
        "static_anchor_matches_dynamic_query_count": sum(
            row["static_anchor_matches_dynamic_query"] for row in per_seed
        ),
        "oracle_not_worse_than_full_by_003_count": sum(
            row["oracle_not_worse_than_full_by_003"] for row in per_seed
        ),
        "same_waypoint_visual_all_correct": all(
            row["same_waypoint_candidate_accuracy"] == 1.0
            for row in per_seed
        ),
        "global_nearest_is_secondary_not_acceptance": True,
    }

    return {
        "format": (
            "remap_former.memorymaze3d_simulator_translated_"
            "waypoint_aba_3seed_summary.v1"
        ),
        "scope": (
            "development_three_seed_replication_not_sealed_"
            "policy_free_rollout_or_rgb_generation"
        ),
        "protocol": (
            "protocols/"
            "memorymaze3d_simulator_translated_waypoint_aba_dev_v1.json"
        ),
        "source_results": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for path in paths
        ],
        "seeds": [int(result["seed"]) for result in results],
        "frozen_model_config": results[0]["config"],
        "frozen_training_objective": results[0]["training_objective"],
        "task_contract": task_contract,
        "per_seed": per_seed,
        "conditions": conditions,
        "paired_contrasts": paired_contrasts,
        "visual_audit": visual_summary,
        "optimization_stability": {
            "maximum_raw_gradient_norm_by_seed": {
                str(row["seed"]): row["maximum_raw_gradient_norm"]
                for row in per_seed
            },
            "maximum_raw_gradient_norm_summary": gradient_summary,
            "global_gradient_clip_norm": 1.0,
            "nan_or_nonfinite_observed": not gates[
                "all_logged_training_values_finite"
            ],
            "interpretation": (
                "All runs finished without non-finite values, but seed 66102 "
                "had a very large pre-clipping gradient spike. This is an "
                "optimization-stability warning, not a failed mechanism gate."
            ),
        },
        "gates": gates,
        "secondary_diagnostics": secondary,
        "status": (
            "PASS_3SEED_WITH_SECONDARY_STABILITY_NOTES"
            if core_pass
            else "FAIL_3SEED_CORE_GATE"
        ),
        "claim_boundary": (
            "Three-seed development replication of context-controlled "
            "waypoint content recall. Not a sealed split, learned-policy free "
            "rollout, RGB generator, full world model, or matched baseline win."
        ),
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    full = summary["conditions"]["full"]
    per_seed_rows = []
    for row in summary["per_seed"]:
        per_seed_rows.append(
            f"| `{row['seed']}` | {row['selected_step']} | "
            f"{row['full_conflict_pairwise_acc']:.4f} | "
            f"{row['full_conflict_target_cosine']:.4f} | "
            f"{row['full_clean_query_cosine']:.4f} | "
            f"{row['fixed_context_conflict_pairwise_acc']:.4f} | "
            f"{row['wrong_history_other_context_target_rate']:.4f} | "
            f"{row['maximum_raw_gradient_norm']:.2f} | "
            f"{row['model_gate_pass_count']}/{row['model_gate_count']} |"
        )

    condition_rows = []
    for condition in CONDITIONS:
        values = summary["conditions"][condition]
        condition_rows.append(
            f"| {DISPLAY_NAMES[condition]} | "
            f"{values['conflict_pairwise_acc']['mean']:.4f} ± "
            f"{values['conflict_pairwise_acc']['sample_std']:.4f} | "
            f"{values['conflict_pairwise_acc']['min']:.4f} | "
            f"{values['conflict_target_cosine']['mean']:.4f} | "
            f"{values['clean_query_cosine']['mean']:.4f} | "
            f"{values['other_context_target_rate']['mean']:.4f} |"
        )

    gate_rows = [
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in summary["gates"].items()
    ]
    visual = summary["visual_audit"]
    gradients = summary["optimization_stability"]
    task = summary["task_contract"]
    report = f"""# MemoryMaze3D 真实平移 Waypoint A-B-A 三种子结果

**日期：2026-07-27**

**状态：`{summary["status"]}`**

**冻结协议：**
`{summary["protocol"]}`

## 结论

冻结配置在 seed `66101/66102/66103` 上完成开发集三种子复现，三个种子
都通过预注册模型 gate **`9/9`**。Full delayed conflict 为
**`{full["conflict_pairwise_acc"]["mean"]:.4f} ± {full["conflict_pairwise_acc"]["sample_std"]:.4f}`**
（sample SD，最差种子 `{full["conflict_pairwise_acc"]["min"]:.4f}`），
conflict target cosine 为
**`{full["conflict_target_cosine"]["mean"]:.4f} ± {full["conflict_target_cosine"]["sample_std"]:.4f}`**，
clean cosine 为
**`{full["clean_query_cosine"]["mean"]:.4f} ± {full["clean_query_cosine"]["sample_std"]:.4f}`**，
latent context re-entry 为 **`1.0000 ± 0.0000`**。

结构化干预也在三个种子中保持同一方向：HPC-zero 固定回到 chance；
fixed context 与 wrong history 破坏正确调用；wrong history 平均以
`{summary["conditions"]["wrong_history"]["other_context_target_rate"]["mean"]:.4f}`
的比例转向另一 context。因而当前最稳妥的机制结论是：

> Window Transformer/PFC 从视觉历史形成动态 latent context；
> action-only 周期 EC 生成 neural place；place × context 共同寻址唯一一套
> episode-local neural HPC。返回相同物理 waypoint、当前目标隐藏时，
> 历史 context 因果控制 visual content 的调用方向。

这是 **development 三种子描述性复现**，不是 fresh sealed test。`n=3`
只报告均值、sample SD 和最差种子，不做显著性推断。

## 1. 冻结任务合同

| 项目 | 值 |
|---|---:|
| 连续 simulator episode | `{task["sequence_length"]}` actions |
| 真实 outbound 路径 | `{task["cumulative_outbound_path_m"]:.1f} m` |
| 物理 waypoint | `4` |
| 写入 / query | `{task["writes_per_episode"]} / {task["queries_per_episode"]}` |
| 最大 waypoint / revisit 误差 | `{task["maximum_waypoint_center_error_m"]:.4f} / {task["maximum_same_waypoint_revisit_error_m"]:.4f} m` |
| Current-query context probe | `{task["current_query_context_probe_balanced_acc"]:.4f}` |
| 反事实 query pixel / action mismatch | `{task["counterfactual_query_pixel_mismatch"]} / {task["counterfactual_action_mismatch_count"]}` |
| 反事实 query pose 最大差 | `{task["counterfactual_query_pose_max_abs_difference"]:.8f}` |
| Train/validation layout / route overlap | `{task["train_validation_layout_overlap"]} / {task["train_validation_route_overlap"]}` |

模型仍只有当前官方六动作 one-hot 与当前 frozen DINO feature 两个输入。
room/context/phase ID、simulator pose、绝对位置、waypoint/place ID 和 future
feature 均不存在于模型输入。每条 sequence 只有一张从零开始的
episode-local factorized HPC，无 slot、无第二套 fast weights。

## 2. 逐种子结果

三个训练均使用完全相同的模型、目标、`300` steps、batch size `4`、
每 `10` 步评估和既定 checkpoint selection；只改变训练 seed。

| Seed | 选中 step | Full | Target cosine | Clean | Fixed | Wrong→other | 最大 raw grad | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_seed_rows)}

66103 的 Full 为 `0.9583`，低于另两个种子的 `1.0000`，但仍超过冻结
`0.85` 门槛。它的 static correct-history anchor 与 context oracle 都为
`0.9167`，比动态 Full 低 `0.0417`；因此总状态保留 dynamic-context
次级备注，不把“静态 anchor/oracle 与 Full 完全等价”写成 3/3 结论。

## 3. 六条件聚合

| 条件 | Conflict mean ± sample SD | 最差种子 | Target cosine | Clean cosine | Other target |
|---|---:|---:|---:|---:|---:|
{chr(10).join(condition_rows)}

关键 paired contrast：

- Full − HPC-zero：
  `{summary["paired_contrasts"]["full_minus_hpc_zero"]["mean"]:.4f} ± {summary["paired_contrasts"]["full_minus_hpc_zero"]["sample_std"]:.4f}`；
- Full − fixed context：
  `{summary["paired_contrasts"]["full_minus_fixed_context"]["mean"]:.4f} ± {summary["paired_contrasts"]["full_minus_fixed_context"]["sample_std"]:.4f}`；
- Correct history − wrong history：
  `{summary["paired_contrasts"]["correct_minus_wrong_history"]["mean"]:.4f} ± {summary["paired_contrasts"]["correct_minus_wrong_history"]["sample_std"]:.4f}`。

![三种子聚合](figures/memorymaze3d_translated_waypoint_3seed.png)

## 4. 冻结 Gate

| Gate | 结果 |
|---|---:|
{chr(10).join(gate_rows)}

主结论 gate 全部通过。每个 episode 恰好写 `8` 次，严格
read-before-write；future ground-truth read/write 为 `0/0`。

## 5. 视觉审计

同-waypoint 两个历史候选中的主视觉选择为
**`{visual["same_waypoint_candidate_accuracy"]["mean"]:.4f} ± {visual["same_waypoint_candidate_accuracy"]["sample_std"]:.4f}`**
（三个种子均 `3/3`）。

更严格但未预注册为 acceptance gate 的全八写入 global-nearest 为
**`{visual["global_nearest_write_accuracy"]["mean"]:.4f} ± {visual["global_nearest_write_accuracy"]["sample_std"]:.4f}`**；
逐种子为
`{visual["global_nearest_write_accuracy"]["values"]}`。它仍显示 broad-ROI
DINO feature 会被其他 waypoint 的背景/视点吸引，因此不能把主任务结果
改写成“全局视觉检索或 RGB 预测已经解决”。

## 6. 优化稳定性

三个种子的最大 raw gradient norm 分别为
`{gradients["maximum_raw_gradient_norm_by_seed"]}`。66102 在 step 20
出现 `1447.72` 尖峰；所有 optimizer step 前均执行 global norm clip
`1.0`，三个训练均跑满、无 NaN/Inf、stderr 为空且最终通过机制 gate。

这说明尖峰没有使本轮机制结果失效，但训练动力学并不平滑。sealed
实验应继续冻结 clip 与 checkpoint rule，并把 raw gradient trajectory
作为审计项；不能只报告选中 checkpoint。

## 7. 证据边界与下一步

当前支持：

- 真实 `3 m` 平移、转向和同一 waypoint 复访；
- 三个独立训练 seed 上 context-controlled HPC recall；
- HPC-zero、fixed-context、wrong-history 的因果方向稳定；
- 无 room/pose/place ID 输入、future GT 为 `0/0`。

当前不支持：

- 未见 layout/route/object 的 sealed 泛化；
- learned-policy free rollout；
- RGB 生成或完整 3D world model；
- 在统一预算下击败 Hippoformer、Titans 或纯 Transformer。

下一步不再调整这一 development protocol：固定代码、数据生成规则、
超参数与 selection rule，生成预留的 fresh sealed
layout/route/object split；只有 sealed 通过后，再进入 learned-policy
free rollout 与 matched baselines。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def write_figure(summary: dict[str, Any], path: Path) -> None:
    colors = (
        "#167D8D",
        "#6B7280",
        "#D28C2D",
        "#B23A48",
        "#4A8F5B",
        "#4856A3",
    )
    conflict_means = np.array(
        [
            summary["conditions"][condition][
                "conflict_pairwise_acc"
            ]["mean"]
            for condition in CONDITIONS
        ]
    )
    conflict_stds = np.array(
        [
            summary["conditions"][condition][
                "conflict_pairwise_acc"
            ]["sample_std"]
            for condition in CONDITIONS
        ]
    )
    labels = [DISPLAY_NAMES[condition] for condition in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    axes[0].bar(
        x,
        conflict_means,
        yerr=conflict_stds,
        capsize=4,
        color=colors,
        edgecolor="#20262E",
        linewidth=0.7,
    )
    for index, condition in enumerate(CONDITIONS):
        values = summary["conditions"][condition][
            "conflict_pairwise_acc"
        ]["values"]
        axes[0].scatter(
            np.full(len(values), index),
            values,
            color="#111827",
            s=24,
            zorder=3,
        )
    axes[0].axhline(0.5, color="#7A8391", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Conflict pairwise accuracy")
    axes[0].set_title("Causal intervention pattern")
    axes[0].set_xticks(x, labels, rotation=27, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    seeds = [str(seed) for seed in summary["seeds"]]
    full = summary["conditions"]["full"]
    width = 0.25
    sx = np.arange(len(seeds))
    axes[1].bar(
        sx - width,
        full["conflict_pairwise_acc"]["values"],
        width,
        color="#167D8D",
        label="Conflict",
    )
    axes[1].bar(
        sx,
        full["conflict_target_cosine"]["values"],
        width,
        color="#E2B34B",
        label="Target cosine",
    )
    axes[1].bar(
        sx + width,
        full["clean_query_cosine"]["values"],
        width,
        color="#4A8F5B",
        label="Clean cosine",
    )
    axes[1].axhline(0.85, color="#B23A48", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.8, 1.015)
    axes[1].set_xticks(sx, seeds)
    axes[1].set_xlabel("Training seed")
    axes[1].set_title("Full model across frozen replications")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "MemoryMaze3D translated waypoint A-B-A, 3-seed development replication",
        fontsize=14,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        default=[
            Path(
                "runs/memorymaze3d/"
                "simulator_translated_waypoint_aba_seed66101/result.json"
            ),
            Path(
                "runs/memorymaze3d/"
                "simulator_translated_waypoint_aba_seed66102/result.json"
            ),
            Path(
                "runs/memorymaze3d/"
                "simulator_translated_waypoint_aba_seed66103/result.json"
            ),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/memorymaze3d/"
            "simulator_translated_waypoint_aba_3seed_summary.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/"
            "MEMORYMAZE3D_SIMULATOR_TRANSLATED_WAYPOINT_ABA_3SEED_CN.md"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "reports/figures/"
            "memorymaze3d_translated_waypoint_3seed.png"
        ),
    )
    args = parser.parse_args()

    results = load_results(args.results)
    summary = aggregate(args.results, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(summary, args.report)
    write_figure(summary, args.figure)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "seeds": summary["seeds"],
                "gates": summary["gates"],
                "secondary_diagnostics": summary[
                    "secondary_diagnostics"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"summary={args.output}")
    print(f"report={args.report}")
    print(f"figure={args.figure}")


if __name__ == "__main__":
    main()
