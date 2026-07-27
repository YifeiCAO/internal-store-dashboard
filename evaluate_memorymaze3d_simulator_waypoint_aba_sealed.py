from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_memorymaze3d_simulator_aba_data import build_route_bank
from generate_memorymaze3d_simulator_waypoint_aba_data import (
    _canonical_assignment_sha256,
)
from remap_former.visual_multiplace_aba import VisualMultiPlaceConfig
from remap_former.visual_simulator_waypoint_aba import (
    SimulatorCoupledWaypointReMAPFormer,
    SimulatorWaypointABADataset,
)
from train_memorymaze3d_simulator_aba import (
    build_secondary_diagnostics,
    evaluate_conditions,
    plot_visual_board,
)
from train_memorymaze3d_simulator_waypoint_aba import (
    build_waypoint_model_gates,
    plot_waypoint_candidate_board,
    run_waypoint_preflight,
)


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

PROTOCOL_FORMAT = (
    "remap_former.memorymaze3d_simulator_translated_"
    "waypoint_aba_sealed_protocol.v1"
)
LOCK_FORMAT = (
    "remap_former.memorymaze3d_simulator_translated_"
    "waypoint_aba_sealed_lock.v1"
)
PREFLIGHT_FORMAT = (
    "remap_former.memorymaze3d_simulator_translated_"
    "waypoint_aba_sealed_preflight.v1"
)
SUMMARY_FORMAT = (
    "remap_former.memorymaze3d_simulator_translated_"
    "waypoint_aba_sealed_3seed_summary.v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_protocol(path: Path) -> tuple[dict[str, Any], Path]:
    protocol = read_json(path)
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"unexpected protocol format: {path}")
    if protocol.get("status") != "frozen_before_test_generation":
        raise ValueError("sealed protocol is not frozen")
    return protocol, path.resolve().parents[1]


def verify_artifacts(
    root: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    verified = []
    for record in records:
        path = resolve(root, record["path"])
        actual = sha256(path)
        expected = str(record["sha256"])
        if actual != expected:
            raise ValueError(
                f"frozen artifact hash mismatch: {path} "
                f"{actual} != {expected}"
            )
        verified.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return verified


def verify_frozen_inputs(
    protocol: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    parent = verify_artifacts(
        root,
        protocol["frozen_parent_artifacts"],
    )
    code = verify_artifacts(
        root,
        protocol["sealed_code_artifacts"],
    )
    checkpoints = verify_artifacts(
        root,
        protocol["frozen_checkpoints"],
    )
    return {
        "frozen_parent_artifacts": parent,
        "sealed_code_artifacts": code,
        "frozen_checkpoints": checkpoints,
    }


def verify_lock(
    protocol_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol, root = load_protocol(protocol_path)
    lock = read_json(lock_path)
    if lock.get("format") != LOCK_FORMAT:
        raise ValueError("unexpected sealed lock format")
    if lock["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("sealed lock does not match the protocol")
    for record in lock["sealed_files"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"sealed file hash mismatch: {path}")
    verify_frozen_inputs(protocol, root)
    return protocol, lock, root


def assignment_hashes(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as source:
        labels = np.asarray(source["labels_by_context"])
    return {
        _canonical_assignment_sha256(episode)
        for episode in labels
    }


def manifest_hashes(path: Path) -> dict[str, set[str]]:
    manifest = read_json(path)
    return {
        "layouts": {
            str(row["layout_sha256"])
            for row in manifest["episodes"]
        },
        "physical_routes": {
            str(row["physical_route_sha256"])
            for row in manifest["episodes"]
        },
        "route_families": {
            str(value)
            for row in manifest["episodes"]
            for value in row["route_family_hashes"]
        },
    }


def build_lock(
    protocol_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    protocol, root = load_protocol(protocol_path)
    frozen = verify_frozen_inputs(protocol, root)
    paths = protocol["paths"]
    sealed_paths = [
        resolve(root, paths["test_data"]),
        resolve(root, paths["test_manifest"]),
        resolve(root, paths["test_cache"]),
        resolve(root, paths["test_cache_manifest"]),
    ]
    records = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sealed_paths
    ]
    manifest = read_json(sealed_paths[1])
    cache_manifest = read_json(sealed_paths[3])
    data_hash = records[0]["sha256"]
    if manifest["split"] != "test":
        raise ValueError("sealed manifest is not the test split")
    if manifest["sha256"] != data_hash:
        raise ValueError("test manifest does not match test NPZ")
    if cache_manifest["source"]["sha256"] != data_hash:
        raise ValueError("test DINO cache does not match test NPZ")
    lock = {
        "format": LOCK_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "sealed_files": records,
        "frozen_inputs": frozen,
        "cross_file_gates": {
            "test_manifest_split_is_test": True,
            "manifest_matches_test_data": True,
            "dino_cache_matches_test_data": True,
        },
        "status": "LOCKED",
    }
    write_json_exclusive(lock_path, lock)
    return lock


def split_audit(
    protocol: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    paths = protocol["paths"]
    train_manifest = resolve(root, paths["dev_train_manifest"])
    validation_manifest = resolve(root, paths["dev_validation_manifest"])
    test_manifest = resolve(root, paths["test_manifest"])
    train_data = resolve(root, paths["dev_train_data"])
    validation_data = resolve(root, paths["dev_validation_data"])
    test_data = resolve(root, paths["test_data"])

    train = manifest_hashes(train_manifest)
    validation = manifest_hashes(validation_manifest)
    test = manifest_hashes(test_manifest)
    development = {
        key: train[key] | validation[key]
        for key in train
    }
    train_assignments = assignment_hashes(train_data)
    validation_assignments = assignment_hashes(validation_data)
    test_assignments = assignment_hashes(test_data)
    development_assignments = (
        train_assignments | validation_assignments
    )

    route_bank = build_route_bank()
    reserved_test_routes = {
        family.canonical_hash for family in route_bank["test"]
    }
    reserved_development_routes = {
        family.canonical_hash
        for split in ("train", "val", "stage_gate")
        for family in route_bank[split]
    }
    manifest = read_json(test_manifest)
    aggregate = manifest["aggregate"]
    values = {
        "test_sequence_count": len(manifest["episodes"]),
        "test_unique_layout_count": len(test["layouts"]),
        "test_unique_physical_route_count": len(
            test["physical_routes"]
        ),
        "test_unique_canonical_assignment_count": len(
            test_assignments
        ),
        "development_test_layout_overlap": len(
            development["layouts"] & test["layouts"]
        ),
        "development_test_physical_route_overlap": len(
            development["physical_routes"] & test["physical_routes"]
        ),
        "development_test_route_family_overlap": len(
            development["route_families"] & test["route_families"]
        ),
        "development_test_canonical_assignment_overlap": len(
            development_assignments & test_assignments
        ),
        "test_routes_outside_reserved_test_bank": len(
            test["route_families"] - reserved_test_routes
        ),
        "reserved_route_banks_overlap": len(
            reserved_test_routes & reserved_development_routes
        ),
        "counterfactual_pair_count": int(
            aggregate["counterfactual_pair_count"]
        ),
        "counterfactual_action_mismatch_count": int(
            aggregate["counterfactual_action_mismatch_count"]
        ),
        "counterfactual_query_pixel_mismatch_count": int(
            aggregate["counterfactual_query_pixel_mismatch_count"]
        ),
        "counterfactual_query_pose_max_abs_difference": float(
            aggregate["counterfactual_query_pose_max_abs_difference"]
        ),
        "query_visible_target_count": int(
            aggregate["query_visible_target_count"]
        ),
        "query_nonhidden_geom_count": int(
            aggregate["query_nonhidden_geom_count"]
        ),
    }
    expected = protocol["test_split"]
    gates = {
        "exact_test_sequence_count": (
            values["test_sequence_count"]
            == int(expected["sequence_count"])
        ),
        "exact_unique_layout_count": (
            values["test_unique_layout_count"]
            == int(expected["unique_layout_count"])
        ),
        "all_physical_routes_unique": (
            values["test_unique_physical_route_count"]
            == int(expected["unique_layout_count"])
        ),
        "all_canonical_assignments_unique": (
            values["test_unique_canonical_assignment_count"]
            == int(expected["unique_layout_count"])
        ),
        "development_test_layout_disjoint": (
            values["development_test_layout_overlap"] == 0
        ),
        "development_test_physical_route_disjoint": (
            values["development_test_physical_route_overlap"] == 0
        ),
        "development_test_route_family_disjoint": (
            values["development_test_route_family_overlap"] == 0
        ),
        "development_test_canonical_assignment_disjoint": (
            values[
                "development_test_canonical_assignment_overlap"
            ]
            == 0
        ),
        "test_uses_only_reserved_route_bank": (
            values["test_routes_outside_reserved_test_bank"] == 0
            and values["reserved_route_banks_overlap"] == 0
        ),
        "exact_counterfactual_pair_count": (
            values["counterfactual_pair_count"]
            == int(expected["unique_layout_count"])
        ),
        "counterfactual_current_inputs_identical": (
            values["counterfactual_action_mismatch_count"] == 0
            and values["counterfactual_query_pixel_mismatch_count"] == 0
            and values[
                "counterfactual_query_pose_max_abs_difference"
            ]
            <= 1e-5
        ),
        "query_targets_hidden": (
            values["query_visible_target_count"] == 0
            and values["query_nonhidden_geom_count"] == 0
        ),
    }
    return {
        "values": values,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAILED",
    }


def load_checkpoint_model(
    record: dict[str, Any],
    root: Path,
    device: torch.device,
) -> tuple[SimulatorCoupledWaypointReMAPFormer, dict[str, Any]]:
    path = resolve(root, record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"checkpoint changed: {path}")
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint["seed"]) != int(record["seed"]):
        raise ValueError(f"checkpoint seed mismatch: {path}")
    config = VisualMultiPlaceConfig(**checkpoint["config"])
    model = SimulatorCoupledWaypointReMAPFormer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def run_preflight_mode(
    protocol_path: Path,
    lock_path: Path,
    preflight_path: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    protocol, lock, root = verify_lock(protocol_path, lock_path)
    paths = protocol["paths"]
    train = SimulatorWaypointABADataset(
        resolve(root, paths["dev_train_data"]),
        resolve(root, paths["dev_train_cache"]),
        split="train",
    )
    test = SimulatorWaypointABADataset(
        resolve(root, paths["test_data"]),
        resolve(root, paths["test_cache"]),
        split="test",
    )
    split = split_audit(protocol, root)
    checkpoint_preflights = []
    for record in protocol["frozen_checkpoints"]:
        model, _ = load_checkpoint_model(record, root, device)
        preflight = run_waypoint_preflight(
            model,
            train,
            test,
            train_manifest=resolve(
                root,
                paths["dev_train_manifest"],
            ),
            validation_manifest=resolve(
                root,
                paths["test_manifest"],
            ),
            device=device,
        )
        checkpoint_preflights.append(
            {
                "seed": int(record["seed"]),
                "checkpoint_sha256": record["sha256"],
                "preflight": preflight,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gates = {
        "lock_status_locked": lock["status"] == "LOCKED",
        "split_audit_pass": split["status"] == "PASS",
        "all_checkpoint_preflights_pass": all(
            row["preflight"]["status"] == "PASS"
            for row in checkpoint_preflights
        ),
        "no_model_performance_read_during_preflight": True,
    }
    payload = {
        "format": PREFLIGHT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "lock": {
            "path": str(lock_path.resolve()),
            "sha256": sha256(lock_path),
        },
        "split_audit": split,
        "checkpoint_preflights": checkpoint_preflights,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAILED",
        "performance_metrics_read": False,
    }
    write_json_exclusive(preflight_path, payload)
    return payload


def verify_preflight(
    preflight_path: Path,
    protocol_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    preflight = read_json(preflight_path)
    if preflight.get("format") != PREFLIGHT_FORMAT:
        raise ValueError("unexpected sealed preflight format")
    if preflight.get("status") != "PASS":
        raise ValueError("sealed preflight did not pass")
    if preflight.get("performance_metrics_read") is not False:
        raise ValueError("preflight is contaminated by performance metrics")
    if preflight["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("preflight protocol hash mismatch")
    if preflight["lock"]["sha256"] != sha256(lock_path):
        raise ValueError("preflight lock hash mismatch")
    return preflight


def aggregate_conditions(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        condition: {
            metric: summary(
                [
                    float(row["conditions"][condition][metric])
                    for row in per_seed
                ]
            )
            for metric in METRICS
        }
        for condition in CONDITIONS
    }


def write_figure(
    payload: dict[str, Any],
    path: Path,
) -> None:
    conditions = payload["conditions"]
    seeds = [str(seed) for seed in payload["seeds"]]
    colors = (
        "#167D8D",
        "#6B7280",
        "#D28C2D",
        "#B23A48",
        "#4A8F5B",
        "#4856A3",
    )
    x = np.arange(len(CONDITIONS))
    means = np.asarray(
        [
            conditions[name]["conflict_pairwise_acc"]["mean"]
            for name in CONDITIONS
        ]
    )
    stds = np.asarray(
        [
            conditions[name]["conflict_pairwise_acc"]["sample_std"]
            for name in CONDITIONS
        ]
    )
    labels = [DISPLAY_NAMES[name] for name in CONDITIONS]
    dev = payload["development_reference"]
    sealed_values = conditions["full"]["conflict_pairwise_acc"]["values"]
    dev_values = dev["conditions"]["full"][
        "conflict_pairwise_acc"
    ]["values"]

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.9))
    axes[0].bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#20262E",
        linewidth=0.7,
    )
    for index, name in enumerate(CONDITIONS):
        values = conditions[name]["conflict_pairwise_acc"]["values"]
        axes[0].scatter(
            np.full(len(values), index),
            values,
            color="#111827",
            s=22,
            zorder=3,
        )
    axes[0].axhline(0.5, color="#7A8391", linestyle="--")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Sealed causal interventions")
    axes[0].set_ylabel("Conflict pairwise accuracy")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    sx = np.arange(len(seeds))
    width = 0.35
    axes[1].bar(
        sx - width / 2,
        dev_values,
        width,
        color="#9AA2AD",
        label="Development",
    )
    axes[1].bar(
        sx + width / 2,
        sealed_values,
        width,
        color="#167D8D",
        label="Fresh sealed",
    )
    axes[1].axhline(0.85, color="#B23A48", linestyle="--")
    axes[1].set_ylim(0.8, 1.015)
    axes[1].set_title("No checkpoint reselection")
    axes[1].set_xticks(sx, seeds)
    axes[1].set_xlabel("Frozen training seed")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    target = conditions["full"]["conflict_target_cosine"]["values"]
    clean = conditions["full"]["clean_query_cosine"]["values"]
    axes[2].bar(
        sx - width / 2,
        target,
        width,
        color="#E2B34B",
        label="Target cosine",
    )
    axes[2].bar(
        sx + width / 2,
        clean,
        width,
        color="#4A8F5B",
        label="Clean cosine",
    )
    axes[2].axhline(0.85, color="#B23A48", linestyle="--")
    axes[2].set_ylim(0.8, 1.015)
    axes[2].set_title("Sealed content fidelity")
    axes[2].set_xticks(sx, seeds)
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "MemoryMaze3D translated waypoint A-B-A: fresh sealed test",
        fontsize=15,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_report(
    payload: dict[str, Any],
    path: Path,
) -> None:
    conditions = payload["conditions"]
    full = conditions["full"]
    rows = []
    for seed in payload["per_seed"]:
        metrics = seed["conditions"]
        rows.append(
            f"| `{seed['seed']}` | "
            f"{metrics['full']['conflict_pairwise_acc']:.4f} | "
            f"{metrics['full']['conflict_target_cosine']:.4f} | "
            f"{metrics['full']['clean_query_cosine']:.4f} | "
            f"{metrics['hpc_zero']['conflict_pairwise_acc']:.4f} | "
            f"{metrics['fixed_context']['conflict_pairwise_acc']:.4f} | "
            f"{metrics['wrong_history']['other_context_target_rate']:.4f} | "
            f"{seed['model_gate_pass_count']}/9 |"
        )
    condition_rows = []
    for name in CONDITIONS:
        metrics = conditions[name]
        condition_rows.append(
            f"| {DISPLAY_NAMES[name]} | "
            f"{metrics['conflict_pairwise_acc']['mean']:.4f} ± "
            f"{metrics['conflict_pairwise_acc']['sample_std']:.4f} | "
            f"{metrics['conflict_target_cosine']['mean']:.4f} | "
            f"{metrics['clean_query_cosine']['mean']:.4f} | "
            f"{metrics['other_context_target_rate']['mean']:.4f} |"
        )
    split = payload["preflight"]["split_audit"]["values"]
    status_text = (
        "通过"
        if payload["status"] == "PASS_SEALED_3SEED"
        else "未通过"
    )
    next_step = (
        "下一步解锁同一 sealed task 的 matched Transformer、普通 "
        "fast-weight、Hippoformer 与 Titans/MAC 基线；模型主线不再回看 "
        "sealed 调参。"
        if payload["status"] == "PASS_SEALED_3SEED"
        else "按冻结规则停止，不允许根据 sealed 结果修改模型后重测同一 "
        "split；失败只能作为正式结果保留，并另立未来协议。"
    )
    report = f"""# MemoryMaze3D 真实平移 Waypoint A-B-A Fresh Sealed 结果

**状态：`{payload["status"]}`（{status_text}）**

**协议 SHA256：** `{payload["protocol"]["sha256"]}`

**数据锁 SHA256：** `{payload["lock"]["sha256"]}`

## 核心结论

三个 development checkpoint 原样冻结，没有重新训练，也没有使用 sealed
test 选择 checkpoint。它们在 `32` 个全新 simulator layout、`32` 条唯一
物理 waypoint 路径、`32` 个 canonical object assignment 和保留 test
route bank 上一次性评估。

Full conflict 为
**`{full["conflict_pairwise_acc"]["mean"]:.4f} ± {full["conflict_pairwise_acc"]["sample_std"]:.4f}`**
（sample SD；最差 seed
`{full["conflict_pairwise_acc"]["min"]:.4f}`），target cosine 为
**`{full["conflict_target_cosine"]["mean"]:.4f} ± {full["conflict_target_cosine"]["sample_std"]:.4f}`**，
clean cosine 为
**`{full["clean_query_cosine"]["mean"]:.4f} ± {full["clean_query_cosine"]["sample_std"]:.4f}`**。

## Sealed 数据合同

| 检查 | 结果 |
|---|---:|
| Sequence / unique layout | `{split["test_sequence_count"]} / {split["test_unique_layout_count"]}` |
| Unique physical route | `{split["test_unique_physical_route_count"]}` |
| Unique canonical assignment | `{split["test_unique_canonical_assignment_count"]}` |
| Dev-test layout overlap | `{split["development_test_layout_overlap"]}` |
| Dev-test physical-route overlap | `{split["development_test_physical_route_overlap"]}` |
| Dev-test context-route overlap | `{split["development_test_route_family_overlap"]}` |
| Dev-test assignment overlap | `{split["development_test_canonical_assignment_overlap"]}` |
| Counterfactual action / pixel mismatch | `{split["counterfactual_action_mismatch_count"]} / {split["counterfactual_query_pixel_mismatch_count"]}` |
| Query target visible / nonhidden geom | `{split["query_visible_target_count"]} / {split["query_nonhidden_geom_count"]}` |

task-only preflight 在读取任何模型正确率前写盘并锁定；三个 checkpoint
均通过 `17/17` 结构与防泄漏 gate。

## 逐 Frozen Checkpoint

| Seed | Full | Target cosine | Clean | HPC-zero | Fixed | Wrong→other | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 六条件聚合

| 条件 | Conflict mean ± sample SD | Target cosine | Clean cosine | Other target |
|---|---:|---:|---:|---:|
{chr(10).join(condition_rows)}

![Fresh sealed 三种子结果](figures/memorymaze3d_translated_waypoint_sealed_3seed.png)

## 因果解释

- HPC-zero 若回到 chance，说明 PFC 不能绕过 memory 直接输出答案。
- Fixed context 若显著下降，说明同一 place 必须由历史 context 重映射。
- Wrong history 若定向选择 other target，说明调用方向由历史控制，而非
  当前 RGB、pose 或 waypoint metadata。
- 每个 checkpoint 的 future ground-truth read/write 仍为 `0/0`。

## 边界

- 这是 fresh sealed associative visual-memory test，不是 RGB 生成。
- checkpoint 来自 development selection，但 sealed 从未参与 selection。
- 物体仍是 MemoryMaze3D 的三类彩色球；本结果不等于未见物体类别 OOD。
- 路径由 generator-only controller 执行，不是 learned-policy free rollout。
- 主指标是所有 hidden conflict query 的 feature pairwise recall；视觉板只
  是少量可读样例，不能替代全量指标。

## 下一步

{next_step}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def run_evaluate_mode(
    protocol_path: Path,
    lock_path: Path,
    preflight_path: Path,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"refusing to repeat sealed evaluation: {summary_path}"
        )
    protocol, lock, root = verify_lock(protocol_path, lock_path)
    preflight = verify_preflight(
        preflight_path,
        protocol_path,
        lock_path,
    )
    paths = protocol["paths"]
    test = SimulatorWaypointABADataset(
        resolve(root, paths["test_data"]),
        resolve(root, paths["test_cache"]),
        split="test",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for record in protocol["frozen_checkpoints"]:
        model, checkpoint = load_checkpoint_model(record, root, device)
        metrics = evaluate_conditions(
            model,
            test,
            device=device,
            conditions=CONDITIONS,
            batch_size=batch_size,
        )
        gates = build_waypoint_model_gates(metrics)
        secondary = build_secondary_diagnostics(metrics)
        seed_dir = output_dir / f"seed{record['seed']}"
        visual = plot_waypoint_candidate_board(
            model,
            test,
            device=device,
            path=seed_dir / "sealed_same_waypoint_visual_board.png",
        )
        global_nearest = plot_visual_board(
            model,
            test,
            device=device,
            path=seed_dir / "sealed_global_nearest_diagnostic.png",
        )
        per_seed.append(
            {
                "seed": int(record["seed"]),
                "checkpoint_path": str(
                    resolve(root, record["path"]).resolve()
                ),
                "checkpoint_sha256": record["sha256"],
                "development_selected_validation": checkpoint[
                    "validation"
                ],
                "conditions": metrics,
                "model_gates": gates,
                "model_gate_pass_count": sum(gates.values()),
                "secondary_diagnostics": secondary,
                "visual_board": visual,
                "global_nearest_write_diagnostic": global_nearest,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    conditions = aggregate_conditions(per_seed)
    acceptance = protocol["sealed_acceptance"]
    gates = {
        "preflight_pass": preflight["status"] == "PASS",
        "all_seed_model_gates_9_of_9": all(
            row["model_gate_pass_count"] == 9
            for row in per_seed
        ),
        "all_full_conflict_at_least_threshold": all(
            row["conditions"]["full"]["conflict_pairwise_acc"]
            >= float(acceptance["full_delayed_conflict_pairwise_min"])
            for row in per_seed
        ),
        "all_target_cosine_at_least_threshold": all(
            row["conditions"]["full"]["conflict_target_cosine"]
            >= float(acceptance["full_conflict_target_cosine_min"])
            for row in per_seed
        ),
        "all_clean_cosine_at_least_threshold": all(
            row["conditions"]["full"]["clean_query_cosine"]
            >= float(acceptance["full_clean_cosine_min"])
            for row in per_seed
        ),
        "all_context_reentry_at_least_threshold": all(
            row["conditions"]["full"]["latent_context_reentry_acc"]
            >= float(acceptance["latent_context_reentry_min"])
            for row in per_seed
        ),
        "all_hpc_zero_at_most_threshold": all(
            row["conditions"]["hpc_zero"]["conflict_pairwise_acc"]
            <= float(acceptance["hpc_zero_conflict_max"])
            for row in per_seed
        ),
        "all_fixed_context_at_most_threshold": all(
            row["conditions"]["fixed_context"]["conflict_pairwise_acc"]
            <= float(acceptance["fixed_context_conflict_max"])
            for row in per_seed
        ),
        "all_wrong_history_other_at_least_threshold": all(
            row["conditions"]["wrong_history"][
                "other_context_target_rate"
            ]
            >= float(
                acceptance["wrong_history_other_context_rate_min"]
            )
            for row in per_seed
        ),
        "all_future_ground_truth_zero_zero": all(
            row["conditions"]["full"]["future_ground_truth_reads"] == 0
            and row["conditions"]["full"]["future_ground_truth_writes"] == 0
            for row in per_seed
        ),
    }
    development_reference = read_json(
        resolve(root, paths["development_3seed_summary"])
    )
    payload = {
        "format": SUMMARY_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_SEALED_3SEED"
            if all(gates.values())
            else "FAIL_SEALED_3SEED"
        ),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "lock": {
            "path": str(lock_path.resolve()),
            "sha256": sha256(lock_path),
        },
        "preflight": preflight,
        "seeds": [row["seed"] for row in per_seed],
        "per_seed": per_seed,
        "conditions": conditions,
        "development_reference": development_reference,
        "gates": gates,
        "claim_boundary": (
            "Fresh sealed layout, physical-route, context-route, and "
            "canonical-assignment evaluation of three frozen development "
            "checkpoints. Not unseen object categories, RGB generation, "
            "learned-policy free rollout, or a matched-baseline result."
        ),
    }
    write_json_exclusive(summary_path, payload)
    figure_path = (
        Path("reports/figures/")
        / "memorymaze3d_translated_waypoint_sealed_3seed.png"
    )
    report_path = (
        Path("reports/")
        / "MEMORYMAZE3D_SIMULATOR_TRANSLATED_WAYPOINT_ABA_SEALED_3SEED_CN.md"
    )
    write_figure(payload, figure_path)
    write_report(payload, report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lock, preflight, or once-only evaluate the translated waypoint "
            "fresh sealed split."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("lock", "preflight", "evaluate"),
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "protocols/"
            "memorymaze3d_simulator_translated_waypoint_aba_sealed_v1.json"
        ),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(
            "runs/memorymaze3d/"
            "simulator_translated_waypoint_aba_sealed_v1_lock.json"
        ),
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "runs/memorymaze3d/"
            "simulator_translated_waypoint_aba_sealed_v1_preflight.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/memorymaze3d/"
            "simulator_translated_waypoint_aba_sealed_v1"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    if args.mode == "lock":
        result = build_lock(args.protocol, args.lock)
    elif args.mode == "preflight":
        result = run_preflight_mode(
            args.protocol,
            args.lock,
            args.preflight,
            device=device,
        )
    else:
        result = run_evaluate_mode(
            args.protocol,
            args.lock,
            args.preflight,
            args.output_dir,
            device=device,
            batch_size=args.batch_size,
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": result["status"],
                "gates": result.get("gates"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
