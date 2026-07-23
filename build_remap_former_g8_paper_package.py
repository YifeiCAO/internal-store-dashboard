from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
TABLES_PATH = ROOT / "runs/remap_former/paper_evidence_g8_tables.json"
MANIFEST_PATH = ROOT / "runs/remap_former/paper_evidence_g8_manifest.json"
REPORT_PATH = ROOT / "reports/REMAP_FORMER_PAPER_EVIDENCE_G8_CN.md"
FIGURE_DIR = ROOT / "reports/figures"
ARCH_PNG = FIGURE_DIR / "remap_former_m1b_architecture_g8.png"
ARCH_PDF = FIGURE_DIR / "remap_former_m1b_architecture_g8.pdf"
CURVE_PNG = FIGURE_DIR / "remap_former_g7_g8_hidden_cutoff.png"
CURVE_PDF = FIGURE_DIR / "remap_former_g7_g8_hidden_cutoff.pdf"

P1_PATH = Path("runs/remap_former/p2_m1b_t1_hierarchical_bootstrap_seed2171601.json")
P3_PATH = Path("runs/remap_former/p3_test2_seed2471601.json")
G7_PROTOCOL_PATH = Path(
    "runs/remap_former/context_need_online_g7_variable_horizon_protocol.json"
)
G7_RESULT_PATH = Path(
    "runs/remap_former/context_need_online_g7_variable_horizon_seed4871701_4922001.json"
)
G8_PROTOCOL_PATH = Path(
    "runs/remap_former/context_need_online_g8_hidden_cutoff_protocol.json"
)
G8_RESULT_PATH = Path(
    "runs/remap_former/context_need_online_g8_hidden_cutoff_seed5471701_5522001.json"
)
LEGACY_P0_PATH = Path("runs/remap_former/p0_truth_freeze_manifest.json")
LEGACY_P5_PATH = Path("runs/remap_former/p5_release_manifest.json")
ARCHITECTURE_CARD_PATH = Path(
    "reports/REMAP_FORMER_M1B_CANONICAL_ARCHITECTURE_CN.md"
)

FROZEN_INPUT_HASHES = {
    P1_PATH: "5dbf74af66b22815c21aa5a3f5b3af376aa29d1554951221d6d92eb2c22b625b",
    P3_PATH: "3bc282f225bc1445f4cef4b80c5e7c7c30f1fda8928477931c8d58f94ca2a065",
    G7_PROTOCOL_PATH: "c24a489cda3f5c6b449d6e0dc9732ae99872a0f15892c787f1e637d08a9be255",
    G7_RESULT_PATH: "daee4912e3e6a62d9daf019a7a7fa78d51327824b24745fff73adaa8d9209871",
    G8_PROTOCOL_PATH: "fe3dccbb8fb5b6718714f5ff71e8b23c5c4dfd22385b9571cdc88b280c210b5d",
    G8_RESULT_PATH: "9ea58a909cd3e9b3976f1a93b7adc541bfeee8fe897e7b30c73e184c77bdf41e",
    LEGACY_P0_PATH: "e4cef9d39041c7d1f4e624f9ba9f1afedc368461bd5163cf63a1e015ae4aa3fa",
    LEGACY_P5_PATH: "cc78b634e970d0ec0614cef23b15bdf797c392c348d946e27e8abecbc60ce30d",
    ARCHITECTURE_CARD_PATH: "8cf0c75706c5c2a9f19558831ec2bc4a80f33f547d8891022014393d20b6e08f",
    Path("evaluate_remap_context_need_online_g7_variable_horizon.py"): "17f51af46a29ddab494ee7d83f262b3949fd9eb622e08c1295614c5be526fd2b",
    Path("evaluate_remap_context_need_online_g8_hidden_cutoff.py"): "67815034cf125fbd09cb309d79f3323265c5c71f29ea214e6eadc42195949f20",
    Path("remap_former/m1.py"): "8f1466a9c332a056b0cf24af832cea998179d18499a7186dc464a8422de0348b",
    Path("remap_former/memory.py"): "0f1a74a6cd12f2ee901293ebccfeea7252fefae44bf08c0ac6a5cdab7d032b44",
    Path("remap_former/pfc.py"): "091a8d9b645c7bba2e186fad25ccd9c151930dce597357d809101d7b96751dc4",
}

BASELINE_ORDER = (
    "hippoformer",
    "hippoformer_hpc_branch_mmtem_ablation",
    "mdelta",
    "short_window_transformer",
    "parameter_matched_transformer",
    "titans_mac_adaptation",
    "m1b_covariance",
)
BASELINE_LABELS = {
    "hippoformer": "Hippoformer",
    "hippoformer_hpc_branch_mmtem_ablation": "Hippoformer HPC branch",
    "mdelta": "M-delta",
    "short_window_transformer": "Window Transformer",
    "parameter_matched_transformer": "Parameter-matched Transformer",
    "titans_mac_adaptation": "Titans-MAC adaptation",
    "m1b_covariance": "ReMAP-Former M1b",
}
BASELINE_STEPS = {
    "hippoformer": 600,
    "hippoformer_hpc_branch_mmtem_ablation": 600,
    "mdelta": 600,
    "short_window_transformer": 1200,
    "parameter_matched_transformer": 1200,
    "titans_mac_adaptation": 1200,
    "m1b_covariance": 600,
}
BASELINE_PROVENANCE = {
    "hippoformer": "task-matched implementation",
    "hippoformer_hpc_branch_mmtem_ablation": "mechanism ablation; not official mmTEM",
    "mdelta": "task-matched no-context baseline",
    "short_window_transformer": "best-effort external baseline",
    "parameter_matched_transformer": "best-effort external baseline",
    "titans_mac_adaptation": "mechanism-aligned adaptation; not official checkpoint",
    "m1b_covariance": "formal frozen model",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def artifact_record(path: Path) -> dict[str, Any]:
    absolute = _root_path(path)
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return {
        "path": absolute.relative_to(ROOT).as_posix(),
        "bytes": absolute.stat().st_size,
        "sha256": file_sha256(absolute),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(_root_path(path).read_text(encoding="utf-8"))


def load_inputs() -> dict[str, dict[str, Any]]:
    mismatches = []
    for path, expected in FROZEN_INPUT_HASHES.items():
        absolute = _root_path(path)
        actual = file_sha256(absolute)
        if actual != expected:
            mismatches.append(
                {"path": path.as_posix(), "expected": expected, "actual": actual}
            )
    if mismatches:
        raise RuntimeError(f"frozen paper input hash mismatch: {mismatches}")

    inputs = {
        "p1": _read_json(P1_PATH),
        "p3": _read_json(P3_PATH),
        "g7": _read_json(G7_RESULT_PATH),
        "g8": _read_json(G8_RESULT_PATH),
        "legacy_p0": _read_json(LEGACY_P0_PATH),
        "legacy_p5": _read_json(LEGACY_P5_PATH),
    }
    if inputs["g7"].get("status") != "CONTEXT_NEED_ONLINE_G7_VARIABLE_HORIZON_PASS":
        raise RuntimeError("G7 formal result is not PASS")
    if inputs["g8"].get("status") != "CONTEXT_NEED_ONLINE_G8_HIDDEN_CUTOFF_PASS":
        raise RuntimeError("G8 formal result is not PASS")
    if not inputs["g8"].get("all_gates_pass"):
        raise RuntimeError("G8 formal gates are not all green")
    if not all(bool(value) for value in inputs["p3"]["integrity"].values()):
        raise RuntimeError("P3 integrity audit is not green")
    if inputs["legacy_p5"].get("status") != "EVIDENCE_COMPLETE_RELEASE_BLOCKED":
        raise RuntimeError("legacy P5 release boundary changed")
    if inputs["legacy_p5"]["gates"]["canonical_p0_hashes_match"]:
        raise RuntimeError("legacy P0 mismatch must remain disclosed")
    return inputs


def _endpoint(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimate": float(value["estimate"]),
        "ci95": [float(value["ci95_low"]), float(value["ci95_high"])],
    }


def build_baseline_table(p1: dict[str, Any], p3: dict[str, Any]) -> dict[str, Any]:
    aggregate = p3["aggregate"]["metrics"]
    intervals = p3["bootstrap"]["intervals"]
    parameter_counts = p3["results"][0]["parameter_counts"]
    strict_4096 = next(
        row for row in p1["horizon_results"] if int(row["horizon"]) == 4096
    )["models"]
    strict_map = {
        "hippoformer": strict_4096["hippoformer"],
        "mdelta": strict_4096["m_delta"],
        "m1b_covariance": strict_4096["m1b_covariance"],
    }
    rows = []
    for condition in BASELINE_ORDER:
        endpoint = intervals[condition]["return_conflict_acc"]
        strict = strict_map.get(condition)
        rows.append(
            {
                "condition": condition,
                "label": BASELINE_LABELS[condition],
                "parameters": int(parameter_counts[condition]),
                "optimizer_steps": BASELINE_STEPS[condition],
                "provenance": BASELINE_PROVENANCE[condition],
                "t1_strict_4096": None if strict is None else _endpoint(strict),
                "t2_clean": float(aggregate[condition]["clean_acc"]["mean"]),
                "t2_conflict": float(aggregate[condition]["conflict_acc"]["mean"]),
                "t2_return_conflict": {
                    "estimate": float(endpoint["point"]),
                    "ci95": [float(endpoint["ci_low"]), float(endpoint["ci_high"])],
                },
            }
        )
    return {
        "title": "Unified frozen baseline table",
        "rows": rows,
        "disclosures": [
            "T1 strict 4096 was run only for Hippoformer, M-delta, and M1b; other rows are N/A.",
            "Transformer and Titans rows use 1200-step best-effort training; the three original baselines use 600 steps.",
            "Titans-MAC is a task adaptation, not an official author checkpoint.",
            "Hippoformer HPC branch is a mechanism ablation, not an official mmTEM reproduction.",
        ],
    }


def _g7_row(g7: dict[str, Any], horizon: int) -> dict[str, float]:
    row = g7["pooled"]["by_horizon"][str(horizon)]
    return {
        "cutoff": horizon,
        "dynamic_coverage": float(row["t2_observed"]["dynamic_selection_rate"]),
        "return_conflict": float(row["t2_observed"]["return_conflict_accuracy"]),
        "t1_strict_4096": float(row["t1_strict"]["final_accuracy"]),
        "t2_clean": float(row["t2_observed"]["clean_accuracy"]),
        "online_prefix_passes": float(row["t2_observed"]["mean_prefix_hpc_passes"]),
        "fixed_prefix_passes": float(
            row["t2_observed"]["fixed_g5_mean_prefix_hpc_passes"]
        ),
    }


def _g8_row(g8: dict[str, Any], horizon: int) -> dict[str, float]:
    row = g8["pooled"]["by_cutoff"][str(horizon)]
    return {
        "cutoff": horizon,
        "dynamic_coverage": float(row["t2_dynamic_coverage"]),
        "return_conflict": float(row["t2_return_conflict_accuracy"]),
        "t1_strict_4096": float(row["t1_final_accuracy"]),
        "t2_clean": float(row["t2_clean_accuracy"]),
        "online_prefix_passes": float(row["online_prefix_hpc_passes"]),
        "fixed_prefix_passes": float(row["fixed_prefix_hpc_passes"]),
    }


def build_controller_table(g7: dict[str, Any], g8: dict[str, Any]) -> dict[str, Any]:
    horizons = [32, 48, 60, 72]
    expected = g8["expected_uniform_mixture"]
    realized = g8["pooled"]["mixture"]
    return {
        "title": "G7 paired curve and G8 randomized hidden-cutoff confirmation",
        "cutoff_values": horizons,
        "g7_by_horizon": [_g7_row(g7, horizon) for horizon in horizons],
        "g8_by_cutoff": [_g8_row(g8, horizon) for horizon in horizons],
        "g7_preregistered_uniform_mixture": {
            "dynamic_coverage": float(expected["t2_dynamic_coverage"]),
            "return_conflict": float(expected["t2_return_conflict_accuracy"]),
            "online_prefix_passes": float(expected["online_prefix_hpc_passes"]),
            "t1_strict_4096": float(expected["t1_final_accuracy_4096"]),
            "t2_clean": float(expected["t2_clean_accuracy"]),
        },
        "g8_realized_hidden_mixture": {
            "dynamic_coverage": float(realized["t2_dynamic_coverage"]),
            "return_conflict": float(realized["t2_return_conflict_accuracy"]),
            "online_prefix_passes": float(realized["online_prefix_hpc_passes"]),
            "fixed_prefix_passes": float(realized["fixed_prefix_hpc_passes"]),
            "t1_strict_4096": float(realized["t1_final_accuracy"]),
            "t2_clean": float(realized["t2_clean_accuracy"]),
        },
        "absolute_error": {
            "dynamic_coverage": float(realized["dynamic_coverage_abs_error"]),
            "return_conflict": float(realized["return_abs_error"]),
            "online_prefix_passes": float(realized["online_passes_abs_error"]),
        },
        "audits": {
            "g7_gates": len(g7["gates"]),
            "g8_gates": len(g8["gates"]),
            "g8_shadow_batches_passed": int(g8["shadow_audit"]["passed"]),
            "g8_shadow_batches_total": int(g8["shadow_audit"]["total"]),
            "g8_future_crossings_censored": int(g8["shadow_audit"]["censored_count"]),
            "minimum_decision_agreement": float(
                g8["equivalence"]["minimum_decision_agreement"]
            ),
            "maximum_state_logit_error": float(
                g8["equivalence"]["maximum_state_logit_error"]
            ),
            "coverage_spearman": float(g8["trend"]["dynamic_coverage_spearman"]),
            "return_spearman": float(g8["trend"]["return_spearman"]),
        },
    }


def build_tables(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p1 = inputs["p1"]
    legacy_p5 = inputs["legacy_p5"]
    return {
        "format": "remap_former.paper_evidence_g8_tables.v1",
        "date": "2026-07-18",
        "formal_model": {
            "name": "ReMAP-Former M1b covariance",
            "trainable_parameters": 1_398_289,
            "pfc": "one-layer causal Window Transformer, d=128, 8 heads, W=32",
            "context": "16-d action-history state -> 16-d latent context",
            "ec_place": "64-d neural grid EC -> 256-unit sparse place",
            "address": "place x context -> 4096-d multiplicative key",
            "hpc": "one episode-local covariance-corrected fast-weight matrix",
            "memory_initialization": "zero per episode; read-before-write",
            "forbidden_or_absent": [
                "room/context/position/place ID inputs",
                "slot bank",
                "second fast-weight system",
                "learned write gate",
                "iterative HPC reread",
            ],
        },
        "table_1_unified_baselines": build_baseline_table(inputs["p1"], inputs["p3"]),
        "table_2_hidden_cutoff_controller": build_controller_table(
            inputs["g7"], inputs["g8"]
        ),
        "table_3_claim_boundaries": {
            "strict_rollout_full_curve_h1_confirmed": bool(
                p1["decisions"]["h1_confirmed"]
            ),
            "m1b_minus_hippoformer_log_horizon_auc": _endpoint(
                p1["primary_endpoints"]["m1b_minus_hippoformer_auc"]
            ),
            "m1b_minus_hippoformer_4096": _endpoint(
                p1["primary_endpoints"]["m1b_minus_hippoformer_4096"]
            ),
            "legacy_p5_status": legacy_p5["status"],
            "legacy_p5_scientific_evidence_complete": bool(
                legacy_p5["gates"]["scientific_evidence_complete"]
            ),
            "legacy_p5_canonical_p0_hashes_match": bool(
                legacy_p5["gates"]["canonical_p0_hashes_match"]
            ),
            "legacy_p5_release_ready": bool(legacy_p5["gates"]["release_ready"]),
            "legacy_p0_mismatch_paths": [
                row["path"] for row in legacy_p5["audits"]["p0"]["mismatches"]
            ],
        },
        "frozen_inputs": [
            artifact_record(path) for path in FROZEN_INPUT_HASHES
        ],
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _save_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    png_temp = png.with_name(png.stem + ".tmp.png")
    pdf_temp = pdf.with_name(pdf.stem + ".tmp.pdf")
    fig.savefig(
        png_temp,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "build_remap_former_g8_paper_package.py"},
    )
    fig.savefig(
        pdf_temp,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "build_remap_former_g8_paper_package.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    os.replace(png_temp, png)
    os.replace(pdf_temp, pdf)
    plt.close(fig)


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    size: float = 7.2,
    linestyle: str = "-",
    linewidth: float = 1.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        linespacing=1.25,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#3D4652",
    linewidth: float = 1.0,
    connectionstyle: str = "arc3",
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 1,
            "shrinkB": 1,
            "connectionstyle": connectionstyle,
        },
    )


def plot_architecture() -> None:
    fig, axis = plt.subplots(figsize=(7.15, 4.05))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fig.patch.set_facecolor("white")

    colors = {
        "pfc": ("#DCEAF7", "#356A99"),
        "state": ("#E7E2F2", "#6E5A91"),
        "ec": ("#DCEFE5", "#3E7F5C"),
        "key": ("#F4E7CF", "#9B6B25"),
        "hpc": ("#F7DCDC", "#A54646"),
        "readout": ("#E5E8EB", "#4F5965"),
    }

    axis.text(
        0.5,
        0.97,
        "ReMAP-Former M1b: Window-Transformer PFC with episode-local neural HPC",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
    )
    axis.text(0.015, 0.78, "action $a_t$", fontsize=8, va="center")
    axis.text(0.015, 0.66, "lagged sensory\n$o_{t-1}$", fontsize=8, va="center")
    axis.text(0.015, 0.25, "observed value\n$o_t$", fontsize=8, va="center")

    _box(
        axis,
        0.15,
        0.68,
        0.27,
        0.19,
        "PFC = causal Window Transformer\n1 layer, $d=128$, 8 heads, $W=32$\n$h_t^{PFC}$ + short prediction",
        face=colors["pfc"][0],
        edge=colors["pfc"][1],
        size=7.4,
        linewidth=1.2,
    )
    _box(
        axis,
        0.15,
        0.42,
        0.27,
        0.15,
        "action-only retention state\ncyclic signature, 16-d",
        face=colors["state"][0],
        edge=colors["state"][1],
    )
    _box(
        axis,
        0.47,
        0.42,
        0.15,
        0.15,
        "latent context\n$c_t$, 16-d",
        face=colors["state"][0],
        edge=colors["state"][1],
    )
    _box(
        axis,
        0.15,
        0.15,
        0.19,
        0.15,
        "neural grid EC\n64-d structure",
        face=colors["ec"][0],
        edge=colors["ec"][1],
    )
    _box(
        axis,
        0.40,
        0.15,
        0.18,
        0.15,
        "sparse neural place\n$p_t$, 256 units",
        face=colors["ec"][0],
        edge=colors["ec"][1],
    )
    _box(
        axis,
        0.66,
        0.15,
        0.25,
        0.15,
        "multiplicative address\n$p_t \\times c_t$, 4096-d",
        face=colors["key"][0],
        edge=colors["key"][1],
    )
    _box(
        axis,
        0.66,
        0.40,
        0.28,
        0.19,
        "covariance-corrected HPC\none fast-weight matrix per episode\nread-before-write; ridge 0.03",
        face=colors["hpc"][0],
        edge=colors["hpc"][1],
        size=7.2,
        linewidth=1.2,
    )
    _box(
        axis,
        0.60,
        0.68,
        0.34,
        0.19,
        "retrieval projection + residual refinement\n$f(h_t^{PFC}, r_t)$ -> shared sensory decoder\n$\\hat o_t$",
        face=colors["readout"][0],
        edge=colors["readout"][1],
        size=7.3,
    )

    _arrow(axis, (0.10, 0.78), (0.15, 0.78))
    _arrow(axis, (0.10, 0.66), (0.15, 0.72))
    _arrow(axis, (0.10, 0.76), (0.15, 0.49), connectionstyle="arc3,rad=-0.12")
    _arrow(axis, (0.10, 0.74), (0.15, 0.23), connectionstyle="arc3,rad=-0.18")
    _arrow(axis, (0.42, 0.495), (0.47, 0.495))
    _arrow(axis, (0.34, 0.225), (0.40, 0.225))
    _arrow(axis, (0.58, 0.225), (0.66, 0.225))
    _arrow(axis, (0.545, 0.42), (0.73, 0.30), connectionstyle="arc3,rad=0.12")
    _arrow(axis, (0.79, 0.30), (0.79, 0.40))
    _arrow(axis, (0.10, 0.25), (0.66, 0.48), connectionstyle="arc3,rad=-0.14")
    _arrow(axis, (0.80, 0.59), (0.80, 0.68))
    _arrow(axis, (0.42, 0.78), (0.60, 0.78))

    axis.text(0.49, 0.80, "$h_t^{PFC}$", fontsize=7.5, color=colors["pfc"][1])
    axis.text(0.81, 0.63, "retrieval $r_t$", fontsize=7.2, ha="left")
    axis.text(0.43, 0.34, "address only", fontsize=6.8, color=colors["state"][1])
    axis.text(
        0.50,
        0.055,
        "Absent: room/position/place IDs, slot bank, second fast weights, learned write gate. "
        "Memory and context covariance start at zero and are discarded after each episode.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#4F5965",
    )
    _save_figure(fig, ARCH_PNG, ARCH_PDF)


def plot_hidden_cutoff(controller: dict[str, Any]) -> None:
    horizons = np.asarray(controller["cutoff_values"], dtype=float)
    mix_x = 86.0
    metrics = (
        ("dynamic_coverage", "Dynamic coverage", (0.0, 1.05), "{:.3f}"),
        ("return_conflict", "Return-conflict accuracy", (0.0, 0.92), "{:.3f}"),
        ("online_prefix_passes", "Online prefix passes", (0.95, 1.92), "{:.3f}"),
    )
    g7_rows = controller["g7_by_horizon"]
    g8_rows = controller["g8_by_cutoff"]
    expected = controller["g7_preregistered_uniform_mixture"]
    realized = controller["g8_realized_hidden_mixture"]
    errors = controller["absolute_error"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.6,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.72))
    g7_color = "#087E8B"
    g8_color = "#D1495B"
    for index, (key, label, limits, formatter) in enumerate(metrics):
        axis = axes[index]
        g7_values = [float(row[key]) for row in g7_rows]
        g8_values = [float(row[key]) for row in g8_rows]
        axis.plot(
            horizons,
            g7_values,
            color=g7_color,
            marker="o",
            linewidth=1.7,
            markersize=3.8,
            label="G7 paired curve",
        )
        axis.plot(
            horizons,
            g8_values,
            color=g8_color,
            marker="s",
            linewidth=1.35,
            markersize=3.5,
            linestyle="--",
            label="G8 hidden strata",
        )
        axis.scatter(
            [mix_x],
            [expected[key]],
            marker="D",
            s=28,
            facecolors="white",
            edgecolors=g7_color,
            linewidths=1.2,
            zorder=4,
            label="G7 expected mix",
        )
        axis.scatter(
            [mix_x],
            [realized[key]],
            marker="D",
            s=28,
            color=g8_color,
            zorder=5,
            label="G8 realized mix",
        )
        axis.axvline(79, color="#AAB2BA", linewidth=0.7, linestyle=":")
        axis.set_xlim(28, 90)
        axis.set_ylim(*limits)
        axis.set_xticks([32, 48, 60, 72, mix_x], ["32", "48", "60", "72", "mix"])
        axis.set_xlabel("Environment cutoff $H$")
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.annotate(
            f"|d|={formatter.format(errors[key])}",
            xy=(mix_x, realized[key]),
            xytext=(-3, 10 if realized[key] >= expected[key] else -14),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
            color=g8_color,
        )
        if index == 0:
            axis.legend(frameon=False, loc="upper left", handlelength=2.1)

    fig.suptitle(
        "Paired evidence-accumulation curves predict randomized hidden-cutoff deployment",
        fontsize=9.4,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "Cutoff is environment-only, never a model token. G8: 54/54 shadow audits; "
        "826 post-cutoff crossings correctly censored; T1-4096=0.9978, clean=0.9982.",
        ha="center",
        va="top",
        fontsize=6.6,
        color="#4F5965",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.0)
    _save_figure(fig, CURVE_PNG, CURVE_PDF)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_endpoint(value: dict[str, Any] | None) -> str:
    if value is None:
        return "N/A"
    low, high = value["ci95"]
    return f"{_fmt(value['estimate'])} [{_fmt(low)}, {_fmt(high)}]"


def render_report(tables: dict[str, Any]) -> str:
    baseline = tables["table_1_unified_baselines"]
    controller = tables["table_2_hidden_cutoff_controller"]
    boundary = tables["table_3_claim_boundaries"]
    expected = controller["g7_preregistered_uniform_mixture"]
    realized = controller["g8_realized_hidden_mixture"]
    errors = controller["absolute_error"]
    audits = controller["audits"]

    lines = [
        "# ReMAP-Former：G8 论文证据包",
        "",
        "> 本报告由冻结 JSON 自动生成。旧 P0/P5 release block 原样保留；本包不覆盖或改判历史 manifest。",
        "",
        "## 一句话主张",
        "",
        "正式 M1b 用 Window Transformer 作为 PFC 主干，以 action-history latent context 调制 neural place address，并调用一个 episode-local covariance fast-weight HPC。G7/G8 进一步证明：合法 retrieval evidence 尚未出现时保持 null，首次 crossing 后再调用 dynamic context；随机隐藏 cutoff 不需要 room、position 或 cutoff token。",
        "",
        "## Figure 1：正式 M1b 架构",
        "",
        "![M1b architecture](figures/remap_former_m1b_architecture_g8.png)",
        "",
        "- PFC：1-layer causal Window Transformer，`d=128`、8 heads、window 32。",
        "- EC/place：64-d neural grid EC -> 256-unit sparse place。",
        "- Address：place x 16-d context -> 4096-d key。",
        "- HPC：每 episode 从零建立一套 covariance-corrected fast weights，read-before-write，episode 后丢弃。",
        "- 不存在：room/position/place ID、slot bank、第二套 fast weights、learned write gate。",
        "",
        "## Figure 2：G7 曲线预测 G8 隐藏流",
        "",
        "![G7 G8 hidden cutoff](figures/remap_former_g7_g8_hidden_cutoff.png)",
        "",
        "| 指标 | G7 预注册均匀混合 | G8 hidden realized | 绝对差 |",
        "|---|---:|---:|---:|",
        f"| Dynamic coverage | {_fmt(expected['dynamic_coverage'])} | {_fmt(realized['dynamic_coverage'])} | {_fmt(errors['dynamic_coverage'])} |",
        f"| Return-conflict | {_fmt(expected['return_conflict'])} | {_fmt(realized['return_conflict'])} | {_fmt(errors['return_conflict'])} |",
        f"| Online passes | {_fmt(expected['online_prefix_passes'])} | {_fmt(realized['online_prefix_passes'])} | {_fmt(errors['online_prefix_passes'])} |",
        f"| T1 strict 4096 | {_fmt(expected['t1_strict_4096'])} | {_fmt(realized['t1_strict_4096'])} | 未设误差门 |",
        f"| T2 clean | {_fmt(expected['t2_clean'])} | {_fmt(realized['t2_clean'])} | 未设误差门 |",
        "",
        f"G8 为 59/59 gates；shadow audit `{audits['g8_shadow_batches_passed']}/{audits['g8_shadow_batches_total']}`，正确截断 `{audits['g8_future_crossings_censored']}` 个 H 后 crossing；online/fixed agreement `{audits['minimum_decision_agreement']:.1f}`，最大 state/logit error `{audits['maximum_state_logit_error']:.2e}`。",
        "",
        "## Table 1：统一冻结基线",
        "",
        "| 模型 | 参数 | steps | T1 strict 4096 (95% CI) | T2 clean | T2 return-conflict (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline["rows"]:
        lines.append(
            f"| {row['label']} | {row['parameters']:,} | {row['optimizer_steps']} | "
            f"{_fmt_endpoint(row['t1_strict_4096'])} | {_fmt(row['t2_clean'])} | "
            f"{_fmt_endpoint(row['t2_return_conflict'])} |"
        )
    lines.extend(
        [
            "",
            "边界：T1 4096 只评测了 Hippoformer、M-delta、M1b，其他行必须写 N/A；Transformer/Titans 使用 1200-step best-effort，不能写成完全预算匹配。Titans-MAC 是机制对齐的任务适配版，不是官方 checkpoint；Hippoformer HPC branch 是消融，不是官方 mmTEM。",
            "",
            "## 不能隐藏的负结果",
            "",
            f"- Strict rollout 全曲线 H1：`{boundary['strict_rollout_full_curve_h1_confirmed']}`。",
            f"- M1b-Hippo log-horizon AUC：`{boundary['m1b_minus_hippoformer_log_horizon_auc']['estimate']:+.4f}`，95% CI `{boundary['m1b_minus_hippoformer_log_horizon_auc']['ci95']}`。",
            f"- M1b-Hippo 4096 endpoint：`{boundary['m1b_minus_hippoformer_4096']['estimate']:+.4f}`，95% CI `{boundary['m1b_minus_hippoformer_4096']['ci95']}`。",
            f"- 旧 P5 状态：`{boundary['legacy_p5_status']}`；P0 hashes match=`{boundary['legacy_p5_canonical_p0_hashes_match']}`，release ready=`{boundary['legacy_p5_release_ready']}`。",
            f"- 旧 mismatch 路径：`{', '.join(boundary['legacy_p0_mismatch_paths'])}`。本包只披露，不重写旧 manifest。",
            "",
            "## 当前论文口径",
            "",
            "可以主张：在无 oracle 的 hidden-context re-entry 任务中，context-conditioned covariance HPC 显著解决所有健康基线失败的 return-conflict；paired horizon 曲线能够预测随机隐藏 cutoff 部署。",
            "",
            "不能主张：通用 autonomous rollout 已解决、Titans/mmTEM 官方复现完成、任意 cutoff policy 均稳定、或旧 P0 release block 已解除。",
            "",
            "## 产物",
            "",
            "- 机器表：`runs/remap_former/paper_evidence_g8_tables.json`",
            "- 增量 manifest：`runs/remap_former/paper_evidence_g8_manifest.json`",
            "- 架构图：`reports/figures/remap_former_m1b_architecture_g8.png/.pdf`",
            "- G7/G8 主图：`reports/figures/remap_former_g7_g8_hidden_cutoff.png/.pdf`",
        ]
    )
    return "\n".join(lines) + "\n"


def image_audit(path: Path) -> dict[str, Any]:
    image = mpimg.imread(path)
    rgb = image[..., :3].astype(np.float64)
    if rgb.max() > 1.0:
        rgb /= 255.0
    nonwhite = np.any(rgb < 0.985, axis=-1)
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "pixel_std": float(rgb.std()),
        "nonwhite_fraction": float(nonwhite.mean()),
        "nonblank": bool(rgb.std() > 0.02 and nonwhite.mean() > 0.01),
    }


def build_manifest(tables: dict[str, Any]) -> dict[str, Any]:
    controller = tables["table_2_hidden_cutoff_controller"]
    legacy = tables["table_3_claim_boundaries"]
    architecture_audit = image_audit(ARCH_PNG)
    curve_audit = image_audit(CURVE_PNG)
    generated = [TABLES_PATH, REPORT_PATH, ARCH_PNG, ARCH_PDF, CURVE_PNG, CURVE_PDF]
    return {
        "format": "remap_former.paper_evidence_g8_manifest.v1",
        "date": "2026-07-18",
        "scope": "Additive G8 paper evidence package; does not replace legacy P0/P5.",
        "builder": artifact_record(Path(__file__).resolve()),
        "frozen_inputs": [artifact_record(path) for path in FROZEN_INPUT_HASHES],
        "generated_outputs": [artifact_record(path) for path in generated],
        "audits": {
            "frozen_input_hash_count": len(FROZEN_INPUT_HASHES),
            "g7_status": "CONTEXT_NEED_ONLINE_G7_VARIABLE_HORIZON_PASS",
            "g8_status": "CONTEXT_NEED_ONLINE_G8_HIDDEN_CUTOFF_PASS",
            "g7_gate_count": controller["audits"]["g7_gates"],
            "g8_gate_count": controller["audits"]["g8_gates"],
            "g8_shadow_batches": [
                controller["audits"]["g8_shadow_batches_passed"],
                controller["audits"]["g8_shadow_batches_total"],
            ],
            "architecture_png": architecture_audit,
            "hidden_cutoff_png": curve_audit,
            "pdf_bytes": {
                ARCH_PDF.name: ARCH_PDF.stat().st_size,
                CURVE_PDF.name: CURVE_PDF.stat().st_size,
            },
            "legacy_p5_status": legacy["legacy_p5_status"],
            "legacy_p5_release_ready": legacy["legacy_p5_release_ready"],
            "legacy_p0_mismatch_paths": legacy["legacy_p0_mismatch_paths"],
            "legacy_manifest_preserved": file_sha256(_root_path(LEGACY_P5_PATH))
            == FROZEN_INPUT_HASHES[LEGACY_P5_PATH],
        },
        "gates": {
            "all_frozen_inputs_match": True,
            "g7_g8_formal_results_pass": True,
            "g8_shadow_audit_complete": controller["audits"][
                "g8_shadow_batches_passed"
            ]
            == controller["audits"]["g8_shadow_batches_total"],
            "figures_nonblank": architecture_audit["nonblank"]
            and curve_audit["nonblank"],
            "pdfs_nontrivial": ARCH_PDF.stat().st_size > 5_000
            and CURVE_PDF.stat().st_size > 5_000,
            "legacy_p0_block_disclosed": legacy[
                "legacy_p5_canonical_p0_hashes_match"
            ]
            is False,
            "legacy_p5_not_overwritten": True,
        },
        "status": "PAPER_EVIDENCE_G8_READY_LEGACY_P0_BLOCK_DISCLOSED",
    }


def verify_package() -> None:
    inputs = load_inputs()
    expected_tables = build_tables(inputs)
    expected_report = render_report(expected_tables)
    if json.loads(TABLES_PATH.read_text(encoding="utf-8")) != expected_tables:
        raise RuntimeError("paper tables no longer match frozen inputs")
    if REPORT_PATH.read_text(encoding="utf-8") != expected_report:
        raise RuntimeError("paper report no longer matches frozen inputs")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for row in manifest["generated_outputs"]:
        path = ROOT / row["path"]
        if file_sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"generated output mismatch: {row['path']}")
    if file_sha256(Path(__file__).resolve()) != manifest["builder"]["sha256"]:
        raise RuntimeError("paper package builder hash mismatch")
    if not all(bool(value) for value in manifest["gates"].values()):
        raise RuntimeError("paper package gate failed")
    print(
        f"verified {MANIFEST_PATH.relative_to(ROOT)} "
        f"sha256={file_sha256(MANIFEST_PATH)}"
    )


def build_package() -> None:
    outputs = [
        TABLES_PATH,
        MANIFEST_PATH,
        REPORT_PATH,
        ARCH_PNG,
        ARCH_PDF,
        CURVE_PNG,
        CURVE_PDF,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite paper outputs: {existing}")
    inputs = load_inputs()
    tables = build_tables(inputs)
    _atomic_write_json(TABLES_PATH, tables)
    plot_architecture()
    plot_hidden_cutoff(tables["table_2_hidden_cutoff_controller"])
    _atomic_write_text(REPORT_PATH, render_report(tables))
    manifest = build_manifest(tables)
    if not all(bool(value) for value in manifest["gates"].values()):
        raise RuntimeError(f"paper package gate failed: {manifest['gates']}")
    _atomic_write_json(MANIFEST_PATH, manifest)
    print(manifest["status"])
    for path in outputs:
        print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the additive G8 paper evidence package."
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_package()
    else:
        build_package()


if __name__ == "__main__":
    main()
