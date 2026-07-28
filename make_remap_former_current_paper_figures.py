from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "reports" / "figures"
DATA_PATH = (
    ROOT
    / "data"
    / "memorymaze3d_simulator_translated_waypoint_aba_sealed_v1"
    / "test.npz"
)
REMAP_SEALED_PATH = (
    ROOT
    / "runs"
    / "memorymaze3d"
    / "simulator_translated_waypoint_aba_sealed_v1"
    / "summary.json"
)
FULL_TF_PATH = (
    ROOT
    / "runs"
    / "memorymaze3d"
    / "full_transformer_sealed_v1"
    / "summary.json"
)
HEADLINE_PATH = ROOT / "runs" / "remap_former" / "headline8_test_seed1892.json"
ABLATION_PATH = (
    ROOT
    / "runs"
    / "remap_former"
    / "m1b_component_ablation_validation1893.json"
)
LONG_A_PATH = (
    ROOT / "runs" / "remap_former" / "long_delay_stage_a_validation1894.json"
)
LONG_B_PATH = (
    ROOT / "runs" / "remap_former" / "long_delay_stage_b_validation1895.json"
)

EXPECTED_SHA256 = {
    DATA_PATH: "18547c8a869c46cf0342f6b98624819898e9df2756bc0e542fee292ebdf0f069",
    REMAP_SEALED_PATH: "6607d9b7dc3ace7c6921707704c2d534276ea2124cc236f07d0420948d4f3143",
    FULL_TF_PATH: "8b6327a5206a0da184bfe8ec1795ddce6ef036ca1a194d5b07ca6ce46fa70269",
    HEADLINE_PATH: "6c56c6071b267bdf1a1f87177ebb09cfbce0ea05d0101e23b94f18bbaac8d4dd",
    ABLATION_PATH: "8fff414a20a36cb488aaa04a3672c9ec16ebf17f7df2620d8b8b59b8dccf313f",
    LONG_A_PATH: "b833be2a46ca98120b5803533e7ae5c4a1d98125fba9425d0af8949fa1f8284c",
    LONG_B_PATH: "7cd8cbb2b882639bfa1883c303678dcc5242ba96b2b9ec8f6374eab96909d3ec",
}

INK = "#18202A"
MUTED = "#5B6773"
LIGHT = "#EEF2F4"
BLUE = "#2364AA"
TEAL = "#1B998B"
GREEN = "#4C956C"
GOLD = "#D19A2E"
CORAL = "#D65A4A"
PURPLE = "#7656A8"
GRAY = "#9AA5AE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> None:
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B7C0C7",
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str, *, vector: bool = True) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    if vector:
        fig.savefig(
            FIGURE_DIR / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    *,
    title_size: float = 10.2,
    body_size: float = 7.2,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.35,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y + height - 0.06), width, 0.06, color=color))
    ax.text(
        x + 0.018,
        y + height - 0.08,
        title,
        ha="left",
        va="top",
        color=INK,
        fontsize=title_size,
        fontweight="bold",
    )
    ax.text(
        x + 0.018,
        y + height - 0.12,
        body,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=body_size,
        linespacing=1.22,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    *,
    connectionstyle: str = "arc3",
    linewidth: float = 1.4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 6.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.015,
        0.965,
        "ReMAP-Former: a Transformer controller with one remapped episode-local neural memory",
        color=INK,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.015,
        0.915,
        "Current 3D configuration; all dimensions and pathways correspond to the frozen sealed experiment.",
        color=MUTED,
        fontsize=9.5,
        va="top",
    )

    add_box(
        ax,
        (0.015, 0.61),
        0.125,
        0.22,
        "Causal inputs",
        "action  a_t  (6D)\nlagged DINO feature\nx_{t-1}  (384D)\nwrite event  w_t",
        BLUE,
    )
    add_box(
        ax,
        (0.18, 0.61),
        0.155,
        0.22,
        "PFC Transformer",
        "causal self-attention\nwindow W = 32\nhidden h_t in R^96\n4 heads",
        BLUE,
    )
    add_box(
        ax,
        (0.375, 0.61),
        0.155,
        0.22,
        "Retention state",
        "rolling history W = 128\npersistent state s_t in R^16\nlearned retention gate",
        PURPLE,
    )
    add_box(
        ax,
        (0.57, 0.61),
        0.14,
        0.22,
        "Latent context",
        "c_t = norm(tanh(W_c s_t))\nc_t in R^8\nno context label",
        GOLD,
    )

    add_box(
        ax,
        (0.18, 0.22),
        0.155,
        0.22,
        "Action-only EC",
        "egocentric SE(2)\nintegration\nperiodic grid g_t in R^32\nfrozen after calibration",
        TEAL,
    )
    add_box(
        ax,
        (0.375, 0.22),
        0.155,
        0.22,
        "Sparse place code",
        "p_t = sparsemax(W_gp g_t / tau)\np_t in R^64, tau = 0.25\nfrozen neural projection",
        GREEN,
        body_size=7.1,
    )
    add_box(
        ax,
        (0.57, 0.22),
        0.14,
        0.22,
        "Conjunctive key",
        "k_t = norm(vec(p_t c_t^T))\nk_t in R^512\nplace and context factorized",
        GOLD,
        body_size=7.1,
    )
    add_box(
        ax,
        (0.75, 0.28),
        0.16,
        0.47,
        "Episode-local HPC",
        "one dense fast-weight matrix\nM_t in R^{384 x 512}\n\nread before write:\nr_t = M_t k_t\n\nerror-correcting delta write\nwith dual place/context keys\n\nM_0 = 0; discard at episode end",
        CORAL,
        body_size=7.2,
    )
    add_box(
        ax,
        (0.94, 0.40),
        0.05,
        0.23,
        "Read",
        "retrieved\n384D\nDINO\nfeature",
        INK,
        title_size=9.2,
        body_size=6.6,
    )

    add_arrow(ax, (0.14, 0.72), (0.18, 0.72), BLUE)
    add_arrow(ax, (0.335, 0.72), (0.375, 0.72), PURPLE)
    add_arrow(ax, (0.53, 0.72), (0.57, 0.72), GOLD)
    add_arrow(ax, (0.078, 0.61), (0.18, 0.33), TEAL, connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.335, 0.33), (0.375, 0.33), GREEN)
    add_arrow(ax, (0.53, 0.33), (0.57, 0.33), GOLD)
    add_arrow(ax, (0.64, 0.61), (0.64, 0.44), GOLD)
    add_arrow(ax, (0.71, 0.33), (0.75, 0.43), CORAL)
    add_arrow(ax, (0.71, 0.72), (0.75, 0.64), CORAL)
    add_arrow(ax, (0.91, 0.515), (0.94, 0.515), INK)
    add_arrow(ax, (0.69, 0.12), (0.75, 0.31), CORAL, connectionstyle="arc3,rad=-0.08", linewidth=1.0)
    ax.text(
        0.42,
        0.105,
        "write value x_t only when the externally matched event flag w_t = 1",
        color=CORAL,
        fontsize=8.1,
        ha="center",
        va="center",
    )
    ax.text(
        0.015,
        0.025,
        "Absent from the model input: room/context ID, simulator pose, waypoint/place ID, route ID, target label, future observation. "
        "Absent from the architecture: memory slots, a persistent content table, or a second fast-weight system.",
        color=MUTED,
        fontsize=8.2,
        va="bottom",
    )
    save_figure(fig, "remap_former_current_architecture")


def add_paper_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    *,
    title_size: float = 6.6,
    body_size: float = 5.8,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.009,rounding_size=0.015",
        linewidth=1.1,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y + height - 0.035), width, 0.035, color=color))
    ax.text(
        x + 0.012,
        y + height - 0.055,
        title,
        color=INK,
        fontsize=title_size,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        x + 0.012,
        y + height - 0.115,
        body,
        color=MUTED,
        fontsize=body_size,
        ha="left",
        va="top",
        linespacing=1.18,
    )


def make_architecture_paper_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_paper_box(
        ax,
        (0.01, 0.61),
        0.13,
        0.27,
        "Causal inputs",
        "PFC: a_t, x_{t-1}\nHPC: write flag w_t",
        BLUE,
    )
    add_paper_box(
        ax,
        (0.18, 0.61),
        0.14,
        0.27,
        "Transformer PFC",
        "W = 32, h = 96\n4 heads",
        BLUE,
    )
    add_paper_box(
        ax,
        (0.36, 0.61),
        0.13,
        0.27,
        "Retention",
        "history W = 128\ns_t: 16D",
        PURPLE,
    )
    add_paper_box(
        ax,
        (0.53, 0.61),
        0.13,
        0.27,
        "Context",
        "tanh projection\nc_t: 8D",
        GOLD,
        body_size=5.6,
    )
    add_paper_box(
        ax,
        (0.18, 0.12),
        0.14,
        0.27,
        "Action-only EC",
        "periodic SE(2)\ng_t: 32D",
        TEAL,
    )
    add_paper_box(
        ax,
        (0.36, 0.12),
        0.13,
        0.27,
        "Sparse place",
        "sparsemax(W g)\np_t: 64D",
        GREEN,
        body_size=5.6,
    )
    add_paper_box(
        ax,
        (0.53, 0.12),
        0.13,
        0.27,
        "Remapped key",
        "place x context\nk_t: 512D",
        GOLD,
        body_size=5.5,
    )
    add_paper_box(
        ax,
        (0.70, 0.18),
        0.20,
        0.60,
        "Episode-local HPC",
        "M_0 = 0\nM_t: 384 x 512\n\nread before write\nr_t = M_t k_t\n\ndual-key delta update\ndiscard after episode",
        CORAL,
        body_size=5.7,
    )
    add_paper_box(
        ax,
        (0.93, 0.35),
        0.06,
        0.28,
        "Recall",
        "384D\nDINO\nfeature",
        INK,
        title_size=6.2,
        body_size=5.5,
    )

    add_arrow(ax, (0.14, 0.745), (0.18, 0.745), BLUE, linewidth=1.1)
    add_arrow(ax, (0.32, 0.745), (0.36, 0.745), PURPLE, linewidth=1.1)
    add_arrow(ax, (0.49, 0.745), (0.53, 0.745), GOLD, linewidth=1.1)
    add_arrow(ax, (0.075, 0.61), (0.18, 0.255), TEAL, connectionstyle="arc3,rad=0.12", linewidth=1.1)
    add_arrow(ax, (0.32, 0.255), (0.36, 0.255), GREEN, linewidth=1.1)
    add_arrow(ax, (0.49, 0.255), (0.53, 0.255), GOLD, linewidth=1.1)
    add_arrow(ax, (0.595, 0.61), (0.595, 0.39), GOLD, linewidth=1.1)
    add_arrow(ax, (0.66, 0.255), (0.70, 0.35), CORAL, linewidth=1.1)
    add_arrow(ax, (0.66, 0.745), (0.70, 0.65), CORAL, linewidth=1.1)
    add_arrow(ax, (0.90, 0.49), (0.93, 0.49), INK, linewidth=1.1)
    ax.text(
        0.41,
        0.025,
        "No context/room/pose/place/route/target ID. No slots, persistent content table, or second fast-weight system.",
        color=MUTED,
        fontsize=5.7,
        ha="center",
        va="bottom",
    )
    save_figure(fig, "remap_former_current_architecture_paper")


def make_model_family_comparison() -> None:
    rows = [
        (
            "Full-context\nTransformer",
            "All prior tokens through\ncausal self-attention",
            "No explicit episode-local\nassociative matrix",
            "Content attention;\ndirect prediction",
            "3D sealed: 0.530",
            BLUE,
        ),
        (
            "Hippoformer\n(mm-TEM + TF)",
            "Transformer augmented by\na structural-memory module",
            "mm-TEM relational and\nfast-weight machinery",
            "Learned structural code;\ncontent/structure interaction",
            "2D internal adaptation only",
            PURPLE,
        ),
        (
            "Titans / Gated\nDeltaNet family",
            "Attention plus a neural\nlong-term matrix",
            "Test-time gradient or\ngated delta updates",
            "Predominantly\ncontent-derived keys",
            "Mechanism comparison",
            TEAL,
        ),
        (
            "ReMAP-Former",
            "Windowed PFC plus a\npersistent retention state",
            "One M_t, initialized to zero\nand discarded per episode",
            "Explicit place x context key;\ndual-key delta write",
            "3D sealed: 0.977",
            CORAL,
        ),
    ]
    columns = [
        ("Model family", 0.02, 0.17),
        ("History access", 0.20, 0.22),
        ("Fast state", 0.43, 0.22),
        ("Address / write rule", 0.66, 0.20),
        ("Evidence in this work", 0.87, 0.11),
    ]
    fig, ax = plt.subplots(figsize=(14.2, 5.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.015,
        0.965,
        "How ReMAP-Former differs from adjacent memory architectures",
        color=INK,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    header_y = 0.82
    for title, x, width in columns:
        ax.add_patch(Rectangle((x, header_y), width, 0.075, color=INK))
        ax.text(
            x + 0.01,
            header_y + 0.0375,
            title,
            color="white",
            fontsize=8.5,
            fontweight="bold",
            va="center",
        )
    row_height = 0.155
    for index, row in enumerate(rows):
        y = header_y - (index + 1) * row_height
        background = "#F7F9FA" if index % 2 == 0 else "white"
        ax.add_patch(Rectangle((0.02, y), 0.96, row_height, color=background))
        model, access, state, addressing, evidence, color = row
        values = [model, access, state, addressing, evidence]
        for (unused, x, width), value in zip(columns, values):
            ax.text(
                x + 0.01,
                y + row_height / 2,
                value,
                color=color if x == 0.02 else INK,
                fontsize=8.35,
                fontweight="bold" if x in {0.02, 0.87} else "normal",
                va="center",
                ha="left",
                linespacing=1.25,
            )
        ax.add_patch(Rectangle((0.02, y), 0.007, row_height, color=color))
    ax.text(
        0.02,
        0.03,
        "Scores are return-context conflict accuracy on the fresh-sealed 3D task. "
        "The Hippoformer entry denotes an internal navigation adaptation in the 2D study, not an official reproduction. "
        "Titans and Gated DeltaNet are discussed as related mechanisms and are not claimed as sealed 3D baselines here.",
        color=MUTED,
        fontsize=8.1,
        va="bottom",
    )
    save_figure(fig, "remap_former_model_family_comparison")


def metric_mean(metrics: dict, model: str, name: str) -> float:
    return float(metrics[model][name]["mean"])


def make_2d_results_figure() -> None:
    headline = load_json(HEADLINE_PATH)
    ablation = load_json(ABLATION_PATH)
    long_a = load_json(LONG_A_PATH)
    long_b = load_json(LONG_B_PATH)

    headline_metrics = headline["aggregate"]["metrics"]
    headline_values = [
        metric_mean(headline_metrics, "hippoformer", "return_conflict_acc"),
        metric_mean(headline_metrics, "mdelta", "return_conflict_acc"),
        metric_mean(headline_metrics, "m1b_covariance", "return_conflict_acc"),
    ]
    headline_sd = [
        float(
            headline_metrics[name]["return_conflict_acc"]["population_std"]
        )
        for name in ["hippoformer", "mdelta", "m1b_covariance"]
    ]

    ablation_metrics = ablation["aggregate"]["metrics"]
    ablation_names = [
        "Full",
        "No covariance",
        "Fixed context",
        "Shuffled context",
        "HPC read = 0",
        "Wrong return context",
        "Correct return context",
    ]
    ablation_keys = [
        "full",
        "no_covariance",
        "fixed_context_all_steps",
        "shuffled_context_all_steps",
        "hpc_read_zero",
        "wrong_return_context",
        "correct_return_context",
    ]
    ablation_values = [
        float(ablation_metrics[key]["return_conflict_acc"]["mean"])
        for key in ablation_keys
    ]

    lengths_a: list[int] = []
    remap_a: list[float] = []
    no_cov_a: list[float] = []
    hippo_a: list[float] = []
    for repeat in ["1", "2", "4", "8", "16"]:
        item = long_a["aggregate"]["horizons"][repeat]
        lengths_a.append(int(item["sequence_length"]))
        metrics = item["metrics"]
        remap_a.append(metric_mean(metrics, "m1b_covariance", "return_conflict_acc"))
        no_cov_a.append(
            metric_mean(metrics, "m1b_no_covariance", "return_conflict_acc")
        )
        hippo_a.append(metric_mean(metrics, "hippoformer", "return_conflict_acc"))
    lengths_b: list[int] = []
    remap_b: list[float] = []
    for repeat in ["16", "32", "64", "96"]:
        item = long_b["aggregate"]["horizons"][repeat]
        lengths_b.append(int(item["sequence_length"]))
        remap_b.append(
            metric_mean(item["metrics"], "m1b_covariance", "return_conflict_acc")
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.45))
    fig.subplots_adjust(wspace=0.38, top=0.77, bottom=0.22)
    fig.suptitle(
        "2D hidden-context A-B-A: the conjunctive fast-weight path is necessary and stable",
        x=0.02,
        y=0.97,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    labels = ["Hippoformer\nadaptation", "M-delta", "ReMAP"]
    colors = [PURPLE, GRAY, CORAL]
    bars = ax.bar(
        np.arange(3),
        headline_values,
        yerr=headline_sd,
        capsize=3,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title("(a) Eight-seed return conflict")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(3), labels)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, headline_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.5,
            fontweight="bold",
        )
    ax.text(
        0.02,
        0.98,
        "8 seeds; 1,024 return-conflict\nprobes per seed",
        transform=ax.transAxes,
        va="top",
        color=MUTED,
        fontsize=7.8,
    )

    ax = axes[1]
    y = np.arange(len(ablation_names))
    bar_colors = [
        CORAL,
        GRAY,
        GOLD,
        PURPLE,
        BLUE,
        "#B54A45",
        GREEN,
    ]
    bars = ax.barh(y, ablation_values, color=bar_colors, height=0.68)
    ax.set_title("(b) Mechanism interventions")
    ax.set_xlabel("Return-conflict accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_yticks(y, ablation_names)
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ablation_values):
        ax.text(
            min(value + 0.018, 0.94),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            color=INK,
            fontsize=7.7,
        )

    ax = axes[2]
    ax.plot(
        lengths_a,
        remap_a,
        color=CORAL,
        marker="o",
        linewidth=2.2,
        label="ReMAP (8 seeds)",
    )
    ax.plot(
        lengths_a,
        no_cov_a,
        color=GRAY,
        marker="s",
        linewidth=1.8,
        label="No covariance",
    )
    ax.plot(
        lengths_a,
        hippo_a,
        color=PURPLE,
        marker="^",
        linewidth=1.8,
        label="Hippoformer adaptation",
    )
    ax.plot(
        lengths_b,
        remap_b,
        color=CORAL,
        marker="D",
        linestyle="--",
        linewidth=1.5,
        label="ReMAP extension (3 seeds)",
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlabel("Observed-history sequence length")
    ax.set_ylabel("Return-conflict accuracy")
    ax.set_title("(c) Delay scaling")
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.2, loc="center right")
    ax.annotate(
        "4,356 steps",
        xy=(lengths_b[-1], remap_b[-1]),
        xytext=(1500, 0.88),
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9},
        fontsize=7.8,
        color=MUTED,
    )
    fig.text(
        0.02,
        0.025,
        "The delay study supplies observations throughout the history and tests delayed context-dependent recall; it is not action-only free rollout. "
        "The Hippoformer result is an internal navigation adaptation.",
        color=MUTED,
        fontsize=8.0,
        va="bottom",
    )
    save_figure(fig, "remap_former_2d_current_results")

    with plt.rc_context(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
        fig.subplots_adjust(wspace=0.55, top=0.86, bottom=0.25, left=0.07, right=0.99)

        ax = axes[0]
        bars = ax.bar(
            np.arange(3),
            headline_values,
            yerr=headline_sd,
            capsize=2,
            color=[PURPLE, GRAY, CORAL],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title("(a) Eight-seed return conflict")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(np.arange(3), ["Hippo.\nadapt.", "M-delta", "ReMAP"])
        ax.grid(axis="y", color=LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, headline_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.04,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.1,
                fontweight="bold",
            )

        ax = axes[1]
        short_names = [
            "Full",
            "No covariance",
            "Fixed context",
            "Shuffled context",
            "HPC read = 0",
            "Wrong context",
            "Correct context",
        ]
        y = np.arange(len(short_names))
        bars = ax.barh(y, ablation_values, color=bar_colors, height=0.68)
        ax.set_title("(b) Mechanism interventions")
        ax.set_xlabel("Return-conflict accuracy")
        ax.set_xlim(0, 1.02)
        ax.set_yticks(y, short_names)
        ax.invert_yaxis()
        ax.grid(axis="x", color=LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, ablation_values):
            ax.text(
                min(value + 0.018, 0.91),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=5.6,
            )

        ax = axes[2]
        ax.plot(lengths_a, remap_a, color=CORAL, marker="o", linewidth=1.5, markersize=3, label="ReMAP, 8 seeds")
        ax.plot(lengths_a, no_cov_a, color=GRAY, marker="s", linewidth=1.2, markersize=3, label="No covariance")
        ax.plot(lengths_a, hippo_a, color=PURPLE, marker="^", linewidth=1.2, markersize=3, label="Hippo. adapt.")
        ax.plot(lengths_b, remap_b, color=CORAL, marker="D", linestyle="--", linewidth=1.1, markersize=3, label="ReMAP, 3 seeds")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.03, 1.02)
        ax.set_xlabel("Observed-history length")
        ax.set_ylabel("Accuracy")
        ax.set_title("(c) Delay scaling")
        ax.grid(color=LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=5.0, loc="center right")
        ax.annotate(
            "4,356",
            xy=(lengths_b[-1], remap_b[-1]),
            xytext=(1900, 0.89),
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.7},
            fontsize=5.8,
            color=MUTED,
        )
        save_figure(fig, "remap_former_2d_current_results_paper")


def draw_grid(ax: plt.Axes, color: str, label: str, hidden: bool = False) -> None:
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    for x in range(6):
        ax.plot([x, x], [0, 5], color="#D8DEE3", linewidth=0.8)
        ax.plot([0, 5], [x, x], color="#D8DEE3", linewidth=0.8)
    ax.plot([0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 1.5, 2.5], color=BLUE, linewidth=2)
    ax.scatter([0.5], [0.5], s=55, color=INK, zorder=3)
    ax.scatter([3.5], [2.5], s=180, color=color, edgecolor="white", linewidth=1.5, zorder=4)
    if hidden:
        ax.text(
            3.5,
            2.5,
            "?",
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=5,
        )
    ax.text(0.5, 0.16, "start", ha="center", va="top", fontsize=7, color=MUTED)
    ax.text(3.5, 2.86, label, ha="center", fontsize=7.2, color=color, fontweight="bold")
    ax.axis("off")


def make_task_overview(data: np.lib.npyio.NpzFile) -> None:
    fig = plt.figure(figsize=(14.2, 4.75))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=[0.18, 0.82],
        width_ratios=[1, 1, 1, 1.32, 1.32, 1.32],
        hspace=0.04,
        wspace=0.12,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.95,
        "Hidden-context re-entry: identical local state, different episodic target",
        color=INK,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    title_ax.text(
        0,
        0.36,
        "The agent encounters the same place in two latent contexts and must retrieve the item bound to the re-entered context.",
        color=MUTED,
        fontsize=9.2,
        va="top",
    )

    grid_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    draw_grid(grid_axes[0], "#3B83BD", "write item A")
    draw_grid(grid_axes[1], "#C7463A", "write item B")
    draw_grid(grid_axes[2], "#3B83BD", "recall item A", hidden=True)
    for ax, title, subtitle in zip(
        grid_axes,
        ["Context A", "Context B", "Return to A"],
        ["same place, first write", "same place, conflicting write", "target is not visible"],
    ):
        ax.set_title(title, color=INK, fontsize=10, fontweight="bold", pad=8)
        ax.text(
            0.5,
            -0.06,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=7.6,
        )
    for left, right in [(grid_axes[0], grid_axes[1]), (grid_axes[1], grid_axes[2])]:
        start = left.transAxes.transform((1.0, 0.52))
        end = right.transAxes.transform((0.0, 0.52))
        inv = fig.transFigure.inverted()
        fig.patches.append(
            FancyArrowPatch(
                inv.transform(start),
                inv.transform(end),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=11,
                color=MUTED,
                linewidth=1.2,
            )
        )

    episode = 0
    times = [30, 158, 285]
    titles = [
        "3D Context A write",
        "3D Context B write",
        "3D Return-to-A query",
    ]
    subtitles = [
        "blue visible at waypoint 2",
        "red visible at the same waypoint",
        "target hidden; correct = blue",
    ]
    for offset, (time_index, title, subtitle) in enumerate(
        zip(times, titles, subtitles), start=3
    ):
        ax = fig.add_subplot(grid[1, offset])
        ax.imshow(data["images"][episode, time_index])
        ax.set_title(title, color=INK, fontsize=9.6, fontweight="bold", pad=7)
        ax.text(
            0.5,
            -0.06,
            f"t={time_index}\n{subtitle}",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=7.2,
            linespacing=1.25,
        )
        ax.axis("off")
    fig.text(
        0.745,
        0.025,
        "Actual fresh-sealed MemoryMaze3D frames",
        color=MUTED,
        fontsize=8,
        ha="center",
        fontstyle="italic",
    )
    save_figure(fig, "remap_former_hidden_context_task_overview", vector=False)


def make_event_board(data: np.lib.npyio.NpzFile) -> None:
    episode = 0
    event_times = [
        [12, 30, 45, 55],
        [140, 158, 170, 178],
        [269, 285, 298, 307],
    ]
    row_titles = [
        "Context A writes",
        "Context B writes",
        "Return to A: hidden queries",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12.6, 8.2))
    fig.subplots_adjust(top=0.86, left=0.08, right=0.985, bottom=0.055, hspace=0.25, wspace=0.06)
    fig.suptitle(
        "Fresh-sealed 3D episode: four physical waypoints, two conflicting contexts",
        x=0.08,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.915,
        "Episode 0 from the frozen test set. Every query target is geometrically hidden.",
        color=MUTED,
        fontsize=9,
        ha="left",
    )
    for row_index, row in enumerate(event_times):
        for col_index, time_index in enumerate(row):
            ax = axes[row_index, col_index]
            ax.imshow(data["images"][episode, time_index])
            target = int(data["target_labels"][episode, time_index])
            context = int(data["context_ids"][episode, time_index])
            if row_index < 2:
                subtitle = f"t={time_index}  context={context}  item={target}"
            else:
                competing = int(data["competing_labels"][episode, time_index])
                subtitle = f"t={time_index}  target={target}  competing={competing}"
            ax.set_title(subtitle, fontsize=8.1, color=INK, pad=4)
            if row_index == 0:
                ax.text(
                    0.5,
                    1.12,
                    f"Waypoint {col_index + 1}",
                    transform=ax.transAxes,
                    ha="center",
                    color=INK,
                    fontsize=10,
                    fontweight="bold",
                )
            if col_index == 0:
                ax.text(
                    -0.2,
                    0.5,
                    row_titles[row_index],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    color=[BLUE, CORAL, GREEN][row_index],
                    fontsize=9.2,
                    fontweight="bold",
                )
            if row_index == 2:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(GREEN)
                    spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])
    save_figure(fig, "memorymaze3d_waypoint_event_board_english", vector=False)


def make_visual_recall_board(
    data: np.lib.npyio.NpzFile, sealed_summary: dict
) -> None:
    seed_result = next(
        item for item in sealed_summary["per_seed"] if int(item["seed"]) == 66102
    )
    rows = seed_result["visual_board"]["rows"]
    fig, axes = plt.subplots(3, 4, figsize=(12.6, 8.35))
    fig.subplots_adjust(top=0.85, left=0.055, right=0.99, bottom=0.06, hspace=0.27, wspace=0.055)
    fig.suptitle(
        "Visual source audit: ReMAP selects the correct same-waypoint episode",
        x=0.055,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.914,
        "The model predicts a 384D DINO feature. The selected source frame below is a diagnostic nearest-candidate match, not generated RGB.",
        color=MUTED,
        fontsize=8.8,
        ha="left",
    )
    column_titles = [
        "Hidden query RGB",
        "Correct write",
        "Model-selected write",
        "Wrong-context write",
    ]
    for row_index, item in enumerate(rows):
        episode = int(item["episode_index"])
        indices = [
            int(item["query_index"]),
            int(item["target_index"]),
            int(item["selected_index"]),
            int(item["competing_index"]),
        ]
        for col_index, time_index in enumerate(indices):
            ax = axes[row_index, col_index]
            ax.imshow(data["images"][episode, time_index])
            if row_index == 0:
                ax.text(
                    0.5,
                    1.12,
                    column_titles[col_index],
                    transform=ax.transAxes,
                    ha="center",
                    color=INK,
                    fontsize=9.2,
                    fontweight="bold",
                )
            if col_index == 0:
                ax.text(
                    0.02,
                    0.04,
                    f"ep {episode}, waypoint {int(item['place_id']) + 1}\nt={time_index}",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=7.8,
                    fontweight="bold",
                    bbox={"facecolor": INK, "alpha": 0.78, "edgecolor": "none", "pad": 3},
                )
            elif col_index == 1:
                ax.set_title(f"t={time_index}; target source", fontsize=7.8, color=GREEN)
            elif col_index == 2:
                margin = float(item["correct_cosine"]) - float(item["competing_cosine"])
                ax.set_title(
                    f"t={time_index}; selected, margin={margin:.3f}",
                    fontsize=7.8,
                    color=GREEN,
                )
            else:
                ax.set_title(
                    f"t={time_index}; competing source",
                    fontsize=7.8,
                    color=CORAL,
                )
            border_color = GREEN if col_index in {1, 2} else (CORAL if col_index == 3 else BLUE)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])
    save_figure(fig, "memorymaze3d_sealed_visual_recall_english", vector=False)


def make_trajectory_figure(data: np.lib.npyio.NpzFile) -> None:
    episode = 0
    positions = np.asarray(data["agent_pos"][episode])
    waypoints = np.asarray(data["waypoint_centers"][episode])
    phase_lengths = np.asarray(data["phase_action_counts"][episode]).astype(int)
    boundaries = np.concatenate([[0], np.cumsum(phase_lengths)])
    colors = [BLUE, CORAL, GREEN]
    labels = ["Context A", "Context B", "Return to A"]

    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    for phase in range(3):
        start = int(boundaries[phase])
        stop = int(boundaries[phase + 1])
        ax.plot(
            positions[start:stop, 0],
            positions[start:stop, 1],
            color=colors[phase],
            linewidth=2.0,
            label=f"{labels[phase]} ({stop - start} actions)",
        )
        ax.scatter(
            positions[start, 0],
            positions[start, 1],
            s=35,
            color=colors[phase],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=95,
        marker="D",
        color=GOLD,
        edgecolor=INK,
        linewidth=0.8,
        zorder=5,
        label="Physical waypoints",
    )
    for index, point in enumerate(waypoints):
        ax.text(
            point[0] + 0.08,
            point[1] + 0.08,
            f"W{index + 1}",
            color=INK,
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title(
        "One continuous 384-action trajectory in a fresh sealed layout",
        loc="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Simulator x")
    ax.set_ylabel("Simulator y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.text(
        0.13,
        0.015,
        "The simulator pose is used only for dataset generation and audit; it is never supplied to the model.",
        color=MUTED,
        fontsize=8,
    )
    save_figure(fig, "memorymaze3d_trajectory_english")


def make_3d_results_figure(sealed_summary: dict, full_tf_summary: dict) -> None:
    conditions = sealed_summary["conditions"]
    names = [
        "ReMAP",
        "Full-context TF",
        "HPC read = 0",
        "Fixed context",
        "Wrong history",
        "Correct history",
        "Context oracle",
    ]
    remap_values = [
        float(conditions["full"]["conflict_pairwise_acc"]["mean"]),
        float(full_tf_summary["aggregate"]["conflict_pairwise_acc"]["mean"]),
        float(conditions["hpc_zero"]["conflict_pairwise_acc"]["mean"]),
        float(conditions["fixed_context"]["conflict_pairwise_acc"]["mean"]),
        float(conditions["wrong_history"]["conflict_pairwise_acc"]["mean"]),
        float(conditions["correct_history"]["conflict_pairwise_acc"]["mean"]),
        float(conditions["orthogonal_context_oracle"]["conflict_pairwise_acc"]["mean"]),
    ]
    sds = [
        float(conditions["full"]["conflict_pairwise_acc"]["sample_std"]),
        float(full_tf_summary["aggregate"]["conflict_pairwise_acc"]["sample_sd"]),
        float(conditions["hpc_zero"]["conflict_pairwise_acc"]["sample_std"]),
        float(conditions["fixed_context"]["conflict_pairwise_acc"]["sample_std"]),
        float(conditions["wrong_history"]["conflict_pairwise_acc"]["sample_std"]),
        float(conditions["correct_history"]["conflict_pairwise_acc"]["sample_std"]),
        float(
            conditions["orthogonal_context_oracle"]["conflict_pairwise_acc"][
                "sample_std"
            ]
        ),
    ]
    colors = [CORAL, BLUE, GRAY, GOLD, PURPLE, GREEN, INK]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.65))
    fig.subplots_adjust(wspace=0.34, top=0.78, bottom=0.25)
    fig.suptitle(
        "Fresh-sealed 3D evaluation isolates context-dependent episodic retrieval",
        x=0.02,
        y=0.97,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    y = np.arange(len(names))
    bars = ax.barh(y, remap_values, xerr=sds, capsize=2.5, color=colors, height=0.7)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Return-context conflict accuracy")
    ax.set_title("(a) Main baseline and causal interventions")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, remap_values):
        if value > 0.9:
            label_x = min(value - 0.015, 0.99)
            horizontal_alignment = "right"
        else:
            label_x = value + 0.018
            horizontal_alignment = "left"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=7.8,
            color=INK,
        )

    ax = axes[1]
    models = ["ReMAP", "Full-context TF"]
    target_cos = [
        float(conditions["full"]["conflict_target_cosine"]["mean"]),
        float(full_tf_summary["aggregate"]["conflict_target_cosine"]["mean"]),
    ]
    margins = [
        float(conditions["full"]["conflict_cosine_margin"]["mean"]),
        float(full_tf_summary["aggregate"]["conflict_cosine_margin"]["mean"]),
    ]
    x = np.arange(2)
    width = 0.35
    bars_a = ax.bar(x - width / 2, target_cos, width, color=[CORAL, BLUE], alpha=0.92)
    bars_b = ax.bar(x + width / 2, margins, width, color=[CORAL, BLUE], alpha=0.48)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Cosine score")
    ax.set_title("(b) Feature similarity versus disambiguation")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        [bars_a[0], bars_b[0]],
        ["Target cosine", "Target-minus-competing margin"],
        frameon=False,
        fontsize=8,
        loc="center",
        bbox_to_anchor=(0.53, 0.58),
    )
    for bars_group in [bars_a, bars_b]:
        for bar in bars_group:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=INK,
            )
    fig.text(
        0.02,
        0.025,
        "Mean +/- sample SD across three frozen checkpoints. Test: 32 new layouts, 64 paired sequences, 192 delayed conflict queries. "
        "Both learned models receive the same binary write-event flag; neither receives context, pose, waypoint, route, or target IDs.",
        color=MUTED,
        fontsize=8,
        va="bottom",
    )
    save_figure(fig, "remap_former_3d_sealed_results")

    with plt.rc_context(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
        fig.subplots_adjust(wspace=0.42, top=0.86, bottom=0.23, left=0.16, right=0.99)
        ax = axes[0]
        y = np.arange(len(names))
        bars = ax.barh(y, remap_values, xerr=sds, capsize=2, color=colors, height=0.68)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.03)
        ax.set_xlabel("Return-context conflict accuracy")
        ax.set_title("(a) Baseline and interventions")
        ax.grid(axis="x", color=LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, remap_values):
            if value > 0.9:
                label_x = min(value - 0.015, 0.99)
                horizontal_alignment = "right"
            else:
                label_x = value + 0.04
                horizontal_alignment = "left"
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha=horizontal_alignment,
                fontsize=5.7,
            )

        ax = axes[1]
        x = np.arange(2)
        width = 0.35
        bars_a = ax.bar(x - width / 2, target_cos, width, color=[CORAL, BLUE], alpha=0.92)
        bars_b = ax.bar(x + width / 2, margins, width, color=[CORAL, BLUE], alpha=0.48)
        ax.set_xticks(x, ["ReMAP", "Full-context TF"])
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Cosine score")
        ax.set_title("(b) Similarity vs. disambiguation")
        ax.grid(axis="y", color=LIGHT, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(
            [bars_a[0], bars_b[0]],
            ["Target cosine", "Target - competing"],
            frameon=False,
            fontsize=5.8,
            loc="center",
            bbox_to_anchor=(0.53, 0.58),
        )
        for bars_group in [bars_a, bars_b]:
            for bar in bars_group:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                )
        save_figure(fig, "remap_former_3d_sealed_results_paper")


def main() -> None:
    verify_sources()
    configure_matplotlib()
    sealed_summary = load_json(REMAP_SEALED_PATH)
    full_tf_summary = load_json(FULL_TF_PATH)
    archive = np.load(DATA_PATH)
    data = {
        "images": archive["images"][:3],
        "agent_pos": archive["agent_pos"][:3],
        "waypoint_centers": archive["waypoint_centers"][:3],
        "phase_action_counts": archive["phase_action_counts"][:3],
        "target_labels": archive["target_labels"][:3],
        "competing_labels": archive["competing_labels"][:3],
        "context_ids": archive["context_ids"][:3],
    }
    archive.close()
    make_architecture_figure()
    make_architecture_paper_figure()
    make_model_family_comparison()
    make_2d_results_figure()
    make_task_overview(data)
    make_event_board(data)
    make_visual_recall_board(data, sealed_summary)
    make_trajectory_figure(data)
    make_3d_results_figure(sealed_summary, full_tf_summary)
    print("Generated current-paper figures in", FIGURE_DIR)


if __name__ == "__main__":
    main()
