"""Nature-style multi-target trajectory comparison with explicit graph context."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


TARGET_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
HISTORY_COLOR = "#B8B8B8"
EDGE_COLOR = "#8A8A8A"


def configure_style(font_dir: Path) -> None:
    font_files = sorted(font_dir.glob("times*.ttf"))
    if not font_files:
        raise FileNotFoundError(f"Times New Roman font files were not found in {font_dir}")
    for font_file in font_files:
        font_manager.fontManager.addfont(str(font_file))
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data(path: Path):
    data = np.load(path)
    arrays = {
        name: np.asarray(data[name], dtype=np.float64)
        for name in ("clean_history", "future", "independent", "proposed")
    }
    if arrays["clean_history"].ndim != 3 or arrays["clean_history"].shape[1:] != (4, 2):
        raise ValueError(f"Expected history [H, 4, 2], got {arrays['clean_history'].shape}")
    if arrays["future"].shape != arrays["independent"].shape:
        raise ValueError("Independent prediction shape does not match ground truth")
    if arrays["future"].shape != arrays["proposed"].shape:
        raise ValueError("Proposed prediction shape does not match ground truth")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite values were found")
    origin = arrays["clean_history"][-1].mean(axis=0)
    for name in arrays:
        arrays[name] = arrays[name] - origin
        # Display the road-aligned coordinates in the conventional top-view
        # orientation: lateral position on the horizontal axis and the main
        # longitudinal travel direction on the vertical axis.  This is an
        # axis permutation only; distances, errors, and physical geometry are
        # unchanged.
        arrays[name] = arrays[name][..., [1, 0]]
    return arrays, int(data["scene_index"]), int(data["snr_db"])


def graph_edges(current: np.ndarray, neighbours: int = 1):
    distance = np.linalg.norm(current[:, None, :] - current[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    edges = set()
    for source in range(current.shape[0]):
        for target in np.argsort(distance[source])[:neighbours]:
            edges.add(tuple(sorted((source, int(target)))))
    return sorted(edges)


def calculate_metrics(arrays: dict):
    rows = []
    for method, prediction in (
        ("Independent Transformer", arrays["independent"]),
        ("Proposed", arrays["proposed"]),
    ):
        error = np.linalg.norm(prediction - arrays["future"], axis=-1)
        rows.append(
            {
                "method": method,
                "ade_m": float(error.mean()),
                "fde_m": float(error[-1].mean()),
                "targets_with_lower_ade_than_independent": "" if method.startswith("Independent") else int(
                    np.sum(
                        np.linalg.norm(arrays["proposed"] - arrays["future"], axis=-1).mean(axis=0)
                        < np.linalg.norm(arrays["independent"] - arrays["future"], axis=-1).mean(axis=0)
                    )
                ),
            }
        )
    return rows


def common_limits(arrays: dict):
    points = np.concatenate(
        [arrays[name].reshape(-1, 2) for name in ("clean_history", "future", "independent", "proposed")],
        axis=0,
    )
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    padding = np.maximum(0.08 * span, np.array([0.7, 0.5]))
    return lower - padding, upper + padding


def plot_panel(ax, arrays: dict, prediction_key: str, panel_label: str, method_label: str):
    history = arrays["clean_history"]
    current = history[-1]
    truth = np.concatenate([current[None], arrays["future"]], axis=0)
    prediction = np.concatenate([current[None], arrays[prediction_key]], axis=0)

    if prediction_key == "proposed":
        for source, target in graph_edges(current):
            ax.plot(
                current[[source, target], 0],
                current[[source, target], 1],
                color=EDGE_COLOR,
                linewidth=0.75,
                linestyle=(0, (1.2, 1.6)),
                alpha=0.75,
                zorder=1,
            )

    for target, color in enumerate(TARGET_COLORS):
        ax.plot(
            history[:, target, 0],
            history[:, target, 1],
            color=HISTORY_COLOR,
            linewidth=0.8,
            zorder=1,
        )
        ax.plot(
            truth[:, target, 0],
            truth[:, target, 1],
            color=color,
            linewidth=1.15,
            zorder=3,
        )
        ax.plot(
            prediction[:, target, 0],
            prediction[:, target, 1],
            color=color,
            linewidth=1.35 if prediction_key == "independent" else 2.0,
            linestyle=(0, (2.2, 1.7)),
            marker="^" if prediction_key == "independent" else "o",
            markevery=5,
            markersize=3.3,
            markerfacecolor="white",
            markeredgewidth=0.75,
            zorder=4,
        )
        ax.scatter(
            current[target, 0],
            current[target, 1],
            s=24,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=5,
        )

    ax.text(
        0.50,
        1.035,
        method_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.3,
    )
    ax.text(
        0.035,
        0.975,
        panel_label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def plot_horizon_error(ax, arrays: dict):
    steps = np.arange(1, arrays["future"].shape[0] + 1)
    independent_error = np.linalg.norm(
        arrays["independent"] - arrays["future"], axis=-1
    ).mean(axis=1)
    proposed_error = np.linalg.norm(
        arrays["proposed"] - arrays["future"], axis=-1
    ).mean(axis=1)

    ax.plot(
        steps,
        independent_error,
        color="#666666",
        linewidth=1.35,
        linestyle=(0, (2.2, 1.7)),
        marker="^",
        markevery=4,
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Independent prediction",
        zorder=3,
    )
    ax.plot(
        steps,
        proposed_error,
        color="#0072B2",
        linewidth=2.0,
        marker="o",
        markevery=4,
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Proposed joint prediction",
        zorder=4,
    )
    ax.scatter(
        [steps[-1], steps[-1]],
        [independent_error[-1], proposed_error[-1]],
        s=18,
        facecolors="white",
        edgecolors=["#666666", "#0072B2"],
        linewidths=0.9,
        zorder=5,
    )
    ax.annotate(
        f"{independent_error[-1]:.2f} m",
        (steps[-1], independent_error[-1]),
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    ax.annotate(
        f"{proposed_error[-1]:.2f} m",
        (steps[-1], proposed_error[-1]),
        xytext=(-5, -6),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#0072B2",
    )
    ax.text(
        0.012,
        0.96,
        "c",
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.set_xlim(1, steps[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Mean displacement error (m)")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, linestyle=(0, (2, 2)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.legend(loc="upper left", bbox_to_anchor=(0.08, 0.98), frameon=False, ncol=2)


def create_figure(arrays: dict):
    fig = plt.figure(figsize=(4.4, 7.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.8, 1.25], hspace=0.40, wspace=0.42)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    error_ax = fig.add_subplot(grid[1, :])
    plot_panel(axes[0], arrays, "independent", "a", "Independent prediction")
    plot_panel(axes[1], arrays, "proposed", "b", "Proposed joint prediction")
    plot_horizon_error(error_ax, arrays)
    lower, upper = common_limits(arrays)
    for ax in axes:
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
    fig.text(0.5, 0.335, "Relative lateral position (m)", ha="center", va="center", fontsize=8.0)
    fig.text(
        0.025,
        0.61,
        "Relative longitudinal position (m)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.0,
    )

    handles = [
        Line2D([0], [0], color=HISTORY_COLOR, linewidth=0.8, label="History"),
        Line2D([0], [0], color="#333333", linewidth=1.15, label="Ground truth"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.7,
            linestyle=(0, (2.2, 1.7)),
            marker="o",
            markersize=3.3,
            markerfacecolor="white",
            label="Prediction",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor="#333333",
            linewidth=0,
            label="Prediction start",
        ),
        Line2D(
            [0],
            [0],
            color=EDGE_COLOR,
            linewidth=0.75,
            linestyle=(0, (1.2, 1.6)),
            label="Target-interaction edge",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.085, top=0.86)
    return fig


def save_metrics(path: Path, metrics: list, scene_index: int, snr: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "scene_index", "snr_db", "ade_m", "fde_m", "targets_with_lower_ade_than_independent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow({"scene_index": scene_index, "snr_db": snr, **row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/multitarget_snr/joint_multitarget_trajectory_scene.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/second_point_snr"))
    parser.add_argument("--font-dir", type=Path, default=Path("figure_tools/fonts"))
    parser.add_argument("--tool-dir", type=Path, default=Path("figure_tools"))
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    configure_style(args.font_dir)
    arrays, scene_index, snr = load_data(args.input)
    metrics = calculate_metrics(arrays)
    for row in metrics:
        print(row)
    fig = create_figure(arrays)

    sys.path.insert(0, str(args.tool_dir.resolve()))
    from visual_qa import audit_layout, print_report, render_preview

    preview_path = Path("/tmp/fig_multitarget_joint_trajectory_nature_preview.png")
    render_preview(fig, str(preview_path), dpi=180)
    from PIL import Image

    grayscale_path = Path("/tmp/fig_multitarget_joint_trajectory_nature_grayscale.png")
    with Image.open(preview_path) as image:
        image.convert("L").save(grayscale_path)
    issues = audit_layout(fig)
    print_report(issues)
    if any(severity == "FAIL" for severity, _ in issues):
        raise SystemExit("Layout audit failed")
    print(f"preview={preview_path}")
    print(f"grayscale_preview={grayscale_path}")

    if not args.preview_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"fig_snr{snr}_multitarget_joint_trajectory_nature"
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        save_metrics(stem.with_name(stem.name + "_metrics.csv"), metrics, scene_index, snr)
        print(f"exported={stem}")
    plt.close(fig)


if __name__ == "__main__":
    main()
