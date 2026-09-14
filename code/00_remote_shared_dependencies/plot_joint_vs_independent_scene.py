"""Publication figure contrasting independent and joint multi-target prediction."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np


TARGET_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
GRID_COLOR = "#D8D8D8"
INDEPENDENT_COLOR = "#6F6F6F"
PROPOSED_COLOR = "#D55E00"


def configure_style(font_dir: Path) -> None:
    font_files = sorted(font_dir.glob("times*.ttf"))
    if not font_files:
        raise FileNotFoundError(f"Times New Roman font files were not found in {font_dir}")
    for font_file in font_files:
        font_manager.fontManager.addfont(str(font_file))
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
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
        key: np.asarray(data[key], dtype=np.float64)
        for key in (
            "clean_history",
            "future",
            "independent",
            "proposed",
            "ground_truth_pair_distance",
            "independent_pair_distance",
            "proposed_pair_distance",
        )
    }
    history = arrays["clean_history"]
    future = arrays["future"]
    if history.ndim != 3 or history.shape[1:] != (4, 2):
        raise ValueError(f"Expected history [H, 4, 2], got {history.shape}")
    if future.shape != arrays["independent"].shape or future.shape != arrays["proposed"].shape:
        raise ValueError("Future and prediction shapes are inconsistent")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite values were found")
    return arrays, int(data["snr_db"])


def trajectory_limits(arrays: dict):
    points = np.concatenate(
        [
            arrays["clean_history"].reshape(-1, 2),
            arrays["future"].reshape(-1, 2),
            arrays["independent"].reshape(-1, 2),
            arrays["proposed"].reshape(-1, 2),
        ],
        axis=0,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, np.asarray([1.0, 1.0]))
    padding = np.maximum(0.065 * span, np.asarray([1.2, 0.8]))
    return minimum - padding, maximum + padding


def interaction_zoom_spec(arrays: dict):
    current = arrays["clean_history"][-1]
    truth_paths = np.concatenate([current[None], arrays["future"]], axis=0)
    independent_paths = np.concatenate([current[None], arrays["independent"]], axis=0)
    proposed_paths = np.concatenate([current[None], arrays["proposed"]], axis=0)
    pair_distance = np.linalg.norm(truth_paths[:, 0] - truth_paths[:, 1], axis=-1)
    closest = int(np.argmin(pair_distance[1:])) + 1
    zoom_start = max(1, closest - 2)
    zoom_end = min(truth_paths.shape[0] - 1, closest + 2)
    zoom_points = np.concatenate(
        [
            truth_paths[zoom_start : zoom_end + 1, :2].reshape(-1, 2),
            independent_paths[zoom_start : zoom_end + 1, :2].reshape(-1, 2),
            proposed_paths[zoom_start : zoom_end + 1, :2].reshape(-1, 2),
        ],
        axis=0,
    )
    zoom_minimum = zoom_points.min(axis=0)
    zoom_maximum = zoom_points.max(axis=0)
    zoom_span = np.maximum(zoom_maximum - zoom_minimum, np.asarray([0.5, 0.5]))
    zoom_padding = np.maximum(0.16 * zoom_span, np.asarray([0.35, 0.30]))
    return {
        "closest": closest,
        "start": zoom_start,
        "end": zoom_end,
        "lower": zoom_minimum - zoom_padding,
        "upper": zoom_maximum + zoom_padding,
    }


def plot_trajectory_panel(
    ax,
    history: np.ndarray,
    future: np.ndarray,
    prediction: np.ndarray,
    panel_label: str,
    method_name: str,
    prediction_width: float,
    zoom_spec: dict,
) -> None:
    current = history[-1]
    truth_paths = np.concatenate([current[None], future], axis=0)
    prediction_paths = np.concatenate([current[None], prediction], axis=0)
    for target, color in enumerate(TARGET_COLORS):
        interacting = target < 2
        draw_color = color if interacting else "#A7A7A7"
        ax.plot(
            history[:, target, 0],
            history[:, target, 1],
            color=draw_color,
            linewidth=0.72 if interacting else 0.55,
            alpha=0.27 if interacting else 0.14,
            zorder=1,
        )
        ax.plot(
            truth_paths[:, target, 0],
            truth_paths[:, target, 1],
            color=draw_color,
            linewidth=1.0 if interacting else 0.72,
            alpha=1.0 if interacting else 0.36,
            zorder=3,
        )
        ax.plot(
            prediction_paths[:, target, 0],
            prediction_paths[:, target, 1],
            color=draw_color,
            linewidth=prediction_width if interacting else 0.9,
            linestyle=(0, (5.0, 2.2)),
            alpha=1.0 if interacting else 0.34,
            zorder=4,
        )
        ax.scatter(
            current[target, 0],
            current[target, 1],
            s=20 if interacting else 13,
            facecolor="white",
            edgecolor=draw_color,
            linewidth=0.85 if interacting else 0.65,
            alpha=1.0 if interacting else 0.48,
            zorder=5,
        )

    closest = zoom_spec["closest"]
    zoom_start = zoom_spec["start"]
    zoom_end = zoom_spec["end"]
    zoom_truth = truth_paths[zoom_start : zoom_end + 1, :2]
    zoom_prediction = prediction_paths[zoom_start : zoom_end + 1, :2]
    zoom_lower = zoom_spec["lower"]
    zoom_upper = zoom_spec["upper"]
    ax.add_patch(
        Rectangle(
            zoom_lower,
            zoom_upper[0] - zoom_lower[0],
            zoom_upper[1] - zoom_lower[1],
            fill=False,
            edgecolor="#777777",
            linewidth=0.65,
            linestyle=(0, (2.0, 1.6)),
            zorder=2,
        )
    )
    inset = ax.inset_axes([0.055, 0.055, 0.40, 0.34])
    for target in range(2):
        color = TARGET_COLORS[target]
        inset.plot(
            zoom_truth[:, target, 0],
            zoom_truth[:, target, 1],
            color=color,
            linewidth=0.95,
            zorder=3,
        )
        inset.plot(
            zoom_prediction[:, target, 0],
            zoom_prediction[:, target, 1],
            color=color,
            linewidth=prediction_width,
            linestyle=(0, (4.2, 1.9)),
            zorder=4,
        )
        closest_local = closest - zoom_start
        inset.scatter(
            zoom_truth[closest_local, target, 0],
            zoom_truth[closest_local, target, 1],
            s=10,
            marker="o",
            color=color,
            zorder=5,
        )
        inset.scatter(
            zoom_prediction[closest_local, target, 0],
            zoom_prediction[closest_local, target, 1],
            s=11,
            marker="s",
            facecolor="white",
            edgecolor=color,
            linewidth=0.65,
            zorder=5,
        )
    inset.set_xlim(zoom_lower[0], zoom_upper[0])
    inset.set_ylim(zoom_lower[1], zoom_upper[1])
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.65)

    ax.set_title(f"{panel_label}  {method_name}", pad=4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, color=GRID_COLOR, linewidth=0.42, linestyle="--", zorder=0)
    ax.tick_params(direction="out", length=3.0, width=0.7)


def create_figure(arrays: dict, dt: float = 0.1):
    fig = plt.figure(figsize=(7.16, 5.05))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.40, wspace=0.18)
    ax_independent = fig.add_subplot(grid[0, 0])
    ax_proposed = fig.add_subplot(grid[0, 1], sharex=ax_independent, sharey=ax_independent)
    ax_distance = fig.add_subplot(grid[1, :])
    zoom_spec = interaction_zoom_spec(arrays)

    plot_trajectory_panel(
        ax_independent,
        arrays["clean_history"],
        arrays["future"],
        arrays["independent"],
        "(a)",
        "Independent single-target prediction",
        1.50,
        zoom_spec,
    )
    plot_trajectory_panel(
        ax_proposed,
        arrays["clean_history"],
        arrays["future"],
        arrays["proposed"],
        "(b)",
        "Proposed joint prediction",
        2.05,
        zoom_spec,
    )
    lower, upper = trajectory_limits(arrays)
    for ax in (ax_independent, ax_proposed):
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_xlabel("Longitudinal position (m)")
    ax_independent.set_ylabel("Lateral position (m)")

    horizon = np.arange(arrays["ground_truth_pair_distance"].shape[0]) * dt
    ax_distance.plot(
        horizon,
        arrays["ground_truth_pair_distance"],
        color="black",
        linewidth=1.25,
        label="Ground truth",
        zorder=4,
    )
    ax_distance.plot(
        horizon,
        arrays["independent_pair_distance"],
        color=INDEPENDENT_COLOR,
        linewidth=1.35,
        linestyle=(0, (1.2, 1.8)),
        marker="^",
        markevery=4,
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.75,
        label="Independent Transformer",
        zorder=3,
    )
    ax_distance.plot(
        horizon,
        arrays["proposed_pair_distance"],
        color=PROPOSED_COLOR,
        linewidth=2.0,
        linestyle="-",
        marker="o",
        markevery=4,
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.85,
        label="Proposed",
        zorder=5,
    )
    closest = int(np.argmin(arrays["ground_truth_pair_distance"][1:])) + 1
    half_step = 0.5 * dt
    ax_distance.axvspan(
        max(horizon[0], horizon[closest] - half_step),
        min(horizon[-1], horizon[closest] + half_step),
        color="#ECECEC",
        zorder=0,
    )
    ax_distance.axvline(horizon[closest], color="#A8A8A8", linewidth=0.75, linestyle="--", zorder=1)
    maximum_distance = max(
        arrays["ground_truth_pair_distance"].max(),
        arrays["independent_pair_distance"].max(),
        arrays["proposed_pair_distance"].max(),
    )
    ax_distance.set_xlim(horizon[0], horizon[-1])
    ax_distance.set_ylim(0.0, maximum_distance * 1.12)
    ax_distance.set_xlabel("Prediction horizon (s)")
    ax_distance.set_ylabel("Inter-target distance (m)")
    ax_distance.set_title("(c)  Inter-target distance of the highlighted pair", pad=4.0)
    ax_distance.grid(True, color=GRID_COLOR, linewidth=0.42, linestyle="--", zorder=0)
    ax_distance.tick_params(direction="out", length=3.0, width=0.7)
    ax_distance.legend(loc="upper right", frameon=False, ncol=3, handlelength=2.6, columnspacing=1.1)

    trajectory_handles = [
        Line2D([0], [0], color="black", linewidth=0.72, alpha=0.27, label="History"),
        Line2D([0], [0], color="black", linewidth=1.0, label="Ground truth"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle=(0, (5.0, 2.2)), label="Prediction"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            markeredgewidth=0.85,
            linewidth=0,
            markersize=4.8,
            label="Prediction start",
        ),
    ]
    fig.legend(
        handles=trajectory_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.09, top=0.89)
    return fig


def calculate_metrics(arrays: dict):
    output = []
    for method, prediction, distance in (
        ("Independent Transformer", arrays["independent"], arrays["independent_pair_distance"]),
        ("Proposed", arrays["proposed"], arrays["proposed_pair_distance"]),
    ):
        errors = np.linalg.norm(prediction - arrays["future"], axis=-1)
        pair_error = np.mean(np.abs(distance[1:] - arrays["ground_truth_pair_distance"][1:]))
        output.append(
            {
                "method": method,
                "ade_m": float(errors.mean()),
                "fde_m": float(errors[-1].mean()),
                "pair_distance_error_m": float(pair_error),
            }
        )
    return output


def save_metrics(path: Path, metrics: list, snr: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "snr_db", "ade_m", "fde_m", "pair_distance_error_m"],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "method": row["method"],
                    "snr_db": snr,
                    "ade_m": f"{row['ade_m']:.6f}",
                    "fde_m": f"{row['fde_m']:.6f}",
                    "pair_distance_error_m": f"{row['pair_distance_error_m']:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/multitarget_snr/joint_vs_independent_scene.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/second_point_snr"))
    parser.add_argument("--font-dir", type=Path, default=Path("figure_tools/fonts"))
    parser.add_argument("--tool-dir", type=Path, default=Path("figure_tools"))
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    configure_style(args.font_dir)
    arrays, snr = load_data(args.input)
    metrics = calculate_metrics(arrays)
    for row in metrics:
        print(
            f"method={row['method']}, ADE={row['ade_m']:.4f}, FDE={row['fde_m']:.4f}, "
            f"pair_distance_error={row['pair_distance_error_m']:.4f}"
        )
    fig = create_figure(arrays)

    sys.path.insert(0, str(args.tool_dir.resolve()))
    from visual_qa import audit_layout, print_report, render_preview

    preview_path = Path("/tmp/fig_joint_vs_independent_preview.png")
    render_preview(fig, str(preview_path), dpi=150)
    from PIL import Image

    grayscale_path = Path("/tmp/fig_joint_vs_independent_grayscale.png")
    with Image.open(preview_path) as preview_image:
        preview_image.convert("L").save(grayscale_path)
    issues = audit_layout(fig)
    print_report(issues)
    if any(severity == "FAIL" for severity, _ in issues):
        raise SystemExit("Layout audit failed")
    print(f"preview={preview_path}")
    print(f"grayscale_preview={grayscale_path}")

    if not args.preview_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"fig_snr{snr}_joint_vs_independent_interaction"
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        save_metrics(stem.with_name(stem.name + "_metrics.csv"), metrics, snr)
        print(f"exported={stem}")
    plt.close(fig)


if __name__ == "__main__":
    main()
