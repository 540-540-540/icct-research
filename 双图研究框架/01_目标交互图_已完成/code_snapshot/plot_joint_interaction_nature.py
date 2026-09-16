"""Nature-style evidence figure for joint multi-target prediction."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


BLACK = "#111111"
GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
ORANGE = "#D55E00"


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


def load_scene(path: Path):
    data = np.load(path)
    names = (
        "ground_truth_pair_distance",
        "independent_pair_distance",
        "proposed_pair_distance",
    )
    arrays = {name: np.asarray(data[name], dtype=np.float64) for name in names}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite values were found in the selected scene")
    return arrays, int(data["scene_index"]), int(data["snr_db"])


def load_population(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "scene_index",
        "independent_pair_distance_error_m",
        "proposed_pair_distance_error_m",
        "pair_distance_error_gain_pct",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values were found in the population metrics")
    if (frame[["independent_pair_distance_error_m", "proposed_pair_distance_error_m"]] <= 0).any().any():
        raise ValueError("Pair-distance errors must be positive for logarithmic comparison")
    return frame


def clean_log_label(value: float, _position: float) -> str:
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.1g}"


def create_figure(arrays: dict, population: pd.DataFrame, scene_index: int, dt: float = 0.1):
    fig, (ax_distance, ax_population) = plt.subplots(
        1,
        2,
        figsize=(7.18, 3.05),
        gridspec_kw={"width_ratios": [1.0, 1.02], "wspace": 0.34},
    )

    horizon = np.arange(arrays["ground_truth_pair_distance"].size) * dt
    ax_distance.plot(
        horizon,
        arrays["ground_truth_pair_distance"],
        color=BLACK,
        linewidth=1.25,
        label="Ground truth",
        zorder=4,
    )
    ax_distance.plot(
        horizon,
        arrays["independent_pair_distance"],
        color=GREY,
        linewidth=1.25,
        linestyle=(0, (2.0, 1.8)),
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
        color=ORANGE,
        linewidth=2.0,
        marker="o",
        markevery=4,
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=0.85,
        label="Proposed",
        zorder=5,
    )
    ax_distance.set_xlim(horizon[0], horizon[-1])
    ymax = max(value.max() for value in arrays.values())
    ax_distance.set_ylim(0.0, ymax * 1.08)
    ax_distance.set_xlabel("Prediction horizon (s)")
    ax_distance.set_ylabel("Inter-target distance (m)")
    ax_distance.grid(axis="y", color="#E5E5E5", linewidth=0.45, zorder=0)

    independent = population["independent_pair_distance_error_m"].to_numpy()
    proposed = population["proposed_pair_distance_error_m"].to_numpy()
    lower = min(independent.min(), proposed.min()) * 0.72
    upper = max(independent.max(), proposed.max()) * 1.25
    identity = np.geomspace(lower, upper, 256)
    ax_population.fill_between(
        identity,
        lower,
        identity,
        color=ORANGE,
        alpha=0.045,
        linewidth=0,
        zorder=0,
    )
    ax_population.plot(identity, identity, color=BLACK, linewidth=0.8, zorder=1)
    ax_population.scatter(
        independent,
        proposed,
        s=10,
        color=LIGHT_GREY,
        alpha=0.52,
        edgecolors="none",
        zorder=2,
    )
    selected = population.loc[population["scene_index"].astype(int) == scene_index]
    if selected.empty:
        raise ValueError(f"Selected scene {scene_index} was not found in the population table")
    ax_population.scatter(
        selected["independent_pair_distance_error_m"],
        selected["proposed_pair_distance_error_m"],
        s=38,
        facecolor=ORANGE,
        edgecolor=BLACK,
        linewidth=0.75,
        zorder=4,
    )
    ax_population.set_xscale("log")
    ax_population.set_yscale("log")
    ax_population.set_xlim(lower, upper)
    ax_population.set_ylim(lower, upper)
    ax_population.set_aspect("equal", adjustable="box")
    formatter = FuncFormatter(clean_log_label)
    ax_population.xaxis.set_major_formatter(formatter)
    ax_population.yaxis.set_major_formatter(formatter)
    ax_population.set_xlabel("Independent pair-distance error (m)")
    ax_population.set_ylabel("Proposed pair-distance error (m)")
    improved_pct = 100.0 * float(np.mean(proposed < independent))
    median_gain = float(np.median(population["pair_distance_error_gain_pct"]))
    ax_population.text(
        0.045,
        0.955,
        f"n = {len(population)}\nProposed lower: {improved_pct:.1f}%\nMedian reduction: {median_gain:.1f}%",
        transform=ax_population.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        linespacing=1.25,
    )
    ax_population.text(
        0.965,
        0.055,
        "Proposed better",
        transform=ax_population.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=ORANGE,
    )

    for label, ax in zip(("a", "b"), (ax_distance, ax_population)):
        ax.text(
            -0.18,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    legend_handles = [
        Line2D([0], [0], color=BLACK, linewidth=1.25, label="Ground truth"),
        Line2D(
            [0],
            [0],
            color=GREY,
            linewidth=1.25,
            linestyle=(0, (2.0, 1.8)),
            marker="^",
            markersize=3.5,
            markerfacecolor="white",
            label="Independent Transformer",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            linewidth=2.0,
            marker="o",
            markersize=3.4,
            markerfacecolor="white",
            label="Proposed",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=4.8,
            markerfacecolor=ORANGE,
            markeredgecolor=BLACK,
            markeredgewidth=0.75,
            linewidth=0,
            label="Example in a",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.84)
    summary = {
        "strong_interaction_scene_count": len(population),
        "proposed_lower_pair_error_pct": improved_pct,
        "median_pair_error_reduction_pct": median_gain,
        "independent_pair_error_median_m": float(np.median(independent)),
        "proposed_pair_error_median_m": float(np.median(proposed)),
        "selected_scene_index": scene_index,
    }
    return fig, summary


def save_summary(path: Path, summary: dict, snr: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "snr_db"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value, "snr_db": snr})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("results/multitarget_snr/joint_vs_independent_scene.npz"),
    )
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("results/multitarget_snr/joint_vs_independent_candidates.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/second_point_snr"))
    parser.add_argument("--font-dir", type=Path, default=Path("figure_tools/fonts"))
    parser.add_argument("--tool-dir", type=Path, default=Path("figure_tools"))
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    configure_style(args.font_dir)
    arrays, scene_index, snr = load_scene(args.scene)
    population = load_population(args.population)
    fig, summary = create_figure(arrays, population, scene_index)
    print(population[["independent_pair_distance_error_m", "proposed_pair_distance_error_m"]].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).to_string())
    for key, value in summary.items():
        print(f"{key}={value}")

    sys.path.insert(0, str(args.tool_dir.resolve()))
    from visual_qa import audit_layout, print_report, render_preview

    preview_path = Path("/tmp/fig_joint_interaction_nature_preview.png")
    render_preview(fig, str(preview_path), dpi=180)
    from PIL import Image

    grayscale_path = Path("/tmp/fig_joint_interaction_nature_grayscale.png")
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
        stem = args.output_dir / f"fig_snr{snr}_joint_interaction_nature"
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        save_summary(stem.with_name(stem.name + "_metrics.csv"), summary, snr)
        print(f"exported={stem}")
    plt.close(fig)


if __name__ == "__main__":
    main()
