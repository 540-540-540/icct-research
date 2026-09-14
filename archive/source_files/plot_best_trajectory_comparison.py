"""Plot four diverse full-test cases where Proposed wins both ADE and FDE."""

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


METHODS = ("lstm", "transformer", "target_interaction_gnn", "proposed")
PLOTTED_CASES = (0, 2)
METHOD_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "target_interaction_gnn": "Target GNN",
    "proposed": "Proposed",
}
STYLES = {
    "lstm": dict(color="#7A7A7A", linestyle=":", linewidth=0.9, marker="v", zorder=2),
    "transformer": dict(color="#D55E00", linestyle=(0, (5, 2, 1, 2)), linewidth=0.95, marker="D", zorder=2),
    "target_interaction_gnn": dict(color="#009E73", linestyle="--", linewidth=1.05, marker="s", zorder=3),
    "proposed": dict(color="#0072B2", linestyle=(0, (5, 2)), linewidth=2.15, marker="P", zorder=6),
}
GRID_COLOR = "#D8D8D8"


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
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def compute_errors(data):
    future = data["future"]
    out = {}
    for method in METHODS:
        distances = np.linalg.norm(data[method] - future, axis=-1)
        out[method] = {"ade": distances.mean(axis=1), "fde": distances[:, -1]}
    return out


def verify_cases(data):
    errors = compute_errors(data)
    case_count = data["future"].shape[0]
    if case_count != 4:
        raise RuntimeError(f"Expected exactly four selected cases, found {case_count}")
    for case in range(case_count):
        proposed_ade = errors["proposed"]["ade"][case]
        proposed_fde = errors["proposed"]["fde"][case]
        if proposed_ade > min(errors[m]["ade"][case] for m in METHODS) + 1.0e-8:
            raise RuntimeError(f"Case {case} does not have the lowest Proposed ADE")
        if proposed_fde > min(errors[m]["fde"][case] for m in METHODS) + 1.0e-8:
            raise RuntimeError(f"Case {case} does not have the lowest Proposed FDE")
    return errors


def plot_case(ax, data, case: int) -> None:
    history_local = data["clean_history"][case, -8:, :2]
    truth_local = np.vstack([np.zeros((1, 2)), data["future"][case, :, :2]])

    ax.plot(
        history_local[:, 0], history_local[:, 1],
        color="#B0B0B0", linestyle="-", linewidth=0.8, alpha=0.75, zorder=1,
    )
    for method in METHODS:
        prediction_local = np.vstack(
            [np.zeros((1, 2)), data[method][case, :, :2]]
        )
        style = STYLES[method]
        ax.plot(
            prediction_local[:, 0], prediction_local[:, 1],
            color=style["color"], linestyle=style["linestyle"],
            linewidth=style["linewidth"], marker=style["marker"],
            markevery=[5, 10, 15, 20], markersize=3.4 if method != "proposed" else 4.5,
            markerfacecolor="white", markeredgewidth=0.75,
            zorder=style["zorder"],
        )
    ax.plot(
        truth_local[:, 0], truth_local[:, 1],
        color="black", linestyle="-", linewidth=1.05, zorder=5,
    )
    ax.scatter(
        0.0, 0.0, facecolor="white", edgecolor="black",
        marker="o", s=20, linewidth=0.8, zorder=7,
    )

    all_local = [history_local, truth_local]
    for method in METHODS:
        all_local.append(
            np.vstack([np.zeros((1, 2)), data[method][case, :, :2]])
        )
    points = np.concatenate(all_local, axis=0)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    half_span = 0.60 * span
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", zorder=0)
    ax.tick_params(direction="out", length=3.0, width=0.7)


def create_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55))
    for case, ax in zip(PLOTTED_CASES, axes):
        plot_case(ax, data, case)

    axes[0].set_ylabel("Y displacement (m)")
    fig.supxlabel("X displacement (m)", fontsize=8.5, y=0.025)

    handles = [
        Line2D([0], [0], color="#B0B0B0", linewidth=0.8, label="History"),
        Line2D([0], [0], color="black", linewidth=1.05, label="Ground truth"),
    ]
    for method in METHODS:
        style = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], marker=style["marker"],
                markerfacecolor="white", markeredgewidth=0.75,
                markersize=3.8 if method != "proposed" else 4.8,
                label=METHOD_LABELS[method],
            )
        )
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=6, frameon=False, handlelength=2.5,
        columnspacing=0.85, handletextpad=0.42,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.80, wspace=0.22)
    return fig


def save_selection(path: Path, data, error_tables) -> None:
    fields = [
        "panel", "snr_db", "scene_index", "target_index", "target_id",
        "motion_type", "method", "ade_m", "fde_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for panel, case in enumerate(PLOTTED_CASES, start=1):
            for method in METHODS:
                writer.writerow(
                    {
                        "panel": panel,
                        "snr_db": int(data["snr_db"]),
                        "scene_index": int(data["scene_indices"][case]),
                        "target_index": int(data["target_indices"][case]),
                        "target_id": int(data["target_ids"][case]),
                        "motion_type": str(data["motion_types"][case]),
                        "method": method,
                        "ade_m": f"{error_tables[method]['ade'][case]:.6f}",
                        "fde_m": f"{error_tables[method]['fde'][case]:.6f}",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/multitarget_snr/diverse_best_trajectory_cases.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/second_point_snr"))
    parser.add_argument("--font-dir", type=Path, default=Path("figure_tools/fonts"))
    parser.add_argument("--tool-dir", type=Path, default=Path("figure_tools"))
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    configure_style(args.font_dir)
    data = np.load(args.input)
    error_tables = verify_cases(data)
    for panel, case in enumerate(PLOTTED_CASES, start=1):
        proposed_ade = error_tables["proposed"]["ade"][case]
        proposed_fde = error_tables["proposed"]["fde"][case]
        print(
            f"panel={panel}, source_case={case + 1}, snr={int(data['snr_db'])}, "
            f"scene_index={int(data['scene_indices'][case])}, "
            f"motion_type={str(data['motion_types'][case])}, "
            f"proposed_ADE={proposed_ade:.6f}, proposed_FDE={proposed_fde:.6f}, "
            "criterion=lowest_ADE_and_FDE"
        )

    fig = create_figure(data)
    sys.path.insert(0, str(args.tool_dir.resolve()))
    from visual_qa import audit_layout, print_report, render_preview
    from PIL import Image

    preview_path = Path("/tmp/fig_best_trajectory_comparison_preview.png")
    grayscale_path = Path("/tmp/fig_best_trajectory_comparison_grayscale.png")
    render_preview(fig, str(preview_path), dpi=150)
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
        stem = args.output_dir / "fig_snr_trajectory_prediction_comparison"
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
        save_selection(
            args.output_dir / "trajectory_best_case_selection_metrics.csv",
            data, error_tables,
        )
        print(f"exported={stem}")
    plt.close(fig)


if __name__ == "__main__":
    main()
