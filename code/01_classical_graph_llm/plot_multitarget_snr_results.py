"""Generate publication-ready multi-SNR comparison, ablation, and trajectory figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
    "gray": "#6F6F6F",
    "black": "#111111",
}
GRID = "#D9D9D9"


def configure_tools(tool_dir: Path) -> None:
    sys.path.insert(0, str(tool_dir.resolve()))


def audit_and_export(
    fig, basename: Path, size: tuple[float, float], panel_axes=None, top_rect: float = 0.88
) -> dict:
    from export_figure import export_figure
    from PIL import Image
    from visual_qa import audit_layout, print_report, render_preview

    fig.set_layout_engine(None)
    fig.tight_layout(rect=(0.015, 0.015, 0.995, top_rect))
    if panel_axes:
        for label, ax in zip("abcdefghijklmnopqrstuvwxyz", panel_axes):
            ax.text(
                -0.13, 0.98, f"({label})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold", clip_on=False,
            )
    preview = str(basename) + "_preview.png"
    render_preview(fig, preview, dpi=180)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if verdict == "FAIL":
        raise RuntimeError(f"Visual QA failed for {basename}")
    paths = export_figure(
        fig, str(basename), formats=("pdf", "svg", "png"),
        size_inches=size, dpi=600, grayscale_preview=False, tight=False,
    )
    gray = str(basename) + "_grayscale.png"
    Image.open(str(basename) + ".png").convert("L").save(gray, dpi=(600, 600))
    paths.append(gray)
    plt.close(fig)
    return {"basename": str(basename), "verdict": verdict, "issues": issues, "paths": paths}


def line_panel(ax, df: pd.DataFrame, key_col: str, metric: str, styles: dict, order: list[str]) -> None:
    for key in order:
        sub = df[df[key_col] == key].sort_values("snr_db")
        style = styles[key]
        ax.plot(
            sub.snr_db, sub[metric], label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"], linewidth=style["linewidth"],
            markersize=style.get("markersize", 4.2), markerfacecolor="white",
            markeredgewidth=0.9, zorder=style.get("zorder", 3),
        )
    ax.set_xticks([5, 10, 15, 20])
    ax.set_xlabel("SNR (dB)")
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", zorder=0)


def comparison_figure(df: pd.DataFrame, out_dir: Path) -> dict:
    order = ["lstm", "tcn", "transformer", "target_interaction_gnn", "graph_motion_token_gpt2"]
    styles = {
        "lstm": {"label": "LSTM", "color": PALETTE["gray"], "linestyle": ":", "marker": "v", "linewidth": 1.05},
        "tcn": {"label": "TCN", "color": PALETTE["green"], "linestyle": "--", "marker": "^", "linewidth": 1.05},
        "transformer": {"label": "Transformer", "color": PALETTE["purple"], "linestyle": "-.", "marker": "D", "linewidth": 1.05},
        "target_interaction_gnn": {"label": "Target GNN", "color": PALETTE["blue"], "linestyle": "--", "marker": "s", "linewidth": 1.35},
        "graph_motion_token_gpt2": {"label": "Proposed", "color": PALETTE["orange"], "linestyle": "-", "marker": "o", "linewidth": 1.8, "zorder": 5},
    }
    size = (7.16, 3.45)
    fig, axes = plt.subplots(1, 2, figsize=size)
    line_panel(axes[0], df, "model_key", "ade_m", styles, order)
    line_panel(axes[1], df, "model_key", "fde_m", styles, order)
    axes[0].set_ylabel("ADE (m)")
    axes[1].set_ylabel("FDE (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False,
               handlelength=2.4, columnspacing=1.15)
    return audit_and_export(fig, out_dir / "fig_snr_model_comparison", size, list(axes), top_rect=0.86)


def ablation_figure(df: pd.DataFrame, out_dir: Path) -> dict:
    order = ["full_model", "without_uncertainty", "without_soft_token", "without_graph_context", "without_token_loss", "without_lora"]
    styles = {
        "full_model": {"label": "Full model", "color": PALETTE["orange"], "linestyle": "-", "marker": "o", "linewidth": 1.8, "zorder": 6},
        "without_uncertainty": {"label": "w/o uncertainty", "color": PALETTE["sky"], "linestyle": "--", "marker": "v", "linewidth": 1.0},
        "without_soft_token": {"label": "w/o soft token", "color": PALETTE["green"], "linestyle": "-.", "marker": "^", "linewidth": 1.0},
        "without_graph_context": {"label": "w/o graph context", "color": PALETTE["black"], "linestyle": ":", "marker": "s", "linewidth": 1.2},
        "without_token_loss": {"label": "w/o token loss", "color": PALETTE["purple"], "linestyle": (0, (5, 2)), "marker": "D", "linewidth": 1.0},
        "without_lora": {"label": "w/o LoRA", "color": PALETTE["yellow"], "linestyle": (0, (3, 1, 1, 1)), "marker": "P", "linewidth": 1.0},
    }
    size = (7.16, 3.75)
    fig, axes = plt.subplots(1, 2, figsize=size)
    line_panel(axes[0], df, "variant_key", "ade_m", styles, order)
    line_panel(axes[1], df, "variant_key", "fde_m", styles, order)
    axes[0].set_ylabel("ADE (m)")
    axes[1].set_ylabel("FDE (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
               handlelength=2.5, columnspacing=1.4)
    return audit_and_export(fig, out_dir / "fig_snr_ablation", size, list(axes), top_rect=0.79)


def trajectory_figure(npz_path: Path, out_dir: Path) -> dict:
    data = np.load(npz_path)
    size = (7.16, 4.75)
    fig, axes = plt.subplots(2, 4, figsize=size, sharex="col", sharey="col")
    clean_history = data["clean_history"]
    future = data["future"]
    marker_positions = [4, 9, 14, 19]
    method_styles = {
        "lstm": {
            "label": "LSTM", "color": PALETTE["gray"], "linestyle": ":",
            "marker": "v", "linewidth": 0.9, "alpha": 0.88, "zorder": 2,
        },
        "transformer": {
            "label": "Transformer", "color": PALETTE["orange"],
            "linestyle": (0, (5, 2, 1, 2)), "marker": "D",
            "linewidth": 0.95, "alpha": 0.9, "zorder": 2,
        },
        "target_interaction_gnn": {
            "label": "Target GNN", "color": PALETTE["green"], "linestyle": "--",
            "marker": "s", "linewidth": 1.05, "alpha": 0.94, "zorder": 3,
        },
        "proposed": {
            "label": "Proposed", "color": PALETTE["blue"], "linestyle": "-.",
            "marker": "P", "linewidth": 1.9, "alpha": 1.0, "zorder": 4,
        },
    }
    for row, snr in enumerate((5, 20)):
        lstm = data[f"lstm_snr{snr}"]
        transformer = data[f"transformer_snr{snr}"]
        gnn = data[f"target_interaction_gnn_snr{snr}"]
        proposed = data[f"proposed_snr{snr}"]
        predictions = {
            "lstm": lstm,
            "transformer": transformer,
            "target_interaction_gnn": gnn,
            "proposed": proposed,
        }
        for col in range(4):
            ax = axes[row, col]
            origin = clean_history[-1, col, :2]
            history_local = clean_history[-6:, col, :2] - origin
            future_local = future[:, col, :2] - origin
            ax.plot(
                history_local[:, 0], history_local[:, 1], color="#B0B0B0",
                linestyle="-", linewidth=0.75, alpha=0.75, zorder=1,
            )
            for key, prediction in predictions.items():
                style = method_styles[key]
                local = prediction[:, col, :2] - origin
                ax.plot(
                    local[:, 0], local[:, 1], color=style["color"],
                    linestyle=style["linestyle"], linewidth=style["linewidth"],
                    alpha=style["alpha"], marker=style["marker"],
                    markevery=marker_positions, markersize=3.1 if key != "proposed" else 4.0,
                    markerfacecolor="white", markeredgewidth=0.7,
                    zorder=style["zorder"],
                )
            ax.plot(
                future_local[:, 0], future_local[:, 1], color=PALETTE["black"],
                linestyle="-", linewidth=1.0, zorder=5,
            )
            ax.scatter(
                0.0, 0.0, facecolor="white", edgecolor=PALETTE["black"],
                marker="o", s=15, linewidth=0.75, zorder=6,
            )
            ax.margins(0.08)
            ax.grid(color=GRID, linewidth=0.4, linestyle="--", zorder=0)
            if col == 0:
                ax.set_ylabel(f"SNR = {snr} dB\nLateral displacement (m)")
    fig.supxlabel("Longitudinal displacement (m)", fontsize=8, y=0.015)
    method_handles = [
        Line2D([0], [0], color="#B0B0B0", linestyle="-", linewidth=0.75, label="Clean history"),
        Line2D([0], [0], color=PALETTE["black"], linestyle="-", linewidth=1.0, label="Ground truth"),
        Line2D([0], [0], color=PALETTE["gray"], linestyle=":", linewidth=0.9,
               marker="v", markerfacecolor="white", markersize=3.8, label="LSTM"),
        Line2D([0], [0], color=PALETTE["orange"], linestyle=(0, (5, 2, 1, 2)), linewidth=0.95,
               marker="D", markerfacecolor="white", markersize=3.4, label="Transformer"),
        Line2D([0], [0], color=PALETTE["green"], linestyle="--", linewidth=1.05,
               marker="s", markerfacecolor="white", markersize=3.8, label="Target GNN"),
        Line2D([0], [0], color=PALETTE["blue"], linestyle="-.", linewidth=1.9,
               marker="P", markerfacecolor="white", markersize=4.8, label="Proposed"),
    ]
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6,
               frameon=False, handlelength=2.4, columnspacing=0.95, handletextpad=0.45)
    return audit_and_export(
        fig, out_dir / "fig_snr_trajectory_prediction_comparison", size,
        panel_axes=None, top_rect=0.91,
    )


def trajectory_scene_overview(npz_path: Path, out_dir: Path) -> dict:
    data = np.load(npz_path)
    target_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
    size = (7.16, 3.45)
    fig, axes = plt.subplots(1, 2, figsize=size, sharex=True, sharey=True)
    clean_history = data["clean_history"]
    future = data["future"]
    for ax, snr in zip(axes, (5, 20)):
        proposed = data[f"proposed_snr{snr}"]
        for col, color in enumerate(target_colors):
            ax.plot(
                clean_history[:, col, 0], clean_history[:, col, 1], color=color,
                linestyle="-", linewidth=0.75, alpha=0.42, zorder=1,
            )
            ax.plot(
                proposed[:, col, 0], proposed[:, col, 1], color=color,
                linestyle="-.", linewidth=2.0, zorder=3,
            )
            ax.plot(
                future[:, col, 0], future[:, col, 1], color=color,
                linestyle="-", linewidth=1.0, zorder=4,
            )
            ax.scatter(
                clean_history[-1, col, 0], clean_history[-1, col, 1],
                facecolor="white", edgecolor=color, marker="o", s=17,
                linewidth=0.75, zorder=5,
            )
        ax.text(
            0.04, 0.95, f"SNR = {snr} dB", transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#A0A0A0", linewidth=0.4, pad=2.0, alpha=0.9),
        )
        ax.set_xlabel("Longitudinal position (m)")
        ax.grid(color=GRID, linewidth=0.45, linestyle="--", zorder=0)
    axes[0].set_ylabel("Lateral position (m)")
    handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=0.75, alpha=0.42, label="Clean history"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, label="Ground truth"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=2.0, label="Proposed"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
               frameon=False, handlelength=2.5, columnspacing=1.4)
    return audit_and_export(
        fig, out_dir / "fig_snr_trajectory_scene_overview", size,
        panel_axes=list(axes), top_rect=0.86,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="figures/second_point_snr")
    parser.add_argument("--result-dir", default="results/multitarget_snr")
    parser.add_argument("--tool-dir", default="figure_tools")
    parser.add_argument("--font-dir", default="figure_tools/fonts")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    configure_tools(Path(args.tool_dir))

    font_files = sorted(Path(args.font_dir).glob("times*.ttf"))
    if not font_files:
        raise FileNotFoundError("Times New Roman font files were not found in " + args.font_dir)
    for font_file in font_files:
        font_manager.fontManager.addfont(str(font_file))
    from setup_style import setup_style
    style_info = setup_style(journal="ieee", lang="en", use_sciplots=False)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.65,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    resolved_font = font_manager.findfont("Times New Roman", fallback_to_default=False)
    print("Resolved Times New Roman:", resolved_font)
    comparison = pd.read_csv(data_dir / "comparison_snr_metrics.csv")
    ablation = pd.read_csv(data_dir / "ablation_snr_metrics.csv")
    reports = {
        "style": style_info,
        "resolved_font": resolved_font,
        "figures": [
            comparison_figure(comparison, data_dir),
            ablation_figure(ablation, data_dir),
            trajectory_figure(Path(args.result_dir) / "trajectory_examples_comparison.npz", data_dir),
            trajectory_scene_overview(Path(args.result_dir) / "trajectory_examples_comparison.npz", data_dir),
        ],
        "statistical_note": "Single-seed point estimates; no error bars or significance claims.",
    }
    (data_dir / "visual_qa_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
