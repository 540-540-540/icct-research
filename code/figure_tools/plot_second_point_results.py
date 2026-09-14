"""Generate IEEE-style publication figures for the second research point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#D55E00"
OKABE_SKY = "#56B4E9"
GRAPH_BLUE = "#0072B2"
PROPOSED_ORANGE = "#D55E00"
CONVENTIONAL_GRAY = "#777777"
LIGHT_GRID = "#D9D9D9"


def configure_tools(tool_dir: Path) -> None:
    sys.path.insert(0, str(tool_dir.resolve()))


def audit_and_export(fig, basename: Path, size: tuple[float, float], panel_axes=None) -> dict:
    from export_figure import export_figure
    from layout_tools import add_panel_labels, finalize_figure
    from PIL import Image
    from visual_qa import audit_layout, print_report, render_preview

    finalize_figure(fig, prefer="constrained")
    if panel_axes:
        add_panel_labels(
            fig,
            axes=panel_axes,
            style="ieee",
            fontsize=9,
            x_offset_pt=-24,
            y_offset_pt=3,
        )
    preview_path = str(basename) + "_preview.png"
    render_preview(fig, preview_path, dpi=180)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if verdict == "FAIL":
        raise RuntimeError(f"Visual QA failed for {basename}")
    paths = export_figure(
        fig,
        str(basename),
        formats=("pdf", "svg", "png"),
        size_inches=size,
        dpi=600,
        grayscale_preview=False,
        tight=False,
    )
    gray_path = str(basename) + "_grayscale.png"
    Image.open(str(basename) + ".png").convert("L").save(gray_path, dpi=(600, 600))
    paths.append(gray_path)
    plt.close(fig)
    return {"basename": str(basename), "verdict": verdict, "issues": issues, "paths": paths}


def draw_error_panel(ax, df: pd.DataFrame, ade_col: str, fde_col: str, xlabel: str, show_labels: bool) -> None:
    y = np.arange(len(df))
    height = 0.32
    ax.axhspan(len(df) - 1.48, len(df) - 0.52, color="#F6E8D9", alpha=0.65, zorder=0)
    ade_bars = ax.barh(
        y - height / 2,
        df[ade_col],
        height=height,
        color=OKABE_BLUE,
        edgecolor="black",
        linewidth=0.45,
        hatch="///",
        label="ADE",
        zorder=3,
    )
    fde_bars = ax.barh(
        y + height / 2,
        df[fde_col],
        height=height,
        color=OKABE_ORANGE,
        edgecolor="black",
        linewidth=0.45,
        hatch="...",
        label="FDE",
        zorder=3,
    )
    for bars in (ade_bars, fde_bars):
        bars[-1].set_linewidth(1.15)
        for bar in bars:
            value = bar.get_width()
            ax.text(
                value + 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=6.5,
            )
    ax.set_xlim(0, 2.22)
    ax.set_xticks(np.arange(0, 2.21, 0.5))
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    if show_labels:
        labels = ax.set_yticklabels(df["short_model"])
        labels[-1].set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.5, linestyle="--", zorder=0)
    ax.grid(axis="y", visible=False)


def make_main_comparison(df: pd.DataFrame, out_dir: Path) -> dict:
    size = (7.16, 4.15)
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    draw_error_panel(axes[0], df, "ade_m", "fde_m", "Overall displacement error (m)", True)
    draw_error_panel(
        axes[1],
        df,
        "interaction_ade_m",
        "interaction_fde_m",
        "Interaction displacement error (m)",
        False,
    )
    legend = [
        Patch(facecolor=OKABE_BLUE, edgecolor="black", linewidth=0.45, hatch="///", label="ADE"),
        Patch(facecolor=OKABE_ORANGE, edgecolor="black", linewidth=0.45, hatch="...", label="FDE"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.34, 1.015),
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.5,
    )
    return audit_and_export(fig, out_dir / "fig_second_point_comparison", size, list(axes))


def make_accuracy_safety(df: pd.DataFrame, out_dir: Path) -> dict:
    size = (3.5, 3.0)
    learned = df[df["model_key"] != "constant_velocity"].copy()
    fig, ax = plt.subplots(figsize=size)
    styles = {
        "independent_gru": ("o", CONVENTIONAL_GRAY, 34),
        "lstm": ("v", CONVENTIONAL_GRAY, 38),
        "tcn": ("^", CONVENTIONAL_GRAY, 38),
        "transformer": ("D", CONVENTIONAL_GRAY, 34),
        "target_interaction_gnn": ("s", GRAPH_BLUE, 46),
        "graph_motion_token_gpt2": ("*", PROPOSED_ORANGE, 100),
    }
    label_offsets = {
        "independent_gru": (-33, -2),
        "lstm": (-13, 7),
        "tcn": (5, 6),
        "transformer": (-42, 8),
        "target_interaction_gnn": (7, -10),
        "graph_motion_token_gpt2": (7, 5),
    }
    label_names = {
        "independent_gru": "GRU",
        "lstm": "LSTM",
        "tcn": "TCN",
        "transformer": "Transformer",
        "target_interaction_gnn": "Target GNN",
        "graph_motion_token_gpt2": "Proposed",
    }
    for row in learned.itertuples(index=False):
        marker, color, size_pt = styles[row.model_key]
        ax.scatter(
            row.ade_m,
            row.excess_collision_rate_pct,
            marker=marker,
            s=size_pt,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=4,
        )
        dx, dy = label_offsets[row.model_key]
        ax.annotate(
            label_names[row.model_key],
            (row.ade_m, row.excess_collision_rate_pct),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )

    gnn = learned[learned.model_key == "target_interaction_gnn"].iloc[0]
    proposed = learned[learned.model_key == "graph_motion_token_gpt2"].iloc[0]
    ax.annotate(
        "",
        xy=(proposed.ade_m, proposed.excess_collision_rate_pct),
        xytext=(gnn.ade_m, gnn.excess_collision_rate_pct),
        arrowprops=dict(arrowstyle="->", color=PROPOSED_ORANGE, lw=1.1),
        zorder=3,
    )
    ax.set_xlim(0.57, 0.715)
    ax.set_ylim(0.15, 0.86)
    ax.set_xlabel("ADE (m)")
    ax.set_ylabel("Excess collision rate (%)")
    ax.grid(color=LIGHT_GRID, linewidth=0.5, linestyle="--", zorder=0)
    return audit_and_export(fig, out_dir / "fig_second_point_accuracy_safety", size)


def make_relative_gain(reductions: pd.DataFrame, out_dir: Path) -> dict:
    order = ["independent_gru", "lstm", "tcn", "transformer", "target_interaction_gnn"]
    names = {
        "independent_gru": "Independent GRU",
        "lstm": "LSTM",
        "tcn": "TCN",
        "transformer": "Transformer",
        "target_interaction_gnn": "Target GNN",
    }
    sub = reductions.set_index("model_key").loc[order].reset_index()
    y = np.arange(len(sub))
    size = (3.5, 2.85)
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(
        sub["ade_reduction_pct"],
        y - 0.12,
        color=OKABE_BLUE,
        edgecolor="black",
        linewidth=0.45,
        marker="o",
        s=34,
        label="ADE reduction",
        zorder=3,
    )
    ax.scatter(
        sub["fde_reduction_pct"],
        y + 0.12,
        color=OKABE_ORANGE,
        edgecolor="black",
        linewidth=0.45,
        marker="s",
        s=30,
        label="FDE reduction",
        zorder=3,
    )
    for i, row in sub.iterrows():
        ax.text(row.ade_reduction_pct + 0.45, i - 0.12, f"{row.ade_reduction_pct:.2f}", va="center", fontsize=6.5)
        ax.text(row.fde_reduction_pct + 0.45, i + 0.12, f"{row.fde_reduction_pct:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(y, [names[key] for key in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 21)
    ax.set_xlabel("Error reduction of proposed method (%)")
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.5, linestyle="--", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    return audit_and_export(fig, out_dir / "fig_second_point_relative_gain", size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="figures/second_point")
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
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    resolved_font = font_manager.findfont("Times New Roman", fallback_to_default=False)
    print("Resolved Times New Roman:", resolved_font)
    comparison = pd.read_csv(data_dir / "comparison_metrics.csv")
    comparison["short_model"] = [
        "Constant velocity",
        "Independent GRU",
        "LSTM",
        "TCN",
        "Transformer",
        "Target GNN",
        "Proposed Graph+LLM",
    ]
    reductions = pd.read_csv(data_dir / "relative_reductions.csv")

    reports = {
        "style": style_info,
        "resolved_font": resolved_font,
        "figures": [
            make_main_comparison(comparison, data_dir),
            make_accuracy_safety(comparison, data_dir),
            make_relative_gain(reductions, data_dir),
        ],
        "statistical_note": "Single-seed benchmark point estimates; no error bars or significance claims.",
    }
    (data_dir / "visual_qa_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
