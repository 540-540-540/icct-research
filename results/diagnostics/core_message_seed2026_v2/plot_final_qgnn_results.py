"""Create the IEEE-style final performance and complexity figure."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
RESULT_DIR = BASE / "final_qgnn_paper_evaluation_seed2026_v1"
TOOLS = BASE / "figure_tools"
sys.path.insert(0, str(TOOLS))

from export_figure import export_figure
from layout_tools import add_panel_labels, finalize_figure
from setup_style import setup_style
from visual_qa import audit_layout, print_report, render_preview

FIGURE_DIR = RESULT_DIR / "figures"
CSV_PATH = FIGURE_DIR / "final_qgnn_plot_data.csv"


def prepare_rows():
    final = json.loads((RESULT_DIR / "final_results.json").read_text(encoding="utf-8"))
    strata = json.loads((RESULT_DIR / "complexity_paired_bootstrap.json").read_text(encoding="utf-8"))["strata"]
    summaries = final["summaries"]
    plain = summaries["plain_gnn"]["metrics"]["aggregate"]
    labels = {
        "physics_classical_dual": "Classical dual GNN",
        "physics_quantum_dual": "Dual QGNN",
        "physics_classical_dual_llm": "Classical dual GNN + LLM",
        "physics_quantum_dual_llm": "Dual QGNN + LLM",
    }
    rows = []
    for key, label in labels.items():
        values = summaries[key]["metrics"]["aggregate"]
        for metric in ("ade", "fde"):
            baseline = plain[f"{metric}_m"]
            value = 100 * (baseline - values[f"{metric}_m"]) / baseline
            rows.append({"panel": "overall", "group": label, "metric": metric.upper(), "value_percent": value, "ci_low": "", "ci_high": "", "n_scenes": 1200})
    for group in ("low", "medium", "high"):
        for metric in ("ade", "fde"):
            item = strata[group][metric]
            rows.append({"panel": "complexity", "group": group.capitalize(), "metric": metric.upper(), "value_percent": item["point_improvement_percent"], "ci_low": item["ci95_percent"][0], "ci_high": item["ci95_percent"][1], "n_scenes": strata[group]["scenes"]})
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rows = prepare_rows()
    if args.prepare_only:
        print(CSV_PATH)
        return

    setup_style(journal="ieee", lang="en", use_sciplots=True)
    colors = {"ADE": "#0072B2", "FDE": "#D55E00"}
    markers = {"ADE": "o", "FDE": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), gridspec_kw={"width_ratios": [1.12, 1.0]})

    overall_labels = ["Classical dual GNN", "Dual QGNN", "Classical dual GNN + LLM", "Dual QGNN + LLM"]
    y = np.arange(len(overall_labels))[::-1]
    for metric, offset in (("ADE", 0.10), ("FDE", -0.10)):
        values = [next(row["value_percent"] for row in rows if row["panel"] == "overall" and row["group"] == label and row["metric"] == metric) for label in overall_labels]
        axes[0].scatter(values, y + offset, s=34, color=colors[metric], marker=markers[metric], edgecolor="white", linewidth=0.5, label=metric, zorder=3)
        for value, ypos in zip(values, y + offset):
            axes[0].text(value + 0.08, ypos, f"{value:.2f}", va="center", ha="left", fontsize=7, color=colors[metric])
    axes[0].axvline(0, color="0.25", linewidth=0.8)
    axes[0].set_yticks(y, overall_labels)
    axes[0].set_xlim(0, 4.5)
    axes[0].set_title("Error reduction vs. plain GNN (%)", fontsize=8, pad=3)
    axes[0].grid(axis="x", color="0.88", linewidth=0.5)
    axes[0].legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.35, columnspacing=0.9)

    complexity_labels = ["Low", "Medium", "High"]
    y2 = np.arange(len(complexity_labels))[::-1]
    for metric, offset in (("ADE", 0.10), ("FDE", -0.10)):
        subset = [next(row for row in rows if row["panel"] == "complexity" and row["group"] == label and row["metric"] == metric) for label in complexity_labels]
        values = np.asarray([row["value_percent"] for row in subset])
        low = np.asarray([row["ci_low"] for row in subset])
        high = np.asarray([row["ci_high"] for row in subset])
        axes[1].errorbar(values, y2 + offset, xerr=np.vstack([values - low, high - values]), fmt=markers[metric], color=colors[metric], markerfacecolor=colors[metric], markeredgecolor="white", markeredgewidth=0.5, markersize=5.5, elinewidth=1.0, capsize=2.2, label=metric, zorder=3)
    axes[1].axvline(0, color="0.25", linewidth=0.9, linestyle="--")
    axes[1].set_yticks(y2, [f"{label} (n=400)" for label in complexity_labels])
    axes[1].set_xlim(-4.0, 2.6)
    axes[1].set_title("QGNN gain vs. matched classical (%)", fontsize=8, pad=3)
    axes[1].grid(axis="x", color="0.88", linewidth=0.5)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
    finalize_figure(fig, prefer="constrained")
    add_panel_labels(fig, axes=axes, style="ieee", x_offset_pt=-22, y_offset_pt=3)
    preview = render_preview(fig, str(FIGURE_DIR / "final_qgnn_performance_preview.png"), dpi=180)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    plt.rcParams["savefig.bbox"] = None
    basename = str(FIGURE_DIR / "final_qgnn_performance")
    exports = export_figure(fig, basename, formats=["pdf", "svg", "png"], dpi=600, size_inches=(7.16, 3.15), grayscale_preview=False, tight=False)
    grayscale = basename + "_grayscale.png"
    Image.open(basename + ".png").convert("L").save(grayscale, dpi=(600, 600))
    exports.append(grayscale)
    qa = {"verdict": verdict, "issues": issues, "preview": preview, "exports": exports}
    (FIGURE_DIR / "visual_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
