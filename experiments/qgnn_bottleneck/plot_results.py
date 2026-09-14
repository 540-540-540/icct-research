"""Render the four BDX-01 figures exclusively from saved diagnostic tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


MODELS = ("Q_best", "G_best", "A_best", "CV_hat")
COLORS = dict(Q_best="#3B6FB6", G_best="#E28E2C", A_best="#4A9D75", CV_hat="#777777")
LABELS = dict(Q_best="QGNN", G_best="Strong GNN", A_best="Directed A", CV_hat="CV_hat")


def _read(path, required):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    missing = set(required) - fields
    if missing or not rows:
        raise ValueError(f"{path}: missing columns {sorted(missing)} or no rows")
    return rows


def _number(row, key):
    value = row.get(key, "")
    if value in (None, ""):
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite {key}: {value}")
    return result


def _unique(rows, **where):
    found = [row for row in rows if all(row.get(key) == value for key, value in where.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {where}, found {len(found)}")
    return found[0]


def _save(fig, figures, stem):
    figures.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", dict(dpi=220)), ("svg", {})):
        fig.savefig(figures / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)


def best_latest(tables, figures):
    rows = _read(tables / "best_latest.csv", ("model", "family", "checkpoint_stage", "split", "ADE", "FDE"))
    families = ("Q", "A", "G")
    family_labels = ("QGNN", "Directed A", "Strong GNN")
    styles = (("train", "best", "#4878A8"), ("train", "latest", "#9CC1DD"),
              ("V_select", "best", "#D97828"), ("V_select", "latest", "#F2B277"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)
    x = np.arange(len(families), dtype=float)
    width = .19
    for axis, metric in zip(axes, ("ADE", "FDE")):
        for offset, (split, stage, color) in zip((-.285, -.095, .095, .285), styles):
            values = [_number(_unique(rows, family=family, checkpoint_stage=stage, split=split), metric)
                      for family in families]
            bars = axis.bar(x + offset, values, width, color=color, edgecolor="white",
                            label=f"{split} / {stage}")
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)
        axis.set(title=f"{metric} at frozen checkpoints", ylabel=f"{metric} (m)",
                 xticks=x, xticklabels=family_labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=.7, alpha=.8)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    fig.suptitle("Best versus latest: identical eval-mode metric hierarchy", fontsize=13)
    fig.tight_layout()
    _save(fig, figures, "01_best_latest")


def _horizon_series(rows, model, split, field, reducer="mean"):
    result = []
    for step in range(1, 21):
        part = [row for row in rows if row["model"] == model and row["split"] == split
                and int(float(row["horizon_steps"])) == step]
        if len(part) != 4 or {int(float(row["snr_db"])) for row in part} != {5, 10, 15, 20}:
            raise ValueError(f"Incomplete four-SNR horizon rows: {model}/{split}/{step}")
        values = [_number(row, field) for row in part]
        present = [value for value in values if value is not None]
        result.append(None if not present else (sum(present) if reducer == "sum" else float(np.mean(present))))
    return np.asarray(result, dtype=float)


def horizon_error(tables, figures):
    required = ("model", "split", "snr_db", "horizon_steps", "horizon_seconds", "ADE", "FDE",
                "fixed_ADE", "fixed_FDE", "ade_targets", "fde_targets", "fixed_complete_targets")
    rows = _read(tables / "horizon_metrics.csv", required)
    available = tuple(model for model in MODELS if any(row["model"] == model for row in rows))
    if available != MODELS:
        raise ValueError(f"Horizon figure requires {MODELS}, found {available}")
    seconds = np.arange(1, 21) * .1
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1, .62]})
    for row_index, split in enumerate(("train", "V_select")):
        for column, metric in enumerate(("ADE", "FDE")):
            axis = axes[row_index, column]
            for model in MODELS:
                axis.plot(seconds, _horizon_series(rows, model, split, metric), color=COLORS[model],
                          linewidth=2, label=LABELS[model])
                fixed = _horizon_series(rows, model, split, f"fixed_{metric}")
                if np.isfinite(fixed).any():
                    axis.plot(seconds, fixed, color=COLORS[model], linewidth=1.1,
                              linestyle=":", alpha=.8)
            axis.set(title=f"{split}: {metric}", ylabel=f"{metric} (m)")
            axis.grid(color="#D9D9D9", linewidth=.7, alpha=.8)
            axis.set_axisbelow(True)
    for column, split in enumerate(("train", "V_select")):
        axis = axes[2, column]
        reference = MODELS[0]
        ade = _horizon_series(rows, reference, split, "ade_targets", "sum")
        fde = _horizon_series(rows, reference, split, "fde_targets", "sum")
        fixed = _horizon_series(rows, reference, split, "fixed_complete_targets", "sum")
        for model in MODELS[1:]:
            if not (np.array_equal(ade, _horizon_series(rows, model, split, "ade_targets", "sum"))
                    and np.array_equal(fde, _horizon_series(rows, model, split, "fde_targets", "sum"))
                    and np.array_equal(fixed, _horizon_series(rows, model, split, "fixed_complete_targets", "sum"))):
                raise ValueError(f"Model denominators differ for {split}")
        axis.plot(seconds, ade, color="#3B6FB6", label="ADE targets")
        axis.plot(seconds, fde, color="#D97828", label="FDE targets")
        axis.plot(seconds, fixed, color="#777777", linestyle=":", label="complete targets")
        axis.set(title=f"{split}: summed denominators over four SNRs",
                 xlabel="Forecast horizon (s)", ylabel="Target count")
        axis.grid(color="#D9D9D9", linewidth=.7, alpha=.8)
        axis.legend(frameon=False, fontsize=8)
    model_handles = [Line2D([0], [0], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in MODELS]
    style_handles = [Line2D([0], [0], color="#333333", linewidth=2, label="Main eligible-target metric"),
                     Line2D([0], [0], color="#333333", linewidth=1.2, linestyle=":",
                            label="Fixed complete-target auxiliary")]
    fig.suptitle("Horizon error and explicit evaluation denominators", fontsize=13, y=.995)
    fig.legend(handles=model_handles + style_handles, loc="upper center",
               bbox_to_anchor=(.5, .965), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .90))
    _save(fig, figures, "02_horizon_error")


def paired_errors(tables, figures):
    fields = ("Q_best_ADE20", "G_best_ADE20", "Q_best_FDE20", "G_best_FDE20")
    rows = _read(tables / "target_metrics_V_select.csv", ("origin", "slot", *fields))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    for axis, metric in zip(axes, ("ADE20", "FDE20")):
        pairs = [(_number(row, f"Q_best_{metric}"), _number(row, f"G_best_{metric}")) for row in rows]
        pairs = np.asarray([(q, g) for q, g in pairs if q is not None and g is not None], dtype=float)
        if len(pairs) < 2:
            raise ValueError(f"Too few paired {metric} targets")
        low, high = float(pairs.min()), float(pairs.max())
        pad = max((high - low) * .04, .005)
        axis.scatter(pairs[:, 1], pairs[:, 0], s=11, alpha=.32, color="#3B6FB6", edgecolors="none")
        axis.plot([low-pad, high+pad], [low-pad, high+pad], color="#333333", linestyle="--", linewidth=1)
        correlation = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
        axis.text(.03, .97, f"n={len(pairs):,}\nPearson r={correlation:.3f}", transform=axis.transAxes,
                  va="top", fontsize=9, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        axis.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad), aspect="equal",
                 title=f"Paired target {metric}", xlabel="Strong GNN error (m)", ylabel="QGNN error (m)")
        axis.grid(color="#E2E2E2", linewidth=.6, alpha=.7)
        axis.set_axisbelow(True)
    fig.suptitle("QGNN and strong-GNN errors on identical V_select target-origin pairs", fontsize=13)
    fig.tight_layout()
    _save(fig, figures, "03_paired_errors")


def _json_object(row, key):
    try:
        value = json.loads(row[key])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON object in {key}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{key} must contain a JSON object")
    return value


def context_intervention(tables, figures):
    required = ("model", "split", "snr_db", "off_minus_on_ADE", "off_minus_on_FDE", "prediction_change_m")
    rows = _read(tables / "context_intervention.csv", required)
    combinations = (("A_best", "train"), ("A_best", "V_select"),
                    ("A_latest", "train"), ("A_latest", "V_select"))
    labels = ("A best\ntrain", "A best\nV_select", "A latest\ntrain", "A latest\nV_select")
    for model, split in combinations:
        part = [row for row in rows if row["model"] == model and row["split"] == split]
        if len(part) != 4 or {int(float(row["snr_db"])) for row in part} != {5, 10, 15, 20}:
            raise ValueError(f"Incomplete context-intervention rows: {model}/{split}")
    x = np.arange(len(combinations), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    delta_ade, delta_fde, changes = [], [], {key: [] for key in ("median", "p90", "p99")}
    for model, split in combinations:
        part = [row for row in rows if row["model"] == model and row["split"] == split]
        delta_ade.append(1000 * np.mean([_number(row, "off_minus_on_ADE") for row in part]))
        delta_fde.append(1000 * np.mean([_number(row, "off_minus_on_FDE") for row in part]))
        objects = [_json_object(row, "prediction_change_m") for row in part]
        for key in changes:
            values = [float(value[key]) for value in objects if value.get(key) is not None]
            changes[key].append(1000 * float(np.mean(values)) if values else np.nan)
    width = .34
    axes[0].bar(x-width/2, delta_ade, width, color="#3B6FB6", label="ADE")
    axes[0].bar(x+width/2, delta_fde, width, color="#D97828", label="FDE")
    axes[0].axhline(0, color="#333333", linewidth=.9)
    axes[0].set(title="Metric change after switching A off", ylabel="off - on (mm)",
                xticks=x, xticklabels=labels)
    axes[0].legend(frameon=False)
    change_colors = dict(median="#4A9D75", p90="#8C6BB1", p99="#C85A5A")
    for offset, key in zip((-.24, 0, .24), ("median", "p90", "p99")):
        axes[1].bar(x + offset, changes[key], .23, color=change_colors[key], label=key.upper())
    axes[1].set(title="Prediction change despite metric cancellation", ylabel="mean across SNR summaries (mm)",
                xticks=x, xticklabels=labels)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=.7, alpha=.8)
        axis.set_axisbelow(True)
    fig.suptitle("Directed-context on/off intervention", fontsize=13)
    fig.tight_layout()
    _save(fig, figures, "04_context_intervention")


def main(report_root=None):
    if report_root is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("report_root", type=Path)
        report_root = parser.parse_args().report_root
    report_root = Path(report_root)
    tables, figures = report_root / "tables", report_root / "figures"
    best_latest(tables, figures)
    horizon_error(tables, figures)
    paired_errors(tables, figures)
    context_intervention(tables, figures)
    return [figures / f"{name}.{suffix}" for name in
            ("01_best_latest", "02_horizon_error", "03_paired_errors", "04_context_intervention")
            for suffix in ("png", "svg")]


if __name__ == "__main__":
    main()
