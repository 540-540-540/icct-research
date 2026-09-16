"""Paired time-block bootstrap for retrained LLM component ablations."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
FULL = BASE / "future_token_qgnn_llm_seed2026_v1/future_token_qgnn_llm_full_dev.npz"
ARMS = {
    "no_motion": BASE / "ablation_no_motion_qgnn_llm_seed2026_v3/full_dev_predictions.npz",
    "no_graph": BASE / "ablation_no_graph_qgnn_llm_seed2026_v3/full_dev_predictions.npz",
    "no_queries": BASE / "ablation_no_queries_qgnn_llm_seed2026_v3/full_dev_predictions.npz",
}
OUTPUT = BASE / "llm_component_ablation_seed2026_v3.json"
SNRS = ("5", "10", "15", "20")
REPLICATES = 10000


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {snr: data[snr].copy() for snr in SNRS}, data["indices"].copy()


def aggregate(arrays, rows):
    metrics = []
    for snr in SNRS:
        selected = arrays[snr][rows]
        denominator = selected[:, 2].sum()
        metrics.append((selected[:, 0].sum() / (20.0 * denominator), selected[:, 1].sum() / denominator))
    return np.asarray(metrics).mean(axis=0)


def compare(full, ablated, blocks):
    all_rows = np.arange(len(blocks))
    point_full = aggregate(full, all_rows)
    point_ablated = aggregate(ablated, all_rows)
    # Positive means the complete model is better than the ablated model.
    point = 100.0 * (point_ablated - point_full) / point_ablated
    unique = np.unique(blocks)
    rows_by_block = {block: np.flatnonzero(blocks == block) for block in unique}
    rng = np.random.default_rng(20260910)
    samples = np.empty((REPLICATES, 2), dtype=np.float64)
    for iteration in range(REPLICATES):
        sampled = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([rows_by_block[block] for block in sampled])
        metric_full = aggregate(full, rows)
        metric_ablated = aggregate(ablated, rows)
        samples[iteration] = 100.0 * (metric_ablated - metric_full) / metric_ablated
    return {
        "full": {"ade_m": float(point_full[0]), "fde_m": float(point_full[1])},
        "ablated": {"ade_m": float(point_ablated[0]), "fde_m": float(point_ablated[1])},
        "full_improvement_percent": {"ade": float(point[0]), "fde": float(point[1])},
        "ci95_percent": {
            "ade": np.percentile(samples[:, 0], [2.5, 97.5]).tolist(),
            "fde": np.percentile(samples[:, 1], [2.5, 97.5]).tolist(),
        },
        "probability_full_better": {
            "ade": float(np.mean(samples[:, 0] > 0)),
            "fde": float(np.mean(samples[:, 1] > 0)),
        },
    }


def main():
    assert Path.cwd() == ROOT and platform.node() == "jscn"
    full, indices = load(FULL)
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz", allow_pickle=False) as cache:
        starts = cache["train_start_index"][indices]
    blocks = (starts // 200).astype(np.int64)
    comparisons = {}
    for name, path in ARMS.items():
        ablated, ablated_indices = load(path)
        assert np.array_equal(indices, ablated_indices)
        for snr in SNRS:
            assert np.array_equal(full[snr][:, 2], ablated[snr][:, 2])
        comparisons[name] = compare(full, ablated, blocks)
    result = {
        "seed": 2026,
        "replicates": REPLICATES,
        "time_blocks": int(len(np.unique(blocks))),
        "interpretation": "Positive improvement means the complete model outperforms the ablated model.",
        "limitation": "Intervals condition on one trained seed and measure paired scene/time-block variation only.",
        "comparisons": comparisons,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
