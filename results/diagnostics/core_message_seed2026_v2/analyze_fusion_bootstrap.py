"""Paired time-block bootstrap for quantum and non-quantum fusion controls."""
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
Q_PATH = BASE / "two_hop_quantum_fusion_seed2026_v1/fusion_confirmation.npz"
P_PATH = BASE / "two_hop_plain_fusion_control_seed2026_v1/fusion_confirmation.npz"
OUTPUT = BASE / "two_hop_fusion_paired_bootstrap_seed2026_v1.json"
SEED = 2026
REPLICATES = 10000


def load(path, prefix):
    with np.load(path, allow_pickle=False) as data:
        return {snr: data[f"{prefix}_{snr}"].copy() for snr in ("5", "10", "15", "20")}, data["starts"].copy()


def aggregate(arrays, rows):
    ade, fde = [], []
    for snr in ("5", "10", "15", "20"):
        selected = arrays[snr][rows]
        denominator = selected[:, 2].sum()
        ade.append(selected[:, 0].sum() / (denominator * 20.0))
        fde.append(selected[:, 1].sum() / denominator)
    return np.asarray([np.mean(ade), np.mean(fde)])


def compare(candidate, baseline, blocks):
    all_rows = np.arange(len(blocks))
    point_candidate = aggregate(candidate, all_rows)
    point_baseline = aggregate(baseline, all_rows)
    point = 100.0 * (point_baseline - point_candidate) / point_baseline
    unique_blocks = np.unique(blocks)
    rows_by_block = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(SEED)
    samples = np.empty((REPLICATES, 2), dtype=np.float64)
    for iteration in range(REPLICATES):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        rows = np.concatenate([rows_by_block[block] for block in sampled_blocks])
        candidate_metric = aggregate(candidate, rows)
        baseline_metric = aggregate(baseline, rows)
        samples[iteration] = 100.0 * (baseline_metric - candidate_metric) / baseline_metric
    return {
        "point_improvement_percent": {"ade": float(point[0]), "fde": float(point[1])},
        "ci95_percent": {
            "ade": np.quantile(samples[:, 0], [0.025, 0.975]).tolist(),
            "fde": np.quantile(samples[:, 1], [0.025, 0.975]).tolist(),
        },
        "bootstrap_probability_positive": {
            "ade": float(np.mean(samples[:, 0] > 0)),
            "fde": float(np.mean(samples[:, 1] > 0)),
        },
    }


def main():
    assert Path.cwd() == ROOT and platform.node() == "jscn"
    quantum_fusion, q_starts = load(Q_PATH, "fusion")
    classical, c_starts = load(Q_PATH, "classical")
    plain_fusion, p_starts = load(P_PATH, "fusion")
    assert np.array_equal(q_starts, c_starts) and np.array_equal(q_starts, p_starts)
    blocks = q_starts // 200
    result = {
        "seed": SEED,
        "replicates": REPLICATES,
        "time_blocks": int(len(np.unique(blocks))),
        "quantum_fusion_vs_classical": compare(quantum_fusion, classical, blocks),
        "plain_fusion_vs_classical": compare(plain_fusion, classical, blocks),
        "quantum_fusion_vs_plain_fusion": compare(quantum_fusion, plain_fusion, blocks),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
