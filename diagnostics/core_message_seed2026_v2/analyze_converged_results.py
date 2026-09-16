"""Paired development-set analysis for the convergence-controlled run."""
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "converged_protocol_seed2026_v1"
SNRS = (5, 10, 15, 20)
BOOTSTRAPS = 10000


def load(arm, snr):
    return np.load(RESULTS / f"{arm}_graph_full_dev.npz")[str(snr)]


def metrics(array, rows):
    selected = array[rows]
    denominator = selected[:, 2].sum()
    return np.array(
        [selected[:, 0].sum() / (20 * denominator), selected[:, 1].sum() / denominator]
    )


def compare(reference, blocks, unique_blocks, rng):
    quantum = {snr: load("quantum", snr) for snr in SNRS}
    baseline = {snr: load(reference, snr) for snr in SNRS}
    for snr in SNRS:
        assert np.array_equal(quantum[snr][:, 2], baseline[snr][:, 2])
    all_rows = np.arange(len(blocks))
    q_point = np.mean([metrics(quantum[snr], all_rows) for snr in SNRS], axis=0)
    b_point = np.mean([metrics(baseline[snr], all_rows) for snr in SNRS], axis=0)
    point = 100 * (b_point - q_point) / b_point
    bootstrap = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(unique_blocks, len(unique_blocks), replace=True)
        rows = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
        q_value = np.mean([metrics(quantum[snr], rows) for snr in SNRS], axis=0)
        b_value = np.mean([metrics(baseline[snr], rows) for snr in SNRS], axis=0)
        bootstrap.append(100 * (b_value - q_value) / b_value)
    bootstrap = np.asarray(bootstrap)
    return {
        "reference": reference,
        "reduction_percent_ADE_FDE": point.tolist(),
        "block_bootstrap_95_percentile_ci_ADE_FDE": np.percentile(
            bootstrap, [2.5, 97.5], axis=0
        ).T.tolist(),
    }


def main():
    indices = np.load(RESULTS / "quantum_graph_full_dev.npz")["indices"]
    with np.load("/home/js_cn/sensing/data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
        starts = data["train_start_index"][indices]
    blocks = (starts // 200).astype(int)
    unique_blocks = np.unique(blocks)
    rng = np.random.default_rng(20260907)
    report = {
        "training_seed": 2026,
        "development_scenes": int(len(indices)),
        "block_width_frames": 200,
        "unique_blocks": int(len(unique_blocks)),
        "bootstrap_replicates": BOOTSTRAPS,
        "warning": "Intervals condition on one trained seed and quantify paired development scene/time-block variation, not training-seed uncertainty.",
        "comparisons": {
            "quantum_vs_plain": compare("plain", blocks, unique_blocks, rng),
            "quantum_vs_classical": compare("classical", blocks, unique_blocks, rng),
        },
    }
    output = RESULTS / "paired_block_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
