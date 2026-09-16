"""Paired scene bootstrap for the high-complexity routed-fusion comparisons."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
RUN = BASE / "attention_routed_qgnn_llm_seed2026_v1"
sys.path.insert(0, str(BASE))
from analyze_interaction_strata import scene_features

SNRS = ("5", "10", "15", "20")
REPETITIONS = 10000
SEED = 2026


def arrays(path):
    data = np.load(path)
    return {snr: data[snr] for snr in SNRS}, data["indices"]


def metrics(data, selected):
    values = []
    for snr in SNRS:
        x = data[snr][selected]
        denominator = x[:, 2].sum()
        values.append([x[:, 0].sum() / (denominator * 20), x[:, 1].sum() / denominator])
    return np.asarray(values).mean(0)


def bootstrap(reference, candidate, indices, rng):
    point = 100 * (metrics(reference, indices) - metrics(candidate, indices)) / metrics(reference, indices)
    draws = np.empty((REPETITIONS, 2))
    for repetition in range(REPETITIONS):
        sampled = indices[rng.integers(0, len(indices), len(indices))]
        ref, cand = metrics(reference, sampled), metrics(candidate, sampled)
        draws[repetition] = 100 * (ref - cand) / ref
    return {
        name: {
            "point_improvement_percent": float(point[column]),
            "median_percent": float(np.median(draws[:, column])),
            "ci95_percent": [float(v) for v in np.quantile(draws[:, column], [0.025, 0.975])],
            "probability_improvement": float((draws[:, column] > 0).mean()),
        }
        for column, name in enumerate(("ade", "fde"))
    }


def main():
    quantum, indices = arrays(RUN / "attention_routed_full_dev.npz")
    classical, ci = arrays(BASE / "attention_routed_classical_llm_seed2026_v1/attention_routed_full_dev.npz")
    qgraph, qi = arrays(BASE / "physics_aligned_dual_seed2026_v1/physics_quantum_dual_graph_full_dev.npz")
    assert np.array_equal(indices, ci) and np.array_equal(indices, qi)
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        features = scene_features(data["train_states"][indices, :20], data["train_mask"][indices])
    cut = np.quantile(features["composite"], 2 / 3)
    high = np.flatnonzero(features["composite"] > cut)
    rng = np.random.default_rng(SEED)
    result = {
        "seed": SEED, "repetitions": REPETITIONS, "high_complexity_scenes": int(len(high)),
        "definition": "Outcome-blind upper tertile of the pre-existing interaction complexity composite",
        "quantum_routed_vs_classical_routed": bootstrap(classical, quantum, high, rng),
        "quantum_routed_vs_qgnn_graph_only": bootstrap(qgraph, quantum, high, rng),
    }
    (RUN / "high_complexity_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
