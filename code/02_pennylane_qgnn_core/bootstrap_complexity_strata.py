"""Paired scene bootstrap for outcome-blind interaction-complexity strata."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
EVAL = BASE / "final_qgnn_paper_evaluation_seed2026_v1"
sys.path.insert(0, str(BASE))

from analyze_interaction_strata import scene_features

SEED = 2026
REPS = 10000


def metric(arrays, indices):
    values = []
    for snr in ("5", "10", "15", "20"):
        x = arrays[snr][indices]
        den = x[:, 2].sum()
        values.append((x[:, 0].sum() / (den * 20), x[:, 1].sum() / den))
    return np.mean(values, axis=0)


def improvement(reference, candidate, indices):
    ref = metric(reference, indices)
    cand = metric(candidate, indices)
    return 100 * (ref - cand) / ref


def summarize(samples):
    return {
        name: {
            "point_improvement_percent": float(samples["point"][j]),
            "bootstrap_median_percent": float(np.median(samples["draws"][:, j])),
            "ci95_percent": [float(v) for v in np.quantile(samples["draws"][:, j], [0.025, 0.975])],
            "bootstrap_probability_improvement": float((samples["draws"][:, j] > 0).mean()),
        }
        for j, name in enumerate(("ade", "fde"))
    }


def main():
    classical_npz = np.load(EVAL / "physics_classical_dual_llm.npz")
    quantum_npz = np.load(EVAL / "physics_quantum_dual_llm.npz")
    classical = {s: classical_npz[s] for s in ("5", "10", "15", "20")}
    quantum = {s: quantum_npz[s] for s in ("5", "10", "15", "20")}
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        history = data["test_states"][:, :20]
        mask = data["test_mask"]
    score = scene_features(history, mask)["composite"]
    cuts = np.quantile(score, [1 / 3, 2 / 3])
    groups = {
        "low": np.flatnonzero(score <= cuts[0]),
        "medium": np.flatnonzero((score > cuts[0]) & (score <= cuts[1])),
        "high": np.flatnonzero(score > cuts[1]),
    }
    rng = np.random.default_rng(SEED)
    raw = {}
    for name, idx in groups.items():
        draws = np.empty((REPS, 2), dtype=np.float64)
        for b in range(REPS):
            sampled = idx[rng.integers(0, len(idx), len(idx))]
            draws[b] = improvement(classical, quantum, sampled)
        raw[name] = {"point": improvement(classical, quantum, idx), "draws": draws}

    interaction = raw["high"]["draws"] - raw["low"]["draws"]
    result = {
        "seed": SEED,
        "bootstrap_repetitions": REPS,
        "definition": "Outcome-blind tertiles of the fixed interaction-complexity composite; paired resampling of scenes within each stratum and four-SNR macro averaging.",
        "strata": {name: {"scenes": int(len(groups[name])), **summarize(raw[name])} for name in groups},
        "high_minus_low_interaction": {
            metric_name: {
                "median_percentage_points": float(np.median(interaction[:, j])),
                "ci95_percentage_points": [float(v) for v in np.quantile(interaction[:, j], [0.025, 0.975])],
                "bootstrap_probability_positive": float((interaction[:, j] > 0).mean()),
            }
            for j, metric_name in enumerate(("ade", "fde"))
        },
    }
    out = EVAL / "complexity_paired_bootstrap.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
