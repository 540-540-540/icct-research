"""Validate the development-selected physical complexity rule on the fixed test arrays."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
EVAL = BASE / "final_qgnn_paper_evaluation_seed2026_v1"
RULE_PATH = BASE / "physical_complexity_rule_seed2026_v1/selected_rule.json"
OUT = BASE / "physical_complexity_rule_seed2026_v1/test_validation.json"
sys.path.insert(0, str(BASE))

from analyze_interaction_strata import scene_features

SEED = 2026
REPS = 10000
SNRS = ("5", "10", "15", "20")


def metric(arrays, indices):
    values = []
    for snr in SNRS:
        x = arrays[snr][indices]
        denominator = x[:, 2].sum()
        values.append((x[:, 0].sum() / (denominator * 20), x[:, 1].sum() / denominator))
    return np.mean(values, axis=0)


def summarize(reference, candidate, indices, rng):
    c = metric(reference, indices)
    q = metric(candidate, indices)
    point = 100 * (c - q) / c
    draws = np.empty((REPS, 2), dtype=np.float64)
    for repetition in range(REPS):
        sampled = indices[rng.integers(0, len(indices), len(indices))]
        cb = metric(reference, sampled)
        qb = metric(candidate, sampled)
        draws[repetition] = 100 * (cb - qb) / cb
    result = {
        "scenes": int(len(indices)),
        "fraction": float(len(indices) / len(reference["5"])),
        "classical": {"ade_m": float(c[0]), "fde_m": float(c[1])},
        "quantum": {"ade_m": float(q[0]), "fde_m": float(q[1])},
    }
    for column, key in enumerate(("ade", "fde")):
        result[key] = {
            "point_improvement_percent": float(point[column]),
            "bootstrap_median_percent": float(np.median(draws[:, column])),
            "ci95_percent": [float(value) for value in np.quantile(draws[:, column], [0.025, 0.975])],
            "bootstrap_probability_improvement": float((draws[:, column] > 0).mean()),
        }
    return result, draws


def per_snr(reference, candidate, indices):
    output = {}
    for snr in SNRS:
        c = reference[snr][indices]
        q = candidate[snr][indices]
        c_metric = np.asarray((c[:, 0].sum() / (c[:, 2].sum() * 20), c[:, 1].sum() / c[:, 2].sum()))
        q_metric = np.asarray((q[:, 0].sum() / (q[:, 2].sum() * 20), q[:, 1].sum() / q[:, 2].sum()))
        improvement = 100 * (c_metric - q_metric) / c_metric
        output[snr] = {
            "classical": {"ade_m": float(c_metric[0]), "fde_m": float(c_metric[1])},
            "quantum": {"ade_m": float(q_metric[0]), "fde_m": float(q_metric[1])},
            "quantum_improvement_percent": {"ade": float(improvement[0]), "fde": float(improvement[1])},
        }
    return output


def main():
    selected_rule = json.loads(RULE_PATH.read_text(encoding="utf-8"))["chosen"]
    rule = selected_rule["rule"]
    assert rule["feature"] == "maneuver" and rule["operator"] == ">="
    threshold = float(rule["threshold"])
    classic_npz = np.load(EVAL / "physics_classical_dual_llm.npz")
    quantum_npz = np.load(EVAL / "physics_quantum_dual_llm.npz")
    classical = {snr: classic_npz[snr] for snr in SNRS}
    quantum = {snr: quantum_npz[snr] for snr in SNRS}
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        features = scene_features(data["test_states"][:, :20], data["test_mask"])
    selected = np.flatnonzero(features["maneuver"] >= threshold)
    complement = np.flatnonzero(features["maneuver"] < threshold)
    rng = np.random.default_rng(SEED)
    selected_result, selected_draws = summarize(classical, quantum, selected, rng)
    complement_result, complement_draws = summarize(classical, quantum, complement, rng)
    interaction = selected_draws - complement_draws
    result = {
        "status": "completed",
        "rule_locked_on_development": rule,
        "test_rule_prevalence": selected_result["fraction"],
        "selected": {**selected_result, "by_snr": per_snr(classical, quantum, selected)},
        "complement": complement_result,
        "selected_minus_complement_interaction": {
            key: {
                "median_percentage_points": float(np.median(interaction[:, column])),
                "ci95_percentage_points": [float(value) for value in np.quantile(interaction[:, column], [0.025, 0.975])],
                "bootstrap_probability_positive": float((interaction[:, column] > 0).mean()),
            }
            for column, key in enumerate(("ade", "fde"))
        },
        "paired_scene_bootstrap_repetitions": REPS,
        "single_seed": True,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
