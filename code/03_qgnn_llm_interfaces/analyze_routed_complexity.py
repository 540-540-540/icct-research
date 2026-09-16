"""Outcome-blind complexity analysis for the attention-routed development run."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(BASE))
from analyze_interaction_strata import scene_features


def aggregate(arrays, selected):
    metrics = []
    for snr in ("5", "10", "15", "20"):
        x = arrays[snr][selected]
        denominator = x[:, 2].sum()
        metrics.append([
            x[:, 0].sum() / (denominator * 20),
            x[:, 1].sum() / denominator,
        ])
    return np.asarray(metrics).mean(0)


def main():
    run_dir = BASE / "attention_routed_qgnn_llm_seed2026_v1"
    paths = {
        "qgnn_llm_before_routing": BASE / "future_token_qgnn_llm_seed2026_v1/future_token_qgnn_llm_full_dev.npz",
        "qgnn_llm_attention_routed": run_dir / "attention_routed_full_dev.npz",
        "classical_llm_matched": BASE / "future_token_classical_llm_seed2026_v1/future_token_qgnn_llm_full_dev.npz",
        "classical_llm_attention_routed": BASE / "attention_routed_classical_llm_seed2026_v1/attention_routed_full_dev.npz",
        "qgnn_graph_only": BASE / "physics_aligned_dual_seed2026_v1/physics_quantum_dual_graph_full_dev.npz",
        "classical_graph_only": BASE / "physics_aligned_dual_seed2026_v1/physics_classical_dual_graph_full_dev.npz",
    }
    loaded = {name: np.load(path) for name, path in paths.items()}
    indices = loaded["qgnn_llm_attention_routed"]["indices"]
    assert all(np.array_equal(data["indices"], indices) for data in loaded.values())
    arrays = {name: {snr: data[snr] for snr in ("5", "10", "15", "20")} for name, data in loaded.items()}
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        features = scene_features(data["train_states"][indices, :20], data["train_mask"][indices])
    cuts = np.quantile(features["composite"], [1 / 3, 2 / 3])
    groups = {
        "low": features["composite"] <= cuts[0],
        "medium": (features["composite"] > cuts[0]) & (features["composite"] <= cuts[1]),
        "high": features["composite"] > cuts[1],
        "all": np.ones(len(indices), dtype=bool),
    }
    result = {"definition": "Outcome-blind tertiles of target count, close pairs, predicted conflicts, closing strength, and maneuver", "groups": {}}
    for group_name, selected in groups.items():
        values = {name: aggregate(data, selected) for name, data in arrays.items()}
        base, routed = values["qgnn_llm_before_routing"], values["qgnn_llm_attention_routed"]
        classical = values["classical_llm_matched"]
        classical_routed = values["classical_llm_attention_routed"]
        qgraph = values["qgnn_graph_only"]
        result["groups"][group_name] = {
            "scenes": int(selected.sum()),
            "metrics": {name: {"ade_m": float(v[0]), "fde_m": float(v[1])} for name, v in values.items()},
            "routing_improvement_over_qgnn_llm_percent": {"ade": float(100 * (base[0] - routed[0]) / base[0]), "fde": float(100 * (base[1] - routed[1]) / base[1])},
            "routed_quantum_improvement_over_classical_percent": {"ade": float(100 * (classical[0] - routed[0]) / classical[0]), "fde": float(100 * (classical[1] - routed[1]) / classical[1])},
            "routed_quantum_improvement_over_matched_routed_classical_percent": {"ade": float(100 * (classical_routed[0] - routed[0]) / classical_routed[0]), "fde": float(100 * (classical_routed[1] - routed[1]) / classical_routed[1])},
            "llm_plus_routing_improvement_over_qgnn_graph_percent": {"ade": float(100 * (qgraph[0] - routed[0]) / qgraph[0]), "fde": float(100 * (qgraph[1] - routed[1]) / qgraph[1])},
        }
    (run_dir / "complexity_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
