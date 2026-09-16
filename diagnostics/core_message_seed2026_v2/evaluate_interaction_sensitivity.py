"""Exploratory all-component sensitivity checks after primary stratification."""
import json
from pathlib import Path

import numpy as np

import evaluate_interaction_strata as primary


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "converged_protocol_seed2026_v1"
COMPONENTS = (
    "active_targets",
    "mean_neighbor_degree",
    "nearest_pair_proximity",
    "closing_ttc_risk",
)


def exact_thirds(values, indices):
    order = np.lexsort((indices, values))
    labels = np.empty(len(values), dtype=np.int64)
    cuts = np.linspace(0, len(values), 4, dtype=np.int64)
    for level in range(3):
        labels[order[cuts[level] : cuts[level + 1]]] = level
    return labels


def main():
    dev_indices, starts, _, _, features, _, _ = primary.build_strata()
    predictions = {arm: primary.load_arm(arm, dev_indices) for arm in primary.ARMS}
    blocks = (starts // 200).astype(np.int64)
    report = {
        "status": "exploratory_sensitivity_after_primary_composite_analysis",
        "policy": "all four prespecified components are reported; none selected using model outcomes",
        "components": {},
        "no_retraining": True,
        "no_llm": True,
        "no_test": True,
    }
    for component in COMPONENTS:
        labels = exact_thirds(features[component], dev_indices)
        component_result = {}
        for level, name in enumerate(("low", "medium", "high")):
            rows = np.flatnonzero(labels == level)
            values = features[component][rows]
            metrics = {arm: primary.metrics(predictions[arm], rows) for arm in primary.ARMS}
            q_vs_c = primary.reduction(metrics["classical"]["aggregate"], metrics["quantum"]["aggregate"])
            q_vs_p = primary.reduction(metrics["plain"]["aggregate"], metrics["quantum"]["aggregate"])
            component_result[name] = {
                "scenes": int(len(rows)),
                "component": primary.summarize_values(values),
                "quantum_vs_classical_reduction_percent": q_vs_c,
                "quantum_vs_plain_reduction_percent": q_vs_p,
                "quantum_vs_classical_paired_time_block_bootstrap_95_ci": primary.bootstrap(
                    predictions["classical"], predictions["quantum"], rows, blocks
                ),
            }
        report["components"][component] = component_result
    output = RESULTS / "interaction_component_sensitivity.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
