"""Prepare tidy CSV tables from the formal multi-target comparison JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DISPLAY_NAMES = {
    "constant_velocity": "Constant velocity",
    "independent_gru": "Independent GRU",
    "lstm": "LSTM",
    "tcn": "TCN",
    "transformer": "Transformer",
    "target_interaction_gnn": "Target-interaction GNN",
    "graph_motion_token_gpt2": "Proposed Graph+LLM",
}

FAMILIES = {
    "constant_velocity": "Physics baseline",
    "independent_gru": "Independent temporal",
    "lstm": "Independent temporal",
    "tcn": "Independent temporal",
    "transformer": "Independent temporal",
    "target_interaction_gnn": "Graph model",
    "graph_motion_token_gpt2": "Proposed",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/multitarget_comparisons_final/comparison_results.json",
    )
    parser.add_argument("--output-dir", default="figures/second_point")
    args = parser.parse_args()

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, metrics in result["test"].items():
        rows.append(
            {
                "model_key": key,
                "model": DISPLAY_NAMES[key],
                "family": FAMILIES[key],
                "ade_m": metrics["ade_m"],
                "fde_m": metrics["fde_m"],
                "interaction_ade_m": metrics["interaction_ade_m"],
                "interaction_fde_m": metrics["interaction_fde_m"],
                "excess_collision_rate_pct": metrics["excess_collision_rate"] * 100.0,
                "is_proposed": key == "graph_motion_token_gpt2",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison_metrics.csv", index=False)

    reduction_rows = []
    for key, metrics in result["proposed_reduction_percent"].items():
        reduction_rows.append(
            {
                "model_key": key,
                "model": DISPLAY_NAMES[key],
                "ade_reduction_pct": metrics["ade_reduction_percent"],
                "fde_reduction_pct": metrics["fde_reduction_percent"],
                "interaction_ade_reduction_pct": metrics["interaction_ade_reduction_percent"],
            }
        )
    pd.DataFrame(reduction_rows).to_csv(output_dir / "relative_reductions.csv", index=False)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
