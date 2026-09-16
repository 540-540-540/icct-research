"""Convert multi-SNR experiment JSON files to tidy plotting tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_test(payload: dict, key_name: str) -> pd.DataFrame:
    rows = []
    for snr_text, groups in payload["test"].items():
        for key, metrics in groups.items():
            rows.append({"snr_db": int(snr_text), key_name: key, **metrics})
    return pd.DataFrame(rows).sort_values([key_name, "snr_db"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/multitarget_snr")
    parser.add_argument("--output-dir", default="figures/second_point_snr")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((result_dir / "comparison_snr_results.json").read_text(encoding="utf-8"))
    ablation = json.loads((result_dir / "ablation_snr_results.json").read_text(encoding="utf-8"))

    comparison_df = flatten_test(comparison, "model_key")
    ablation_df = flatten_test(ablation, "variant_key")
    comparison_df.to_csv(output_dir / "comparison_snr_metrics.csv", index=False)
    ablation_df.to_csv(output_dir / "ablation_snr_metrics.csv", index=False)

    proposed = comparison_df[comparison_df.model_key == "graph_motion_token_gpt2"].set_index("snr_db")
    gnn = comparison_df[comparison_df.model_key == "target_interaction_gnn"].set_index("snr_db")
    gain = pd.DataFrame(
        {
            "snr_db": proposed.index,
            "ade_reduction_vs_gnn_pct": 100.0 * (gnn.ade_m - proposed.ade_m) / gnn.ade_m,
            "fde_reduction_vs_gnn_pct": 100.0 * (gnn.fde_m - proposed.fde_m) / gnn.fde_m,
        }
    ).reset_index(drop=True)
    gain.to_csv(output_dir / "proposed_gain_vs_gnn.csv", index=False)

    metadata = {
        "snr_db": comparison["snr_db"],
        "noise_map": comparison["noise_map"],
        "proposed_best_on_ade_and_fde": comparison["proposed_best_on_ade_and_fde"],
        "trajectory_example": comparison["trajectory_example"],
        "comparison_rows": len(comparison_df),
        "ablation_rows": len(ablation_df),
        "statistical_note": "Single-seed point estimates; no error bars or significance claims.",
    }
    (output_dir / "figure_data_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison_df.groupby("model_key")[["ade_m", "fde_m"]].agg(["min", "max"]))
    print(ablation_df.groupby("variant_key")[["ade_m", "fde_m"]].agg(["min", "max"]))
    print(gain)


if __name__ == "__main__":
    main()
