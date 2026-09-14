"""Diagnose the selected trajectory figure without changing model outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from multitarget_scene_dataset import MultiTargetSceneDataset


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def roughness(position: np.ndarray, dt: float) -> float:
    if position.shape[0] < 3:
        return float("nan")
    acceleration = np.diff(position, n=2, axis=0) / (dt * dt)
    return rms(acceleration)


def displacement_metrics(prediction: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(prediction - future, axis=-1)
    return distance.mean(axis=0), distance[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--trajectory", default="results/multitarget_snr/trajectory_examples.npz")
    parser.add_argument("--output", default="results/multitarget_snr/trajectory_diagnostics.json")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    dataset = MultiTargetSceneDataset(str(project / args.cache), "test")
    plotted = np.load(project / args.trajectory)
    scene_index = int(plotted["scene_index"])
    selected = plotted["selected_indices"].astype(int)
    target_ids = plotted["target_ids"].astype(int)
    item = dataset[scene_index]
    dt = float(dataset.metadata["dt"])
    clean_history = item["history"][:, selected, :2].numpy()
    clean_future = item["future"][:, selected, :2].numpy()
    stored_future = plotted["future"]

    report = {
        "scene_index": scene_index,
        "selected_indices": selected.tolist(),
        "target_ids": target_ids.tolist(),
        "dt_s": dt,
        "shapes": {
            "clean_history": list(clean_history.shape),
            "clean_future": list(clean_future.shape),
            "stored_future": list(stored_future.shape),
        },
        "stored_future_matches_dataset": bool(np.allclose(clean_future, stored_future)),
        "targets": {},
    }

    for snr in (5, 20):
        observed = plotted[f"history_snr{snr}"]
        gnn = plotted[f"target_interaction_gnn_snr{snr}"]
        proposed = plotted[f"graph_motion_token_gpt2_snr{snr}"]
        gnn_ade, gnn_fde = displacement_metrics(gnn, clean_future)
        proposed_ade, proposed_fde = displacement_metrics(proposed, clean_future)
        snr_report = {}
        for j, target_id in enumerate(target_ids):
            observed_error = observed[:, j] - clean_history[:, j]
            snr_report[str(target_id)] = {
                "history_position_error_rmse_m": rms(observed_error),
                "clean_history_roughness_mps2": roughness(clean_history[:, j], dt),
                "observed_history_roughness_mps2": roughness(observed[:, j], dt),
                "clean_history_to_future_gap_m": float(
                    np.linalg.norm(clean_future[0, j] - clean_history[-1, j])
                ),
                "observed_history_to_future_gap_m": float(
                    np.linalg.norm(clean_future[0, j] - observed[-1, j])
                ),
                "target_gnn_ade_m": float(gnn_ade[j]),
                "target_gnn_fde_m": float(gnn_fde[j]),
                "proposed_ade_m": float(proposed_ade[j]),
                "proposed_fde_m": float(proposed_fde[j]),
                "proposed_better_ade": bool(proposed_ade[j] < gnn_ade[j]),
                "proposed_better_fde": bool(proposed_fde[j] < gnn_fde[j]),
            }
        report["targets"][str(snr)] = snr_report

    output = project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
