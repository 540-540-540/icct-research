"""One-shot paired test evaluation of the frozen two-local-layer Raj QGNN."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "code" / "00_remote_shared_dependencies"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCHIVE))

from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, parameter_count
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN
from prediction.senior_raj_backbone import SeniorRajQGNN


def load_state(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)["model_state"]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dataset = MultiTargetSceneDataset("data/multitarget_lankershim_v1.npz", "test")
    metadata = dataset.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    q_paths = {
        2026: ROOT / "results/senior_raj_two_classical_screen/seed2026_matched/raj_qgnn.pt",
        2027: ROOT / "results/senior_raj_two_classical_screen/seed2027/raj_qgnn.pt",
        2028: ROOT / "results/senior_raj_two_classical_screen/seed2028/raj_qgnn.pt",
    }
    runs = []
    for seed in (2026, 2027, 2028):
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        classical = TargetInteractionGNN(config).to(device)
        classical.load_state_dict(
            load_state(ROOT / f"results/senior_raj_hybrid_matched/seed{seed}/target_interaction_gnn.pt")
        )
        quantum = SeniorRajQGNN(
            config,
            quantum_scale=0.05,
            projection_hidden=128,
            projection_depth=1,
            classical_layers=2,
            quantum_first=True,
        ).to(device)
        quantum.load_state_dict(load_state(q_paths[seed]))
        classical_test = evaluate(classical, loader, device, 0.35, 0.20, seed + 2000)
        quantum_test = evaluate(quantum, loader, device, 0.35, 0.20, seed + 2000)
        gain = {
            metric: 100.0 * (classical_test[metric] - quantum_test[metric]) / classical_test[metric]
            for metric in ("ade_m", "fde_m")
        }
        runs.append(
            {
                "seed": seed,
                "target_interaction_gnn": classical_test,
                "raj_two_local_layer_qgnn": quantum_test,
                "qgnn_gain_percent": gain,
            }
        )

    summary = {
        "status": "engineering_confirmation_not_blind_test",
        "protocol": {
            "test_evaluated_after_three_seed_validation_freeze": True,
            "test_split_was_seen_by_earlier_rejected_architectures": True,
            "position_noise": 0.35,
            "velocity_noise": 0.20,
            "quantum_scale": 0.05,
            "classical_layers": 2,
            "quantum_first": True,
        },
        "parameters": {
            "target_interaction_gnn": parameter_count(classical),
            "raj_two_local_layer_qgnn": parameter_count(quantum),
        },
        "runs": runs,
    }
    for arm in ("target_interaction_gnn", "raj_two_local_layer_qgnn"):
        summary[arm] = {}
        for metric in ("ade_m", "fde_m"):
            values = [run[arm][metric] for run in runs]
            summary[arm][metric] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
    summary["mean_qgnn_gain_percent"] = {}
    for metric in ("ade_m", "fde_m"):
        classical_mean = summary["target_interaction_gnn"][metric]["mean"]
        quantum_mean = summary["raj_two_local_layer_qgnn"][metric]["mean"]
        summary["mean_qgnn_gain_percent"][metric] = (
            100.0 * (classical_mean - quantum_mean) / classical_mean
        )
    summary["qgnn_wins_both_metrics_all_seeds"] = all(
        run["qgnn_gain_percent"]["ade_m"] > 0 and run["qgnn_gain_percent"]["fde_m"] > 0
        for run in runs
    )
    output = ROOT / "results/senior_raj_two_classical_final/three_seed_test_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
