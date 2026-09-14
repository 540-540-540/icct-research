"""Robustness sweep over missed-detection and false-alarm rates.

The association networks are not retrained for individual conditions.  This is
an out-of-distribution test of the checkpoints trained at miss=0.10 and FAR=1.5.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from measurement_track_association import AssociationGraphConfig, MeasurementTrackAssociationNet
from measurement_track_dataset import AssociationFrameDataset, AssociationSimulationConfig, load_snr_noise
from run_measurement_track_experiment import evaluate_frames, evaluate_recursive_tracking, set_seed


def load_model(path: Path, graph_layers: int, device: torch.device) -> torch.nn.Module:
    model = MeasurementTrackAssociationNet(
        AssociationGraphConfig(hidden_dim=96, graph_layers=graph_layers)
    )
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload.get("model_state", payload) if isinstance(payload, dict) else payload)
    return model.to(device).eval()


def condition_name(miss_probability: float, false_alarm_rate: float) -> str:
    return f"miss_{miss_probability:.2f}__far_{false_alarm_rate:.2f}"


def write_csv(path: Path, payload: dict) -> None:
    rows = []
    for condition, condition_result in payload["conditions"].items():
        setting = condition_result["setting"]
        for snr, snr_result in condition_result["snr"].items():
            for method, frame in snr_result["frame_association"].items():
                recursive = snr_result["recursive_tracking"][method]
                rows.append(
                    {
                        "condition": condition,
                        "miss_probability": setting["miss_probability"],
                        "false_alarm_rate": setting["false_alarm_rate"],
                        "snr_db": int(snr),
                        "method": method,
                        "association_f1": frame["association_f1"],
                        "track_decision_accuracy": frame["track_decision_accuracy"],
                        "identity_f1": recursive["identity_f1"],
                        "id_switches": recursive["id_switches"],
                        "id_switches_per_1000_detections": recursive[
                            "id_switches_per_1000_detections"
                        ],
                        "position_rmse_m": recursive["position_rmse_m"],
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--checkpoint-dir", default="results/measurement_track_association")
    parser.add_argument("--output-dir", default="results/measurement_track_robustness")
    parser.add_argument("--frame-samples", type=int, default=6000)
    parser.add_argument("--sequence-scenes", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the remote execution policy")
    set_seed(args.seed)
    device = torch.device("cuda")
    checkpoint_dir = Path(args.checkpoint_dir)
    models = {
        "edge_mlp": load_model(checkpoint_dir / "edge_mlp.pt", 0, device),
        "association_gnn": load_model(checkpoint_dir / "association_gnn.pt", 2, device),
    }
    with np.load(args.cache, allow_pickle=False) as cache:
        states = cache["test_states"].copy().astype(np.float32)
        masks = cache["test_mask"].copy().astype(np.bool_)
        metadata = json.loads(str(cache["metadata"].item()))
    noise_map = load_snr_noise(args.calibration)

    # One-factor-at-a-time sweeps around the training condition (0.10, 1.5).
    settings = []
    for miss in (0.05, 0.10, 0.20, 0.30):
        settings.append((miss, 1.5))
    for false_alarm in (0.5, 1.0, 3.0, 5.0):
        settings.append((0.10, false_alarm))
    settings = list(dict.fromkeys(settings))

    payload = {
        "experiment": "measurement_track_association_robustness_sweep",
        "status": "formal_single_seed_robustness_without_retraining",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "training_condition": {"miss_probability": 0.10, "false_alarm_rate": 1.5},
        "frame_samples_per_snr_condition": args.frame_samples,
        "recursive_scenes_per_snr_condition": args.sequence_scenes,
        "snr_db": [5, 10, 15, 20],
        "hungarian_stabilization": {
            "explicit_null_rejection_mahalanobis_sq": 16.0,
            "position_covariance_floor_sigma_m": 0.50,
            "velocity_covariance_floor_sigma_mps": 0.25,
            "track_recovery_after_consecutive_misses": 3,
            "recovery_position_sigma_m": 3.0,
            "recovery_velocity_sigma_mps": 1.5,
            "uses_ground_truth_identity": False,
        },
        "conditions": {},
    }

    methods = (
        "gated_hungarian_ekf_legacy",
        "gated_hungarian_ekf",
        "edge_mlp",
        "association_gnn",
    )
    for miss_probability, false_alarm_rate in settings:
        simulation = AssociationSimulationConfig(
            dt=float(metadata["dt"]),
            miss_probability=miss_probability,
            false_alarm_rate=false_alarm_rate,
            seed=args.seed,
        )
        name = condition_name(miss_probability, false_alarm_rate)
        payload["conditions"][name] = {
            "setting": {
                "miss_probability": miss_probability,
                "false_alarm_rate": false_alarm_rate,
            },
            "snr": {},
        }
        for snr in (5, 10, 15, 20):
            dataset = AssociationFrameDataset(
                args.cache,
                "test",
                args.calibration,
                simulation,
                snr_values=(snr,),
                max_samples=args.frame_samples,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=args.workers > 0,
            )
            frame_results = {}
            recursive_results = {}
            for method in methods:
                model = models.get(method)
                frame_results[method] = evaluate_frames(
                    model, loader, device, method, simulation.max_measurements
                )
                recursive_results[method] = evaluate_recursive_tracking(
                    method,
                    model,
                    states,
                    masks,
                    noise_map,
                    snr,
                    simulation,
                    device,
                    args.sequence_scenes,
                )
                print(
                    json.dumps(
                        {
                            "condition": name,
                            "snr_db": snr,
                            "method": method,
                            "frame_f1": frame_results[method]["association_f1"],
                            "identity_f1": recursive_results[method]["identity_f1"],
                            "id_switches": recursive_results[method]["id_switches"],
                            "position_rmse_m": recursive_results[method]["position_rmse_m"],
                        }
                    ),
                    flush=True,
                )
            payload["conditions"][name]["snr"][str(snr)] = {
                "frame_association": frame_results,
                "recursive_tracking": recursive_results,
            }

    comparisons = {}
    for condition, condition_result in payload["conditions"].items():
        comparisons[condition] = {}
        for snr, result in condition_result["snr"].items():
            legacy = result["recursive_tracking"]["gated_hungarian_ekf_legacy"]
            stable = result["recursive_tracking"]["gated_hungarian_ekf"]
            graph = result["recursive_tracking"]["association_gnn"]
            comparisons[condition][snr] = {
                "stabilized_hungarian_reduces_id_switches": stable["id_switches"]
                < legacy["id_switches"],
                "stabilized_hungarian_reduces_rmse": stable["position_rmse_m"]
                < legacy["position_rmse_m"],
                "association_gnn_higher_idf1_than_stabilized_hungarian": graph["identity_f1"]
                > stable["identity_f1"],
                "association_gnn_fewer_id_switches_than_stabilized_hungarian": graph["id_switches"]
                < stable["id_switches"],
            }
    payload["comparisons"] = comparisons

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "association_robustness_results.json"
    csv_path = output_dir / "association_robustness_summary.csv"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, payload)
    print(json.dumps({"result_path": str(result_path), "conditions": len(settings)}), flush=True)


if __name__ == "__main__":
    main()
