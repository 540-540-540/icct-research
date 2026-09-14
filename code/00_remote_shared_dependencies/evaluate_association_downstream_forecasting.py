"""Evaluate how measurement-track association affects downstream forecasting.

All measurement sets are identity-free model inputs. Ground-truth owners are used
only by the oracle upper-bound and offline metrics. No figure is produced.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from measurement_track_association import AssociationGraphConfig, MeasurementTrackAssociationNet
from measurement_track_dataset import (
    AssociationSimulationConfig,
    build_association_features,
    load_snr_noise,
    simulate_measurements,
)
from MultiTargetTimeLLM import MultiTargetGraphLLM
from run_measurement_track_experiment import (
    decode_hungarian_baseline,
    decode_neural,
    make_single_batch,
    set_seed,
)
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def load_checkpoint(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def reconstruct_histories(
    method: str,
    association_model: Optional[torch.nn.Module],
    states: np.ndarray,
    masks: np.ndarray,
    noise_map: Dict[int, tuple],
    snr_db: int,
    simulation: AssociationSimulationConfig,
    device: torch.device,
    number_scenes: int,
) -> Dict[str, np.ndarray | float]:
    """Run recursive tracking and retain the estimated state at every history step."""
    if association_model is not None:
        association_model.eval()
    scenes = min(number_scenes, len(states))
    history_length = 20
    histories = np.zeros((scenes, history_length, simulation.max_tracks, 4), dtype=np.float32)
    entropy_history = np.zeros((scenes, history_length, simulation.max_tracks), dtype=np.float32)
    process_covariance = np.asarray(
        [
            simulation.process_position_sigma_m**2,
            simulation.process_position_sigma_m**2,
            simulation.process_velocity_sigma_mps**2,
            simulation.process_velocity_sigma_mps**2,
        ],
        dtype=np.float32,
    )

    for scene in range(scenes):
        valid = masks[scene]
        rng_init = np.random.default_rng(simulation.seed + 7_000_000 + scene)
        track_state = states[scene, 0].copy()
        track_state[:, :2] += rng_init.normal(
            0.0, simulation.process_position_sigma_m, size=track_state[:, :2].shape
        ).astype(np.float32)
        track_state[:, 2:4] += rng_init.normal(
            0.0, simulation.process_velocity_sigma_mps, size=track_state[:, 2:4].shape
        ).astype(np.float32)
        track_state[~valid] = 0.0
        track_covariance = np.tile(process_covariance, (simulation.max_tracks, 1))
        miss_count = np.zeros(simulation.max_tracks, dtype=np.float32)
        confidence = np.ones(simulation.max_tracks, dtype=np.float32)
        histories[scene, 0] = track_state

        for frame in range(1, history_length):
            rng = np.random.default_rng(
                simulation.seed + 8_000_000 + snr_db * 100_000 + scene * 101 + frame
            )
            measurement = simulate_measurements(
                states[scene, frame], valid, snr_db, noise_map, simulation, rng
            )
            predicted_state = track_state.copy()
            predicted_state[:, :2] += track_state[:, 2:4] * simulation.dt
            predicted_covariance = track_covariance + process_covariance
            if method == "gated_hungarian_ekf":
                covariance_floor = np.asarray(
                    [
                        simulation.covariance_floor_position_m**2,
                        simulation.covariance_floor_position_m**2,
                        simulation.covariance_floor_velocity_mps**2,
                        simulation.covariance_floor_velocity_mps**2,
                    ],
                    dtype=np.float32,
                )
                predicted_covariance = np.maximum(predicted_covariance, covariance_floor)
            features = build_association_features(
                predicted_state,
                predicted_covariance,
                valid,
                np.full(simulation.max_tracks, frame, dtype=np.float32),
                miss_count,
                confidence,
                measurement,
                snr_db,
                simulation,
            )

            if method == "oracle_association":
                assignment = np.full(
                    simulation.max_tracks, simulation.max_measurements, dtype=np.int64
                )
                for track in np.flatnonzero(valid):
                    match = np.flatnonzero(measurement["owner"] == track)
                    if len(match):
                        assignment[track] = int(match[0])
                entropy = None
            elif method == "gated_hungarian_ekf":
                assignment = decode_hungarian_baseline(
                    features["edge_features"],
                    features["candidate_mask"],
                    features["track_mask"],
                    features["measurement_mask"],
                    simulation.max_measurements,
                    rejection_mahalanobis_sq=simulation.hungarian_rejection_mahalanobis_sq,
                )
                entropy = None
            else:
                output = association_model(make_single_batch(features, device))
                probability = output["probabilities"][0].cpu().numpy()
                entropy = output["association_entropy"][0].cpu().numpy()
                assignment = decode_neural(
                    probability,
                    features["candidate_mask"],
                    features["track_mask"],
                    features["measurement_mask"],
                )

            for track in np.flatnonzero(valid):
                match = int(assignment[track])
                if match < len(measurement["owner"]):
                    measurement_covariance = measurement["covariance"][match]
                    gain = predicted_covariance[track] / np.maximum(
                        predicted_covariance[track] + measurement_covariance, 1.0e-6
                    )
                    track_state[track] = predicted_state[track] + gain * (
                        measurement["state"][match] - predicted_state[track]
                    )
                    track_covariance[track] = (1.0 - gain) * predicted_covariance[track]
                    if method == "gated_hungarian_ekf":
                        track_covariance[track] = np.maximum(
                            track_covariance[track], covariance_floor
                        )
                    miss_count[track] = 0.0
                    confidence[track] = min(1.0, confidence[track] + 0.08)
                else:
                    track_state[track] = predicted_state[track]
                    track_covariance[track] = predicted_covariance[track]
                    miss_count[track] += 1.0
                    confidence[track] *= 0.82
                    if (
                        method == "gated_hungarian_ekf"
                        and miss_count[track] >= simulation.reinitialize_after_misses
                    ):
                        track_covariance[track] = np.maximum(
                            track_covariance[track],
                            np.asarray(
                                [
                                    simulation.reinitialization_position_sigma_m**2,
                                    simulation.reinitialization_position_sigma_m**2,
                                    simulation.reinitialization_velocity_sigma_mps**2,
                                    simulation.reinitialization_velocity_sigma_mps**2,
                                ],
                                dtype=np.float32,
                            ),
                        )
                        confidence[track] = min(confidence[track], 0.25)
            track_state[~valid] = 0.0
            histories[scene, frame] = track_state
            if entropy is not None:
                entropy_history[scene, frame] = entropy

    valid_state = masks[:scenes, None, :, None]
    difference = (histories - states[:scenes, :history_length]) * valid_state
    position_squared = np.sum(difference[..., :2] ** 2, axis=-1)
    velocity_squared = np.sum(difference[..., 2:4] ** 2, axis=-1)
    denominator = float(masks[:scenes].sum() * history_length)
    return {
        "history": histories,
        "history_position_rmse_m": float(np.sqrt(position_squared.sum() / denominator)),
        "history_velocity_rmse_mps": float(np.sqrt(velocity_squared.sum() / denominator)),
        "mean_association_entropy": float(
            entropy_history[:, 1:][np.broadcast_to(masks[:scenes, None, :], entropy_history[:, 1:].shape)].mean()
        )
        if method in ("edge_mlp", "association_gnn")
        else 0.0,
    }


@torch.inference_mode()
def forecast_metrics(
    model: torch.nn.Module,
    histories: np.ndarray,
    futures: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    total_distance = 0.0
    total_final = 0.0
    total_points = 0
    total_targets = 0
    step_sums = np.zeros(futures.shape[1], dtype=np.float64)
    for start in range(0, len(histories), batch_size):
        stop = min(start + batch_size, len(histories))
        history = torch.from_numpy(histories[start:stop]).to(device)
        mask = torch.from_numpy(masks[start:stop]).to(device)
        truth = torch.from_numpy(futures[start:stop, :, :, :2]).to(device)
        prediction = model(history, mask)["future_position"]
        distance = torch.linalg.vector_norm(prediction - truth, dim=-1)
        valid = mask[:, None, :].expand_as(distance)
        total_distance += float(distance[valid].sum())
        total_final += float(distance[:, -1][mask].sum())
        total_points += int(valid.sum())
        total_targets += int(mask.sum())
        step_sums += (distance * valid).sum(dim=(0, 2)).cpu().numpy()
    per_step_denominator = max(total_targets, 1)
    return {
        "ade_m": total_distance / max(total_points, 1),
        "fde_m": total_final / max(total_targets, 1),
        "ade_by_step_m": (step_sums / per_step_denominator).tolist(),
        "valid_target_trajectories": total_targets,
    }


def write_csv(path: Path, payload: dict) -> None:
    rows = []
    for snr, methods in payload["results"].items():
        for method, values in methods.items():
            for predictor, metrics in values["forecast"].items():
                rows.append(
                    {
                        "snr_db": snr,
                        "association_method": method,
                        "predictor": predictor,
                        "history_position_rmse_m": values["history_position_rmse_m"],
                        "history_velocity_rmse_mps": values["history_velocity_rmse_mps"],
                        "ade_m": metrics["ade_m"],
                        "fde_m": metrics["fde_m"],
                        "valid_target_trajectories": metrics["valid_target_trajectories"],
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
    parser.add_argument("--association-dir", default="results/measurement_track_association")
    parser.add_argument("--output-dir", default="results/measurement_track_association")
    parser.add_argument("--scenes", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the remote execution policy")
    set_seed(args.seed)
    device = torch.device("cuda")
    with np.load(args.cache, allow_pickle=False) as cache:
        states = cache["test_states"].copy().astype(np.float32)
        masks = cache["test_mask"].copy().astype(np.bool_)
        metadata = json.loads(str(cache["metadata"].item()))
    scenes = min(args.scenes, len(states))
    states = states[:scenes]
    masks = masks[:scenes]
    simulation = AssociationSimulationConfig(seed=args.seed, dt=float(metadata["dt"]))
    noise_map = load_snr_noise(args.calibration)

    association_dir = Path(args.association_dir)
    association_models = {
        "edge_mlp": load_checkpoint(
            MeasurementTrackAssociationNet(AssociationGraphConfig(hidden_dim=96, graph_layers=0)),
            association_dir / "edge_mlp.pt",
        ).to(device).eval(),
        "association_gnn": load_checkpoint(
            MeasurementTrackAssociationNet(AssociationGraphConfig(hidden_dim=96, graph_layers=2)),
            association_dir / "association_gnn.pt",
        ).to(device).eval(),
    }

    forecast_config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    target_gnn = load_checkpoint(
        TargetInteractionGNN(forecast_config),
        Path("results/multitarget_phase1/target_interaction_gnn.pt"),
    )
    graph_llm_backbone = load_checkpoint(
        TargetInteractionGNN(forecast_config),
        Path("results/multitarget_phase1/target_interaction_gnn.pt"),
    )
    graph_llm = load_checkpoint(
        MultiTargetGraphLLM(graph_llm_backbone, forecast_config, llm_layers=4, lora_rank=8),
        Path("results/multitarget_ablation/full_retrained/graph_motion_token_gpt2.pt"),
    )
    predictors = {
        "target_interaction_gnn": target_gnn.to(device).eval(),
        "graph_motion_token_gpt2": graph_llm.to(device).eval(),
    }

    payload = {
        "experiment": "association_to_multi_target_forecasting_end_to_end",
        "status": "initial_downstream_validation_no_visualization",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "evaluated_scenes": scenes,
        "snr_db": [5, 10, 15, 20],
        "protocol": {
            "shared_measurements_across_association_methods": True,
            "vehicle_id_used_as_model_input": False,
            "history_frames": 20,
            "forecast_frames": 20,
            "forecast_horizon_s": float(metadata["prediction_length"] * metadata["dt"]),
            "association_entropy_used_by_current_llm": False,
            "note": "Entropy is exported as an interface for later joint training; this run isolates history-reconstruction quality.",
        },
        "results": {},
    }

    methods = ("clean_history", "oracle_association", "gated_hungarian_ekf", "edge_mlp", "association_gnn")
    future = states[:, 20:40]
    for snr in (5, 10, 15, 20):
        payload["results"][str(snr)] = {}
        for method in methods:
            if method == "clean_history":
                history_result = {
                    "history": states[:, :20].copy(),
                    "history_position_rmse_m": 0.0,
                    "history_velocity_rmse_mps": 0.0,
                    "mean_association_entropy": 0.0,
                }
            else:
                history_result = reconstruct_histories(
                    method,
                    association_models.get(method),
                    states,
                    masks,
                    noise_map,
                    snr,
                    simulation,
                    device,
                    scenes,
                )
            forecast = {
                name: forecast_metrics(
                    model,
                    history_result["history"],
                    future,
                    masks,
                    device,
                    args.batch_size,
                )
                for name, model in predictors.items()
            }
            record = {
                "history_position_rmse_m": history_result["history_position_rmse_m"],
                "history_velocity_rmse_mps": history_result["history_velocity_rmse_mps"],
                "mean_association_entropy": history_result["mean_association_entropy"],
                "forecast": forecast,
            }
            payload["results"][str(snr)][method] = record
            print(json.dumps({"snr_db": snr, "method": method, **record}), flush=True)

    payload["performance_gates"] = {
        "association_gnn_history_rmse_below_hungarian_at_each_snr": {
            snr: payload["results"][snr]["association_gnn"]["history_position_rmse_m"]
            < payload["results"][snr]["gated_hungarian_ekf"]["history_position_rmse_m"]
            for snr in payload["results"]
        },
        "association_gnn_graph_llm_ade_fde_below_hungarian_at_each_snr": {
            snr: (
                payload["results"][snr]["association_gnn"]["forecast"]["graph_motion_token_gpt2"]["ade_m"]
                < payload["results"][snr]["gated_hungarian_ekf"]["forecast"]["graph_motion_token_gpt2"]["ade_m"]
                and payload["results"][snr]["association_gnn"]["forecast"]["graph_motion_token_gpt2"]["fde_m"]
                < payload["results"][snr]["gated_hungarian_ekf"]["forecast"]["graph_motion_token_gpt2"]["fde_m"]
            )
            for snr in payload["results"]
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "downstream_forecasting_results.json"
    csv_path = output_dir / "downstream_forecasting_summary.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, payload)
    print(json.dumps({"result_path": str(json_path), "performance_gates": payload["performance_gates"]}), flush=True)


if __name__ == "__main__":
    main()
