"""Train and evaluate measurement-to-track association models on the GPU server."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.utils.data import DataLoader

from measurement_track_association import AssociationGraphConfig, MeasurementTrackAssociationNet
from measurement_track_dataset import (
    AssociationFrameDataset,
    AssociationSimulationConfig,
    build_association_features,
    load_snr_noise,
    simulate_measurements,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def decode_neural(
    probabilities: np.ndarray,
    candidate_mask: np.ndarray,
    track_mask: np.ndarray,
    measurement_mask: np.ndarray,
) -> np.ndarray:
    tracks = np.flatnonzero(track_mask)
    measurements = np.flatnonzero(measurement_mask)
    dustbin = probabilities.shape[-1] - 1
    prediction = np.full(len(track_mask), dustbin, dtype=np.int64)
    if not len(tracks) or not len(measurements):
        return prediction
    real_probability = probabilities[np.ix_(tracks, measurements)]
    cost = -np.log(np.maximum(real_probability, 1.0e-12))
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        track = int(tracks[row])
        measurement = int(measurements[column])
        if (
            candidate_mask[track, measurement]
            and probabilities[track, measurement] > probabilities[track, dustbin]
        ):
            prediction[track] = measurement
    return prediction


def decode_hungarian_baseline(
    edge_features: np.ndarray,
    candidate_mask: np.ndarray,
    track_mask: np.ndarray,
    measurement_mask: np.ndarray,
    dustbin: int,
    rejection_mahalanobis_sq: Optional[float] = None,
) -> np.ndarray:
    tracks = np.flatnonzero(track_mask)
    measurements = np.flatnonzero(measurement_mask)
    prediction = np.full(len(track_mask), dustbin, dtype=np.int64)
    if not len(tracks) or not len(measurements):
        return prediction
    # Edge channel 5 is clipped squared Mahalanobis distance divided by 10.
    cost = edge_features[np.ix_(tracks, measurements, [5])].squeeze(-1) * 10.0
    allowed = candidate_mask[np.ix_(tracks, measurements)]
    cost = np.where(allowed, cost, 1.0e6)
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        accepted = cost[row, column] < 1.0e5
        if rejection_mahalanobis_sq is not None:
            accepted &= cost[row, column] <= rejection_mahalanobis_sq
        if accepted:
            prediction[int(tracks[row])] = int(measurements[column])
    return prediction


class AssociationAccumulator:
    def __init__(self, dustbin: int):
        self.dustbin = dustbin
        self.valid_tracks = 0
        self.gt_real = 0
        self.gt_miss = 0
        self.pred_real = 0
        self.correct = 0
        self.correct_miss = 0
        self.wrong_real = 0
        self.false_assignment = 0
        self.entropy_sum = 0.0
        self.entropy_count = 0

    def update(
        self,
        prediction: np.ndarray,
        labels: np.ndarray,
        track_mask: np.ndarray,
        entropy: Optional[np.ndarray] = None,
    ) -> None:
        for track in np.flatnonzero(track_mask):
            truth = int(labels[track])
            predicted = int(prediction[track])
            self.valid_tracks += 1
            if truth == self.dustbin:
                self.gt_miss += 1
                if predicted == self.dustbin:
                    self.correct_miss += 1
                else:
                    self.false_assignment += 1
            else:
                self.gt_real += 1
                if predicted == truth:
                    self.correct += 1
                elif predicted != self.dustbin:
                    self.wrong_real += 1
            if predicted != self.dustbin:
                self.pred_real += 1
            if entropy is not None:
                self.entropy_sum += float(entropy[track])
                self.entropy_count += 1

    def result(self) -> Dict[str, float]:
        precision = self.correct / max(self.pred_real, 1)
        recall = self.correct / max(self.gt_real, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        return {
            "association_precision": precision,
            "association_recall": recall,
            "association_f1": f1,
            "track_decision_accuracy": (self.correct + self.correct_miss) / max(self.valid_tracks, 1),
            "miss_class_accuracy": self.correct_miss / max(self.gt_miss, 1),
            "wrong_real_rate": self.wrong_real / max(self.gt_real, 1),
            "false_assignment_rate": self.false_assignment / max(self.gt_miss, 1),
            "mean_association_entropy": self.entropy_sum / max(self.entropy_count, 1),
            "valid_tracks": self.valid_tracks,
            "ground_truth_associations": self.gt_real,
        }


@torch.inference_mode()
def evaluate_frames(
    model: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    method: str,
    max_measurements: int,
) -> Dict[str, float]:
    accumulator = AssociationAccumulator(max_measurements)
    if model is not None:
        model.eval()
    for batch in loader:
        numpy_batch = {key: value.numpy() for key, value in batch.items()}
        if model is not None:
            output = model(to_device(batch, device))
            probabilities = output["probabilities"].cpu().numpy()
            entropy = output["association_entropy"].cpu().numpy()
        else:
            probabilities = None
            entropy = None
        for index in range(len(batch["labels"])):
            if method == "gated_hungarian_ekf":
                prediction = decode_hungarian_baseline(
                    numpy_batch["edge_features"][index],
                    numpy_batch["candidate_mask"][index],
                    numpy_batch["track_mask"][index],
                    numpy_batch["measurement_mask"][index],
                    max_measurements,
                    rejection_mahalanobis_sq=16.0,
                )
            elif method == "gated_hungarian_ekf_legacy":
                prediction = decode_hungarian_baseline(
                    numpy_batch["edge_features"][index],
                    numpy_batch["candidate_mask"][index],
                    numpy_batch["track_mask"][index],
                    numpy_batch["measurement_mask"][index],
                    max_measurements,
                )
            else:
                prediction = decode_neural(
                    probabilities[index],
                    numpy_batch["candidate_mask"][index],
                    numpy_batch["track_mask"][index],
                    numpy_batch["measurement_mask"][index],
                )
            accumulator.update(
                prediction,
                numpy_batch["labels"][index],
                numpy_batch["track_mask"][index],
                None if entropy is None else entropy[index],
            )
    return accumulator.result()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_tracks = 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
            loss = nn.functional.cross_entropy(
                output["logits"].reshape(-1, output["logits"].shape[-1]),
                batch["labels"].reshape(-1),
                ignore_index=-100,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        valid = int((batch["labels"] != -100).sum())
        total_loss += float(loss.detach()) * valid
        total_tracks += valid
    return total_loss / max(total_tracks, 1)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    patience: int,
    max_measurements: int,
) -> Dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda")
    best_state = None
    best_f1 = -math.inf
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        metrics = evaluate_frames(model, validation_loader, device, name, max_measurements)
        row = {"epoch": epoch, "train_loss": loss, **metrics}
        history.append(row)
        print(json.dumps({"model": name, **row}), flush=True)
        if metrics["association_f1"] > best_f1 + 1.0e-5:
            best_f1 = metrics["association_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output_dir / f"{name}.pt"
    torch.save(best_state, checkpoint)
    with (output_dir / f"{name}_training.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")
    return {"checkpoint": str(checkpoint), "best_validation_f1": best_f1, "epochs_ran": len(history)}


def make_single_batch(features: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = (
        "track_features",
        "measurement_features",
        "edge_features",
        "candidate_mask",
        "track_mask",
        "measurement_mask",
    )
    return {key: torch.from_numpy(features[key])[None].to(device) for key in keys}


@torch.inference_mode()
def evaluate_recursive_tracking(
    method: str,
    model: Optional[nn.Module],
    states: np.ndarray,
    masks: np.ndarray,
    noise_map: Dict[int, tuple],
    snr_db: int,
    simulation: AssociationSimulationConfig,
    device: torch.device,
    number_scenes: int,
) -> Dict[str, float]:
    if model is not None:
        model.eval()
    process_covariance = np.asarray(
        [
            simulation.process_position_sigma_m**2,
            simulation.process_position_sigma_m**2,
            simulation.process_velocity_sigma_mps**2,
            simulation.process_velocity_sigma_mps**2,
        ],
        dtype=np.float32,
    )
    correct = wrong = missed = predicted_real = detected = correct_miss = 0
    id_switches = fragments = 0
    squared_error = []
    entropy_values = []
    scenes = min(number_scenes, len(states))
    for scene in range(scenes):
        valid = masks[scene]
        rng_init = np.random.default_rng(simulation.seed + 7_000_000 + scene)
        track_state = states[scene, 0].copy()
        track_state[:, :2] += rng_init.normal(
            0.0, simulation.process_position_sigma_m, size=track_state[:, :2].shape
        )
        track_state[:, 2:4] += rng_init.normal(
            0.0, simulation.process_velocity_sigma_mps, size=track_state[:, 2:4].shape
        )
        track_covariance = np.tile(process_covariance, (simulation.max_tracks, 1))
        miss_count = np.zeros(simulation.max_tracks, dtype=np.float32)
        confidence = np.ones(simulation.max_tracks, dtype=np.float32)
        ever_correct = np.zeros(simulation.max_tracks, dtype=np.bool_)
        previously_correct = np.zeros(simulation.max_tracks, dtype=np.bool_)
        for frame in range(1, min(20, states.shape[1])):
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
                assignment = np.full(simulation.max_tracks, simulation.max_measurements, dtype=np.int64)
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
            elif method == "gated_hungarian_ekf_legacy":
                assignment = decode_hungarian_baseline(
                    features["edge_features"],
                    features["candidate_mask"],
                    features["track_mask"],
                    features["measurement_mask"],
                    simulation.max_measurements,
                )
                entropy = None
            else:
                output = model(make_single_batch(features, device))
                probability = output["probabilities"][0].cpu().numpy()
                entropy = output["association_entropy"][0].cpu().numpy()
                assignment = decode_neural(
                    probability,
                    features["candidate_mask"],
                    features["track_mask"],
                    features["measurement_mask"],
                )
                entropy_values.extend(entropy[valid].tolist())

            current_correct = np.zeros(simulation.max_tracks, dtype=np.bool_)
            owners = measurement["owner"]
            for track in np.flatnonzero(valid):
                gt_match = np.flatnonzero(owners == track)
                gt_detected = len(gt_match) > 0
                if gt_detected:
                    detected += 1
                match = int(assignment[track])
                if match < len(owners):
                    predicted_real += 1
                    if int(owners[match]) == int(track):
                        correct += 1
                        current_correct[track] = True
                    else:
                        wrong += 1
                        id_switches += 1
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
                    if gt_detected:
                        missed += 1
                    else:
                        correct_miss += 1
                    track_state[track] = predicted_state[track]
                    track_covariance[track] = predicted_covariance[track]
                    miss_count[track] += 1.0
                    confidence[track] *= 0.82
                    if (
                        method == "gated_hungarian_ekf"
                        and miss_count[track] >= simulation.reinitialize_after_misses
                    ):
                        # Identity-free track recovery: retain the predicted
                        # state but reopen its uncertainty so a later valid
                        # measurement can pass the statistical gate.
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
                if current_correct[track] and ever_correct[track] and not previously_correct[track]:
                    fragments += 1
                ever_correct[track] |= current_correct[track]
            previously_correct = current_correct
            error = track_state[valid, :2] - states[scene, frame, valid, :2]
            squared_error.extend(np.sum(error**2, axis=-1).tolist())

    precision = correct / max(predicted_real, 1)
    recall = correct / max(detected, 1)
    identity_f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "association_precision": precision,
        "association_recall": recall,
        "identity_f1": identity_f1,
        "id_switches": id_switches,
        "id_switches_per_1000_detections": 1000.0 * id_switches / max(detected, 1),
        "track_fragmentations": fragments,
        "missed_detected_targets": missed,
        "correct_miss_decisions": correct_miss,
        "position_rmse_m": float(np.sqrt(np.mean(squared_error))) if squared_error else float("nan"),
        "mean_association_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "evaluated_scenes": scenes,
        "detected_target_instances": detected,
    }


def write_summary_csv(path: Path, payload: Dict[str, object]) -> None:
    rows = []
    for snr, methods in payload["recursive_tracking"].items():
        for method, metrics in methods.items():
            rows.append({"snr_db": int(snr), "method": method, **metrics})
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--output-dir", default="results/measurement_track_association")
    parser.add_argument("--train-samples", type=int, default=48000)
    parser.add_argument("--validation-samples", type=int, default=10000)
    parser.add_argument("--test-samples-per-snr", type=int, default=8000)
    parser.add_argument("--sequence-scenes", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--miss-probability", type=float, default=0.10)
    parser.add_argument("--false-alarm-rate", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-graph-best", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the remote execution policy")
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation = AssociationSimulationConfig(
        miss_probability=args.miss_probability,
        false_alarm_rate=args.false_alarm_rate,
        seed=args.seed,
    )
    train_dataset = AssociationFrameDataset(
        args.cache,
        "train",
        args.calibration,
        simulation,
        snr_values=(5, 10, 15, 20),
        max_samples=args.train_samples,
    )
    validation_dataset = AssociationFrameDataset(
        args.cache,
        "val",
        args.calibration,
        simulation,
        snr_values=(5, 10, 15, 20),
        max_samples=args.validation_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model_specs = {
        "edge_mlp": AssociationGraphConfig(hidden_dim=args.hidden_dim, graph_layers=0),
        "association_gnn": AssociationGraphConfig(hidden_dim=args.hidden_dim, graph_layers=2),
    }
    models = {}
    training = {}
    for name, config in model_specs.items():
        model = MeasurementTrackAssociationNet(config)
        training[name] = train_model(
            name,
            model,
            train_loader,
            validation_loader,
            device,
            output_dir,
            args.epochs,
            args.learning_rate,
            args.patience,
            simulation.max_measurements,
        )
        models[name] = model.to(device).eval()

    frame_results = {}
    for snr in (5, 10, 15, 20):
        test_dataset = AssociationFrameDataset(
            args.cache,
            "test",
            args.calibration,
            simulation,
            snr_values=(snr,),
            max_samples=args.test_samples_per_snr,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        frame_results[str(snr)] = {
            "gated_hungarian_ekf": evaluate_frames(
                None, test_loader, device, "gated_hungarian_ekf", simulation.max_measurements
            )
        }
        for name, model in models.items():
            frame_results[str(snr)][name] = evaluate_frames(
                model, test_loader, device, name, simulation.max_measurements
            )
        print(json.dumps({"snr_db": snr, "frame_results": frame_results[str(snr)]}), flush=True)

    with np.load(args.cache, allow_pickle=False) as cache:
        test_states = cache["test_states"].copy().astype(np.float32)
        test_masks = cache["test_mask"].copy().astype(np.bool_)
        metadata = json.loads(str(cache["metadata"].item()))
    simulation.dt = float(metadata["dt"])
    noise_map = load_snr_noise(args.calibration)
    recursive = {}
    for snr in (5, 10, 15, 20):
        recursive[str(snr)] = {}
        for method in ("oracle_association", "gated_hungarian_ekf", "edge_mlp", "association_gnn"):
            recursive[str(snr)][method] = evaluate_recursive_tracking(
                method,
                models.get(method),
                test_states,
                test_masks,
                noise_map,
                snr,
                simulation,
                device,
                args.sequence_scenes,
            )
            print(
                json.dumps(
                    {"snr_db": snr, "method": method, **recursive[str(snr)][method]}
                ),
                flush=True,
            )

    graph_best_frame = {
        snr: frame_results[snr]["association_gnn"]["association_f1"]
        > max(
            frame_results[snr]["gated_hungarian_ekf"]["association_f1"],
            frame_results[snr]["edge_mlp"]["association_f1"],
        )
        for snr in frame_results
    }
    graph_best_recursive = {
        snr: recursive[snr]["association_gnn"]["identity_f1"]
        > max(
            recursive[snr]["gated_hungarian_ekf"]["identity_f1"],
            recursive[snr]["edge_mlp"]["identity_f1"],
        )
        and recursive[snr]["association_gnn"]["id_switches"]
        < min(
            recursive[snr]["gated_hungarian_ekf"]["id_switches"],
            recursive[snr]["edge_mlp"]["id_switches"],
        )
        for snr in recursive
    }
    payload = {
        "experiment": "uncertainty_aware_measurement_track_bipartite_graph",
        "status": "initial_measurement-association_and_recursive-tracking_results",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "data": {
            "cache": args.cache,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "test_samples_per_snr": args.test_samples_per_snr,
            "recursive_test_scenes": args.sequence_scenes,
            "snr_db": [5, 10, 15, 20],
            "vehicle_id_used_as_model_input": False,
        },
        "simulation": vars(simulation),
        "model": {
            "node_types": ["predicted_track", "identity_free_fused_measurement"],
            "graph_layers": 2,
            "hidden_dim": args.hidden_dim,
            "matching": "gated neural edge probabilities plus one-to-one Hungarian decoding",
            "uncertainty_outputs": ["association_probability", "normalized_association_entropy"],
        },
        "training": training,
        "frame_association": frame_results,
        "recursive_tracking": recursive,
        "performance_gates": {
            "graph_best_frame_f1_at_each_snr": graph_best_frame,
            "graph_best_recursive_identity_and_switches_at_each_snr": graph_best_recursive,
        },
        "limitations": [
            "The current synchronized cache contains fixed target membership within each 40-frame window.",
            "Birth/death lifecycle labels require a later cache with targets entering and leaving inside a window.",
            "This stage uses fused state-space measurements rather than learning directly from raw RD/RA peak nodes.",
            "HOTA and GOSPA are deferred until the variable-membership tracking cache is added.",
        ],
    }
    result_path = output_dir / "measurement_track_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "recursive_tracking_summary.csv", payload)
    print(json.dumps({"result_path": str(result_path), "performance_gates": payload["performance_gates"]}), flush=True)
    if args.require_graph_best and not (all(graph_best_frame.values()) and all(graph_best_recursive.values())):
        raise SystemExit("Association GNN did not pass all requested performance gates")


if __name__ == "__main__":
    main()
