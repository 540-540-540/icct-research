"""Online unknown-measurement simulation for measurement-to-track association.

The synchronized multi-target cache supplies ground-truth states and vehicle IDs.
IDs are used only to construct supervision/evaluation labels and are never exposed
in node or edge features.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


TRACK_DIM = 11
MEASUREMENT_DIM = 14
EDGE_DIM = 10


@dataclass
class AssociationSimulationConfig:
    max_tracks: int = 8
    max_measurements: int = 12
    dt: float = 0.1
    miss_probability: float = 0.10
    false_alarm_rate: float = 1.5
    process_position_sigma_m: float = 0.45
    process_velocity_sigma_mps: float = 0.30
    gate_mahalanobis_sq: float = 25.0
    gate_distance_m: float = 12.0
    # Stabilized Hungarian-EKF settings.  A covariance floor prevents the
    # filter from becoming unrealistically overconfident after several clean
    # updates.  Tracks that remain unmatched are widened so that they can be
    # reacquired without using ground-truth identity.
    hungarian_rejection_mahalanobis_sq: float = 16.0
    covariance_floor_position_m: float = 0.50
    covariance_floor_velocity_mps: float = 0.25
    reinitialize_after_misses: int = 3
    reinitialization_position_sigma_m: float = 3.0
    reinitialization_velocity_sigma_mps: float = 1.5
    seed: int = 2026


def load_snr_noise(calibration_path: str, velocity_sigma_20db: float = 0.20) -> Dict[int, Tuple[float, float]]:
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    position = {
        int(key): float(value["position_per_axis_rmse_m"])
        for key, value in payload["calibration"].items()
    }
    reference = position[20]
    return {
        snr: (sigma, velocity_sigma_20db * sigma / reference)
        for snr, sigma in sorted(position.items())
    }


def _visibility(rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    number = int(rng.choice([1, 2, 3], p=[0.08, 0.27, 0.65]))
    selected = rng.choice(3, size=number, replace=False)
    mask = np.zeros(3, dtype=np.float32)
    mask[selected] = 1.0
    return number, mask


def simulate_measurements(
    truth_state: np.ndarray,
    valid_mask: np.ndarray,
    snr_db: int,
    noise_map: Dict[int, Tuple[float, float]],
    config: AssociationSimulationConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Create a shuffled, identity-free measurement set for one frame."""
    position_sigma, velocity_sigma = noise_map[int(snr_db)]
    entries = []
    valid_indices = np.flatnonzero(valid_mask)
    for target in valid_indices:
        if rng.random() < config.miss_probability:
            continue
        bs_count, bs_mask = _visibility(rng)
        # More BS support modestly lowers effective covariance.
        support_scale = np.sqrt(3.0 / float(bs_count))
        pos_sigma = position_sigma * support_scale
        vel_sigma = velocity_sigma * support_scale
        covariance = np.asarray(
            [pos_sigma**2, pos_sigma**2, vel_sigma**2, vel_sigma**2], dtype=np.float32
        )
        noise = rng.normal(0.0, np.sqrt(covariance)).astype(np.float32)
        measured = truth_state[target, :4].astype(np.float32) + noise
        normalized_noise = float(np.sqrt(np.mean((noise**2) / np.maximum(covariance, 1.0e-8))))
        confidence = float(np.clip(np.exp(-0.35 * normalized_noise), 0.15, 0.99))
        entries.append((measured, covariance, confidence, bs_count, bs_mask, int(target)))

    number_false = min(
        int(rng.poisson(config.false_alarm_rate)),
        max(0, config.max_measurements - len(entries)),
    )
    valid_truth = truth_state[valid_indices, :4]
    if len(valid_truth):
        lower_position = valid_truth[:, :2].min(axis=0) - 12.0
        upper_position = valid_truth[:, :2].max(axis=0) + 12.0
        lower_velocity = valid_truth[:, 2:4].min(axis=0) - 4.0
        upper_velocity = valid_truth[:, 2:4].max(axis=0) + 4.0
    else:
        lower_position = np.asarray([-50.0, -50.0])
        upper_position = np.asarray([50.0, 50.0])
        lower_velocity = np.asarray([-15.0, -15.0])
        upper_velocity = np.asarray([15.0, 15.0])
    for _ in range(number_false):
        measured = np.concatenate(
            [
                rng.uniform(lower_position, upper_position),
                rng.uniform(lower_velocity, upper_velocity),
            ]
        ).astype(np.float32)
        covariance = np.asarray(
            [position_sigma**2 * 2.25] * 2 + [velocity_sigma**2 * 2.25] * 2,
            dtype=np.float32,
        )
        bs_count, bs_mask = _visibility(rng)
        confidence = float(rng.uniform(0.08, 0.55))
        entries.append((measured, covariance, confidence, bs_count, bs_mask, -1))

    if entries:
        order = rng.permutation(len(entries))[: config.max_measurements]
        entries = [entries[int(index)] for index in order]
    state = np.stack([entry[0] for entry in entries]) if entries else np.zeros((0, 4), np.float32)
    covariance = np.stack([entry[1] for entry in entries]) if entries else np.zeros((0, 4), np.float32)
    confidence = np.asarray([entry[2] for entry in entries], dtype=np.float32)
    bs_count = np.asarray([entry[3] for entry in entries], dtype=np.float32)
    bs_mask = np.stack([entry[4] for entry in entries]) if entries else np.zeros((0, 3), np.float32)
    owner = np.asarray([entry[5] for entry in entries], dtype=np.int64)
    return {
        "state": state,
        "covariance": covariance,
        "confidence": confidence,
        "bs_count": bs_count,
        "bs_mask": bs_mask,
        "owner": owner,
    }


def build_association_features(
    track_state: np.ndarray,
    track_covariance: np.ndarray,
    track_mask: np.ndarray,
    track_age: np.ndarray,
    miss_count: np.ndarray,
    track_confidence: np.ndarray,
    measurement: Dict[str, np.ndarray],
    snr_db: int,
    config: AssociationSimulationConfig,
) -> Dict[str, np.ndarray]:
    """Build padded track, measurement, and candidate-edge tensors."""
    max_tracks = config.max_tracks
    max_measurements = config.max_measurements
    number_measurements = min(len(measurement["state"]), max_measurements)
    measurement_mask = np.zeros(max_measurements, dtype=np.bool_)
    measurement_mask[:number_measurements] = True

    tracks = np.zeros((max_tracks, TRACK_DIM), dtype=np.float32)
    tracks[:, :4] = track_state / np.asarray([50.0, 50.0, 20.0, 20.0], dtype=np.float32)
    tracks[:, 4:8] = np.log1p(np.maximum(track_covariance, 0.0)) / 4.0
    tracks[:, 8] = np.clip(track_age / 20.0, 0.0, 2.0)
    tracks[:, 9] = np.clip(miss_count / 5.0, 0.0, 2.0)
    tracks[:, 10] = track_confidence

    measurements = np.zeros((max_measurements, MEASUREMENT_DIM), dtype=np.float32)
    if number_measurements:
        measurements[:number_measurements, :4] = measurement["state"][:number_measurements] / np.asarray(
            [50.0, 50.0, 20.0, 20.0], dtype=np.float32
        )
        measurements[:number_measurements, 4:8] = (
            np.log1p(np.maximum(measurement["covariance"][:number_measurements], 0.0)) / 4.0
        )
        measurements[:number_measurements, 8] = float(snr_db) / 20.0
        measurements[:number_measurements, 9] = measurement["confidence"][:number_measurements]
        measurements[:number_measurements, 10] = measurement["bs_count"][:number_measurements] / 3.0
        measurements[:number_measurements, 11:14] = measurement["bs_mask"][:number_measurements]

    edges = np.zeros((max_tracks, max_measurements, EDGE_DIM), dtype=np.float32)
    candidate = np.zeros((max_tracks, max_measurements), dtype=np.bool_)
    if number_measurements:
        innovation = measurement["state"][None, :number_measurements] - track_state[:, None, :]
        innovation_covariance = (
            track_covariance[:, None, :] + measurement["covariance"][None, :number_measurements, :]
        )
        mahalanobis = np.sum(innovation**2 / np.maximum(innovation_covariance, 1.0e-6), axis=-1)
        distance = np.linalg.norm(innovation[..., :2], axis=-1)
        speed_residual = np.linalg.norm(innovation[..., 2:4], axis=-1)
        edges[:, :number_measurements, 0:4] = innovation / np.asarray(
            [10.0, 10.0, 5.0, 5.0], dtype=np.float32
        )
        edges[:, :number_measurements, 4] = distance / 15.0
        edges[:, :number_measurements, 5] = np.minimum(mahalanobis, 100.0) / 10.0
        edges[:, :number_measurements, 6] = speed_residual / 5.0
        edges[:, :number_measurements, 7] = (
            np.log1p(np.mean(innovation_covariance, axis=-1)) / 4.0
        )
        edges[:, :number_measurements, 8] = float(snr_db) / 20.0
        edges[:, :number_measurements, 9] = measurement["confidence"][None, :number_measurements]
        candidate[:, :number_measurements] = (
            (mahalanobis <= config.gate_mahalanobis_sq)
            | (distance <= config.gate_distance_m)
        )
    candidate &= track_mask[:, None] & measurement_mask[None, :]

    labels = np.full(max_tracks, -100, dtype=np.int64)
    owners = measurement["owner"][:number_measurements]
    for track in np.flatnonzero(track_mask):
        match = np.flatnonzero(owners == int(track))
        if len(match) and candidate[track, int(match[0])]:
            labels[track] = int(match[0])
        else:
            labels[track] = max_measurements
    owner_padded = np.full(max_measurements, -2, dtype=np.int64)
    owner_padded[:number_measurements] = owners
    return {
        "track_features": tracks,
        "measurement_features": measurements,
        "edge_features": edges,
        "candidate_mask": candidate,
        "track_mask": track_mask.astype(np.bool_),
        "measurement_mask": measurement_mask,
        "labels": labels,
        "measurement_owner": owner_padded,
    }


class AssociationFrameDataset(Dataset):
    """Deterministic online frame samples derived from synchronized scenes."""

    def __init__(
        self,
        cache_path: str,
        split: str,
        calibration_path: str,
        config: AssociationSimulationConfig,
        snr_values: Sequence[int] = (5, 10, 15, 20),
        max_samples: Optional[int] = None,
    ) -> None:
        with np.load(cache_path, allow_pickle=False) as cache:
            self.states = cache[f"{split}_states"].copy().astype(np.float32)
            self.mask = cache[f"{split}_mask"].copy().astype(np.bool_)
            self.target_ids = cache[f"{split}_target_ids"].copy().astype(np.int64)
            metadata = json.loads(str(cache["metadata"].item()))
        self.history_length = int(metadata["history_length"])
        self.config = config
        self.config.dt = float(metadata["dt"])
        self.noise_map = load_snr_noise(calibration_path)
        self.snr_values = tuple(int(value) for value in snr_values)
        available = len(self.states) * (self.history_length - 1)
        self.sample_count = available if max_samples is None else min(available, int(max_samples))
        self.split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        frame_count = self.history_length - 1
        scene_index = index // frame_count
        frame = index % frame_count + 1
        rng = np.random.default_rng(self.config.seed + self.split_offset + index * 37)
        snr_db = self.snr_values[index % len(self.snr_values)]
        valid = self.mask[scene_index]
        previous = self.states[scene_index, frame - 1]
        current = self.states[scene_index, frame]
        track_state = previous.copy()
        track_state[:, :2] += previous[:, 2:4] * self.config.dt
        track_state[:, :2] += rng.normal(
            0.0, self.config.process_position_sigma_m, size=track_state[:, :2].shape
        ).astype(np.float32)
        track_state[:, 2:4] += rng.normal(
            0.0, self.config.process_velocity_sigma_mps, size=track_state[:, 2:4].shape
        ).astype(np.float32)
        track_covariance = np.tile(
            np.asarray(
                [
                    self.config.process_position_sigma_m**2,
                    self.config.process_position_sigma_m**2,
                    self.config.process_velocity_sigma_mps**2,
                    self.config.process_velocity_sigma_mps**2,
                ],
                dtype=np.float32,
            ),
            (self.config.max_tracks, 1),
        )
        measurement = simulate_measurements(
            current,
            valid,
            snr_db,
            self.noise_map,
            self.config,
            rng,
        )
        features = build_association_features(
            track_state=track_state,
            track_covariance=track_covariance,
            track_mask=valid,
            track_age=np.full(self.config.max_tracks, frame, dtype=np.float32),
            miss_count=np.zeros(self.config.max_tracks, dtype=np.float32),
            track_confidence=np.ones(self.config.max_tracks, dtype=np.float32),
            measurement=measurement,
            snr_db=snr_db,
            config=self.config,
        )
        output = {key: torch.from_numpy(value) for key, value in features.items()}
        output["snr_db"] = torch.tensor(snr_db, dtype=torch.int64)
        output["scene_index"] = torch.tensor(scene_index, dtype=torch.int64)
        output["frame_index"] = torch.tensor(frame, dtype=torch.int64)
        return output
