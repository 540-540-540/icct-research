"""Multi-target scene construction for the Lankershim NGSIM data set.

This module deliberately keeps a common physical time axis.  It does not use the
single-vehicle samples produced by ``gen_lankershim_data_v2.py`` because those
samples lose the simultaneity needed by an interaction graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEET_TO_METERS = 0.3048


def _vehicle_record(times: np.ndarray, positions: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return time indices and [x, y, vx, vy] without differentiating across gaps."""
    velocity = np.zeros_like(positions, dtype=np.float32)
    if len(times) > 1:
        contiguous = np.diff(times) == 1
        delta = np.diff(positions, axis=0) / dt
        velocity[1:][contiguous] = delta[contiguous]
        velocity[:-1][contiguous] = 0.5 * (velocity[:-1][contiguous] + delta[contiguous])

        # Fill segment boundaries from their only valid one-sided derivative.
        starts = np.r_[True, ~contiguous]
        ends = np.r_[~contiguous, True]
        for index in np.flatnonzero(starts):
            if index + 1 < len(times) and times[index + 1] == times[index] + 1:
                velocity[index] = (positions[index + 1] - positions[index]) / dt
        for index in np.flatnonzero(ends):
            if index > 0 and times[index] == times[index - 1] + 1:
                velocity[index] = (positions[index] - positions[index - 1]) / dt
    states = np.concatenate([positions.astype(np.float32), velocity], axis=-1)
    return times.astype(np.int32), states


def _extract_contiguous(record: Tuple[np.ndarray, np.ndarray], start: int, length: int) -> Optional[np.ndarray]:
    times, states = record
    offset = int(np.searchsorted(times, start))
    stop = offset + length
    if stop > len(times):
        return None
    selected = times[offset:stop]
    if selected[0] != start or selected[-1] != start + length - 1:
        return None
    if not np.all(np.diff(selected) == 1):
        return None
    return states[offset:stop]


def _to_anchor_frame(states: np.ndarray, history_length: int) -> np.ndarray:
    """Translate and rotate a scene into target 0's last-history coordinate frame."""
    output = states.copy()
    origin = output[history_length - 1, 0, :2].copy()
    anchor_velocity = output[history_length - 1, 0, 2:4]
    heading = float(np.arctan2(anchor_velocity[1], anchor_velocity[0]))
    if float(np.linalg.norm(anchor_velocity)) < 0.5:
        heading = 0.0
    cosine, sine = np.cos(heading), np.sin(heading)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    output[..., :2] = (output[..., :2] - origin) @ rotation.T
    output[..., 2:4] = output[..., 2:4] @ rotation.T
    return output


def _split_bounds(number_of_times: int) -> Dict[str, Tuple[int, int]]:
    train_end = int(number_of_times * 0.70)
    validation_end = int(number_of_times * 0.85)
    return {
        "train": (0, train_end),
        "val": (train_end, validation_end),
        "test": (validation_end, number_of_times),
    }


def build_scene_cache(
    csv_path: str,
    output_path: str,
    history_length: int = 20,
    prediction_length: int = 20,
    max_targets: int = 8,
    min_targets: int = 2,
    radius_m: float = 45.0,
    stride: int = 3,
    scenes_per_time: int = 2,
    max_train_scenes: int = 6000,
    max_val_scenes: int = 1200,
    max_test_scenes: int = 1200,
    seed: int = 2026,
) -> Dict[str, int]:
    """Build a leakage-safe cache of simultaneous multi-vehicle scenes."""
    columns = ["Vehicle_ID", "Global_Time", "Local_X", "Local_Y"]
    frame = pd.read_csv(csv_path, usecols=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().drop_duplicates(["Vehicle_ID", "Global_Time"], keep="first")
    frame = frame.sort_values(["Global_Time", "Vehicle_ID"], kind="mergesort").reset_index(drop=True)
    frame["Vehicle_ID"] = frame["Vehicle_ID"].astype(np.int64)

    unique_times = np.sort(frame["Global_Time"].unique())
    if len(unique_times) < history_length + prediction_length:
        raise ValueError("The CSV is too short for the requested history/prediction window.")
    native_delta_ms = float(np.median(np.diff(unique_times)))
    dt = native_delta_ms / 1000.0
    time_index = pd.Series(np.arange(len(unique_times), dtype=np.int32), index=unique_times)
    frame["time_index"] = frame["Global_Time"].map(time_index).astype(np.int32)
    frame["x_m"] = frame["Local_X"].astype(np.float32) * FEET_TO_METERS
    frame["y_m"] = frame["Local_Y"].astype(np.float32) * FEET_TO_METERS

    records: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for vehicle_id, vehicle_frame in frame.groupby("Vehicle_ID", sort=False):
        records[int(vehicle_id)] = _vehicle_record(
            vehicle_frame["time_index"].to_numpy(np.int32),
            vehicle_frame[["x_m", "y_m"]].to_numpy(np.float32),
            dt,
        )

    frames: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for index, time_frame in frame.groupby("time_index", sort=False):
        frames[int(index)] = (
            time_frame["Vehicle_ID"].to_numpy(np.int64),
            time_frame[["x_m", "y_m"]].to_numpy(np.float32),
        )

    rng = np.random.default_rng(seed)
    total_length = history_length + prediction_length
    limits = {"train": max_train_scenes, "val": max_val_scenes, "test": max_test_scenes}
    arrays: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    for split, (lower, upper) in _split_bounds(len(unique_times)).items():
        split_states = []
        split_masks = []
        split_ids = []
        split_starts = []
        signatures = set()
        candidate_starts = np.arange(lower, upper - total_length + 1, stride, dtype=np.int32)
        rng.shuffle(candidate_starts)

        for start_value in candidate_starts:
            start = int(start_value)
            last_history = start + history_length - 1
            vehicle_ids, positions = frames[last_history]
            eligible = []
            for vehicle_id, position in zip(vehicle_ids, positions):
                trajectory = _extract_contiguous(records[int(vehicle_id)], start, total_length)
                if trajectory is not None:
                    eligible.append((int(vehicle_id), position, trajectory))
            if len(eligible) < min_targets:
                continue

            anchor_order = rng.permutation(len(eligible))[:scenes_per_time]
            for anchor_index in anchor_order:
                anchor_id, anchor_position, _ = eligible[int(anchor_index)]
                neighbors = []
                for vehicle_id, position, trajectory in eligible:
                    distance = float(np.linalg.norm(position - anchor_position))
                    if distance <= radius_m:
                        neighbors.append((distance, vehicle_id, trajectory))
                neighbors.sort(key=lambda item: (0 if item[1] == anchor_id else 1, item[0]))
                selected = neighbors[:max_targets]
                if len(selected) < min_targets:
                    continue
                signature = (start, tuple(sorted(item[1] for item in selected)))
                if signature in signatures:
                    continue
                signatures.add(signature)

                state = np.zeros((total_length, max_targets, 4), dtype=np.float32)
                mask = np.zeros(max_targets, dtype=np.bool_)
                ids = np.full(max_targets, -1, dtype=np.int64)
                # target 0 is always the anchor because of the sort key above
                for target_index, (_, vehicle_id, trajectory) in enumerate(selected):
                    state[:, target_index] = trajectory
                    mask[target_index] = True
                    ids[target_index] = vehicle_id
                split_states.append(_to_anchor_frame(state, history_length))
                split_masks.append(mask)
                split_ids.append(ids)
                split_starts.append(start)
                if len(split_states) >= limits[split]:
                    break
            if len(split_states) >= limits[split]:
                break

        if not split_states:
            raise RuntimeError("No scenes were constructed for split %s." % split)
        arrays[split + "_states"] = np.stack(split_states)
        arrays[split + "_mask"] = np.stack(split_masks)
        arrays[split + "_target_ids"] = np.stack(split_ids)
        arrays[split + "_start_index"] = np.asarray(split_starts, dtype=np.int32)
        counts[split] = len(split_states)

    metadata = {
        "source_csv": str(csv_path),
        "history_length": history_length,
        "prediction_length": prediction_length,
        "max_targets": max_targets,
        "min_targets": min_targets,
        "radius_m": radius_m,
        "stride": stride,
        "scenes_per_time": scenes_per_time,
        "dt": dt,
        "native_delta_ms": native_delta_ms,
        "seed": seed,
        "temporal_split": [0.70, 0.15, 0.15],
        "counts": counts,
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return counts


class MultiTargetSceneDataset(Dataset):
    """Thin tensor dataset backed by a cached ``npz`` split."""

    def __init__(self, cache_path: str, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        with np.load(cache_path, allow_pickle=False) as cache:
            self.states = torch.from_numpy(cache[split + "_states"].copy()).float()
            self.mask = torch.from_numpy(cache[split + "_mask"].copy()).bool()
            self.target_ids = torch.from_numpy(cache[split + "_target_ids"].copy()).long()
            self.start_index = torch.from_numpy(cache[split + "_start_index"].copy()).long()
            self.metadata = json.loads(str(cache["metadata"].item()))
        self.history_length = int(self.metadata["history_length"])

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        state = self.states[index]
        return {
            "history": state[: self.history_length],
            "future": state[self.history_length :],
            "mask": self.mask[index],
            "target_ids": self.target_ids[index],
            "start_index": self.start_index[index],
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build multi-target NGSIM scene cache")
    parser.add_argument("--csv", default="Lankershim_Vehicle_Trajectories.csv")
    parser.add_argument("--output", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--prediction-length", type=int, default=20)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--min-targets", type=int, default=2)
    parser.add_argument("--radius-m", type=float, default=45.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--scenes-per-time", type=int, default=2)
    parser.add_argument("--max-train-scenes", type=int, default=6000)
    parser.add_argument("--max-val-scenes", type=int, default=1200)
    parser.add_argument("--max-test-scenes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    counts = build_scene_cache(
        csv_path=args.csv,
        output_path=args.output,
        history_length=args.history_length,
        prediction_length=args.prediction_length,
        max_targets=args.max_targets,
        min_targets=args.min_targets,
        radius_m=args.radius_m,
        stride=args.stride,
        scenes_per_time=args.scenes_per_time,
        max_train_scenes=args.max_train_scenes,
        max_val_scenes=args.max_val_scenes,
        max_test_scenes=args.max_test_scenes,
        seed=args.seed,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
