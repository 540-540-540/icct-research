"""Synthetic calibration of tracker process noise q_a (SENS-REBUILD-03B section 18).

Uses only synthetic constant-velocity and mild-acceleration trajectories with detector-derived
measurement covariance. No downstream metric and no dataset is involved.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/p4_tracker/tracker_calibration.json"
CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
FRAMES = 80
WARMUP = 10
DT_NS = 100_000_000


def representative_covariance(config: dict) -> np.ndarray:
    from frontend.sensing import coords

    lut = _common.load_lut(ROOT)
    if lut is None:
        raise SystemExit("covariance LUT missing; run calibrate_covariance first")
    covariance = coords.covariance_from_lut(150.0, math.sin(math.radians(10.0)), 25.0, 0.0, lut,
                                            config["detector"]["covariance_floor_m2"])
    return covariance.numpy()


def make_observations(truth: list[tuple[float, float]], covariance: np.ndarray, seed: int) -> list[dict]:
    generator = np.random.default_rng(seed)
    observations = []
    for index, position in enumerate(truth):
        noise = generator.multivariate_normal(np.zeros(2), covariance)
        observations.append({
            "time_ns": index * DT_NS,
            "x_m": float(position[0] + noise[0]),
            "y_m": float(position[1] + noise[1]),
            "C_xy": covariance.tolist(),
            "bs_mask": 0b011,
            "n_bs": 2,
            "quality_db": 25.0,
        })
    return observations


def trajectory(kind: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Analytic synthetic truth: positions and velocities are integrated together."""
    position = np.array([-20.0, 0.0])
    velocity = np.array([12.0, -6.0])
    positions, velocities = [], []
    for frame in range(FRAMES):
        positions.append(tuple(position.tolist()))
        velocities.append(tuple(velocity.tolist()))
        if kind == "cv":
            acceleration = np.zeros(2)
        elif frame < 26:
            acceleration = np.array([1.2, -0.8])
        elif frame < 51:
            acceleration = np.array([-1.0, 0.6])
        else:
            acceleration = np.zeros(2)
        position = position + velocity * 0.1 + 0.5 * acceleration * 0.1 ** 2
        velocity = velocity + acceleration * 0.1
    return positions, velocities


def run_candidate(q_a: float, observations_by_kind: dict, truth_by_kind: dict, config: dict,
                  covariance: np.ndarray) -> dict:
    from frontend.tracking.cv_kf import CvKalmanTracker

    local = copy.deepcopy(config)
    local["tracker"]["q_a_m2_s3"] = q_a
    local["tracker"]["confirm_hits"] = 1
    local["tracker"]["confirm_requires_nbs2"] = False
    metrics = {}
    for kind, observations in observations_by_kind.items():
        tracker = CvKalmanTracker(local)
        truth_points = truth_by_kind[kind]["positions"]
        truth_velocity = truth_by_kind[kind]["velocities"]
        position_errors, velocity_errors = [], []
        keys = set()
        for frame, observation in enumerate(observations):
            records = tracker.step([observation], observation["time_ns"])
            if not records:
                continue
            truth_position = truth_points[frame]
            record = min(records, key=lambda entry: math.hypot(entry["state_hat"][0] - truth_position[0],
                                                               entry["state_hat"][1] - truth_position[1]))
            keys.add(record["track_key"])
            if frame >= WARMUP:
                position_errors.append(math.hypot(record["state_hat"][0] - truth_position[0],
                                                  record["state_hat"][1] - truth_position[1]))
                velocity_errors.append(np.array(record["state_hat"][2:]) - np.array(truth_velocity[frame]))
        metrics[kind] = {
            "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None,
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(velocity_errors)))))
            if velocity_errors else None,
            "track_keys": sorted(keys),
            "finite": all(math.isfinite(value) for value in (position_errors + [0.0])),
        }
    return metrics


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("calibration requires CUDA only for the LUT-derived covariance")
    config = _common.load_frontend_config()
    covariance = representative_covariance(config)
    cv_points, cv_velocities = trajectory("cv")
    ca_points, ca_velocities = trajectory("ca")
    observations_by_kind = {"cv": make_observations(cv_points, covariance, 2026),
                            "ca": make_observations(ca_points, covariance, 2027)}
    truth_by_kind = {"cv": {"positions": cv_points, "velocities": cv_velocities},
                     "ca": {"positions": ca_points, "velocities": ca_velocities}}
    table = {}
    for candidate in CANDIDATES:
        table[str(candidate)] = run_candidate(candidate, observations_by_kind, truth_by_kind, config, covariance)

    def score(entry: dict) -> float:
        return (entry["ca"]["position_rmse_m"] + entry["ca"]["velocity_rmse_mps"]
                + entry["cv"]["velocity_rmse_mps"])

    chosen = min(CANDIDATES, key=lambda candidate: score(table[str(candidate)]))
    updated = copy.deepcopy(config)
    updated["tracker"]["q_a_m2_s3"] = chosen
    _common.write_json(ROOT / "configs/shared_frontend.json", updated)
    report = {"method": "synthetic CV + mild-CA trajectories at 10 Hz with analytic truth positions/velocities "
                        "and LUT-derived measurement covariance; q_a selected by minimizing "
                        "(CA position RMSE + CA velocity RMSE + CV velocity RMSE); no downstream metric used",
              "candidates_m2_s3": CANDIDATES, "chosen_q_a_m2_s3": chosen, "table": table,
              "measurement_covariance": covariance.tolist(), "frames": FRAMES, "warmup_frames": WARMUP,
              "config_hash": hashlib.sha256((ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest()}
    _common.write_json(OUT, report)
    print(json.dumps({"chosen_q_a_m2_s3": chosen, "table": table}, ensure_ascii=False))


if __name__ == "__main__":
    main()