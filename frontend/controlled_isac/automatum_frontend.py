"""Route-B Automatum controlled ISAC frontend.

For every ground-truth vehicle of a canonical frame, generate independent 3-BS noisy polar
measurements (range / bearing / radial velocity) and fuse them into one state estimate:

    GT [x, y, vx, vy]
      -> per visible BS: noisy [r_hat, bearing_hat, vr_hat] -> [x_hat_bs, y_hat_bs]
      -> covariance-weighted position fusion (frontend.controlled_isac.cross_bs_association.fuse_positions)
      -> multi-BS radial-velocity fusion (frontend.controlled_isac.velocity_fusion.recover_velocity)
      -> [x_hat, y_hat, vx_hat, vy_hat]

Target detection and identity association are assumed successful: each vehicle keeps its
``vehicle_id``, which is an index only and never a model feature. No detector, no temporal
tracker, no Kalman filter, no per-target tuning.
"""
from __future__ import annotations

import math

import numpy as np

from .automatum_measurement import make_measurement
from .cross_bs_association import fuse_positions
from .velocity_fusion import recover_velocity

STATE_FIELDS = ("x_hat", "y_hat", "vx_hat", "vy_hat")


def sense_vehicle(scene_id: int, frame: int, vehicle_id: int, position, velocity, snr_db: float,
                  calibration: dict, setup: dict) -> dict:
    """Fuse all geometrically visible BS measurements of one vehicle at one SNR level."""
    measurements = []
    for bs_id in range(3):
        measurement = make_measurement(scene_id, frame, vehicle_id, position, velocity, bs_id,
                                       snr_db, calibration, setup)
        if measurement is not None:
            measurements.append(measurement)
    n_bs = len(measurements)
    if n_bs == 0:
        raise ValueError(f"vehicle {vehicle_id} of scene {scene_id} frame {frame} has no visible BS; "
                         "the geometry audit must report this instead of fabricating a state")
    if n_bs == 1:
        position_hat = np.asarray([measurements[0]["x_hat_bs"], measurements[0]["y_hat_bs"]],
                                  dtype=float)
        covariance = np.asarray(measurements[0]["position_covariance"], dtype=float)
    else:
        position_hat, covariance = fuse_positions([
            {"x": measurement["x_hat_bs"], "y": measurement["y_hat_bs"],
             "covariance": measurement["position_covariance"]}
            for measurement in measurements])

    height = float(setup["height"])
    members = []
    for measurement in measurements:
        delta = position_hat - np.asarray(measurement["station"], dtype=float)
        r_3d = math.sqrt(float(delta @ delta) + height ** 2)
        members.append({"station": measurement["station"],
                        "radial_velocity": measurement["vr_hat"],
                        "direction": delta / max(r_3d, 1e-9)})
    sigma_radial = measurements[0]["sigma"]["radial_velocity"]
    velocity = recover_velocity(position_hat, members, sigma_radial,
                                velocity_prior_sigma=float(calibration["velocity_prior_sigma_mps"]))
    return {
        "scene_id": int(scene_id), "frame": int(frame), "vehicle_id": int(vehicle_id),
        "snr_db": float(snr_db),
        "x_hat": float(position_hat[0]), "y_hat": float(position_hat[1]),
        "vx_hat": float(velocity["vx"]), "vy_hat": float(velocity["vy"]),
        "n_bs": n_bs, "rank": int(velocity["rank"]), "condition_ratio": float(velocity["condition_ratio"]),
        "position_covariance": covariance, "per_bs_measurements": measurements,
        "sigma": measurements[0]["sigma"],
    }


def sense_frame(scene_frame, scene_id: int, snr_db: float, calibration: dict, setup: dict) -> list[dict]:
    """One fused state per ground-truth vehicle of the frame (no vehicle is dropped)."""
    states = []
    for vehicle_id, state in zip(scene_frame.vehicle_ids.tolist(), scene_frame.states):
        states.append(sense_vehicle(scene_id, int(scene_frame.frame), int(vehicle_id),
                                    state[:2], state[2:], snr_db, calibration, setup))
    return states