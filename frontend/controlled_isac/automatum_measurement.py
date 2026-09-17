"""Route-B Automatum controlled measurement model: one vehicle, one BS, one SNR level.

Reuses the verified polar mathematics of ``frontend/controlled_isac/measurement.py``
(``truth_measurement``, ``visible``, ``polar_to_xy``, ``xy_covariance``): planar bearing from
boresight, 3D range ``r = sqrt(rho^2 + dh^2)`` and toward-station-positive radial velocity.

Noise model (V1): zero-mean Gaussian, independent across frame / BS / vehicle / channel.
The base standard normal of every ``(scene_id, frame, bs_id, vehicle_id, channel)`` is drawn
from a deterministic seed that does NOT contain the SNR, so all SNR levels reuse the same
realisation and only the error standard deviation changes:

    sigma_q(gamma) = sqrt(floor_q^2 + (a_q * 10^(-gamma/20))^2)

This module never reads ground-truth identities from the dataset and never calls a detector.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .measurement import polar_to_xy, truth_measurement, visible, wrap_angle, xy_covariance

CHANNELS = ("range", "bearing", "radial_velocity")
SEED_BASE = 2026


def setup_from_config(config: dict) -> dict:
    return {
        "scenes": {
            int(scene["scene_id"]): {
                "stations": np.asarray(scene["stations_xy_m"], dtype=float),
                "boresights": np.radians(np.asarray(scene["boresights_deg"], dtype=float)),
            }
            for scene in config["scenes"]
        },
        "height": float(config["visibility"]["height_difference_m"]),
        "range_min": float(config["visibility"]["range_min_m"]),
        "range_max": float(config["visibility"]["range_max_m"]),
        "half_angle_rad": math.radians(float(config["visibility"]["fov_half_angle_deg"])),
    }


def calibration_from_config(config: dict) -> dict:
    measurement = config["measurement"]
    return {
        "a": {channel: float(measurement["a"][channel]) for channel in CHANNELS},
        "floors": {channel: float(measurement["floors"][channel]) for channel in CHANNELS},
        "covariance_floor_m2": float(measurement.get("covariance_floor_m2", 1e-8)),
        "velocity_prior_sigma_mps": float(config["fusion"]["velocity_prior_sigma_mps"]),
    }


@lru_cache(maxsize=None)
def base_noise(scene_id: int, frame: int, bs_id: int, vehicle_id: int, channel: str) -> float:
    """SNR-independent standard normal for one measurement channel (seed excludes SNR)."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel}")
    sequence = np.random.SeedSequence([SEED_BASE, int(scene_id), int(frame), int(bs_id),
                                       int(vehicle_id), CHANNELS.index(channel)])
    return float(np.random.default_rng(sequence).standard_normal())


def sigma_for(channel: str, snr_db: float, calibration: dict) -> float:
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel}")
    a_q = float(calibration["a"][channel])
    floor_q = float(calibration["floors"][channel])
    if a_q < 0 or floor_q < 0:
        raise ValueError("calibration scales must be non-negative")
    scaled = a_q * 10 ** (-float(snr_db) / 20.0)
    return math.sqrt(floor_q ** 2 + scaled ** 2)


def make_measurement(scene_id: int, frame: int, vehicle_id: int, position, velocity, bs_id: int,
                     snr_db: float, calibration: dict, setup: dict) -> dict | None:
    """One noisy BS polar measurement, or None when the BS does not geometrically see target."""
    scene = setup["scenes"][int(scene_id)]
    station = scene["stations"][int(bs_id)]
    boresight = float(scene["boresights"][int(bs_id)])
    truth = truth_measurement(position, velocity, station, boresight, setup["height"])
    if not visible(truth, setup):
        return None
    sigma = {channel: sigma_for(channel, snr_db, calibration) for channel in CHANNELS}
    z = {channel: base_noise(scene_id, frame, bs_id, vehicle_id, channel) for channel in CHANNELS}
    r_hat = truth["r"] + sigma["range"] * z["range"]
    bearing_hat = wrap_angle(truth["bearing"] + sigma["bearing"] * z["bearing"])
    radial_hat = truth["radial_velocity"] + sigma["radial_velocity"] * z["radial_velocity"]
    if r_hat <= setup["height"]:
        r_hat = setup["height"] + 1e-6
    x_hat, y_hat = polar_to_xy(r_hat, bearing_hat, station, boresight, setup["height"])
    covariance = xy_covariance(r_hat, bearing_hat, sigma["range"], sigma["bearing"], station,
                               boresight, setup["height"], calibration["covariance_floor_m2"])
    return {
        "vehicle_id": int(vehicle_id), "bs_id": int(bs_id), "station": station,
        "r_true": float(truth["r"]), "bearing_true": float(truth["bearing"]),
        "vr_true": float(truth["radial_velocity"]),
        "r_hat": float(r_hat), "bearing_hat": float(bearing_hat), "vr_hat": float(radial_hat),
        "x_hat_bs": float(x_hat), "y_hat_bs": float(y_hat),
        "sigma": sigma, "base_noise": z, "position_covariance": covariance,
    }


def selfcheck() -> dict:
    config = {
        "scenes": [{"scene_id": 0, "stations_xy_m": [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]],
                    "boresights_deg": [0.0, 180.0, -90.0]}],
        "visibility": {"height_difference_m": 5.0, "range_min_m": 5.0, "range_max_m": 150.0,
                       "fov_half_angle_deg": 70.0},
        "measurement": {"a": {"range": 0.1, "bearing": 0.002, "radial_velocity": 0.1},
                        "floors": {"range": 0.0, "bearing": 0.0, "radial_velocity": 0.0}},
        "fusion": {"velocity_prior_sigma_mps": 200.0},
    }
    setup = setup_from_config(config)
    calibration = calibration_from_config(config)
    low = make_measurement(0, 0, 1, [20.0, 0.0], [-5.0, 0.0], 0, -10.0, calibration, setup)
    high = make_measurement(0, 0, 1, [20.0, 0.0], [-5.0, 0.0], 0, 10.0, calibration, setup)
    if low["base_noise"] != high["base_noise"]:
        raise SystemExit("base noise depends on SNR")
    ratio = ((low["r_hat"] - low["r_true"]) / low["sigma"]["range"]) / \
        ((high["r_hat"] - high["r_true"]) / high["sigma"]["range"])
    assert abs(ratio - 1.0) < 1e-12
    assert sigma_for("range", -10.0, calibration) > sigma_for("range", 10.0, calibration)
    print({"base_noise": low["base_noise"], "sigma_ratio": sigma_for("range", -10.0, calibration)
           / sigma_for("range", 10.0, calibration)})
    return {"ok": True}


if __name__ == "__main__":
    print(selfcheck())