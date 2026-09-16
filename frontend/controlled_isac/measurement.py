"""Controlled measurement model: frozen BS geometry, polar truth, calibrated noise injection.

Truth polar conventions replicate the frozen production formulas (``frontend/sensing/coords.py``
and ``simulator.py``): planar rho/bearing, 3D range ``r = sqrt(rho^2 + height^2)``, bearing from
boresight, and radial velocity ``dot(station - position, velocity) / r`` (toward-station positive).
The measurement noise model is (range, bearing, radial velocity) with a fixed base standard normal
per (episode, frame, BS, vehicle) shared by all SNR channels and only rescaled per SNR.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIGMA_KEYS = ("range", "bearing", "radial_velocity")


def load_setup(root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    config = json.loads((root / "configs/shared_frontend.json").read_text())
    geometry = json.loads((root / "reports/f01a/geometry.json").read_text())
    config_height = float(config["visibility"]["height_difference_m"])
    geometry_height = float(geometry["station_height_m"] - geometry["vehicle_height_m"])
    if config_height != geometry_height:
        raise ValueError("height mismatch between production config and A01 geometry")
    if (list(config["visibility"]["range_m"])
            != [float(geometry["range_m"][0]), float(geometry["range_m"][1])]):
        raise ValueError("range gate mismatch between production config and A01 geometry")
    if float(config["visibility"]["half_angle_deg"]) != float(geometry["fov_half_angle_deg"]):
        raise ValueError("FOV mismatch between production config and A01 geometry")
    return {
        "stations": np.asarray(geometry["stations_xy_m"], dtype=float),
        "boresights": np.asarray(geometry["boresight_rad"], dtype=float),
        "height": geometry_height,
        "range_min": float(config["visibility"]["range_m"][0]),
        "range_max": float(config["visibility"]["range_m"][1]),
        "half_angle_rad": math.radians(float(config["visibility"]["half_angle_deg"])),
        "gate_chi2": float(config["association"]["gate_chi2_2dof"]),
        "covariance_floor_m2": float(config["detector"].get("covariance_floor_m2", 0.01)),
    }


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def truth_measurement(position, velocity, station, boresight: float, height: float) -> dict:
    delta = np.asarray(position, dtype=float) - np.asarray(station, dtype=float)
    rho = float(np.linalg.norm(delta))
    r = math.sqrt(rho ** 2 + height ** 2)
    bearing = wrap_angle(math.atan2(delta[1], delta[0]) - float(boresight))
    radial_velocity = float(np.dot(np.asarray(station, dtype=float) - np.asarray(position, dtype=float),
                                   np.asarray(velocity, dtype=float)) / max(r, 1e-9))
    return {"rho": rho, "r": r, "bearing": bearing, "u": math.sin(bearing),
            "radial_velocity": radial_velocity}


def visible(truth: dict, setup: dict) -> bool:
    return (setup["range_min"] <= truth["r"] <= setup["range_max"]
            and abs(truth["bearing"]) <= setup["half_angle_rad"])


def recoverable_geometry(position, setup: dict, minimum_condition_ratio: float = 0.01) -> tuple:
    """(visible_bs_count, condition_ratio) of ``sum(u u^T)`` over geometrically visible BS.

    A ratio below ``minimum_condition_ratio`` means the visible BS directions are nearly
    parallel and the transverse velocity is unobservable; such windows do not serve the stated
    multi-BS 2D velocity recovery purpose and are excluded during window selection.
    """
    position = np.asarray(position, dtype=float)
    matrix = np.zeros((2, 2), dtype=float)
    count = 0
    for bs in range(len(setup["stations"])):
        delta = position - np.asarray(setup["stations"][bs], dtype=float)
        rho = float(np.linalg.norm(delta))
        if rho < 1e-9:
            continue
        r = math.sqrt(rho ** 2 + float(setup["height"]) ** 2)
        bearing = wrap_angle(math.atan2(delta[1], delta[0]) - float(setup["boresights"][bs]))
        if not (setup["range_min"] <= r <= setup["range_max"]
                and abs(bearing) <= setup["half_angle_rad"]):
            continue
        direction = delta / rho
        matrix += np.outer(direction, direction)
        count += 1
    if count < 2:
        return count, 0.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    ratio = float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 0.0
    return count, ratio


def polar_to_xy(r: float, bearing: float, station, boresight: float, height: float) -> tuple:
    rho = math.sqrt(max(r * r - height * height, 1e-9))
    phi = float(boresight) + float(bearing)
    return (float(np.asarray(station, dtype=float)[0] + rho * math.cos(phi)),
            float(np.asarray(station, dtype=float)[1] + rho * math.sin(phi)))


def xy_covariance(r: float, bearing: float, sigma_r: float, sigma_bearing: float,
                  station, boresight: float, height: float, floor_m2: float = 0.01) -> np.ndarray:
    rho = math.sqrt(max(r * r - height * height, 1e-9))
    phi = float(boresight) + float(bearing)
    sigma_rho = float(sigma_r) * r / max(rho, 1e-9)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    jacobian = np.asarray([[cos_phi, -rho * sin_phi], [sin_phi, rho * cos_phi]], dtype=float)
    covariance = jacobian @ np.diag([sigma_rho ** 2, float(sigma_bearing) ** 2]) @ jacobian.T
    return covariance + float(floor_m2) * np.eye(2)


def base_disturbance(episode_index: int, deadline_ms: int, bs: int, vehicle_key: int) -> tuple:
    """Fixed standard-normal base draw for one (episode, frame, BS, vehicle); SNR-independent."""
    sequence = np.random.SeedSequence([2029, int(episode_index), int(deadline_ms) // 100, int(bs),
                                       int(vehicle_key)])
    draw = np.random.default_rng(sequence).standard_normal(3)
    return float(draw[0]), float(draw[1]), float(draw[2])


def shuffle_order(episode_index: int, deadline_ms: int, bs: int, count: int) -> np.ndarray:
    """Deterministic anonymization order of the BS target list (SNR-independent)."""
    sequence = np.random.SeedSequence([2028, int(episode_index), int(deadline_ms) // 100, int(bs)])
    return np.random.default_rng(sequence).permutation(int(count))


def make_measurement(episode_index: int, deadline_ms: int, bs: int, vehicle_key: int,
                     position, velocity, sigma, setup: dict) -> dict | None:
    """One anonymous BS measurement with calibrated noise, or None when geometrically invisible."""
    truth = truth_measurement(position, velocity, setup["stations"][bs], setup["boresights"][bs],
                              setup["height"])
    if not visible(truth, setup):
        return None
    z_range, z_bearing, z_radial = base_disturbance(episode_index, deadline_ms, bs, vehicle_key)
    r_hat = truth["r"] + float(sigma["range"]) * z_range
    bearing_hat = truth["bearing"] + float(sigma["bearing"]) * z_bearing
    radial_hat = truth["radial_velocity"] + float(sigma["radial_velocity"]) * z_radial
    if r_hat <= setup["height"]:
        r_hat = setup["height"] + 1e-6
    x, y = polar_to_xy(r_hat, bearing_hat, setup["stations"][bs], setup["boresights"][bs],
                       setup["height"])
    covariance = xy_covariance(r_hat, bearing_hat, sigma["range"], sigma["bearing"],
                               setup["stations"][bs], setup["boresights"][bs], setup["height"],
                               setup["covariance_floor_m2"])
    return {"bs": int(bs), "source_key": int(vehicle_key), "r": r_hat, "bearing": bearing_hat,
            "radial_velocity": radial_hat, "x": x, "y": y, "covariance": covariance,
            "station": np.asarray(setup["stations"][bs], dtype=float)}