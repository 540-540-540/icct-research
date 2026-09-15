"""Sequential Hungarian 3-BS association and inverse-covariance fusion (B domain).

Frozen design: DECISIONS.md D09, SYSTEM_MODEL.md section 6, SENS-REBUILD-03B sections 5-12.
This module never receives target lists, identities or truth states.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_CHI2_DEFAULT = 9.21
BIG = 1e9


def _detection_sort_key(detection: dict):
    grid = tuple(int(value) for value in detection["grid_index"])
    return (-float(detection["peak_power"]), grid)


def _state(detection: dict) -> tuple[np.ndarray, np.ndarray]:
    position = np.array([float(detection["x_m"]), float(detection["y_m"])], dtype=float)
    covariance = np.asarray(detection["C_xy"], dtype=float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("Detection C_xy must be a finite 2x2 matrix")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Detection covariance must be positive definite")
    return position, covariance


def _fuse(members: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inverse_sum = np.zeros((2, 2), dtype=float)
    weighted = np.zeros(2, dtype=float)
    for member in members:
        position, covariance = _state(member)
        inverse = np.linalg.inv(covariance)
        inverse_sum += inverse
        weighted += inverse @ position
    fused_covariance = np.linalg.inv(inverse_sum)
    fused_position = fused_covariance @ weighted
    return fused_position, fused_covariance


def _mahalanobis(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    delta = first[0] - second[0]
    combined = first[1] + second[1]
    return float(delta @ np.linalg.solve(combined, delta))


def _match(left_states: list[tuple[np.ndarray, np.ndarray]],
           right_states: list[tuple[np.ndarray, np.ndarray]],
           gate: float) -> tuple[list[tuple[int, int]], dict]:
    n_left, n_right = len(left_states), len(right_states)
    diagnostics = {"gate_rejections": 0, "postfilter_rejections": 0}
    if n_left == 0 or n_right == 0:
        return [], diagnostics
    cost = np.full((n_left, n_right), BIG, dtype=float)
    for i, left in enumerate(left_states):
        for j, right in enumerate(right_states):
            distance = _mahalanobis(left, right)
            if distance <= gate:
                cost[i, j] = distance
            else:
                diagnostics["gate_rejections"] += 1
    rows, columns = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(rows.tolist(), columns.tolist()):
        if cost[i, j] < BIG / 2:
            pairs.append((i, j))
        else:
            diagnostics["postfilter_rejections"] += 1
    return pairs, diagnostics


def fuse_frame(detections_by_bs: dict[int, list[dict]], config: dict) -> tuple[list[dict], dict]:
    gate = float(config.get("association", {}).get("gate_chi2_2dof", GATE_CHI2_DEFAULT))
    if not math.isfinite(gate) or gate <= 0:
        raise ValueError("Invalid association gate")
    ordered = {bs: sorted(detections_by_bs.get(bs, []), key=_detection_sort_key) for bs in (0, 1, 2)}

    diagnostics = {"gate_rejections": 0, "postfilter_rejections": 0,
                   "input_count_by_bs": {bs: len(ordered[bs]) for bs in (0, 1, 2)}}

    def singleton(member: dict) -> dict:
        position, covariance = _state(member)
        return {"members": [member], "position": position, "covariance": covariance, "mask": 1 << int(member["station_id"])}

    left_states = [_state(d) for d in ordered[0]]
    right_states = [_state(d) for d in ordered[1]]
    pairs, stage_diagnostics = _match(left_states, right_states, gate)
    diagnostics["gate_rejections"] += stage_diagnostics["gate_rejections"]
    diagnostics["postfilter_rejections"] += stage_diagnostics["postfilter_rejections"]
    matched_right = {j for _, j in pairs}
    groups = []
    for i, j in pairs:
        members = [ordered[0][i], ordered[1][j]]
        position, covariance = _fuse(members)
        groups.append({"members": members, "position": position, "covariance": covariance,
                       "mask": (1 << 0) | (1 << 1)})
    for i, detection in enumerate(ordered[0]):
        if all(i != pair[0] for pair in pairs):
            groups.append(singleton(detection))
    for j, detection in enumerate(ordered[1]):
        if j not in matched_right:
            groups.append(singleton(detection))

    left_states = [(group["position"], group["covariance"]) for group in groups]
    right_states = [_state(d) for d in ordered[2]]
    pairs, stage_diagnostics = _match(left_states, right_states, gate)
    diagnostics["gate_rejections"] += stage_diagnostics["gate_rejections"]
    diagnostics["postfilter_rejections"] += stage_diagnostics["postfilter_rejections"]
    matched_right = {j for _, j in pairs}
    for i, j in pairs:
        groups[i]["members"].append(ordered[2][j])
        groups[i]["mask"] |= 1 << 2
        groups[i]["position"], groups[i]["covariance"] = _fuse(groups[i]["members"])
    for j, detection in enumerate(ordered[2]):
        if j not in matched_right:
            groups.append(singleton(detection))

    observations = []
    for group in groups:
        members = group["members"]
        if len({int(member["station_id"]) for member in members}) != len(members):
            raise ValueError("A group may contain at most one detection per BS")
        covariance = group["covariance"]
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("Fused covariance must be finite positive definite")
        time_values = {int(member["time_ns"]) for member in members}
        if len(time_values) != 1:
            raise ValueError("All members of a fused observation must share time_ns")
        observations.append({
            "time_ns": int(time_values.pop()),
            "x_m": float(group["position"][0]),
            "y_m": float(group["position"][1]),
            "C_xy": covariance.tolist(),
            "bs_mask": int(group["mask"]),
            "n_bs": len(members),
            "quality_db": float(max(float(member["peak_to_noise_db"]) for member in members)),
        })
    observations.sort(key=lambda observation: (-observation["n_bs"], observation["bs_mask"],
                                               round(observation["x_m"], 9), round(observation["y_m"], 9),
                                               -observation["quality_db"]))
    diagnostics["groups_2bs"] = sum(1 for observation in observations if observation["n_bs"] == 2)
    diagnostics["groups_3bs"] = sum(1 for observation in observations if observation["n_bs"] == 3)
    diagnostics["singletons"] = sum(1 for observation in observations if observation["n_bs"] == 1)
    diagnostics["observations"] = len(observations)
    return observations, diagnostics