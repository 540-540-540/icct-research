"""Cross-BS anonymous target association and same-frame position fusion.

First version (probe): position Mahalanobis distance with the frozen chi-square gate and
one-to-one Hungarian assignment, processed BS0-BS1 then fused groups against BS2, exactly in the
spirit of the frozen ``frontend/fusion/association.py`` but implemented additively for the
controlled measurement model. GT keys are carried only for the offline audit and are never used by
the association itself.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9


def mahalanobis(first, second) -> float:
    delta = np.asarray(first["position"], dtype=float) - np.asarray(second["position"], dtype=float)
    combined = np.asarray(first["covariance"], dtype=float) + np.asarray(second["covariance"],
                                                                         dtype=float)
    return float(delta @ np.linalg.solve(combined, delta))


def fuse_positions(members: list) -> tuple:
    information = np.zeros((2, 2), dtype=float)
    weighted = np.zeros(2, dtype=float)
    for member in members:
        inverse = np.linalg.inv(np.asarray(member["covariance"], dtype=float))
        information += inverse
        weighted += inverse @ np.asarray([member["x"], member["y"]], dtype=float)
    covariance = np.linalg.inv(information)
    position = covariance @ weighted
    return position, covariance


def _state_of(item: dict) -> dict:
    if "position" in item:
        position = np.asarray(item["position"], dtype=float)
    else:
        position = np.asarray([item["x"], item["y"]], dtype=float)
    return {"position": position,
            "covariance": np.asarray(item["covariance"], dtype=float)}


def _match(left: list, right: list, gate_chi2: float) -> tuple:
    diagnostics = {"gate_rejections": 0, "candidate_pairs": len(left) * len(right)}
    if not left or not right:
        return [], diagnostics
    left_states = [_state_of(item) for item in left]
    right_states = [_state_of(item) for item in right]
    cost = np.full((len(left), len(right)), BIG, dtype=float)
    for i, first in enumerate(left_states):
        for j, second in enumerate(right_states):
            distance = mahalanobis(first, second)
            if distance <= float(gate_chi2):
                cost[i, j] = distance
            else:
                diagnostics["gate_rejections"] += 1
    rows, columns = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows.tolist(), columns.tolist())
             if cost[i, j] < BIG / 2]
    return pairs, diagnostics


def _group(members: list) -> dict:
    position, covariance = fuse_positions(members)
    return {"members": list(members), "position": position, "covariance": covariance,
            "n_bs": len({int(member["bs"]) for member in members})}


def associate_frame(measurements_by_bs: dict, gate_chi2: float) -> tuple:
    """Return (groups, diagnostics) for one frame; groups are anonymous fused targets."""
    diagnostics = {"gate_rejections": 0, "candidate_pairs": 0, "stage1_pairs": 0, "stage2_pairs": 0}
    left = measurements_by_bs.get(0, [])
    right = measurements_by_bs.get(1, [])
    pairs, stage = _match(left, right, gate_chi2)
    diagnostics["gate_rejections"] += stage["gate_rejections"]
    diagnostics["candidate_pairs"] += stage["candidate_pairs"]
    diagnostics["stage1_pairs"] += len(pairs)
    matched_right = {j for _, j in pairs}
    groups = []
    for i, j in pairs:
        groups.append(_group([left[i], right[j]]))
    for i, measurement in enumerate(left):
        if all(i != pair[0] for pair in pairs):
            groups.append(_group([measurement]))
    for j, measurement in enumerate(right):
        if j not in matched_right:
            groups.append(_group([measurement]))

    third = measurements_by_bs.get(2, [])
    states = [{"position": group["position"], "covariance": group["covariance"]}
              for group in groups]
    pairs, stage = _match(states, third, gate_chi2)
    diagnostics["gate_rejections"] += stage["gate_rejections"]
    diagnostics["candidate_pairs"] += stage["candidate_pairs"]
    diagnostics["stage2_pairs"] += len(pairs)
    matched_right = {j for _, j in pairs}
    for i, j in pairs:
        groups[i]["members"].append(third[j])
        groups[i] = _group(groups[i]["members"])
    for j, measurement in enumerate(third):
        if j not in matched_right:
            groups.append(_group([measurement]))
    return groups, diagnostics