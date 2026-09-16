"""Temporal (frame-to-frame) slot linking without any smoothing.

The tracker only decides "which vehicle is this"; the state stored for a slot is always the
current frame's fused measurement, copied unchanged. Previous position and velocity are used
solely to predict where a slot should reappear for the one-to-one matching (dt = 0.1 s).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class SlotTracker:
    def __init__(self, gate_m: float = 5.0, dt: float = 0.1):
        self.gate_m = float(gate_m)
        self.dt = float(dt)
        self.states = []

    def initialize(self, targets: list) -> list:
        """First frame: every target becomes a slot in the given order (arbitrary but fixed)."""
        self.states = [{"position": np.asarray(target["position"], dtype=float).copy(),
                        "velocity": np.asarray(target["velocity"], dtype=float).copy()}
                       for target in targets]
        return list(range(len(targets)))

    def step(self, targets: list) -> dict:
        """Return {'slot_to_target': [...], 'births': [...], 'deaths': [...]}."""
        slot_to_target = [-1] * len(self.states)
        taken = set()
        if self.states and targets:
            predictions = [state["position"] + state["velocity"] * self.dt
                           for state in self.states]
            cost = np.full((len(self.states), len(targets)), 1e9, dtype=float)
            for i, prediction in enumerate(predictions):
                for j, target in enumerate(targets):
                    distance = float(np.linalg.norm(prediction
                                                    - np.asarray(target["position"], dtype=float)))
                    if distance <= self.gate_m:
                        cost[i, j] = distance
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows.tolist(), columns.tolist()):
                if cost[i, j] < 1e9 / 2:
                    slot_to_target[i] = int(j)
                    taken.add(int(j))
        deaths = [i for i, target in enumerate(slot_to_target) if target < 0]
        births = [j for j in range(len(targets)) if j not in taken]
        for slot, target_index in enumerate(slot_to_target):
            if target_index >= 0:
                target = targets[target_index]
                self.states[slot]["position"] = np.asarray(target["position"], dtype=float).copy()
                self.states[slot]["velocity"] = np.asarray(target["velocity"], dtype=float).copy()
        return {"slot_to_target": slot_to_target, "births": births, "deaths": deaths}