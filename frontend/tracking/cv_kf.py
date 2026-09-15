"""Anonymous constant-velocity Kalman tracker with fixed 8-slot policy (B domain).

Frozen design: DECISIONS.md D11-D14, SYSTEM_MODEL.md sections 7-8, SENS-REBUILD-03B sections 16-27.
This module never receives target lists, identities or truth states.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_CHI2_DEFAULT = 9.21
BIG = 1e9
INITIAL_VELOCITY_VARIANCE = 100.0


class CvKalmanTracker:
    def __init__(self, config: dict):
        settings = config["tracker"]
        self.dt = float(settings["dt"])
        if abs(self.dt - 0.1) > 1e-9:
            raise ValueError("Frozen tracker uses dt=0.1 s")
        if settings.get("q_a_m2_s3") is None:
            raise ValueError("tracker.q_a_m2_s3 must be calibrated before use")
        self.q_a = float(settings["q_a_m2_s3"])
        self.birth_inflation = float(settings["birth_inflation"])
        self.confirm_hits = int(settings["confirm_hits"])
        self.confirm_window = int(settings["confirm_window"])
        self.confirm_requires_nbs2 = bool(settings["confirm_requires_nbs2"])
        self.max_missed = int(settings["max_missed"])
        self.max_confirmed = int(settings["max_confirmed"])
        self.slot_cooldown = int(settings["slot_cooldown_cycles"])
        self.gate = float(config.get("association", {}).get("gate_chi2_2dof", GATE_CHI2_DEFAULT))

        self.F = np.eye(4)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        eye2 = np.eye(2)
        self.Q = self.q_a * np.block([[self.dt ** 3 / 3 * eye2, self.dt ** 2 / 2 * eye2],
                                      [self.dt ** 2 / 2 * eye2, self.dt * eye2]])
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])

        self.tracks: dict[int, dict] = {}
        self.next_key = 0
        self.cycle = 0
        self.time_ns: int | None = None
        self.slot_free_at = {slot: -10 ** 9 for slot in range(self.max_confirmed)}
        self.last_records: list[dict] = []

    def _predict(self) -> None:
        for track in self.tracks.values():
            track["x"] = self.F @ track["x"]
            track["P"] = self.F @ track["P"] @ self.F.T + self.Q
            track["P"] = (track["P"] + track["P"].T) / 2

    def _birth(self, observation: dict) -> None:
        position = np.array([float(observation["x_m"]), float(observation["y_m"])])
        covariance = np.asarray(observation["C_xy"], dtype=float)
        state = np.concatenate([position, [0.0, 0.0]])
        P = np.zeros((4, 4))
        P[:2, :2] = self.birth_inflation * covariance
        P[2:, 2:] = INITIAL_VELOCITY_VARIANCE * np.eye(2)
        self.tracks[self.next_key] = {
            "key": self.next_key,
            "x": state, "P": P, "hits": [(True, int(observation["n_bs"]) >= 2)], "misses": 0, "age": 0,
            "confirmed": False, "slot": None, "birth_cycle": self.cycle, "last_n_bs": int(observation["n_bs"]),
        }
        self.next_key += 1

    def _available_slot(self) -> int | None:
        held = {track["slot"] for track in self.tracks.values() if track["slot"] is not None and track["confirmed"]}
        for slot in range(self.max_confirmed):
            if slot not in held and self.slot_free_at[slot] <= self.cycle:
                return slot
        return None

    def _rank(self, track: dict):
        return (track["misses"], float(np.trace(track["P"][:2, :2])), -track["age"], track.get("key", 0))

    def _confirm_if_ready(self, track: dict) -> None:
        hits = track["hits"][-self.confirm_window:]
        hit_count = sum(1 for hit, _ in hits if hit)
        nbs2 = any(flag for _, flag in hits)
        if not track["confirmed"] and hit_count >= self.confirm_hits and (not self.confirm_requires_nbs2 or nbs2):
            track["confirmed"] = True
            slot = self._available_slot()
            if slot is not None:
                track["slot"] = slot

    def _delete(self, key: int) -> None:
        track = self.tracks.pop(key)
        if track["slot"] is not None:
            self.slot_free_at[track["slot"]] = self.cycle + self.slot_cooldown

    def step(self, observations: list[dict], time_ns: int) -> list[dict]:
        time_ns = int(time_ns)
        if self.time_ns is not None:
            delta = (time_ns - self.time_ns) * 1e-9
            if delta < 0:
                raise ValueError("Observations must arrive in timestamp order")
            if delta > 0 and abs(delta - self.dt) > 1e-6:
                raise ValueError("Frozen tracker advances by exactly dt=0.1 s")
        advancing = self.time_ns is None or time_ns > self.time_ns
        self.time_ns = time_ns
        if advancing:
            self._predict()

        keys = sorted(self.tracks)
        states = [(self.tracks[key]["x"], self.tracks[key]["P"]) for key in keys]
        measurement_states = []
        for observation in observations:
            covariance = np.asarray(observation["C_xy"], dtype=float)
            if covariance.shape != (2, 2) or np.linalg.eigvalsh(covariance).min() <= 0:
                raise ValueError("Observation covariance must be positive definite 2x2")
            measurement_states.append(
                (np.array([float(observation["x_m"]), float(observation["y_m"])]), covariance))

        cost = np.full((len(keys), len(measurement_states)), BIG, dtype=float)
        caches = {}
        for i, key in enumerate(keys):
            track = self.tracks[key]
            predicted = self.H @ track["x"]
            for j, (position, covariance) in enumerate(measurement_states):
                innovation = position - predicted
                innovation_covariance = self.H @ track["P"] @ self.H.T + covariance
                distance = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
                if distance <= self.gate:
                    cost[i, j] = distance
                    caches[(i, j)] = (innovation, innovation_covariance, covariance)
        matched_observations = set()
        assigned_rows = set()
        if keys and measurement_states:
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows.tolist(), columns.tolist()):
                if cost[i, j] < BIG / 2:
                    key = keys[i]
                    track = self.tracks[key]
                    innovation, innovation_covariance, covariance = caches[(i, j)]
                    kalman_gain = np.linalg.solve(innovation_covariance, self.H @ track["P"]).T
                    track["x"] = track["x"] + kalman_gain @ innovation
                    track["P"] = (np.eye(4) - kalman_gain @ self.H) @ track["P"] @ (np.eye(4) - kalman_gain @ self.H).T \
                        + kalman_gain @ covariance @ kalman_gain.T
                    track["P"] = (track["P"] + track["P"].T) / 2
                    track["hits"].append((True, int(observations[j]["n_bs"]) >= 2))
                    track["misses"] = 0
                    track["last_n_bs"] = int(observations[j]["n_bs"])
                    assigned_rows.add(i)
                    matched_observations.add(j)
        for i, key in enumerate(keys):
            if i not in assigned_rows:
                track = self.tracks[key]
                track["hits"].append((False, False))
                track["misses"] += 1
        for key in keys:
            track = self.tracks[key]
            track["age"] += 1
        for j, observation in enumerate(observations):
            if j not in matched_observations:
                self._birth(observation)

        self.cycle += 1
        for key in list(self.tracks):
            track = self.tracks[key]
            self._confirm_if_ready(track)
            expired_tentative = (not track["confirmed"] and self.cycle - track["birth_cycle"] >= self.confirm_window)
            if expired_tentative or track["misses"] > self.max_missed:
                self._delete(key)
        slotless = [key for key in sorted(self.tracks)
                    if self.tracks[key]["confirmed"] and self.tracks[key]["slot"] is None]
        for key in sorted(slotless, key=lambda value: self._rank(self.tracks[value])):
            slot = self._available_slot()
            if slot is None:
                break
            self.tracks[key]["slot"] = slot

        records = []
        for key in sorted(self.tracks):
            track = self.tracks[key]
            if not track["confirmed"] or track["slot"] is None:
                continue
            detected = bool(track["hits"] and track["hits"][-1][0] and track["misses"] == 0)
            records.append({
                "track_key": int(key),
                "slot": int(track["slot"]),
                "state_hat": [float(value) for value in track["x"]],
                "P": track["P"].tolist(),
                "track_exists": True,
                "detected": detected,
                "confirmed": True,
                "misses": int(track["misses"]),
                "age": int(track["age"]),
                "n_bs": int(track["last_n_bs"]),
            })
        records.sort(key=lambda record: record["slot"])
        self.last_records = records
        return records