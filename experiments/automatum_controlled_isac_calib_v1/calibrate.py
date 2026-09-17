"""Calibration V1: deterministic coarse + coordinate-refinement search of (a_range, a_bearing,
a_radial_velocity) for the Route-B controlled measurement model.

Objective (train, ge2 subset only): bring the provisional per-state quality to the requested
semantics Q(+10 dB) ~ 90 % and Q(-10 dB) ~ 40-50 %, keep the quality curve monotone in SNR,
and keep the physical errors in a sane range. Val/test are never used for the search.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_calib_v1.evaluate import (load_config, run_split,  # noqa: E402
                                                                     scene_frames, summarize)
from frontend.controlled_isac.automatum_measurement import calibration_from_config  # noqa: E402

LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)
CHANNELS = ("range", "bearing", "radial_velocity")

COARSE_GRID = {
    "range": (0.03, 0.06, 0.10, 0.15, 0.25, 0.40),
    "bearing": (0.001, 0.002, 0.004, 0.008, 0.016),
    "radial_velocity": (0.03, 0.06, 0.10, 0.16, 0.25, 0.40),
}
REFINE_FACTORS = (0.7, 0.85, 1.0, 1.15, 1.4)
CANDIDATE_FACTORS = (("A_light", 0.5), ("B_recommended", 1.0),
                     ("C_heavy", 2.0), ("D_very_heavy", 3.0))
QUALITY_TARGET_LOW = 45.0
QUALITY_TARGET_HIGH = 90.0
MONOTONE_PENALTY = 20.0


def calibration_with(config: dict, a: dict) -> dict:
    calibration = calibration_from_config(config)
    calibration["a"] = {channel: float(a[channel]) for channel in CHANNELS}
    return calibration


def evaluate_candidate(config: dict, a: dict, frames_cache, detailed: bool = False) -> dict:
    calibration = calibration_with(config, a)
    rows = run_split(config, calibration, "train", LEVELS, frames_cache=frames_cache)
    ge2 = [row for row in rows if row["n_bs"] >= 2]
    per_snr = {}
    for snr_db in LEVELS:
        selected = [row for row in ge2 if row["snr_db"] == snr_db]
        per_snr[snr_db] = summarize(selected)
    qualities = [per_snr[snr_db]["quality"]["mean"] for snr_db in LEVELS]
    violations = sum(1 for index in range(len(qualities) - 1)
                     if qualities[index] >= qualities[index + 1])
    score = (abs(qualities[0] - QUALITY_TARGET_LOW)
             + abs(qualities[-1] - QUALITY_TARGET_HIGH)
             + MONOTONE_PENALTY * violations)
    result = {"a": {channel: float(a[channel]) for channel in CHANNELS},
              "score": float(score), "quality_mean_by_snr": qualities,
              "monotone_violations": int(violations),
              "position_rmse_by_snr": [per_snr[snr_db]["position"]["rmse"] for snr_db in LEVELS],
              "velocity_rmse_by_snr": [per_snr[snr_db]["velocity"]["rmse"] for snr_db in LEVELS]}
    if detailed:
        result["per_snr"] = {str(snr_db): per_snr[snr_db] for snr_db in LEVELS}
    return result


def search(config: dict, coarse_grid: dict | None = None, refine: bool = True) -> dict:
    grid = coarse_grid or COARSE_GRID
    frames_cache = scene_frames(config, "train")
    evaluations = []
    best = None
    for a_range in grid["range"]:
        for a_bearing in grid["bearing"]:
            for a_radial in grid["radial_velocity"]:
                result = evaluate_candidate(config, {"range": a_range, "bearing": a_bearing,
                                                     "radial_velocity": a_radial}, frames_cache)
                evaluations.append(result)
                if best is None or result["score"] < best["score"]:
                    best = copy.deepcopy(result)
    history = [{"stage": "coarse", "best": best["a"], "score": best["score"]}]
    if refine:
        for _ in range(2):
            for channel in CHANNELS:
                local_best = copy.deepcopy(best)
                for factor in REFINE_FACTORS:
                    candidate = dict(best["a"])
                    candidate[channel] = best["a"][channel] * factor
                    result = evaluate_candidate(config, candidate, frames_cache)
                    evaluations.append(result)
                    if result["score"] < local_best["score"]:
                        local_best = result
                best = local_best
                history.append({"stage": f"refine_{channel}", "best": best["a"],
                                "score": best["score"]})
    candidates = {}
    for name, factor in CANDIDATE_FACTORS:
        parameters = {channel: best["a"][channel] * factor for channel in CHANNELS}
        candidates[name] = {"factor_vs_recommended": factor, "a": parameters,
                            "evaluation": evaluate_candidate(config, parameters, frames_cache,
                                                             detailed=True)}
    return {"recommended": {"a": best["a"], "evaluation": best, "status": "NOT FROZEN"},
            "candidates": candidates, "history": history,
            "coarse_evaluations": evaluations,
            "objective": {"quality_target_low": QUALITY_TARGET_LOW,
                          "quality_target_high": QUALITY_TARGET_HIGH,
                          "monotone_penalty": MONOTONE_PENALTY,
                          "subset": "n_bs >= 2", "split": "train"}}


def main() -> int:
    config = load_config()
    result = search(config)
    recommended = result["recommended"]["evaluation"]
    print("recommended a:", recommended["a"], "score", round(recommended["score"], 3))
    print("quality by SNR:", [round(value, 1) for value in recommended["quality_mean_by_snr"]])
    print("position RMSE:", [round(value, 4) for value in recommended["position_rmse_by_snr"]])
    print("velocity RMSE:", [round(value, 4) for value in recommended["velocity_rmse_by_snr"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())