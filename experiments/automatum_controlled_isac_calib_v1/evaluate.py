"""Evaluation helpers for AUTOMATUM-CONTROLLED-ISAC Calibration V1.

C-domain metrics only: the estimator (automatum_frontend) receives ground-truth vehicle
states and produces estimates; this module compares them and computes coverage/quality.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontend.automatum_scene_source import AutomatumSceneSource  # noqa: E402
from frontend.controlled_isac.automatum_frontend import sense_frame  # noqa: E402
from frontend.controlled_isac.automatum_measurement import setup_from_config  # noqa: E402
from frontend.controlled_isac.measurement import truth_measurement, visible  # noqa: E402
from frontend.controlled_isac.state_quality import quality_from_errors  # noqa: E402

CONFIG_PATH = ROOT / "configs/automatum_controlled_isac.json"
OUT_DIR = ROOT / "reports/isac_calibration/automatum_controlled_isac_v1"
SMOKE01_DIR = ROOT / "reports/isac_smoke/automatum_isac_smoke_01"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def selected_frames(split: str) -> dict[int, list[int]]:
    frames: dict[int, list[int]] = {}
    with (SMOKE01_DIR / f"selection_{split}.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames.setdefault(int(row["scene_id"]), []).append(int(row["frame"]))
    for scene_id in frames:
        frames[scene_id].sort()
    return frames


def scene_frames(config: dict, split: str) -> dict[tuple[int, int], object]:
    source = AutomatumSceneSource(ROOT / config["dataset"][f"{split}_trajectories"],
                                  source_hz=float(config["dataset"]["source_hz"]),
                                  stride=int(config["dataset"]["stride"]))
    frames = {}
    for scene_id, frame_ids in selected_frames(split).items():
        for frame_id in frame_ids:
            frames[(scene_id, frame_id)] = source.at_frame(scene_id, frame_id)
    return frames


def cohort_keys(split: str) -> dict[tuple[int, int], set[int]]:
    samples = np.load(ROOT / "data/automatum_t_crossing/splits" / split / "samples.npz")
    cohort: dict[tuple[int, int], set[int]] = {}
    for index in range(samples["scene_id"].size):
        scene_id = int(samples["scene_id"][index])
        start = int(samples["start_frame"][index])
        ids = samples["vehicle_ids"][index][samples["vehicle_mask"][index]].tolist()
        for frame in range(start, start + 40):
            cohort.setdefault((scene_id, frame), set()).update(int(v) for v in ids)
    return cohort


def run_split(config: dict, calibration: dict, split: str, levels, frames_cache=None) -> list[dict]:
    setup = setup_from_config(config)
    quality_config = config["state_quality"]
    frames = frames_cache if frames_cache is not None else scene_frames(config, split)
    rows = []
    for (scene_id, frame_id), scene_frame in frames.items():
        for snr_db in levels:
            states = sense_frame(scene_frame, scene_id, float(snr_db), calibration, setup)
            for state, truth in zip(states, scene_frame.states):
                dp = math.hypot(state["x_hat"] - truth[0], state["y_hat"] - truth[1])
                dv = math.hypot(state["vx_hat"] - truth[2], state["vy_hat"] - truth[3])
                rows.append({
                    "split": split, "scene_id": scene_id, "frame": frame_id,
                    "timestamp": scene_frame.timestamp, "vehicle_id": int(state["vehicle_id"]),
                    "snr_db": float(snr_db), "n_bs": int(state["n_bs"]),
                    "rank": int(state["rank"]), "condition_ratio": float(state["condition_ratio"]),
                    "x_hat": state["x_hat"], "y_hat": state["y_hat"],
                    "vx_hat": state["vx_hat"], "vy_hat": state["vy_hat"],
                    "x_true": float(truth[0]), "y_true": float(truth[1]),
                    "vx_true": float(truth[2]), "vy_true": float(truth[3]),
                    "position_error": float(dp), "velocity_error": float(dv),
                    "vx_error": float(state["vx_hat"] - truth[2]),
                    "vy_error": float(state["vy_hat"] - truth[3]),
                    "quality": float(quality_from_errors(dp, dv, float(quality_config["p_ref_m"]),
                                                         float(quality_config["v_ref_mps"]))),
                })
    return rows


CHANNELS = ("position", "velocity", "vx", "vy")


def summarize(rows: list[dict]) -> dict:
    summary = {"n": len(rows)}
    for channel in CHANNELS:
        values = np.asarray([row[f"{channel}_error"] for row in rows], dtype=float)
        if values.size == 0:
            continue
        summary[channel] = {
            "mae": float(np.abs(values).mean()), "rmse": float(np.sqrt((values ** 2).mean())),
            "median": float(np.median(np.abs(values))),
            "p95": float(np.percentile(np.abs(values), 95)),
        }
    quality = np.asarray([row["quality"] for row in rows], dtype=float)
    if quality.size:
        summary["quality"] = {"mean": float(quality.mean()), "median": float(np.median(quality)),
                              "p05": float(np.percentile(quality, 5)),
                              "min": float(quality.min())}
    return summary


def aggregate(rows: list[dict], levels, scopes: dict[str, list[int]]) -> list[dict]:
    results = []
    for scope_name, scene_ids in scopes.items():
        scoped = [row for row in rows if row["scene_id"] in scene_ids]
        for subset_name, predicate in (("all", lambda row: True),
                                       ("ge2", lambda row: row["n_bs"] >= 2),
                                       ("eq3", lambda row: row["n_bs"] == 3)):
            for snr_db in levels:
                selected = [row for row in scoped
                            if float(row["snr_db"]) == float(snr_db) and predicate(row)]
                summary = summarize(selected)
                summary.update({"scope": scope_name, "subset": subset_name, "snr_db": float(snr_db)})
                results.append(summary)
    return results


def flatten(summary: dict, prefix: str = "") -> dict:
    flat = {}
    for channel in CHANNELS:
        for stat, value in summary.get(channel, {}).items():
            flat[f"{prefix}{channel}_{stat}"] = value
    for stat, value in summary.get("quality", {}).items():
        flat[f"{prefix}quality_{stat}"] = value
    flat[f"{prefix}n"] = summary.get("n", 0)
    return flat


def geometry_audit(config: dict) -> list[dict]:
    """SNR-independent geometric BS visibility of every selected target state."""
    setup = setup_from_config(config)
    stations = {scene_id: setup["scenes"][scene_id]["stations"] for scene_id in setup["scenes"]}
    boresights = {scene_id: setup["scenes"][scene_id]["boresights"] for scene_id in setup["scenes"]}
    rows = []
    for split in ("train", "val"):
        cohort = cohort_keys(split)
        frames = scene_frames(config, split)
        buckets: dict[tuple, dict] = {}

        def bucket(scope):
            return buckets.setdefault((split, scope), {"split": split, "scope": scope, "n": 0,
                                                       "n_bs_0": 0, "n_bs_1": 0, "n_bs_2": 0,
                                                       "n_bs_3": 0})
        for (scene_id, frame_id), scene_frame in frames.items():
            in_cohort = cohort.get((scene_id, frame_id), set())
            for vehicle_id, state in zip(scene_frame.vehicle_ids.tolist(), scene_frame.states):
                count = 0
                for bs_id in range(3):
                    truth = truth_measurement(state[:2], state[2:], stations[scene_id][bs_id],
                                              float(boresights[scene_id][bs_id]), setup["height"])
                    if visible(truth, setup):
                        count += 1
                for scope in ("overall", f"scene_{scene_id}",
                              "cohort_only" if int(vehicle_id) in in_cohort else None):
                    if scope is None:
                        continue
                    record = bucket(scope)
                    record["n"] += 1
                    record[f"n_bs_{count}"] += 1
        for record in buckets.values():
            n = record["n"]
            record["pct_ge_1bs"] = (record["n_bs_1"] + record["n_bs_2"] + record["n_bs_3"]) / n
            record["pct_ge_2bs"] = (record["n_bs_2"] + record["n_bs_3"]) / n
            record["pct_3bs"] = record["n_bs_3"] / n
            rows.append(record)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")