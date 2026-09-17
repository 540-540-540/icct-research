"""Final Audit A-D: real vehicle motion, displacement, interaction and velocity-change scales
of the de-duplicated prediction-history states (train main, val independent confirmation)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final_audit import common  # noqa: E402

SPEED_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0),
              (20.0, float("inf")))
NEIGHBOR_THRESHOLDS = (5.0, 10.0, 20.0, 30.0)


def speed_bins(speeds: np.ndarray) -> dict:
    bins = {}
    for low, high in SPEED_BINS:
        selected = (speeds >= low) & (speeds < high)
        label = f"[{low:g},{high:g})" if math.isfinite(high) else f"[{low:g},inf)"
        bins[label] = {"count": int(selected.sum()),
                       "fraction": float(selected.mean()) if speeds.size else 0.0}
    return bins


def nearest_neighbor_distances(keys, frames) -> np.ndarray:
    distances = []
    for scene_id, frame_id, vehicle in keys:
        entry = frames.get((scene_id, frame_id))
        if entry is None or entry["ids"].size < 2:
            continue
        index = int(np.flatnonzero(entry["ids"] == vehicle)[0])
        delta = entry["positions"] - entry["positions"][index]
        norms = np.linalg.norm(delta, axis=1)
        norms[index] = np.inf
        distances.append(float(norms.min()))
    return np.asarray(distances, dtype=np.float64)


def split_audit(split: str) -> dict:
    history = common.unique_history(split)
    lookup, frames = common.state_lookup(split)
    keys = history["keys"]
    result = {"unique_states": history["unique_states"], "history_rows": history["history_rows"],
              "duplication_factor": history["duplication_factor"],
              "unique_frames": len({(s, f) for s, f, _ in keys})}
    scene_keys = {0: [], 1: []}
    for key in keys:
        scene_keys[key[0]].append(key)

    speed_all, single_disp, ten_disp = [], [], []
    vchange_1, vchange_10 = [], []
    for key in keys:
        state = lookup.get(key)
        if state is None:
            raise KeyError(f"prediction-history state missing from trajectories.csv: {key}")
        speed_all.append(float(np.hypot(state[2], state[3])))
        nxt = lookup.get((key[0], key[1] + 1, key[2]))
        if nxt is not None:
            single_disp.append(float(np.linalg.norm(nxt[:2] - state[:2])))
            vchange_1.append(float(np.linalg.norm(nxt[2:] - state[2:])))
        tenth = lookup.get((key[0], key[1] + 10, key[2]))
        if tenth is not None:
            ten_disp.append(float(np.linalg.norm(tenth[:2] - state[:2])))
            vchange_10.append(float(np.linalg.norm(tenth[2:] - state[2:])))
    speed_all = np.asarray(speed_all)
    result["speed"] = {"all": common.distribution(speed_all), "bins": speed_bins(speed_all)}
    for scene_id in (0, 1):
        scene_speeds = np.asarray([float(np.hypot(lookup[key][2], lookup[key][3]))
                                   for key in scene_keys[scene_id]])
        result["speed"][f"scene_{scene_id}"] = common.distribution(scene_speeds)
        result["speed"][f"scene_{scene_id}_bins"] = speed_bins(scene_speeds)
    result["single_frame_displacement_m"] = common.distribution(single_disp)
    result["ten_frame_displacement_m"] = common.distribution(ten_disp)
    result["ten_frame_seconds"] = 10 * common.DT
    result["velocity_change_1frame_mps"] = common.distribution(vchange_1)
    result["velocity_change_10frame_mps"] = common.distribution(vchange_10)
    result["acceleration_magnitude_mps2"] = common.distribution(
        [value / common.DT for value in vchange_1])

    history_spans, history_paths, future_spans, future_paths = [], [], [], []
    vchange_20 = []
    for window in history["windows"]:
        scene_id, start, ids = window["scene_id"], window["start_frame"], window["vehicle_ids"]
        for vehicle in ids:
            states = [lookup.get((scene_id, start + offset, vehicle)) for offset in range(40)]
            if any(state is None for state in states):
                raise KeyError(f"sample window state missing: scene {scene_id} start {start} "
                               f"vehicle {vehicle}")
            history_positions = np.asarray([state[:2] for state in states[:20]])
            future_positions = np.asarray([state[:2] for state in states[20:]])
            history_spans.append(float(np.linalg.norm(history_positions[-1] - history_positions[0])))
            history_paths.append(float(np.linalg.norm(np.diff(history_positions, axis=0), axis=1).sum()))
            future_spans.append(float(np.linalg.norm(future_positions[-1] - future_positions[0])))
            future_paths.append(float(np.linalg.norm(np.diff(future_positions, axis=0), axis=1).sum()))
            vchange_20.append(float(np.linalg.norm(states[19][2:] - states[0][2:])))
    result["history_span_20frame_m"] = common.distribution(history_spans)
    result["history_path_20frame_m"] = common.distribution(history_paths)
    result["future_span_20frame_m"] = common.distribution(future_spans)
    result["future_path_20frame_m"] = common.distribution(future_paths)
    result["velocity_change_20frame_mps"] = common.distribution(vchange_20)

    neighbor = nearest_neighbor_distances(keys, frames)
    result["nearest_neighbor_m"] = {"all": common.distribution(neighbor),
                                    "rates": {f"le_{threshold:g}m": common.rate(neighbor, threshold)
                                              for threshold in NEIGHBOR_THRESHOLDS},
                                    "no_neighbor_frames": int(sum(
                                        1 for key in keys
                                        if frames.get((key[0], key[1])) is None
                                        or frames[(key[0], key[1])]["ids"].size < 2))}
    for scene_id in (0, 1):
        scene_neighbors = nearest_neighbor_distances(scene_keys[scene_id], frames)
        result["nearest_neighbor_m"][f"scene_{scene_id}"] = common.distribution(scene_neighbors)
        result["nearest_neighbor_m"][f"scene_{scene_id}_rates"] = {
            f"le_{threshold:g}m": common.rate(scene_neighbors, threshold)
            for threshold in NEIGHBOR_THRESHOLDS}
    return result


def plot_outputs(train: dict, val: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = common.OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for stale in ("speed_distribution.png", "single_frame_displacement_distribution.png",
                  "nearest_neighbor_distribution.png"):
        (plots / stale).unlink(missing_ok=True)

    history_train = common.unique_history("train")
    lookup_train, _ = common.state_lookup("train")
    speeds = np.asarray([float(np.hypot(lookup_train[key][2], lookup_train[key][3]))
                         for key in history_train["keys"]])
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.hist(speeds, bins=60, color="steelblue")
    axis.set_xlabel("speed (m/s)")
    axis.set_ylabel("unique prediction-history states")
    axis.set_title(f"Automatum prediction-history speed distribution (train, n={speeds.size})")
    axis.grid(True, alpha=0.3)
    axis.axvline(float(np.median(speeds)), color="red", linestyle="--", label="median")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "speed_distribution.png", dpi=140)
    plt.close(figure)

    displacement = []
    for key in history_train["keys"]:
        nxt = lookup_train.get((key[0], key[1] + 1, key[2]))
        if nxt is not None:
            displacement.append(float(np.linalg.norm(nxt[:2] - lookup_train[key][:2])))
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.hist(displacement, bins=60, color="seagreen")
    axis.set_xlabel("single-frame displacement (m)")
    axis.set_ylabel("unique prediction-history states")
    axis.set_title("Automatum single-frame displacement distribution (train)")
    axis.grid(True, alpha=0.3)
    axis.axvline(float(np.median(displacement)), color="red", linestyle="--", label="median")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "single_frame_displacement_distribution.png", dpi=140)
    plt.close(figure)

    _, frames_train = common.state_lookup("train")
    neighbor = nearest_neighbor_distances(history_train["keys"], frames_train)
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.hist(neighbor[neighbor <= 150.0], bins=60, color="darkorange")
    axis.set_xlabel("distance to nearest other physical vehicle (m)")
    axis.set_ylabel("unique prediction-history states")
    axis.set_title("Automatum nearest-neighbour distance distribution (train)")
    axis.grid(True, alpha=0.3)
    axis.axvline(float(np.median(neighbor)), color="red", linestyle="--", label="median")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "nearest_neighbor_distribution.png", dpi=140)
    plt.close(figure)


def main() -> int:
    started = time.time()
    payload = {"dt_s": common.DT, "ten_frame_seconds": 10 * common.DT,
               "levels_note": "future windows are used for task-scale statistics only",
               "splits": {}}
    for split in ("train", "val"):
        payload["splits"][split] = split_audit(split)
        print(f"{split}: {payload['splits'][split]['unique_states']} unique history states, "
              f"median speed "
              f"{payload['splits'][split]['speed']['all']['p50']:.2f} m/s")
    common.write_json(common.OUT_DIR / "prediction_cohort_scale.json", payload)
    plot_outputs(payload["splits"]["train"], payload["splits"]["val"])
    print(f"written in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())