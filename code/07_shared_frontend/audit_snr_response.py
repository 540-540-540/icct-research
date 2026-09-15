"""SENS-SNR-AUDIT-04: sensing response across the practical SNR range (audit only).

Levels 1-6 metrics over controlled synthetic scenes and train-subset scenes.
No production config, cache, or model is modified.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/snr_audit_04"
FIGURES = OUT / "figures"
SNR_POINTS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
RANGE_BINS = [10.0, 75.0, 150.0, 225.0, 300.0]
COUNT_BUCKETS = {"low": (1, 3), "mid": (4, 6), "high": (7, 8)}
DT_NS = 100_000_000


def range_bin(r_m: float) -> str:
    for low, high in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        if low <= r_m < high:
            return f"{int(low)}-{int(high)}"
    return f"{int(RANGE_BINS[-2])}-{int(RANGE_BINS[-1])}"


def count_bucket(n: int) -> str:
    for name, (low, high) in COUNT_BUCKETS.items():
        if low <= n <= high:
            return name
    return "high"


def synthetic_snapshot(count: int, range_m: float, bearing_deg: float, close: bool = False) -> dict:
    station = np.array([-64.23372566, -106.90678644])
    rho = math.sqrt(range_m ** 2 - 25.0)
    phi = math.radians(bearing_deg)
    center = station + rho * np.array([math.cos(phi), math.sin(phi)])
    lateral = np.array([-math.sin(phi), math.cos(phi)])
    spacing = 2.0 if close else 18.0
    offsets = (np.arange(count) - (count - 1) / 2) * spacing
    positions = [center + offset * lateral for offset in offsets]
    velocities = [np.array([8.0, -3.0]) + np.array([0.4 * index, -0.2 * index]) for index in range(count)]
    label = f"count{count}_r{int(range_m)}_{'close' if close else 'norm'}_b{int(bearing_deg)}"
    return {"label": label, "type": "synthetic", "count": count, "band": int(range_m),
            "bearing": bearing_deg, "positions": positions, "velocities": velocities}


def synthetic_stream(count: int, frames: int = 12) -> dict:
    starts = [np.array([-30.0 + 18.0 * index, 5.0 - 6.0 * index]) for index in range(count)]
    velocity = np.array([9.0, -2.0])
    positions = [[(start + velocity * 0.1 * frame).tolist() for start in starts] for frame in range(frames)]
    velocities = [[velocity.tolist() for _ in starts] for _ in range(frames)]
    return {"label": f"stream_count{count}", "type": "synthetic_stream", "count": count,
            "positions": positions, "velocities": velocities}


def train_scenes(source) -> tuple[list[dict], list[dict]]:
    train = [(index, episode) for index, episode in enumerate(source.episodes) if episode["split"] == "train"]
    by_count = sorted(train, key=lambda item: len(item[1]["source_keys"]))
    low = next(item for item in by_count if len(item[1]["source_keys"]) <= 3)
    mid = min(by_count, key=lambda item: abs(len(item[1]["source_keys"]) - 5))
    high = max(by_count, key=lambda item: len(item[1]["source_keys"]))
    snapshots, streams = [], []
    for name, (index, episode) in (("low", low), ("mid", mid), ("high", high)):
        keys = [int(k) for k in episode["source_keys"]]
        snapshots.append({"label": f"train_{name}", "type": "train", "count": len(keys),
                          "positions": None, "velocities": None, "episode": index})
        streams.append({"label": f"train_stream_{name}", "type": "train_stream", "count": len(keys),
                        "episode": index, "keys": keys})
    return snapshots, streams


def sample_episode(source, index: int, offset_ms: int, frames: int) -> tuple[list, list]:
    episode = source.episodes[index]
    positions, velocities = [], []
    for frame in range(frames):
        deadline = int(episode["start_ms"]) + offset_ms + frame * 100
        states, _slots, _used = source.at_time(episode, deadline * 1_000_000)
        positions.append(np.asarray(states, dtype=float)[:, :2].tolist())
        velocities.append(np.asarray(states, dtype=float)[:, 2:].tolist())
    return positions, velocities


def nearest_match(detections: list[dict], position: np.ndarray, r_gt: float, u_gt: float,
                  r_gate: float = 8.0, u_gate: float = 0.12):
    best, best_cost = None, None
    for detection in detections:
        if abs(detection["r_m"] - r_gt) > r_gate or abs(detection["u"] - u_gt) > u_gate:
            continue
        cost = abs(detection["r_m"] - r_gt) + 50.0 * abs(detection["u"] - u_gt)
        if best_cost is None or cost < best_cost:
            best, best_cost = detection, cost
    return best


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("SNR audit requires CUDA per frozen design")
    from frontend.echo_source import SourceEpisodes
    from frontend.fusion.association import fuse_frame
    from frontend.sensing import detector as detector_module
    from frontend.tracking.cv_kf import CvKalmanTracker

    device = "cuda:0"
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    _common.assert_height_alignment(config, geometry)
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    lut = _common.load_lut(ROOT)
    if lut is None:
        raise SystemExit("covariance LUT missing")

    source = SourceEpisodes()
    snapshots = []
    for count in (1, 3, 5, 8):
        for range_m in (40.0, 140.0, 280.0):
            for bearing in (0.0, 40.0, 65.0):
                snapshots.append(synthetic_snapshot(count, range_m, bearing))
    snapshots.append(synthetic_snapshot(2, 100.0, 20.0, close=True))
    snapshots.append(synthetic_snapshot(3, 100.0, 0.0, close=True))
    train_snap, train_streams = train_scenes(source)
    snapshots.extend(train_snap)
    streams = [synthetic_stream(3), synthetic_stream(5), synthetic_stream(8)]
    for entry in train_streams:
        positions, velocities = sample_episode(source, entry["episode"], 4000, 12)
        entry["positions"], entry["velocities"] = positions, velocities
        streams.append(entry)

    def run_echo(positions, velocities, keys, snr_ref, episode, frame):
        return _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array, config,
                                  snr_ref, episode, frame, device, height_m=height)

    processing_rows = []
    per_snr = {snr: {"station_pairs": 0, "station_matched": 0, "targets": 0, "targets_matched": 0,
                     "fa": 0, "frames": 0, "detections": 0,
                     "range_abs": [], "u_abs": [], "xy_errors": [], "peak": [], "g_emp": [], "received": [],
                     "range_abs_bin": {}, "station_matched_bin": {}, "station_pairs_bin": {},
                     "nbs": {1: 0, 2: 0, 3: 0}, "gt_match": 0, "obs_total": 0, "obs_matched": 0,
                     "fusion_errors": [], "gate": 0, "postfilter": 0} for snr in SNR_POINTS}

    for scene_index, scene in enumerate(snapshots):
        if scene["type"] == "synthetic":
            positions, velocities = scene["positions"], scene["velocities"]
        else:
            episode = source.episodes[scene["episode"]]
            states, _slots, _used = source.at_time(episode, (int(episode["start_ms"]) + 5000) * 1_000_000)
            positions = [state[:2].tolist() for state in states]
            velocities = [state[2:].tolist() for state in states]
            scene["count"] = len(positions)
        keys = [10000 + scene_index * 100 + index for index in range(len(positions))]
        for snr in SNR_POINTS:
            echo = run_echo(positions, velocities, keys, snr, 9600 + scene_index, 0)
            detections_by_bs, truth_polar = {}, {}
            for bs in range(3):
                detections, _counters, _ = detector_module.detect_from_maps(
                    detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                 config["detector"]),
                    bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                    covariance_lut=lut, height=height)
                detections_by_bs[bs] = detections
            for index, position in enumerate(positions):
                for bs in range(3):
                    rho, r_gt, _bearing, u_gt = _common.truth_polar(position, stations[bs], float(boresights[bs]),
                                                                    height)
                    bearing_gt = math.asin(float(u_gt))
                    visible = 10.0 <= float(r_gt) <= 300.0 and abs(bearing_gt) <= math.radians(70.0)
                    if not visible:
                        continue
                    received = snr + 40.0 * math.log10(100.0 / float(r_gt))
                    per = per_snr[snr]
                    per["station_pairs"] += 1
                    per["received"].append(received)
                    bin_name = range_bin(float(r_gt))
                    per["station_pairs_bin"][bin_name] = per["station_pairs_bin"].get(bin_name, 0) + 1
                    match = nearest_match(detections_by_bs[bs], position, float(r_gt), float(u_gt))
                    if match is not None:
                        per["station_matched"] += 1
                        per["station_matched_bin"][bin_name] = per["station_matched_bin"].get(bin_name, 0) + 1
                        per["range_abs"].append(abs(match["r_m"] - float(r_gt)))
                        per["u_abs"].append(abs(match["u"] - float(u_gt)))
                        per["xy_errors"].append(math.hypot(match["x_m"] - position[0], match["y_m"] - position[1]))
                        per["peak"].append(match["peak_to_noise_db"])
                        per["g_emp"].append(match["peak_to_noise_db"] - received)
                        processing_rows.append({"scene_type": scene["type"], "scene": scene["label"],
                                                "snr_ref_db": snr, "target": index, "bs": bs,
                                                "range_bin": bin_name, "received_snr_db": received,
                                                "peak_to_noise_db": match["peak_to_noise_db"],
                                                "g_emp_db": match["peak_to_noise_db"] - received,
                                                "peak_power": match["peak_power"],
                                                "noise_floor": match["noise_floor"],
                                                "cfar_score": match["cfar_score"]})
            observations, diagnostics = fuse_frame(detections_by_bs, config)
            per = per_snr[snr]
            per["frames"] += 1
            per["detections"] += sum(len(detections) for detections in detections_by_bs.values())
            for observation in observations:
                per["nbs"][observation["n_bs"]] += 1
            matched_gt = 0
            used_observations = set()
            for position in positions:
                near = [(index, observation) for index, observation in enumerate(observations)
                        if math.hypot(observation["x_m"] - position[0], observation["y_m"] - position[1]) < 5.0]
                if near:
                    matched_gt += 1
                    index, observation = min(near, key=lambda item: math.hypot(item[1]["x_m"] - position[0],
                                                                               item[1]["y_m"] - position[1]))
                    used_observations.add(index)
                    per["fusion_errors"].append(math.hypot(observation["x_m"] - position[0],
                                                           observation["y_m"] - position[1]))
            per["targets"] += len(positions)
            per["targets_matched"] += matched_gt
            per["obs_total"] += len(observations)
            per["obs_matched"] += len(used_observations)
            per["gt_match"] += matched_gt
            per["gate"] += diagnostics["gate_rejections"]
            per["postfilter"] += diagnostics["postfilter_rejections"]

    for snr in SNR_POINTS:
        per = per_snr[snr]
        per["fa"] = max(per["detections"] - per["station_matched"], 0)

    tracker_rows = []
    for stream_index, stream in enumerate(streams):
        positions_seq, velocities_seq = stream["positions"], stream["velocities"]
        for snr in SNR_POINTS:
            tracker = CvKalmanTracker(config)
            matched, coast, continuity_hits, continuity_total = 0, 0, 0, 0
            previous_key = {}
            switches = 0
            position_errors, velocity_errors = [], []
            for frame in range(len(positions_seq)):
                positions = positions_seq[frame]
                velocities = velocities_seq[frame]
                keys = [20000 + frame * 100 + index for index in range(len(positions))]
                echo = run_echo(positions, velocities, keys, snr, 9700 + stream_index, frame)
                detections_by_bs = {}
                for bs in range(3):
                    detections, _, _ = detector_module.detect_from_maps(
                        detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                     config["detector"]),
                        bs, frame, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                        covariance_lut=lut, height=height)
                    detections_by_bs[bs] = detections
                observations, _ = fuse_frame(detections_by_bs, config)
                records = tracker.step(observations, frame * DT_NS)
                for target_index, position in enumerate(positions):
                    best, best_distance = None, 5.0
                    for record in records:
                        distance = math.hypot(record["state_hat"][0] - position[0],
                                              record["state_hat"][1] - position[1])
                        if distance < best_distance:
                            best, best_distance = record, distance
                    if frame >= 3:
                        continuity_total += 1
                    if best is None:
                        continue
                    matched += 1
                    if frame >= 3:
                        continuity_hits += 1
                        key = best["track_key"]
                        if target_index in previous_key and previous_key[target_index] != key:
                            switches += 1
                        previous_key[target_index] = key
                    if not best["detected"]:
                        coast += 1
                    truth_velocity = np.asarray(velocities[target_index], dtype=float)
                    position_errors.append(best_distance)
                    velocity_errors.append(np.asarray(best["state_hat"][2:]) - truth_velocity)
            tracker_rows.append({
                "snr_ref_db": snr, "stream": stream["label"], "targets": len(positions_seq[0]),
                "matched_frames": matched, "continuity": continuity_hits / max(continuity_total, 1),
                "coast_ratio": coast / max(matched, 1), "id_switches": switches,
                "track_position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None,
                "track_velocity_rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(velocity_errors)))))
                if velocity_errors else None,
                "confirmed_tracks": len(previous_key)})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "processing_gain.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(processing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(processing_rows)

    response_rows = []
    for snr in SNR_POINTS:
        per = per_snr[snr]
        response_rows.append({
            "snr_ref_db": snr,
            "received_snr_median_db": statistics.median(per["received"]) if per["received"] else None,
            "received_snr_p10_db": quantile(per["received"], 0.10),
            "received_snr_p90_db": quantile(per["received"], 0.90),
            "peak_to_noise_median_db": statistics.median(per["peak"]) if per["peak"] else None,
            "g_emp_median_db": statistics.median(per["g_emp"]) if per["g_emp"] else None,
            "station_recall": per["station_matched"] / max(per["station_pairs"], 1),
            "target_recall": per["targets_matched"] / max(per["targets"], 1),
            "detections_per_bs_frame": per["detections"] / max(per["frames"] * 3, 1),
            "false_alarms_per_bs_frame": per["fa"] / max(per["frames"] * 3, 1),
            "range_rmse_m": float(np.sqrt(np.mean(np.square(per["range_abs"])))) if per["range_abs"] else None,
            "range_median_abs_m": statistics.median(per["range_abs"]) if per["range_abs"] else None,
            "range_p90_abs_m": quantile(per["range_abs"], 0.90),
            "u_rmse": float(np.sqrt(np.mean(np.square(per["u_abs"])))) if per["u_abs"] else None,
            "u_median_abs": statistics.median(per["u_abs"]) if per["u_abs"] else None,
            "single_bs_xy_rmse_m": float(np.sqrt(np.mean(np.square(per["xy_errors"])))) if per["xy_errors"] else None,
            "single_bs_xy_median_m": statistics.median(per["xy_errors"]) if per["xy_errors"] else None,
            "nbs1_fraction": per["nbs"][1] / max(sum(per["nbs"].values()), 1),
            "nbs2_fraction": per["nbs"][2] / max(sum(per["nbs"].values()), 1),
            "nbs3_fraction": per["nbs"][3] / max(sum(per["nbs"].values()), 1),
            "gt_match_rate": per["gt_match"] / max(per["targets"], 1),
            "obs_precision": per["obs_matched"] / max(per["obs_total"], 1),
            "fusion_xy_rmse_m": float(np.sqrt(np.mean(np.square(per["fusion_errors"])))) if per["fusion_errors"] else None,
            "fusion_xy_median_m": statistics.median(per["fusion_errors"]) if per["fusion_errors"] else None,
            "gate_rejections_per_frame": per["gate"] / max(per["frames"], 1),
            "postfilter_rejections_per_frame": per["postfilter"] / max(per["frames"], 1),
        })
    with (OUT / "response_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_rows[0].keys()))
        writer.writeheader()
        writer.writerows(response_rows)

    stratified_rows = []
    for snr in SNR_POINTS:
        per = per_snr[snr]
        for bin_name in [f"{int(RANGE_BINS[i])}-{int(RANGE_BINS[i+1])}" for i in range(len(RANGE_BINS) - 1)]:
            pairs = per["station_pairs_bin"].get(bin_name, 0)
            matched = per["station_matched_bin"].get(bin_name, 0)
            stratified_rows.append({"dimension": "range_bin", "bucket": bin_name, "snr_ref_db": snr,
                                    "pairs_or_targets": pairs, "recall": matched / max(pairs, 1)})
    for bucket, (low, high) in COUNT_BUCKETS.items():
        for snr in SNR_POINTS:
            selected = [row for row in processing_rows if row["snr_ref_db"] == snr]
            stratified_rows.append({"dimension": "density_bucket", "bucket": bucket, "snr_ref_db": snr,
                                    "pairs_or_targets": len(selected), "recall": None})
    with (OUT / "range_stratified.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stratified_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stratified_rows)

    tracker_summary = {}
    for snr in SNR_POINTS:
        rows = [row for row in tracker_rows if row["snr_ref_db"] == snr]
        tracker_summary[str(snr)] = {
            "streams": len(rows),
            "mean_continuity": float(np.mean([row["continuity"] for row in rows])),
            "mean_coast_ratio": float(np.mean([row["coast_ratio"] for row in rows])),
            "total_id_switches": int(sum(row["id_switches"] for row in rows)),
            "mean_track_position_rmse_m": float(np.mean([row["track_position_rmse_m"] for row in rows
                                                         if row["track_position_rmse_m"] is not None])),
            "mean_track_velocity_rmse_mps": float(np.mean([row["track_velocity_rmse_mps"] for row in rows
                                                           if row["track_velocity_rmse_mps"] is not None])),
        }
    with (OUT / "tracker_response.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracker_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tracker_rows)

    summary = {"audit": "SENS-SNR-AUDIT-04", "snr_points": SNR_POINTS,
               "snapshots": len(snapshots), "streams": len(streams),
               "response_table": response_rows, "tracker_summary": tracker_summary,
               "processing_gain_rows": len(processing_rows)}
    _common.write_json(OUT / "summary.json", summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)
        axis = [row["snr_ref_db"] for row in response_rows]

        def save(figure, name):
            figure.tight_layout()
            figure.savefig(FIGURES / name, dpi=150)
            plt.close(figure)

        figure, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(axis, [row["received_snr_median_db"] for row in response_rows], "o-")
        ax.set(xlabel="snr_ref (dB)", ylabel="median received SNR (dB)", title="1. snr_ref -> received SNR")
        ax.grid(alpha=0.3); save(figure, "01_received_snr.png")

        figure, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(axis, [row["peak_to_noise_median_db"] for row in response_rows], "o-")
        ax.set(xlabel="snr_ref (dB)", ylabel="median peak-to-noise (dB)", title="2. snr_ref -> peak-to-noise")
        ax.grid(alpha=0.3); save(figure, "02_peak_to_noise.png")

        figure, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(axis, [row["station_recall"] for row in response_rows], "o-", label="station recall")
        ax.plot(axis, [row["target_recall"] for row in response_rows], "s--", label="target recall")
        ax.set(xlabel="snr_ref (dB)", ylabel="recall", ylim=(0, 1.02), title="3. snr_ref -> detection recall")
        ax.grid(alpha=0.3); ax.legend(fontsize=8); save(figure, "03_detection_recall.png")

        for index, (key, label) in enumerate((("single_bs_xy_median_m", "single-BS position median (m)"),
                                              ("fusion_xy_median_m", "fusion position median (m)"),
                                              ("track_position_rmse_m", "track position RMSE (m)"),
                                              ("track_velocity_rmse_mps", "track velocity RMSE (m/s)"))):
            values = ([row[key] for row in response_rows] if key in response_rows[0]
                      else [tracker_summary[str(snr)][f"mean_{key}"] for snr in SNR_POINTS])
            figure, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(axis, values, "o-")
            ax.set(xlabel="snr_ref (dB)", ylabel=label, title=f"{4+index}. snr_ref -> {key}")
            ax.grid(alpha=0.3); save(figure, f"0{4+index}_{key}.png")

        figure, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(axis, [tracker_summary[str(snr)]["mean_coast_ratio"] for snr in SNR_POINTS], "o-")
        ax.set(xlabel="snr_ref (dB)", ylabel="coast ratio", title="8. snr_ref -> coast ratio")
        ax.grid(alpha=0.3); save(figure, "08_coast_ratio.png")
    except Exception as error:  # noqa: BLE001
        _common.write_json(OUT / "figure_error.json", {"error": repr(error)})

    print(json.dumps({"response_table": response_rows, "tracker_summary": tracker_summary},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()