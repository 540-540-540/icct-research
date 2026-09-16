"""SENS-SNR-AUDIT-04 (rev2): sensing response across the practical SNR range.

Fix pass: strict one-to-one GT<->detection Hungarian matching, identity-stable tracker
evaluation, density stratification, and a detection-independent GT-cell RD diagnostic.
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
from scipy.optimize import linear_sum_assignment

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
GATE_M = 5.0
BIG = 1e9
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


def one_to_one_match(gt_positions, candidate_positions, gate: float = GATE_M):
    """Strict Hungarian matching with a Euclidean gate; each candidate used at most once."""
    count_gt, count_candidate = len(gt_positions), len(candidate_positions)
    if not count_gt or not count_candidate:
        return [], list(range(count_candidate))
    cost = np.full((count_gt, count_candidate), BIG, dtype=float)
    for i, gt in enumerate(gt_positions):
        for j, candidate in enumerate(candidate_positions):
            distance = math.hypot(gt[0] - candidate[0], gt[1] - candidate[1])
            if distance <= gate:
                cost[i, j] = distance
    rows, columns = linear_sum_assignment(cost)
    matched, used = [], set()
    for i, j in zip(rows.tolist(), columns.tolist()):
        if cost[i, j] <= gate:
            matched.append((i, j, cost[i, j]))
            used.add(j)
    unmatched = [j for j in range(count_candidate) if j not in used]
    return matched, unmatched


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
    return {"label": label, "type": "synthetic", "count": count, "positions": positions,
            "velocities": velocities}


def synthetic_stream(count: int, frames: int = 12) -> dict:
    starts = [np.array([-30.0 + 18.0 * index, 5.0 - 6.0 * index]) for index in range(count)]
    velocity = np.array([9.0, -2.0])
    positions = [[(start + velocity * 0.1 * frame).tolist() for start in starts] for frame in range(frames)]
    velocities = [[velocity.tolist() for _ in starts] for _ in range(frames)]
    identities = [list(range(count)) for _ in range(frames)]
    return {"label": f"stream_count{count}", "type": "synthetic_stream", "positions": positions,
            "velocities": velocities, "identities": identities, "keys": [100 + i for i in range(count)]}


def train_scenes(source) -> tuple[list[dict], list[dict]]:
    train = [(index, episode) for index, episode in enumerate(source.episodes) if episode["split"] == "train"]
    by_count = sorted(train, key=lambda item: len(item[1]["source_keys"]))
    low = next(item for item in by_count if len(item[1]["source_keys"]) <= 3)
    mid = min(by_count, key=lambda item: abs(len(item[1]["source_keys"]) - 5))
    high = max(by_count, key=lambda item: len(item[1]["source_keys"]))
    snapshots, streams = [], []
    for name, (index, episode) in (("low", low), ("mid", mid), ("high", high)):
        snapshots.append({"label": f"train_{name}", "type": "train", "count": len(episode["source_keys"]),
                          "episode": index})
        streams.append({"label": f"train_stream_{name}", "type": "train_stream", "episode": index,
                        "count": len(episode["source_keys"])})
    return snapshots, streams


def sample_episode_identity(source, index: int, offset_ms: int, frames: int):
    episode = source.episodes[index]
    positions, velocities, identities, keys = [], [], [], []
    for frame in range(frames):
        deadline = int(episode["start_ms"]) + offset_ms + frame * 100
        states, slots, _used = source.at_time(episode, deadline * 1_000_000)
        positions.append([state[:2].tolist() for state in states])
        velocities.append([state[2:].tolist() for state in states])
        identities.append([int(slot) for slot in slots])
        keys.append([30000 + int(slot) for slot in slots])
    return positions, velocities, identities, keys


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def median_or_none(values):
    return statistics.median(values) if values else None


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
        positions, velocities, identities, keys = sample_episode_identity(source, entry["episode"], 4000, 12)
        entry.update({"positions": positions, "velocities": velocities, "identities": identities, "keys": keys})
        streams.append(entry)

    def run_echo(positions, velocities, keys, snr_ref, episode, frame, noise=True):
        return _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array, config,
                                  snr_ref, episode, frame, device, noise=noise, height_m=height)

    processing_rows = []
    per_snr = {snr: {"station_pairs": 0, "station_matched": 0, "misses": 0, "fa": 0, "frames": 0,
                     "detections": 0, "range_abs": [], "u_abs": [], "xy_errors": [], "peak": [], "g_emp": [],
                     "received": [], "range_bin": {}, "density": {}, "targets": 0, "targets_matched": 0,
                     "obs_total": 0, "obs_matched": 0, "fusion_errors": [], "nbs": {1: 0, 2: 0, 3: 0},
                     "gate": 0, "postfilter": 0} for snr in SNR_POINTS}
    gt_rd_rows = []

    for scene_index, scene in enumerate(snapshots):
        if scene["type"] == "synthetic":
            positions = [np.asarray(position, dtype=float) for position in scene["positions"]]
            velocities = [np.asarray(velocity, dtype=float) for velocity in scene["velocities"]]
        else:
            episode = source.episodes[scene["episode"]]
            states, _slots, _used = source.at_time(episode, (int(episode["start_ms"]) + 5000) * 1_000_000)
            positions = [np.asarray(state[:2], dtype=float) for state in states]
            velocities = [np.asarray(state[2:], dtype=float) for state in states]
            scene["count"] = len(positions)
        bucket = count_bucket(scene["count"])
        keys = [10000 + scene_index * 100 + index for index in range(len(positions))]
        for snr in SNR_POINTS:
            echo = run_echo(positions, velocities, keys, snr, 9600 + scene_index, 0)
            detections_by_bs = {}
            for bs in range(3):
                detections, _, _ = detector_module.detect_from_maps(
                    detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                 config["detector"]),
                    bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                    covariance_lut=lut, height=height)
                detections_by_bs[bs] = detections
            per = per_snr[snr]
            per["frames"] += 1
            detections_total = 0
            for bs in range(3):
                detections = detections_by_bs[bs]
                detections_total += len(detections)
                matched, unmatched = one_to_one_match(positions, [np.array([d["x_m"], d["y_m"]]) for d in detections])
                per["fa"] += len(unmatched)
                matched_targets = {i for i, _, _ in matched}
                for index, position in enumerate(positions):
                    _, r_gt, _, u_gt = _common.truth_polar(position.tolist(), stations[bs],
                                                           float(boresights[bs]), height)
                    bearing_gt = math.asin(float(u_gt))
                    visible = 10.0 <= float(r_gt) <= 300.0 and abs(bearing_gt) <= math.radians(70.0)
                    if not visible:
                        continue
                    received = snr + 40.0 * math.log10(100.0 / float(r_gt))
                    bin_name = range_bin(float(r_gt))
                    per["station_pairs"] += 1
                    per["received"].append(received)
                    per["range_bin"].setdefault(bin_name, {"pairs": 0, "matched": 0, "range_abs": [], "u_abs": [],
                                                           "xy": []})
                    per["range_bin"][bin_name]["pairs"] += 1
                    per["density"].setdefault(bucket, {"pairs": 0, "matched": 0, "range_abs": [], "u_abs": [],
                                                       "xy": []})
                    per["density"][bucket]["pairs"] += 1
                    processing_rows.append({"kind": "input", "scene_type": scene["type"], "scene": scene["label"],
                                            "snr_ref_db": snr, "target": index, "bs": bs, "range_bin": bin_name,
                                            "received_snr_db": received, "peak_to_noise_db": None,
                                            "g_emp_db": None, "gt_rd_snr_db": None, "peak_power": None,
                                            "noise_floor": None, "cfar_score": None})
                    if index not in matched_targets:
                        per["misses"] += 1
                        continue
                    detection = detections[next(j for i, j, _ in matched if i == index)]
                    per["station_matched"] += 1
                    per["range_bin"][bin_name]["matched"] += 1
                    per["density"][bucket]["matched"] += 1
                    error_range = abs(detection["r_m"] - float(r_gt))
                    error_u = abs(detection["u"] - float(u_gt))
                    error_xy = math.hypot(detection["x_m"] - position[0], detection["y_m"] - position[1])
                    per["range_abs"].append(error_range)
                    per["u_abs"].append(error_u)
                    per["xy_errors"].append(error_xy)
                    per["peak"].append(detection["peak_to_noise_db"])
                    per["g_emp"].append(detection["peak_to_noise_db"] - received)
                    per["range_bin"][bin_name]["range_abs"].append(error_range)
                    per["range_bin"][bin_name]["u_abs"].append(error_u)
                    per["range_bin"][bin_name]["xy"].append(error_xy)
                    per["density"][bucket]["range_abs"].append(error_range)
                    per["density"][bucket]["u_abs"].append(error_u)
                    per["density"][bucket]["xy"].append(error_xy)
                    processing_rows.append({"kind": "detection", "scene_type": scene["type"],
                                            "scene": scene["label"], "snr_ref_db": snr, "target": index, "bs": bs,
                                            "range_bin": bin_name, "received_snr_db": received,
                                            "peak_to_noise_db": detection["peak_to_noise_db"],
                                            "g_emp_db": detection["peak_to_noise_db"] - received,
                                            "gt_rd_snr_db": None, "peak_power": detection["peak_power"],
                                            "noise_floor": detection["noise_floor"],
                                            "cfar_score": detection["cfar_score"]})
            per["detections"] += detections_total

            observations, diagnostics = fuse_frame(detections_by_bs, config)
            per["targets"] += len(positions)
            per["obs_total"] += len(observations)
            for observation in observations:
                per["nbs"][observation["n_bs"]] += 1
            matched, unmatched = one_to_one_match(positions,
                                                  [np.array([o["x_m"], o["y_m"]]) for o in observations])
            per["targets_matched"] += len(matched)
            per["obs_matched"] += len(observations) - len(unmatched)
            for _, _, distance in matched:
                per["fusion_errors"].append(distance)
            per["gate"] += diagnostics["gate_rejections"]
            per["postfilter"] += diagnostics["postfilter_rejections"]

            if scene["type"] == "synthetic" and scene["count"] == 1:
                clean = run_echo(positions, velocities, keys, snr, 9600 + scene_index, 0, noise=False)
                maps_clean = detector_module.compute_maps(clean["Y"][0], clean["X"][0], waveform, array,
                                                          config["detector"])
                maps_noise = detector_module.compute_maps(echo["W"][0], clean["X"][0], waveform, array,
                                                          config["detector"])
                position = positions[0]
                delta = stations[0] - torch.tensor(position, dtype=torch.float64)
                radius = float(torch.linalg.vector_norm(delta))
                radial = float(torch.dot(delta, torch.tensor(velocities[0], dtype=torch.float64))) / radius
                _, r_gt, _, _ = _common.truth_polar(position.tolist(), stations[0], float(boresights[0]), height)
                i = int(torch.argmin(torch.abs(maps_clean["ranges"] - float(r_gt))))
                j = int(torch.argmin(torch.abs(maps_clean["velocities"] - radial)))
                window_clean = maps_clean["P_RD"][max(0, i - 1):i + 2, max(0, j - 1):j + 2]
                window_noise = maps_noise["P_RD"][max(0, i - 1):i + 2, max(0, j - 1):j + 2]
                gt_rd_rows.append({"scene": scene["label"], "snr_ref_db": snr,
                                   "r_gt_m": float(r_gt), "radial_gt_mps": radial,
                                   "clean_power_max": float(window_clean.max()),
                                   "noise_power_median": float(window_noise.median()),
                                   "gt_rd_snr_db": float(10 * math.log10(float(window_clean.max())
                                                                         / max(float(window_noise.median()), 1e-300)))})

    tracker_rows = []
    for stream_index, stream in enumerate(streams):
        for snr in SNR_POINTS:
            tracker = CvKalmanTracker(config)
            matched, coast, continuity_hits, continuity_total = 0, 0, 0, 0
            previous_key, switches = {}, 0
            position_errors, velocity_errors = [], []
            for frame in range(len(stream["positions"])):
                positions = [np.asarray(position, dtype=float) for position in stream["positions"][frame]]
                velocities = [np.asarray(velocity, dtype=float) for velocity in stream["velocities"][frame]]
                identities = stream["identities"][frame]
                keys = stream.get("keys", [None] * 12)[frame] if stream["type"] == "train_stream" else \
                    [20000 + index for index in range(len(positions))]
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
                record_positions = [np.array([record["state_hat"][0], record["state_hat"][1]]) for record in records]
                frame_matched, _ = one_to_one_match([position for position in positions], record_positions)
                for gt_index, record_index, distance in frame_matched:
                    identity = int(identities[gt_index])
                    matched += 1
                    if frame >= 3:
                        continuity_hits += 1
                    key = records[record_index]["track_key"]
                    if identity in previous_key and previous_key[identity] != key:
                        switches += 1
                    previous_key[identity] = key
                    if not records[record_index]["detected"]:
                        coast += 1
                    position_errors.append(distance)
                    velocity_errors.append(np.asarray(records[record_index]["state_hat"][2:])
                                           - velocities[gt_index])
                if frame >= 3:
                    continuity_total += len(identities)
            tracker_rows.append({
                "snr_ref_db": snr, "stream": stream["label"], "targets": len(stream["identities"][0]),
                "gt_identities": len(stream["identities"][0]), "matched_frames": matched,
                "continuity": continuity_hits / max(continuity_total, 1),
                "coast_ratio": coast / max(matched, 1), "id_switches": switches,
                "track_position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None,
                "track_velocity_rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(velocity_errors)))))
                if velocity_errors else None})

    for row in gt_rd_rows:
        processing_rows.append({"kind": "gt_rd", "scene_type": "synthetic", "scene": row["scene"],
                                "snr_ref_db": row["snr_ref_db"], "target": 0, "bs": 0, "range_bin": None,
                                "received_snr_db": row["snr_ref_db"] + 40.0 * math.log10(100.0 / row["r_gt_m"]),
                                "peak_to_noise_db": None, "g_emp_db": None,
                                "gt_rd_snr_db": row["gt_rd_snr_db"], "peak_power": row["clean_power_max"],
                                "noise_floor": row["noise_power_median"], "cfar_score": None})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "processing_gain.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(processing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(processing_rows)
    _common.write_json(OUT / "gt_rd_probe.json", {"rows": gt_rd_rows})

    response_rows = []
    for snr in SNR_POINTS:
        per = per_snr[snr]
        response_rows.append({
            "snr_ref_db": snr,
            "received_snr_median_db": median_or_none(per["received"]),
            "received_snr_p10_db": quantile(per["received"], 0.10),
            "received_snr_p90_db": quantile(per["received"], 0.90),
            "peak_to_noise_median_db": median_or_none(per["peak"]),
            "g_emp_median_db": median_or_none(per["g_emp"]),
            "station_recall": per["station_matched"] / max(per["station_pairs"], 1),
            "miss_rate": per["misses"] / max(per["station_pairs"], 1),
            "target_recall": per["targets_matched"] / max(per["targets"], 1),
            "detections_per_bs_frame": per["detections"] / max(per["frames"] * 3, 1),
            "false_alarms_per_bs_frame": per["fa"] / max(per["frames"] * 3, 1),
            "range_rmse_m": float(np.sqrt(np.mean(np.square(per["range_abs"])))) if per["range_abs"] else None,
            "range_median_abs_m": median_or_none(per["range_abs"]),
            "range_p90_abs_m": quantile(per["range_abs"], 0.90),
            "u_rmse": float(np.sqrt(np.mean(np.square(per["u_abs"])))) if per["u_abs"] else None,
            "u_median_abs": median_or_none(per["u_abs"]),
            "single_bs_xy_rmse_m": float(np.sqrt(np.mean(np.square(per["xy_errors"])))) if per["xy_errors"] else None,
            "single_bs_xy_median_m": median_or_none(per["xy_errors"]),
            "nbs1_fraction": per["nbs"][1] / max(sum(per["nbs"].values()), 1),
            "nbs2_fraction": per["nbs"][2] / max(sum(per["nbs"].values()), 1),
            "nbs3_fraction": per["nbs"][3] / max(sum(per["nbs"].values()), 1),
            "gt_match_rate": per["targets_matched"] / max(per["targets"], 1),
            "obs_precision": per["obs_matched"] / max(per["obs_total"], 1),
            "fusion_xy_rmse_m": float(np.sqrt(np.mean(np.square(per["fusion_errors"])))) if per["fusion_errors"] else None,
            "fusion_xy_median_m": median_or_none(per["fusion_errors"]),
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
        for bin_name, entry in sorted(per["range_bin"].items()):
            stratified_rows.append({"dimension": "range_bin", "bucket": bin_name, "snr_ref_db": snr,
                                    "pairs_or_targets": entry["pairs"], "matched": entry["matched"],
                                    "recall": entry["matched"] / max(entry["pairs"], 1),
                                    "range_median_abs_m": median_or_none(entry["range_abs"]),
                                    "u_median_abs": median_or_none(entry["u_abs"]),
                                    "xy_median_m": median_or_none(entry["xy"])})
        for bucket, entry in sorted(per["density"].items()):
            stratified_rows.append({"dimension": "density_bucket", "bucket": bucket, "snr_ref_db": snr,
                                    "pairs_or_targets": entry["pairs"], "matched": entry["matched"],
                                    "recall": entry["matched"] / max(entry["pairs"], 1),
                                    "range_median_abs_m": median_or_none(entry["range_abs"]),
                                    "u_median_abs": median_or_none(entry["u_abs"]),
                                    "xy_median_m": median_or_none(entry["xy"])})
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

    gt_rd_summary = {str(snr): {} for snr in SNR_POINTS}
    for row in gt_rd_rows:
        gt_rd_summary[str(row["snr_ref_db"])][row["scene"]] = row["gt_rd_snr_db"]
    gt_rd_median = [float(np.median([value for value in gt_rd_summary[str(snr)].values()]))
                    for snr in SNR_POINTS]
    summary = {"audit": "SENS-SNR-AUDIT-04 rev2", "matching": "strict one-to-one Hungarian, 5 m gate",
               "snr_points": SNR_POINTS, "snapshots": len(snapshots), "streams": len(streams),
               "response_table": response_rows, "tracker_summary": tracker_summary,
               "gt_rd_probe": gt_rd_summary,
               "g_emp_note": "G_emp is conditional-on-detection; gt_rd_snr_db is detection-independent",
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
        ax.plot(axis, [row["peak_to_noise_median_db"] for row in response_rows], "o-", label="detected peaks")
        ax.plot(axis, gt_rd_median, "s--", label="GT-cell RD (all targets)")
        ax.set(xlabel="snr_ref (dB)", ylabel="dB", title="2. snr_ref -> peak-to-noise / GT-cell RD")
        ax.grid(alpha=0.3); ax.legend(fontsize=8); save(figure, "02_peak_to_noise.png")

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

    print(json.dumps({"response_table": response_rows, "tracker_summary": tracker_summary,
                      "gt_rd_median": gt_rd_median}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()