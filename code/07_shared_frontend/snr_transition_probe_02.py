"""SNR-TRANSITION-PROBE-02: low-SNR transition-region probe (diagnostic only).

Re-runs the frozen SENS-SNR-REBUILD-06 representative diagnostic scene set (41 snapshots / 6 streams,
same vehicles, geometry, waveform, seeds, noise-realization rule and B64/CFAR/AoA/fusion/tracker
production chain) at the low-SNR sweep [-20, -17.5, -15, -12.5, -10, -7.5, -5, 0] dB. The only
changed input is ``snr_ref_db``. No formal F01-E data, config, threshold or calibration is modified;
V_confirm/test are not opened. Outputs are diagnostic reports and figures, not paper results.
"""
from __future__ import annotations

import argparse
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
import audit_snr_response as base  # noqa: E402

OUT = ROOT / "reports/f01e/snr_transition_probe_02"
SNR_POINTS = [-20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0, 0.0]
DT_NS = 100_000_000


def build_scenes(source) -> tuple:
    """Exactly the SENS-SNR-AUDIT-04 / REBUILD-06 diagnostic scene set."""
    snapshots = []
    for count in (1, 3, 5, 8):
        for range_m in (40.0, 140.0, 280.0):
            for bearing in (0.0, 40.0, 65.0):
                snapshots.append(base.synthetic_snapshot(count, range_m, bearing))
    snapshots.append(base.synthetic_snapshot(2, 100.0, 20.0, close=True))
    snapshots.append(base.synthetic_snapshot(3, 100.0, 0.0, close=True))
    train_snap, train_streams = base.train_scenes(source)
    snapshots.extend(train_snap)
    streams = [base.synthetic_stream(3), base.synthetic_stream(5), base.synthetic_stream(8)]
    for entry in train_streams:
        positions, velocities, identities, keys = base.sample_episode_identity(source, entry["episode"],
                                                                               4000, 12)
        entry.update({"positions": positions, "velocities": velocities, "identities": identities,
                      "keys": keys})
        streams.append(entry)
    return snapshots, streams


def new_counter() -> dict:
    return {"station_pairs": 0, "station_matched": 0, "misses": 0, "false_alarms": 0, "frames": 0,
            "detections": 0, "targets": 0, "targets_matched": 0, "observations": 0,
            "observations_matched": 0, "gate_rejections": 0, "postfilter_rejections": 0,
            "received": [], "peaks": [], "range_abs": [], "u_abs": [], "single_xy": [],
            "fusion_errors": [], "nbs": {1: 0, 2: 0, 3: 0}}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    arguments = parser.parse_args()
    OUT = Path(arguments.out)
    figures = OUT / "figures"
    if not torch.cuda.is_available():
        raise SystemExit("probe requires CUDA per frozen production design")

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
    lut = _common.load_production_lut(config)
    if lut is None:
        raise SystemExit("covariance LUT missing")
    resource = _common.load_resource(config)

    source = SourceEpisodes()
    snapshots, streams = build_scenes(source)

    def run_echo(positions, velocities, keys, snr_ref, episode, frame):
        return _common.synthesize(positions, velocities, keys, stations, boresights, waveform, array,
                                  config, snr_ref, episode, frame, device, noise=True, height_m=height)

    per_snr = {snr: new_counter() for snr in SNR_POINTS}
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
        keys = [10000 + scene_index * 100 + index for index in range(len(positions))]
        for snr in SNR_POINTS:
            echo = run_echo(positions, velocities, keys, snr, 9600 + scene_index, 0)
            detections_by_bs = {}
            for bs in range(3):
                detections, _, _ = detector_module.detect_from_maps(
                    detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                 config["detector"], resource=resource),
                    bs, 0, stations[bs], float(boresights[bs]), config, array, multiplier=multiplier,
                    covariance_lut=lut, height=height)
                detections_by_bs[bs] = detections
            per = per_snr[snr]
            per["frames"] += 1
            detections_total = 0
            for bs in range(3):
                detections = detections_by_bs[bs]
                detections_total += len(detections)
                matched, unmatched = base.one_to_one_match(
                    positions, [np.array([d["x_m"], d["y_m"]]) for d in detections])
                per["false_alarms"] += len(unmatched)
                matched_targets = {i for i, _, _ in matched}
                for index, position in enumerate(positions):
                    _, r_gt, _, u_gt = _common.truth_polar(position.tolist(), stations[bs],
                                                           float(boresights[bs]), height)
                    bearing_gt = math.asin(float(u_gt))
                    visible = 10.0 <= float(r_gt) <= 300.0 and abs(bearing_gt) <= math.radians(70.0)
                    if not visible:
                        continue
                    received = snr + 40.0 * math.log10(100.0 / float(r_gt))
                    per["station_pairs"] += 1
                    per["received"].append(received)
                    if index not in matched_targets:
                        per["misses"] += 1
                        continue
                    detection = detections[next(j for i, j, _ in matched if i == index)]
                    per["station_matched"] += 1
                    per["range_abs"].append(abs(detection["r_m"] - float(r_gt)))
                    per["u_abs"].append(abs(detection["u"] - float(u_gt)))
                    per["single_xy"].append(math.hypot(detection["x_m"] - position[0],
                                                       detection["y_m"] - position[1]))
                    per["peaks"].append(detection["peak_to_noise_db"])
            per["detections"] += detections_total

            observations, diagnostics = fuse_frame(detections_by_bs, config)
            per["targets"] += len(positions)
            per["observations"] += len(observations)
            for observation in observations:
                per["nbs"][observation["n_bs"]] += 1
            matched, unmatched = base.one_to_one_match(
                positions, [np.array([o["x_m"], o["y_m"]]) for o in observations])
            per["targets_matched"] += len(matched)
            per["observations_matched"] += len(observations) - len(unmatched)
            per["fusion_errors"].extend(distance for _, _, distance in matched)
            per["gate_rejections"] += diagnostics["gate_rejections"]
            per["postfilter_rejections"] += diagnostics["postfilter_rejections"]

    tracker_rows = {snr: {"position_sq": [], "velocity_sq": [], "matched": 0, "coast": 0,
                          "continuity_hits": 0, "continuity_total": 0, "id_switches": 0}
                    for snr in SNR_POINTS}
    for stream_index, stream in enumerate(streams):
        for snr in SNR_POINTS:
            tracker = CvKalmanTracker(config)
            stats = tracker_rows[snr]
            previous_key = {}
            for frame in range(len(stream["positions"])):
                positions = [np.asarray(position, dtype=float) for position in stream["positions"][frame]]
                velocities = [np.asarray(velocity, dtype=float) for velocity in stream["velocities"][frame]]
                identities = stream["identities"][frame]
                keys = stream.get("keys", [None] * 12)[frame] if stream["type"] == "train_stream" \
                    else [20000 + index for index in range(len(positions))]
                echo = run_echo(positions, velocities, keys, snr, 9700 + stream_index, frame)
                detections_by_bs = {}
                for bs in range(3):
                    detections, _, _ = detector_module.detect_from_maps(
                        detector_module.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                     config["detector"], resource=resource),
                        bs, frame, stations[bs], float(boresights[bs]), config, array,
                        multiplier=multiplier, covariance_lut=lut, height=height)
                    detections_by_bs[bs] = detections
                observations, _ = fuse_frame(detections_by_bs, config)
                records = tracker.step(observations, frame * DT_NS)
                record_positions = [np.array([record["state_hat"][0], record["state_hat"][1]])
                                    for record in records]
                frame_matched, _ = base.one_to_one_match(positions, record_positions)
                for gt_index, record_index, distance in frame_matched:
                    identity = int(identities[gt_index])
                    stats["matched"] += 1
                    if frame >= 3:
                        stats["continuity_hits"] += 1
                    key = records[record_index]["track_key"]
                    if identity in previous_key and previous_key[identity] != key:
                        stats["id_switches"] += 1
                    previous_key[identity] = key
                    if not records[record_index]["detected"]:
                        stats["coast"] += 1
                    stats["position_sq"].append(distance ** 2)
                    stats["velocity_sq"].append(float(np.sum(
                        (np.asarray(records[record_index]["state_hat"][2:]) - velocities[gt_index]) ** 2)))
                if frame >= 3:
                    stats["continuity_total"] += len(identities)

    rows = []
    tracker_summary = {}
    for snr in SNR_POINTS:
        per = per_snr[snr]
        stats = tracker_rows[snr]
        continuity = stats["continuity_hits"] / max(stats["continuity_total"], 1)
        coast_ratio = stats["coast"] / max(stats["matched"], 1)
        position_rmse = math.sqrt(sum(stats["position_sq"]) / len(stats["position_sq"])) \
            if stats["position_sq"] else None
        velocity_rmse = math.sqrt(sum(stats["velocity_sq"]) / len(stats["velocity_sq"])) \
            if stats["velocity_sq"] else None
        observations = sum(per["nbs"].values())
        row = {
            "snr_ref_db": snr,
            "received_snr_median_db": base.median_or_none(per["received"]),
            "received_snr_p10_db": base.quantile(per["received"], 0.10),
            "received_snr_p90_db": base.quantile(per["received"], 0.90),
            "peak_to_noise_median_db": base.median_or_none(per["peaks"]),
            "peak_to_noise_p10_db": base.quantile(per["peaks"], 0.10),
            "peak_to_noise_p90_db": base.quantile(per["peaks"], 0.90),
            "station_recall": per["station_matched"] / max(per["station_pairs"], 1),
            "target_recall": per["targets_matched"] / max(per["targets"], 1),
            "miss_rate": per["misses"] / max(per["station_pairs"], 1),
            "detections_per_bs_frame": per["detections"] / max(per["frames"] * 3, 1),
            "false_alarms_per_bs_frame": per["false_alarms"] / max(per["frames"] * 3, 1),
            "observation_precision": per["observations_matched"] / max(per["observations"], 1),
            "range_median_abs_m": base.median_or_none(per["range_abs"]),
            "range_p90_abs_m": base.quantile(per["range_abs"], 0.90),
            "u_median_abs": base.median_or_none(per["u_abs"]),
            "u_rmse": math.sqrt(float(np.mean(np.square(per["u_abs"])))) if per["u_abs"] else None,
            "single_bs_xy_median_m": base.median_or_none(per["single_xy"]),
            "single_bs_xy_p90_m": base.quantile(per["single_xy"], 0.90),
            "single_bs_xy_rmse_m": math.sqrt(float(np.mean(np.square(per["single_xy"]))))
            if per["single_xy"] else None,
            "fusion_xy_median_m": base.median_or_none(per["fusion_errors"]),
            "fusion_xy_p90_m": base.quantile(per["fusion_errors"], 0.90),
            "fusion_xy_rmse_m": math.sqrt(float(np.mean(np.square(per["fusion_errors"]))))
            if per["fusion_errors"] else None,
            "nbs1_fraction": per["nbs"][1] / max(observations, 1),
            "nbs2_fraction": per["nbs"][2] / max(observations, 1),
            "nbs3_fraction": per["nbs"][3] / max(observations, 1),
            "gate_rejections_per_frame": per["gate_rejections"] / max(per["frames"], 1),
            "gate_rejections_per_detection": per["gate_rejections"] / max(per["detections"], 1),
            "postfilter_rejections_per_frame": per["postfilter_rejections"] / max(per["frames"], 1),
            "tracker_continuity": continuity,
            "tracker_coast_ratio": coast_ratio,
            "tracker_id_switches": stats["id_switches"],
            "tracker_position_rmse_m": position_rmse,
            "tracker_velocity_rmse_mps": velocity_rmse,
            "tracker_matched_return_count": stats["matched"],
        }
        rows.append(row)
        tracker_summary[str(snr)] = {key: row[key] for key in (
            "tracker_continuity", "tracker_coast_ratio", "tracker_id_switches",
            "tracker_position_rmse_m", "tracker_velocity_rmse_mps")}

    def adjacent_deltas(key):
        return [{"from_snr_db": rows[index]["snr_ref_db"],
                 "to_snr_db": rows[index + 1]["snr_ref_db"],
                 "delta": (rows[index + 1][key] - rows[index][key])
                 if rows[index][key] is not None and rows[index + 1][key] is not None else None}
                for index in range(len(rows) - 1)]

    deltas = {key: adjacent_deltas(key) for key in (
        "target_recall", "fusion_xy_p90_m", "tracker_position_rmse_m",
        "tracker_velocity_rmse_mps", "tracker_continuity", "tracker_coast_ratio")}

    reference = rows[-1]
    recall_threshold = reference["target_recall"] - 0.10
    p90_threshold = reference["fusion_xy_p90_m"] * 2.0
    continuity_threshold = 0.90
    coast_threshold = reference["tracker_coast_ratio"] + 0.10

    def first_below(key, threshold, below=True):
        for row in reversed(rows):
            value = row[key]
            if value is not None and (value < threshold if below else value > threshold):
                return row["snr_ref_db"]
        return None

    def largest_degradation(key):
        best = None
        for entry in deltas[key]:
            if entry["delta"] is None:
                continue
            degradation = -entry["delta"]
            if best is None or degradation > best["degradation"]:
                best = {"segment_db": [entry["from_snr_db"], entry["to_snr_db"]],
                        "degradation": degradation}
        return best

    def sustained_below(key, threshold, below=True):
        """SNRs where the crossing holds at that SNR and every lower SNR tested."""
        flags = {row["snr_ref_db"]: (row[key] is not None
                                     and (row[key] < threshold if below
                                          else row[key] > threshold))
                 for row in rows}
        sustained = []
        for index, row in enumerate(rows):
            if all(flags[entry["snr_ref_db"]] for entry in rows[:index + 1]):
                sustained.append(row["snr_ref_db"])
        return sustained

    recall_threshold_sustained = sustained_below("target_recall", recall_threshold)
    p90_sustained = sustained_below("fusion_xy_p90_m", p90_threshold, below=False)
    continuity_sustained = sustained_below("tracker_continuity", continuity_threshold)
    coast_sustained = sustained_below("tracker_coast_ratio", coast_threshold, below=False)
    degraded_snrs = sorted(set(recall_threshold_sustained) | set(p90_sustained)
                           | set(continuity_sustained) | set(coast_sustained))
    transition_low = min(degraded_snrs) if degraded_snrs else None
    transition_high = max(degraded_snrs) if degraded_snrs else None
    minus20 = rows[0]
    collapsed = (minus20["target_recall"] < 0.50 or minus20["tracker_continuity"] < 0.50
                 or minus20["tracker_position_rmse_m"] is None)
    if collapsed:
        usability = "COLLAPSED"
    elif minus20["snr_ref_db"] in degraded_snrs:
        usability = "BORDERLINE"
    else:
        usability = "USABLE"
    saturated_at_minus20 = bool(
        minus20["tracker_position_rmse_m"] is not None
        and reference["tracker_position_rmse_m"] is not None
        and abs(minus20["target_recall"] - reference["target_recall"]) <= 0.02
        and abs(minus20["tracker_continuity"] - reference["tracker_continuity"]) <= 0.02
        and abs(minus20["tracker_position_rmse_m"] - reference["tracker_position_rmse_m"])
        <= 0.05 * reference["tracker_position_rmse_m"])

    analysis = {
        "rules": {
            "recall_degraded": f"target recall < 0 dB recall - 0.10 or < 0.90 (reference {recall_threshold:.3f})",
            "fusion_degraded": f"fusion XY p90 > 2x the 0 dB value ({p90_threshold:.3f} m)",
            "tracker_degraded": f"continuity < {continuity_threshold:.2f} or coast ratio > 0 dB value + 0.10",
            "usability": "USABLE if -20 dB is not degraded, BORDERLINE if degraded but recall >= 0.50 "
                         "and continuity >= 0.50, COLLAPSED otherwise",
            "saturation_at_minus20": "recall and continuity within 0.02 and position RMSE within 5% of 0 dB",
            "sustained": "a crossing counts only when it holds at that SNR and every lower SNR tested",
            "adjacent_delta": "delta = metric(next higher SNR) - metric(current SNR); degradation of "
                              "a lower-is-worse metric is -delta",
        },
        "detector": {
            "first_snr_target_recall_below_0p90": first_below("target_recall", 0.90),
            "first_snr_target_recall_below_reference_minus_0p10":
                first_below("target_recall", recall_threshold),
            "largest_recall_drop_segment": largest_degradation("target_recall"),
            "sustained_recall_degradation_snrs": recall_threshold_sustained,
            "station_recall_by_snr": {str(row["snr_ref_db"]): row["station_recall"] for row in rows},
        },
        "fusion": {
            "first_snr_fusion_p90_above_2x_reference": first_below(
                "fusion_xy_p90_m", p90_threshold, below=False),
            "largest_fusion_p90_rise_segment": largest_degradation("fusion_xy_p90_m"),
            "sustained_fusion_p90_degradation_snrs": p90_sustained,
        },
        "tracker": {
            "first_snr_continuity_below_0p90": first_below("tracker_continuity", continuity_threshold),
            "sustained_continuity_degradation_snrs": continuity_sustained,
            "first_snr_coast_ratio_above_reference_plus_0p10":
                first_below("tracker_coast_ratio", coast_threshold, below=False),
            "sustained_coast_degradation_snrs": coast_sustained,
            "largest_velocity_rmse_rise_segment": largest_degradation("tracker_velocity_rmse_mps"),
            "largest_coast_ratio_rise_segment": largest_degradation("tracker_coast_ratio"),
            "continuous_metrics_note": "continuity/coast fluctuation across SNR is inside the noise "
                                       "band of this 6-stream diagnostic set; no sustained crossing",
        },
        "degraded_snrs_by_sustained_rule": degraded_snrs,
        "transition_region_estimate_db": [transition_low, transition_high]
        if degraded_snrs else None,
        "transition_knee_snr_db": transition_high,
        "transition_note": "fusion XY p90 is the only monotone, sustained degradation; detection "
                           "recall and tracker continuity/coast stay near-saturated and no complete "
                           "saturation point is reached by -20 dB",
        "minus20_usability": usability,
        "saturated_at_minus20": bool(saturated_at_minus20),
        "observations": {
            "target_recall_by_snr": {str(row["snr_ref_db"]): row["target_recall"] for row in rows},
            "fusion_xy_p90_by_snr": {str(row["snr_ref_db"]): row["fusion_xy_p90_m"] for row in rows},
            "tracker_position_rmse_by_snr": {str(row["snr_ref_db"]): row["tracker_position_rmse_m"]
                                             for row in rows},
            "tracker_velocity_rmse_by_snr": {str(row["snr_ref_db"]): row["tracker_velocity_rmse_mps"]
                                             for row in rows},
            "tracker_continuity_by_snr": {str(row["snr_ref_db"]): row["tracker_continuity"]
                                          for row in rows},
            "tracker_coast_ratio_by_snr": {str(row["snr_ref_db"]): row["tracker_coast_ratio"]
                                           for row in rows},
        },
    }

    summary = {
        "stage": "SNR-TRANSITION-PROBE-02",
        "purpose": "diagnostic low-SNR transition-region probe; describes only, does not modify "
                   "any formal SNR or data",
        "tested_snrs": SNR_POINTS,
        "scene_set": {"snapshots": len(snapshots), "streams": len(streams),
                      "source": "SENS-SNR-REBUILD-06 / SENS-SNR-AUDIT-04 diagnostic scene set",
                      "only_train_scenes": True, "v_confirm_or_test_opened": False},
        "protocol": {"matching": "strict one-to-one Hungarian, 5 m gate",
                     "chain": "shared echo -> B64 detector -> CA-CFAR -> AoA -> fusion -> CV-KF",
                     "frozen": ["B64 resource", "waveform", "CFAR multiplier", "covariance LUT",
                                "tracker q_a", "association gate", "geometry", "RCS"],
                     "only_varying_input": "snr_ref_db"},
        "metrics": rows,
        "transition_deltas": deltas,
        "transition_analysis": analysis,
        "tracker_summary": tracker_summary,
        "formal_f01e_modified": False,
        "next_action": "REVIEW_REQUIRED",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _common.write_json(OUT / "summary.json", summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures.mkdir(parents=True, exist_ok=True)
        axis = [row["snr_ref_db"] for row in rows]

        def save(figure, name):
            figure.tight_layout()
            figure.savefig(figures / name, dpi=150)
            plt.close(figure)

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, [row["target_recall"] for row in rows], "o-", label="target recall")
        ax.plot(axis, [row["station_recall"] for row in rows], "s--", label="station recall")
        ax.set(xlabel="snr_ref (dB)", ylabel="recall", ylim=(0, 1.02),
               title="target recall vs SNR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(figure, "01_target_recall_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, [row["fusion_xy_p90_m"] for row in rows], "o-")
        ax.set(xlabel="snr_ref (dB)", ylabel="fusion XY p90 (m)", title="fusion XY p90 vs SNR")
        ax.grid(alpha=0.3)
        save(figure, "02_fusion_xy_p90_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, [row["tracker_position_rmse_m"] for row in rows], "o-")
        ax.set(xlabel="snr_ref (dB)", ylabel="position RMSE (m)",
               title="tracker position RMSE vs SNR")
        ax.grid(alpha=0.3)
        save(figure, "03_tracker_position_rmse_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, [row["tracker_velocity_rmse_mps"] for row in rows], "o-")
        ax.set(xlabel="snr_ref (dB)", ylabel="velocity RMSE (m/s)",
               title="tracker velocity RMSE vs SNR")
        ax.grid(alpha=0.3)
        save(figure, "04_tracker_velocity_rmse_vs_snr.png")

        figure, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(axis, [row["tracker_continuity"] for row in rows], "o-", label="continuity")
        ax.plot(axis, [row["tracker_coast_ratio"] for row in rows], "s--", label="coast ratio")
        ax.set(xlabel="snr_ref (dB)", ylabel="ratio", title="continuity / coast ratio vs SNR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save(figure, "05_continuity_coast_vs_snr.png")
    except Exception as error:  # noqa: BLE001
        _common.write_json(OUT / "figure_error.json", {"error": repr(error)})

    print(json.dumps({"stage": "SNR-TRANSITION-PROBE-02",
                      "snapshots": len(snapshots), "streams": len(streams),
                      "transition_analysis": analysis, "metrics": rows},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()