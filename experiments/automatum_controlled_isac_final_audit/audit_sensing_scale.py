"""Final Audit E-I, K, L: Candidate B sensing errors on the de-duplicated prediction-history
states, normalized against the dataset motion scales, speed-bin behaviour, SNR gradient shape,
temporal stability, BS-count penalty and condition-ratio distribution.

Candidate B values, SNR levels, geometry, fusion and quality code are read-only.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final_audit import common  # noqa: E402
from frontend.controlled_isac.automatum_frontend import sense_vehicle  # noqa: E402
from frontend.controlled_isac.automatum_measurement import setup_from_config  # noqa: E402

SPEED_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0),
              (20.0, float("inf")))
STABILITY_LEVELS = (-10.0, 0.0, 10.0)
LOW_SPEED_LIMIT = 2.0
CONDITION_LOW_THRESHOLD = 0.05


def error_stats(values) -> dict:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)],
                       dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {"n": int(array.size), "mae": float(np.abs(array).mean()),
            "rmse": float(np.sqrt((array ** 2).mean())), "median": float(np.median(np.abs(array))),
            "p75": float(np.percentile(np.abs(array), 75)),
            "p90": float(np.percentile(np.abs(array), 90)),
            "p95": float(np.percentile(np.abs(array), 95)),
            "p99": float(np.percentile(np.abs(array), 99)),
            "max": float(np.abs(array).max())}


def run_sensing(split: str) -> tuple[list[tuple], dict, dict]:
    config = common.load_config()
    calibration = common.load_calibration(config)
    if not common.calibration_is_candidate_b(calibration):
        raise SystemExit("config measurement.a is not Candidate B; audit aborted (read-only)")
    setup = setup_from_config(config)
    history = common.unique_history(split)
    lookup, _ = common.state_lookup(split)
    rows, stability, blind = [], {}, 0
    for scene_id, frame_id, vehicle in history["keys"]:
        state = lookup[(scene_id, frame_id, vehicle)]
        speed = float(np.hypot(state[2], state[3]))
        for level in common.LEVELS:
            try:
                estimate = sense_vehicle(scene_id, frame_id, vehicle, state[:2], state[2:],
                                         level, calibration, setup)
            except ValueError:
                blind += 1
                rows.append((scene_id, frame_id, vehicle, level, 0, 0, 0.0,
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, speed,
                             float(state[2]), float(state[3])))
                continue
            dx = estimate["x_hat"] - float(state[0])
            dy = estimate["y_hat"] - float(state[1])
            dvx = estimate["vx_hat"] - float(state[2])
            dvy = estimate["vy_hat"] - float(state[3])
            rows.append((scene_id, frame_id, vehicle, level, estimate["n_bs"], estimate["rank"],
                         estimate["condition_ratio"], dx, dy, dvx, dvy,
                         float(math.hypot(dx, dy)), float(math.hypot(dvx, dvy)), speed,
                         float(state[2]), float(state[3])))
            if level in STABILITY_LEVELS:
                stability[(scene_id, frame_id, vehicle, level)] = (
                    estimate["x_hat"], estimate["y_hat"], estimate["vx_hat"], estimate["vy_hat"],
                    float(state[0]), float(state[1]), float(state[2]), float(state[3]))
    return rows, stability, {"blind": blind, "states": history["unique_states"]}


def by_snr_rows(split: str, rows: list[tuple]) -> list[dict]:
    output = []
    for scope_name, scene_filter in (("overall", (0, 1)), ("scene_0", (0,)), ("scene_1", (1,))):
        for level in common.LEVELS:
            selected = [row for row in rows
                        if row[0] in scene_filter and row[3] == level and row[4] >= 1]
            entry = {"split": split, "scope": scope_name, "snr_db": level,
                     "n_evaluated": len(selected),
                     "n_blind": sum(1 for row in rows
                                    if row[0] in scene_filter and row[3] == level and row[4] == 0)}
            entry["position"] = error_stats([row[11] for row in selected])
            entry["velocity"] = error_stats([row[12] for row in selected])
            entry["vx"] = error_stats([row[7] for row in selected])
            entry["vy"] = error_stats([row[8] for row in selected])
            output.append(entry)
    return output


def flatten_by_snr(entry: dict) -> dict:
    flat = {"split": entry["split"], "scope": entry["scope"], "snr_db": entry["snr_db"],
            "n_evaluated": entry["n_evaluated"], "n_blind": entry["n_blind"]}
    for key in ("position", "velocity", "vx", "vy"):
        for stat, value in entry[key].items():
            flat[f"{key}_{stat}"] = value
    return flat


def speed_bin_rows(split: str, rows: list[tuple]) -> list[dict]:
    output = []
    for scope_name, scene_filter in (("overall", (0, 1)), ("scene_0", (0,)), ("scene_1", (1,))):
        for low, high in SPEED_BINS:
            label = f"[{low:g},{high:g})" if math.isfinite(high) else f"[{low:g},inf)"
            for level in common.LEVELS:
                selected = [row for row in rows if row[0] in scene_filter and row[3] == level
                            and row[4] >= 1 and low <= row[13] < high]
                entry = {"split": split, "scope": scope_name, "speed_bin_mps": label,
                         "snr_db": level, "n": len(selected)}
                if not selected:
                    output.append(entry)
                    continue
                entry["mean_true_speed_mps"] = float(np.mean([row[13] for row in selected]))
                entry["position_rmse_m"] = float(np.sqrt(np.mean([row[11] ** 2
                                                                  for row in selected])))
                entry["velocity_rmse_mps"] = float(np.sqrt(np.mean([row[12] ** 2
                                                                    for row in selected])))
                entry["velocity_error_median_mps"] = float(np.median([row[12] for row in selected]))
                entry["velocity_error_p95_mps"] = float(np.percentile([row[12] for row in selected],
                                                                       95))
                refined = [row for row in selected if row[4] >= 2]
                entry["n_ge2bs"] = len(refined)
                if refined:
                    entry["position_rmse_m_ge2bs"] = float(np.sqrt(np.mean([row[11] ** 2
                                                                            for row in refined])))
                    entry["velocity_rmse_mps_ge2bs"] = float(np.sqrt(np.mean([row[12] ** 2
                                                                              for row in refined])))
                if low >= LOW_SPEED_LIMIT:
                    entry["velocity_error_over_speed_median"] = float(
                        np.median([row[12] / row[13] for row in selected]))
                    entry["velocity_error_over_speed_p95"] = float(
                        np.percentile([row[12] / row[13] for row in selected], 95))
                    angles, flips = [], 0
                    for row in selected:
                        vx_hat, vy_hat = row[14] + row[7], row[15] + row[8]
                        dot = vx_hat * row[14] + vy_hat * row[15]
                        norm = math.hypot(vx_hat, vy_hat) * row[13]
                        cos_angle = dot / max(norm, 1e-12)
                        angles.append(math.degrees(math.acos(min(max(cos_angle, -1.0), 1.0))))
                        flips += int(cos_angle < 0.0)
                    entry["direction_angle_median_deg"] = float(np.median(angles))
                    entry["direction_flip_fraction"] = flips / len(selected)
                else:
                    entry["velocity_error_over_speed_median"] = None
                    entry["velocity_error_over_speed_p95"] = None
                    entry["direction_angle_median_deg"] = None
                    entry["direction_flip_fraction"] = None
                output.append(entry)
    return output


def gradient_table(rows: list[tuple], split: str) -> list[dict]:
    ordered = sorted(common.LEVELS)
    table = []
    for scope_name, scene_filter in (("overall", (0, 1)), ("scene_0", (0,)), ("scene_1", (1,))):
        rmse = {}
        for level in ordered:
            selected = [row for row in rows if row[0] in scene_filter and row[3] == level
                        and row[4] >= 1]
            rmse[level] = {
                "position": float(np.sqrt(np.mean([row[11] ** 2 for row in selected]))),
                "velocity": float(np.sqrt(np.mean([row[12] ** 2 for row in selected]))),
                "n": len(selected)}
        previous = None
        for level in ordered:
            entry = {"split": split, "scope": scope_name, "snr_db": level,
                     "n": rmse[level]["n"],
                     "position_rmse_m": rmse[level]["position"],
                     "velocity_rmse_mps": rmse[level]["velocity"]}
            if previous is not None:
                entry["position_ratio_vs_higher_snr"] = rmse[level]["position"] / previous["position"]
                entry["velocity_ratio_vs_higher_snr"] = rmse[level]["velocity"] / previous["velocity"]
            table.append(entry)
            previous = rmse[level]
        table.append({"split": split, "scope": scope_name, "snr_db": "overall_+10_to_-10",
                      "position_ratio_vs_higher_snr": rmse[ordered[-1]]["position"]
                      / rmse[ordered[0]]["position"],
                      "velocity_ratio_vs_higher_snr": rmse[ordered[-1]]["velocity"]
                      / rmse[ordered[0]]["velocity"]})
    return table


def stability_table(stability: dict, split: str) -> list[dict]:
    output = []
    for level in STABILITY_LEVELS:
        for speed_class, predicate in (("low_speed_lt2", lambda speed: speed < LOW_SPEED_LIMIT),
                                       ("normal_speed_ge2", lambda speed: speed >= LOW_SPEED_LIMIT)):
            displacement_errors, second_differences, error_deltas = [], [], []
            velocity_angles, flips, speed_ratios, speed_anomalies = [], 0, [], 0
            groups = {}
            for (scene_id, frame_id, vehicle, row_level), value in stability.items():
                if row_level != level:
                    continue
                groups.setdefault((scene_id, vehicle), {})[frame_id] = value
            for (scene_id, vehicle), frames in groups.items():
                for frame_id in sorted(frames):
                    entry = frames[frame_id]
                    speed = math.hypot(entry[6], entry[7])
                    if not predicate(speed):
                        continue
                    if speed >= LOW_SPEED_LIMIT:
                        # direction / speed of the velocity estimate itself (the meaningful check;
                        # the frame-to-frame velocity *change* is not usable at 10 Hz because the
                        # true change is of the same order as the GT velocity noise)
                        hat_speed = math.hypot(entry[2], entry[3])
                        cos_angle = (entry[2] * entry[6] + entry[3] * entry[7]) / max(
                            hat_speed * speed, 1e-12)
                        velocity_angles.append(
                            math.degrees(math.acos(min(max(cos_angle, -1.0), 1.0))))
                        flips += int(cos_angle < 0.0)
                        ratio = hat_speed / speed
                        speed_ratios.append(ratio)
                        speed_anomalies += int(ratio > 1.5 or ratio < 0.5)
                    if frame_id + 1 in frames:
                        nxt = frames[frame_id + 1]
                        displacement_errors.append(math.hypot(
                            (nxt[0] - entry[0]) - (nxt[4] - entry[4]),
                            (nxt[1] - entry[1]) - (nxt[5] - entry[5])))
                        error_deltas.append(math.hypot(
                            (nxt[0] - nxt[4]) - (entry[0] - entry[4]),
                            (nxt[1] - nxt[5]) - (entry[1] - entry[5])))
                    if frame_id + 1 in frames and frame_id + 2 in frames:
                        second = frames[frame_id + 1]
                        third = frames[frame_id + 2]
                        positions = [(entry[4], entry[5]), (second[4], second[5]),
                                     (third[4], third[5])]
                        estimates = [(entry[0], entry[1]), (second[0], second[1]),
                                     (third[0], third[1])]
                        gt_curvature = math.hypot(positions[2][0] - 2 * positions[1][0] + positions[0][0],
                                                  positions[2][1] - 2 * positions[1][1] + positions[0][1])
                        hat_curvature = math.hypot(estimates[2][0] - 2 * estimates[1][0] + estimates[0][0],
                                                   estimates[2][1] - 2 * estimates[1][1] + estimates[0][1])
                        second_differences.append(abs(hat_curvature - gt_curvature))
            output.append({
                "split": split, "snr_db": level, "speed_class": speed_class,
                "position_step_error": error_stats(displacement_errors),
                "position_error_delta": error_stats(error_deltas),
                "second_difference_error_m": error_stats(second_differences),
                "velocity_angle_vs_true_deg": error_stats(velocity_angles),
                "direction_flip_fraction": (flips / len(velocity_angles)) if velocity_angles else None,
                "speed_ratio_median": float(np.median(speed_ratios)) if speed_ratios else None,
                "speed_anomaly_fraction": (speed_anomalies / len(speed_ratios)) if speed_ratios else None,
            })
    return output


def bs_count_rows(rows: list[tuple], split: str) -> list[dict]:
    output = []
    for level in common.LEVELS:
        for count in (1, 2, 3):
            selected = [row for row in rows if row[3] == level and row[4] == count]
            entry = {"split": split, "snr_db": level, "n_bs": count, "n": len(selected)}
            if selected:
                entry["position_rmse_m"] = float(np.sqrt(np.mean([row[11] ** 2
                                                                  for row in selected])))
                entry["velocity_rmse_mps"] = float(np.sqrt(np.mean([row[12] ** 2
                                                                    for row in selected])))
                entry["velocity_rmse_median_mps"] = float(np.median([row[12] for row in selected]))
            output.append(entry)
    return output


def conditioning_rows(rows: list[tuple], split: str) -> list[dict]:
    output = []
    for level in common.LEVELS:
        for count in (2, 3):
            values = np.asarray([row[6] for row in rows if row[3] == level and row[4] == count],
                                dtype=np.float64)
            entry = {"split": split, "snr_db": level, "n_bs": count, "n": int(values.size)}
            if values.size:
                for percentile in (1, 5, 10, 25, 50, 75, 90):
                    entry[f"p{percentile:02d}"] = float(np.percentile(values, percentile))
                entry["min"] = float(values.min())
                entry["fraction_below_0p05"] = float(np.mean(values < CONDITION_LOW_THRESHOLD))
            output.append(entry)
    return output


def normalized_payload(split_rows: dict, scale: dict) -> dict:
    payload = {}
    for split in ("train", "val"):
        scales = scale["splits"][split]
        median_frame = scales["single_frame_displacement_m"]["p50"]
        median_1s = scales["ten_frame_displacement_m"]["p50"]
        median_2s = scales["history_span_20frame_m"]["p50"]
        median_nn = scales["nearest_neighbor_m"]["all"]["p50"]
        median_speed = scales["speed"]["all"]["p50"]
        p25_speed = scales["speed"]["all"]["p25"]
        p75_speed = scales["speed"]["all"]["p75"]
        horizon = 20 * common.DT
        levels = {}
        for level in common.LEVELS:
            selected = [row for row in split_rows[split] if row[3] == level and row[4] >= 1]
            position_errors = np.asarray([row[11] for row in selected])
            velocity_errors = np.asarray([row[12] for row in selected])
            levels[str(level)] = {
                "position_rmse_over_median_frame_displacement": float(
                    np.sqrt((position_errors ** 2).mean()) / median_frame),
                "position_rmse_over_median_1s_displacement": float(
                    np.sqrt((position_errors ** 2).mean()) / median_1s),
                "position_rmse_over_median_2s_displacement": float(
                    np.sqrt((position_errors ** 2).mean()) / median_2s),
                "position_rmse_over_median_nearest_neighbor": float(
                    np.sqrt((position_errors ** 2).mean()) / median_nn),
                "position_p95_over_median_frame_displacement": float(
                    np.percentile(position_errors, 95) / median_frame),
                "position_p95_over_median_nearest_neighbor": float(
                    np.percentile(position_errors, 95) / median_nn),
                "velocity_rmse_over_median_speed": float(
                    np.sqrt((velocity_errors ** 2).mean()) / median_speed),
                "velocity_rmse_over_p25_speed": float(
                    np.sqrt((velocity_errors ** 2).mean()) / p25_speed),
                "velocity_rmse_over_p75_speed": float(
                    np.sqrt((velocity_errors ** 2).mean()) / p75_speed),
                "velocity_error_accumulated_2s_over_median_2s_displacement": float(
                    np.sqrt((velocity_errors ** 2).mean()) * horizon / median_2s),
            }
        payload[split] = {"scales_used": {"median_frame_displacement_m": median_frame,
                                          "median_1s_displacement_m": median_1s,
                                          "median_2s_history_span_m": median_2s,
                                          "median_nearest_neighbor_m": median_nn,
                                          "median_speed_mps": median_speed,
                                          "p25_speed_mps": p25_speed,
                                          "p75_speed_mps": p75_speed},
                          "levels": levels}
    return payload


def plot_outputs(split_rows: dict, speed_rows: list[dict], normalized: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = common.OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    levels = sorted(common.LEVELS)
    for stale in ("snr_vs_position_rmse.png", "snr_vs_velocity_rmse.png",
                  "normalized_error_vs_snr.png", "velocity_error_by_speed_bin.png",
                  "condition_ratio_distribution.png"):
        (plots / stale).unlink(missing_ok=True)
    for metric, filename, label in (("position_rmse_m", "snr_vs_position_rmse.png",
                                     "position RMSE (m)"),
                                    ("velocity_rmse_mps", "snr_vs_velocity_rmse.png",
                                     "velocity RMSE (m/s)")):
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        for split in ("train", "val"):
            for scope, style in (("overall", "-"), ("scene_0", "--"), ("scene_1", ":")):
                series = []
                for level in levels:
                    selected = [row for row in split_rows[split]
                                if row[3] == level and row[4] >= 1 and row[0] in
                                ((0, 1) if scope == "overall" else ((0,) if scope == "scene_0" else (1,)))]
                    series.append(float(np.sqrt(np.mean([row[11] ** 2 if metric.startswith("position")
                                                         else row[12] ** 2 for row in selected]))))
                axis.plot(levels, series, style, alpha=1.0 if split == "train" else 0.45,
                          marker="o", label=f"{split} {scope}")
        axis.set_xlabel("frame-level SNR (dB)")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.set_title(f"Automatum Route-B Candidate B: SNR vs {label}")
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=140)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    labels = ["pos / median\nframe disp", "pos / median\n1s disp", "pos P95 / median\nframe disp",
              "vel / median speed", "vel×2s / median\n2s span"]
    keys = ["position_rmse_over_median_frame_displacement",
            "position_rmse_over_median_1s_displacement",
            "position_p95_over_median_frame_displacement",
            "velocity_rmse_over_median_speed",
            "velocity_error_accumulated_2s_over_median_2s_displacement"]
    for split, style in (("train", "-"), ("val", "--")):
        for key, label in zip(keys, labels):
            series = [normalized[split]["levels"][str(level)][key] for level in levels]
            axis.plot(levels, series, style, marker="o", label=f"{split}: {label}")
    axis.set_xlabel("frame-level SNR (dB)")
    axis.set_ylabel("normalized error ratio")
    axis.set_yscale("log")
    axis.set_title("Automatum Route-B Candidate B: normalized error vs SNR")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(plots / "normalized_error_vs_snr.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    bins = [entry["speed_bin_mps"] for entry in speed_rows if entry["split"] == "train"
            and entry["scope"] == "overall"]
    unique_bins = []
    for label in bins:
        if label not in unique_bins:
            unique_bins.append(label)
    width = 0.15
    for index, level in enumerate(levels):
        values = [next((entry["velocity_rmse_mps"] for entry in speed_rows
                        if entry["split"] == "train" and entry["scope"] == "overall"
                        and entry["snr_db"] == level and entry["speed_bin_mps"] == label), None)
                  for label in unique_bins]
        positions = np.arange(len(unique_bins)) + (index - 2) * width
        axis.bar(positions, values, width=width, label=f"{level:+.0f} dB")
    axis.set_xticks(np.arange(len(unique_bins)))
    axis.set_xticklabels(unique_bins, fontsize=8, rotation=15)
    axis.set_xlabel("true speed bin (m/s)")
    axis.set_ylabel("velocity RMSE (m/s)")
    axis.set_title("Automatum Route-B Candidate B: velocity error by speed bin (train overall)")
    axis.grid(True, alpha=0.3, axis="y")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "velocity_error_by_speed_bin.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for count, color in ((2, "tab:blue"), (3, "tab:green")):
        values = [row[6] for row in split_rows["train"] if row[3] == 0.0 and row[4] == count]
        axis.hist(values, bins=40, alpha=0.6, color=color,
                  label=f"n_bs={count} (n={len(values)})")
    axis.axvline(CONDITION_LOW_THRESHOLD, color="red", linestyle="--",
                 label=f"existing low-condition threshold {CONDITION_LOW_THRESHOLD:g}")
    axis.set_xlabel("velocity-fusion condition ratio")
    axis.set_ylabel("unique prediction-history states")
    axis.set_yscale("log")
    axis.set_title("Automatum Route-B: velocity-fusion condition ratio (train, 0 dB)")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "condition_ratio_distribution.png", dpi=140)
    plt.close(figure)


def main() -> int:
    started = time.time()
    scale = json.loads((common.OUT_DIR / "prediction_cohort_scale.json").read_text())
    split_rows, stability_all = {}, {}
    by_snr, by_speed, gradient, stability, bs_count, conditioning = [], [], [], [], [], []
    for split in ("train", "val"):
        rows, stability_data, meta = run_sensing(split)
        split_rows[split] = rows
        stability_all[split] = stability_data
        if meta["blind"]:
            print(f"WARN {split}: {meta['blind']} blind (0-BS) state-level rows excluded from errors")
        by_snr.extend(by_snr_rows(split, rows))
        by_speed.extend(speed_bin_rows(split, rows))
        gradient.extend(gradient_table(rows, split))
        stability.extend(stability_table(stability_data, split))
        bs_count.extend(bs_count_rows(rows, split))
        conditioning.extend(conditioning_rows(rows, split))
        print(f"{split}: {meta['states']} unique states x {len(common.LEVELS)} levels done")
    normalized = normalized_payload(split_rows, scale)
    common.write_csv(common.OUT_DIR / "sensing_scale_by_snr.csv",
                     [flatten_by_snr(entry) for entry in by_snr])
    common.write_csv(common.OUT_DIR / "sensing_scale_by_speed.csv", by_speed)
    common.write_csv(common.OUT_DIR / "snr_gradient.csv", gradient)
    common.write_csv(common.OUT_DIR / "temporal_stability.csv", stability)
    common.write_csv(common.OUT_DIR / "bs_count_error_scale.csv", bs_count)
    common.write_csv(common.OUT_DIR / "geometry_conditioning.csv", conditioning)
    common.write_json(common.OUT_DIR / "normalized_error_scale.json", normalized)
    plot_outputs(split_rows, by_speed, normalized)
    print(f"written in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())