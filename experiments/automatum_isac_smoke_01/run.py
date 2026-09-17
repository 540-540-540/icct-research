"""AUTOMATUM-ISAC-SMOKE-01 runner.

Scope: Automatum split trajectories -> scene-level frame source -> shared clean echo ->
frame-level SNR noise -> non-CFAR multi-angle single-BS detector -> anonymous measurements ->
offline C-domain evaluation. No cross-BS association, no fusion, no tracking.

Usage (repo root):
    python experiments/automatum_isac_smoke_01/run.py --split both
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.automatum_isac_smoke_01.evaluate import (pair_errors, match_to_truth,  # noqa: E402
                                                          rd_crowded_mask, summarize,
                                                          truth_targets)
from frontend.automatum_scene_source import AutomatumSceneSource  # noqa: E402
from frontend.sensing.detector import detect_anonymous_measurements  # noqa: E402
from frontend.sensing.simulator import scale_shared_noise, synthesize_shared_frame_snr  # noqa: E402
from frontend.sensing.waveform import ArrayConfig, PaperWaveform, SensingResource  # noqa: E402

CONFIG_PATH = ROOT / "configs/automatum_isac_smoke.json"
DEFAULT_OUT = ROOT / "reports/isac_smoke/automatum_isac_smoke_01"
METRIC_KEYS = ("range", "radial_velocity", "bearing", "position")


def rd_processing_gain_db(samples: int) -> float:
    """Bin-centred target peak vs mean noise per RD cell for the periodic-Hann processing.

    With nr=2K and nv=2N, a per-entry target amplitude A gives a peak of N^2*A^2/4 summed
    over the 16 array elements, while the mean noise power per cell is 9*sigma^2/16.
    The value is used only for C-domain diagnosis of missed targets.
    """
    return 10 * math.log10((4.0 * samples ** 2) / 9.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sanitize(value):
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def select_frames(source: AutomatumSceneSource, scene_id: int, junction_center,
                  per_category: int, cap: int) -> tuple[list[int], dict, dict]:
    """Deterministic category union; ties break by smaller canonical frame id."""
    features = source.frame_features(scene_id, junction_center)
    frames = features["frames"]
    counts = features["counts"]
    selected: list[int] = []
    provenance: dict[int, list[str]] = {}

    def take(category: str, order: list[int], limit: int) -> None:
        added = 0
        for index in order:
            frame = int(frames[index])
            if frame not in selected:
                selected.append(frame)
                provenance.setdefault(frame, []).append(category)
                added += 1
            if added >= limit:
                break

    take("high_density", sorted(range(frames.size), key=lambda i: (-int(counts[i]), int(frames[i]))),
         per_category)
    take("low_density", sorted(range(frames.size), key=lambda i: (int(counts[i]), int(frames[i]))),
         per_category)
    median = int(np.median(counts))
    take("median_density",
         sorted(range(frames.size), key=lambda i: (abs(int(counts[i]) - median), int(frames[i]))),
         per_category)
    close = [i for i in range(frames.size)
             if int(counts[i]) >= 2 and math.isfinite(float(features["min_pair_m"][i]))]
    take("closest_pair", sorted(close, key=lambda i: (float(features["min_pair_m"][i]), int(frames[i]))),
         per_category)
    take("junction_center",
         sorted(range(frames.size), key=lambda i: (float(features["mean_junction_m"][i]), int(frames[i]))),
         per_category)
    take("far_target",
         sorted(range(frames.size), key=lambda i: (-float(features["max_junction_m"][i]), int(frames[i]))),
         per_category)
    selected.sort()
    if cap and len(selected) > cap:
        keep = np.floor(np.linspace(0, len(selected) - 1, cap)).astype(int)
        selected = [selected[i] for i in keep]
    return selected, features, provenance


def write_selection(path: Path, source: AutomatumSceneSource, frames_by_scene: dict,
                    features_by_scene: dict, provenance_by_scene: dict) -> None:
    rows = []
    for scene_id, frames in frames_by_scene.items():
        features = features_by_scene[scene_id]
        index = {int(frame): i for i, frame in enumerate(features["frames"])}
        for frame in frames:
            i = index[frame]
            rows.append({"scene_id": scene_id, "frame": frame,
                         "timestamp": source.at_frame(scene_id, frame).timestamp,
                         "vehicles": int(features["counts"][i]),
                         "min_pair_m": features["min_pair_m"][i],
                         "mean_junction_m": features["mean_junction_m"][i],
                         "max_junction_m": features["max_junction_m"][i],
                         "categories": ";".join(provenance_by_scene[scene_id].get(frame, []))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "frame", "timestamp", "vehicles",
                                                    "min_pair_m", "mean_junction_m", "max_junction_m",
                                                    "categories"])
        writer.writeheader()
        writer.writerows(rows)


def run_split(config: dict, split: str, csv_path: Path, frames_by_scene: dict, device: str,
              limit_frames: int = 0) -> dict:
    waveform = PaperWaveform(**config["waveform"])
    array = ArrayConfig(**config["array"])
    resource = SensingResource.from_config(config)
    if resource.mode != "full" or resource.active_symbols != waveform.N:
        raise ValueError("SMOKE-01 requires all 256 communication symbols reused for sensing")
    visibility = config["visibility"]
    detector_params = config["detector"]
    gates = config["smoke"]["match_gates"]
    snr_levels = [float(value) for value in config["snr"]["sweep_db"]]
    scene_map = {int(scene["scene_id"]): scene for scene in config["scenes"]}
    source = AutomatumSceneSource(csv_path, source_hz=float(config["dataset"]["source_hz"]),
                                  stride=int(config["dataset"]["stride"]))
    dtype = torch.float64
    gain_db = rd_processing_gain_db(waveform.N)

    pair_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    unmatched_rows: list[dict] = []
    group: dict[tuple[int, int, float], dict] = {}
    diagnostics: dict[tuple[int, int, int], dict] = {}
    captured_scenes: set[int] = set()
    started = time.time()

    diagnostic_frames = {
        scene_id: max(sorted(frames_by_scene[scene_id]),
                      key=lambda frame: len(source.at_frame(scene_id, frame)))
        for scene_id in frames_by_scene}
    for scene_id in sorted(frames_by_scene):
        scene = scene_map[scene_id]
        stations = np.asarray(scene["stations_xy_m"], dtype=np.float64)
        boresights = np.radians(np.asarray(scene["boresights_deg"], dtype=np.float64))
        frames = list(frames_by_scene[scene_id])[:limit_frames or None]
        for frame in frames:
            scene_frame = source.at_frame(scene_id, frame)
            rcs = [float(config["power"]["fixed_rcs_m2"])] * len(scene_frame)
            base = synthesize_shared_frame_snr(scene_frame.states[:, :2], scene_frame.states[:, 2:],
                                               scene_frame.vehicle_ids,
                                               stations, boresights, waveform, array, rcs,
                                               visibility, scene_id, frame, snr_db=None,
                                               device=device, dtype=dtype)
            visible = base["visible"].cpu().numpy()
            for bs in range(3):
                visible_count = int(visible[bs].sum())
                if visible_count == 0:
                    continue
                truth = truth_targets(scene_frame.states, stations[bs], boresights[bs], visibility)
                for snr_db in snr_levels:
                    noise = scale_shared_noise(base["S"][bs][None], base["W0"][bs][None], snr_db=snr_db,
                                               noise_variance=float(config["snr"].get("noise_variance", 1.0)))
                    Y_b = base["S"][bs] + noise["W"][0]
                    result = detect_anonymous_measurements(
                        Y_b, base["X"][bs], bs, stations[bs], boresights[bs], waveform, array,
                        detector_params, visibility, resource=resource)
                    measurements = result["measurements"]
                    matching = match_to_truth(measurements, truth, gates)
                    crowded_gt = rd_crowded_mask(truth)
                    peak_count: dict[tuple, int] = {}
                    for measurement in measurements:
                        cell = tuple(measurement["grid_index"][:2])
                        peak_count[cell] = peak_count.get(cell, 0) + 1
                    achieved = float(noise["achieved_snr_db"][0])
                    key = (scene_id, bs, snr_db)
                    stats = group.setdefault(key, {
                        "scene_id": scene_id, "station_id": bs, "target_snr_db": snr_db,
                        "frames": 0, "visible_targets": 0, "detections": 0,
                        "mismatch_frames": 0, "mismatch_abs": 0, "matched": 0,
                        "unmatched_detections": 0, "unmatched_targets": 0,
                        "unmatched_targets_crowded": 0, "unmatched_targets_isolated": 0,
                        "achieved_sum": 0.0, "achieved_max_abs_error_db": 0.0,
                        "rows": [], "rd_local_maxima": 0, "nms_suppressed": 0,
                        "roi_rejected": 0, "fov_rejected": 0, "ghost_detections": 0,
                    })
                    stats["frames"] += 1
                    stats["visible_targets"] += visible_count
                    stats["detections"] += len(measurements)
                    mismatch = abs(len(measurements) - visible_count)
                    if mismatch:
                        stats["mismatch_frames"] += 1
                        stats["mismatch_abs"] += mismatch
                        mismatch_rows.append({
                            "split": split, "scene_id": scene_id, "frame": frame, "station_id": bs,
                            "target_snr_db": snr_db, "timestamp": scene_frame.timestamp,
                            "detections": len(measurements), "visible_targets": visible_count,
                            "unmatched_detections": len(matching["unmatched_measurements"]),
                            "unmatched_targets": len(matching["unmatched_targets"]),
                            "unmatched_targets_crowded": int(sum(crowded_gt[t] for t
                                                                 in matching["unmatched_targets"])),
                            "unmatched_targets_isolated": int(sum(not crowded_gt[t] for t
                                                                  in matching["unmatched_targets"])),
                            "rd_local_maxima": result["counters"]["local_maxima"],
                            "nms_suppressed": result["counters"]["nms_suppressed"],
                            "roi_rejected": result["counters"]["roi_rejected"],
                            "fov_rejected": result["counters"]["fov_rejected"],
                            "ghost_detections": sum(1 for measurement in measurements
                                                    if abs(measurement["radial_velocity_hat"]) > 60.0),
                        })
                    if matching["unmatched_targets"]:
                        visible_rows = np.flatnonzero(truth["visible"])
                        for target in matching["unmatched_targets"]:
                            others = visible_rows[visible_rows != target]
                            in_window = ((np.abs(truth["r_m"][others] - truth["r_m"][target]) <= 8.0)
                                         & (np.abs(truth["radial_velocity_mps"][others]
                                                   - truth["radial_velocity_mps"][target]) <= 8.0))
                            if in_window.any():
                                delta = (truth["bearing_rad"][others[in_window]]
                                         - truth["bearing_rad"][target] + math.pi) % (2 * math.pi) - math.pi
                                separation = float(math.degrees(np.abs(delta).min()))
                            else:
                                separation = None
                            alpha = float(config["power"]["fixed_rcs_m2"]) / truth["r_m"][target] ** 2
                            per_target_snr_db = 10 * math.log10(alpha ** 2 / float(noise["signal_power"][0]))
                            margin = (per_target_snr_db + snr_db + gain_db
                                      - float(detector_params["rd_floor_factor_db"]))
                            unmatched_rows.append({
                                "split": split, "scene_id": scene_id, "frame": frame,
                                "station_id": bs, "target_snr_db": snr_db,
                                "timestamp": scene_frame.timestamp,
                                "vehicle_id": int(scene_frame.vehicle_ids[target]),
                                "range_gt_m": float(truth["r_m"][target]),
                                "radial_velocity_gt_mps": float(truth["radial_velocity_mps"][target]),
                                "bearing_gt_deg": float(math.degrees(truth["bearing_rad"][target])),
                                "crowded_gt": bool(crowded_gt[target]),
                                "nearest_rd_window_dbearing_deg": separation,
                                "per_target_share_db": float(per_target_snr_db),
                                "per_target_frame_snr_db": float(per_target_snr_db + snr_db),
                                "predicted_peak_margin_db": float(margin),
                            })
                    stats["matched"] += len(matching["pairs"])
                    stats["unmatched_detections"] += len(matching["unmatched_measurements"])
                    stats["unmatched_targets"] += len(matching["unmatched_targets"])
                    stats["unmatched_targets_crowded"] += int(sum(crowded_gt[t] for t
                                                                  in matching["unmatched_targets"]))
                    stats["unmatched_targets_isolated"] += int(sum(not crowded_gt[t] for t
                                                                   in matching["unmatched_targets"]))
                    stats["achieved_sum"] += achieved
                    stats["achieved_max_abs_error_db"] = max(stats["achieved_max_abs_error_db"],
                                                             abs(achieved - snr_db))
                    stats["rd_local_maxima"] += result["counters"]["local_maxima"]
                    stats["nms_suppressed"] += result["counters"]["nms_suppressed"]
                    stats["roi_rejected"] += result["counters"]["roi_rejected"]
                    stats["fov_rejected"] += result["counters"]["fov_rejected"]
                    stats["ghost_detections"] += sum(1 for measurement in measurements
                                                     if abs(measurement["radial_velocity_hat"]) > 60.0)
                    for row_index, target in matching["pairs"]:
                        cell = tuple(measurements[row_index]["grid_index"][:2])
                        crowded = bool(crowded_gt[target]) or peak_count[cell] > 1
                        errors = pair_errors(measurements[row_index], truth, target, crowded=crowded)
                        errors.update(split=split, scene_id=scene_id, frame=frame, station_id=bs,
                                      target_snr_db=snr_db, timestamp=scene_frame.timestamp,
                                      vehicle_id=int(scene_frame.vehicle_ids[target]),
                                      range_gt_m=float(truth["r_m"][target]),
                                      radial_velocity_gt_mps=float(truth["radial_velocity_mps"][target]),
                                      bearing_gt_deg=float(math.degrees(truth["bearing_rad"][target])))
                        stats["rows"].append(errors)
                        pair_rows.append(errors)
                    if (scene_id not in captured_scenes and snr_db == 0.0
                            and frame == diagnostic_frames[scene_id]):
                        captured_scenes.add(scene_id)
                        peak_counts: dict[tuple, int] = {}
                        for measurement in measurements:
                            cell = tuple(measurement["grid_index"][:2])
                            peak_counts[cell] = peak_counts.get(cell, 0) + 1
                        peak = max(result["rd_peaks"],
                                   key=lambda value: (peak_counts.get(tuple(value["grid_index"]), 0),
                                                      value["peak_power"]), default=None)
                        aoa = None
                        if peak is not None:
                            i, j = peak["grid_index"]
                            snapshot = result["spectrum"][:, i, j]
                            angle_spectrum = np.fft.fftshift(np.fft.fft(snapshot.cpu().numpy(), n=64))
                            aoa = {"u_grid": (2 * np.fft.fftshift(np.fft.fftfreq(64))).tolist(),
                                   "power_db": (20 * np.log10(np.maximum(np.abs(angle_spectrum), 1e-12))).tolist(),
                                   "peak_grid_index": [int(i), int(j)],
                                   "peak_range_m": float(peak["range_m"]),
                                   "peak_radial_velocity_mps": float(peak["radial_velocity_mps"])}
                        diagnostics[(scene_id, frame, bs)] = {
                            "power_rd": result["power_rd"].cpu().numpy(),
                            "ranges": result["ranges"].cpu().numpy(),
                            "velocities": result["velocities"].cpu().numpy(),
                            "detections": measurements, "rd_peaks": result["rd_peaks"],
                            "truth_r_m": truth["r_m"][truth["visible"]].tolist(),
                            "truth_vr_mps": truth["radial_velocity_mps"][truth["visible"]].tolist(),
                            "truth_bearing_rad": truth["bearing_rad"][truth["visible"]].tolist(),
                            "aoa": aoa, "timestamp": scene_frame.timestamp,
                        }
    groups = []
    for key in sorted(group):
        stats = group[key]
        rows = stats.pop("rows")
        stats["matched_pairs"] = stats.pop("matched")
        stats["isolated_pairs"] = sum(not row["crowded"] for row in rows)
        stats["crowded_pairs"] = sum(row["crowded"] for row in rows)
        stats["isolated"] = summarize([row for row in rows if not row["crowded"]])
        stats["crowded"] = summarize([row for row in rows if row["crowded"]])
        stats["achieved_snr_mean_db"] = stats.pop("achieved_sum") / max(stats["frames"], 1)
        stats["achieved_tolerance_db"] = float(config["snr"]["achieved_tolerance_db"])
        stats["achieved_pass"] = stats["achieved_max_abs_error_db"] <= stats["achieved_tolerance_db"]
        stats["mismatch_frame_rate"] = stats["mismatch_frames"] / max(stats["frames"], 1)
        stats.update(summarize(rows))
        groups.append(stats)
    summary = summarize(pair_rows)
    summary.update(frames=sum(g["frames"] for g in groups),
                   visible_targets=sum(g["visible_targets"] for g in groups),
                   detections=sum(g["detections"] for g in groups),
                   mismatch_frames=sum(g["mismatch_frames"] for g in groups),
                   mismatch_abs=sum(g["mismatch_abs"] for g in groups),
                   mismatch_frame_rate=(sum(g["mismatch_frames"] for g in groups)
                                        / max(sum(g["frames"] for g in groups), 1)),
                   achieved_max_abs_error_db=max([g["achieved_max_abs_error_db"] for g in groups] or [0.0]))
    return {"groups": groups, "summary": summary, "pair_rows": pair_rows,
            "mismatch_rows": mismatch_rows, "unmatched_rows": unmatched_rows,
            "diagnostics": diagnostics, "runtime_s": time.time() - started}


def gradient_table(groups: list[dict]) -> dict:
    """Per-scope, per-subset RMSE drop from the lowest to the highest SNR of the sweep.

    Subsets: ``all`` (every matched pair), ``isolated`` (no other visible target in the RD
    resolution window and a single angle measurement from that RD peak) and ``crowded``.
    """
    snr_levels = sorted({float(g["target_snr_db"]) for g in groups})
    scopes: dict[str, list[dict]] = {}
    for group in groups:
        keys = ["overall", f"scene_{group['scene_id']}",
                f"scene_{group['scene_id']}_bs_{group['station_id']}"]
        for scope in keys:
            for subset in ("all", "isolated", "crowded"):
                scopes.setdefault(f"{scope}|{subset}", []).append(group)
    table = {}
    for scope, members in scopes.items():
        subset = scope.split("|")[1]
        entry = {}
        for metric in METRIC_KEYS:
            series = []
            for snr in snr_levels:
                rows = [g for g in members if float(g["target_snr_db"]) == snr]
                pairs = sum(g["matched_pairs"] if subset == "all" else g[f"{subset}_pairs"] for g in rows)
                stats_key = metric if subset == "all" else None
                values = [(g[metric]["rmse"] if subset == "all" else g[subset][metric]["rmse"],
                           g["matched_pairs"] if subset == "all" else g[f"{subset}_pairs"]) for g in rows]
                values = [(value, count) for value, count in values if value is not None and count > 0]
                if pairs == 0 or not values:
                    series.append(None)
                    continue
                mse = sum(float(value) ** 2 * count for value, count in values) / pairs
                series.append(math.sqrt(mse))
            low, high = series[0], series[-1]
            ratio = high / low if low and high is not None else None
            entry[metric] = {"rmse_by_snr": series, "ratio_high_over_low": ratio}
        entry["pairs_total"] = sum(g["matched_pairs"] if subset == "all" else g[f"{subset}_pairs"]
                                   for g in members)
        table[scope] = entry
    return {"snr_levels_db": snr_levels, "scopes": table}


def plot_outputs(out_dir: Path, train: dict, gradient: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for stale in plots.glob("*.png"):
        stale.unlink()
    snr_levels = gradient["snr_levels_db"]
    for metric, label, unit in (("range", "range RMSE", "m"),
                                ("radial_velocity", "radial velocity RMSE", "m/s"),
                                ("bearing", "bearing RMSE", "deg"),
                                ("position", "single-BS position RMSE", "m")):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for scope in ("overall|all", "overall|isolated", "overall|crowded"):
            series = gradient["scopes"][scope][metric]["rmse_by_snr"]
            if any(value is None for value in series):
                continue
            axis.plot(snr_levels, series, marker="o", label=scope)
        axis.set_xlabel("frame-level SNR (dB)")
        axis.set_ylabel(f"{label} ({unit})")
        axis.set_yscale("log")
        axis.set_title(f"Automatum ISAC SMOKE-01: SNR vs {label}")
        axis.grid(True, alpha=0.3, which="both")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots / f"snr_vs_{metric}_rmse.png", dpi=140)
        plt.close(figure)

    for (scene_id, frame, bs), item in sorted(train["diagnostics"].items()):
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        power_db = 10 * np.log10(np.maximum(item["power_rd"], 1e-30))
        rows = (item["ranges"] >= 0.0) & (item["ranges"] <= 160.0)
        columns = (item["velocities"] >= -35.0) & (item["velocities"] <= 35.0)
        crop = power_db[np.ix_(rows, columns)]
        image = axes[0].imshow(crop, aspect="auto", origin="lower",
                               extent=[item["velocities"][columns][0], item["velocities"][columns][-1],
                                       item["ranges"][rows][0], item["ranges"][rows][-1]],
                               cmap="viridis", vmin=crop.max() - 60.0, vmax=crop.max())
        axes[0].scatter([m["radial_velocity_hat"] for m in item["detections"]],
                        [m["range_hat"] for m in item["detections"]], s=34, facecolors="none",
                        edgecolors="red", label="anonymous detections")
        axes[0].scatter(item["truth_vr_mps"], item["truth_r_m"], s=18, marker="x", c="white",
                        label="visible GT")
        axes[0].set_xlabel("radial velocity (m/s)")
        axes[0].set_ylabel("range (m)")
        axes[0].set_title(f"scene {scene_id} frame {frame} BS {bs} RD map @0 dB")
        axes[0].legend(loc="upper right", fontsize=8)
        figure.colorbar(image, ax=axes[0], label="power (dB)")
        aoa = item["aoa"]
        if aoa is not None:
            axes[1].plot(aoa["u_grid"], aoa["power_db"], label="AoA spectrum")
            for measurement in item["detections"]:
                if measurement["grid_index"][:2] == aoa["peak_grid_index"]:
                    axes[1].axvline(measurement["u_hat"], color="red", alpha=0.6, linestyle="--")
            axes[1].set_xlabel("direction cosine u")
            axes[1].set_ylabel("AoA spectrum (dB)")
            axes[1].set_title(f"AoA spectrum, RD peak r={aoa['peak_range_m']:.2f} m, "
                              f"vr={aoa['peak_radial_velocity_mps']:.2f} m/s")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plots / f"scene{scene_id}_frame{frame}_bs{bs}_rd_aoa.png", dpi=140)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-frames", type=int, default=0,
                        help="debug only: truncate the selected frames per scene")
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke = config["smoke"]
    scene_map = {int(scene["scene_id"]): scene for scene in config["scenes"]}
    selections, features, provenance = {}, {}, {}
    for split in ("train", "val"):
        if arguments.split not in ("both", split):
            continue
        csv_path = ROOT / config["dataset"][f"{split}_trajectories"]
        source = AutomatumSceneSource(csv_path, source_hz=float(config["dataset"]["source_hz"]),
                                      stride=int(config["dataset"]["stride"]))
        per_category = int(smoke["train_category_frames"] if split == "train"
                           else smoke["val_category_frames"])
        cap = int(smoke["train_frames_per_scene"] if split == "train"
                  else smoke["val_frames_per_scene"])
        selections[split], features[split], provenance[split] = {}, {}, {}
        for scene_id in sorted(scene_map):
            frames, scene_features, scene_provenance = select_frames(
                source, scene_id, scene_map[scene_id]["junction_center_xy_m"], per_category, cap)
            selections[split][scene_id] = frames
            features[split][scene_id] = scene_features
            provenance[split][scene_id] = scene_provenance
        write_selection(out_dir / f"selection_{split}.csv", source, selections[split],
                        features[split], provenance[split])

    results = {"train": None, "val": None}
    for split in ("train", "val"):
        if split not in selections:
            continue
        csv_path = ROOT / config["dataset"][f"{split}_trajectories"]
        results[split] = run_split(config, split, csv_path, selections[split], arguments.device,
                                   limit_frames=arguments.limit_frames)

    metrics = {
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "git_head": git_head(),
        "device": arguments.device,
        "dtype": "float64",
        "dt_s": float(config["dataset"]["stride"]) / float(config["dataset"]["source_hz"]),
        "waveform": config["waveform"],
        "sensing_resource": config["sensing_resource"],
        "array": config["array"],
        "visibility": config["visibility"],
        "detector": config["detector"],
        "snr_sweep_db": config["snr"]["sweep_db"],
        "selection": {split: {str(scene): frames for scene, frames in scene_frames.items()}
                      for split, scene_frames in selections.items()},
        "splits": {},
        "gradient": {},
        "checks": {},
    }
    for split, result in results.items():
        if result is None:
            continue
        gradient = gradient_table(result["groups"])
        unmatched = result["unmatched_rows"]
        isolated_unmatched = [row for row in unmatched if not row["crowded_gt"]]
        metrics["splits"][split] = {
            "summary": result["summary"],
            "groups": [{key: value for key, value in group.items() if key != "rows"}
                       for group in result["groups"]],
            "unmatched_analysis": {
                "total": len(unmatched),
                "crowded": sum(row["crowded_gt"] for row in unmatched),
                "isolated": len(isolated_unmatched),
                "isolated_negative_predicted_margin": sum(
                    1 for row in isolated_unmatched if row["predicted_peak_margin_db"] < 0),
                "isolated_nonnegative_predicted_margin": sum(
                    1 for row in isolated_unmatched if row["predicted_peak_margin_db"] >= 0),
            },
        }
        metrics["gradient"][split] = gradient
        with (out_dir / f"per_measurement_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["pair_rows"][0].keys())
                                    if result["pair_rows"] else ["split"])
            writer.writeheader()
            writer.writerows(result["pair_rows"])
        with (out_dir / f"count_mismatch_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["mismatch_rows"][0].keys())
                                    if result["mismatch_rows"] else ["split"])
            writer.writeheader()
            writer.writerows(result["mismatch_rows"])
        with (out_dir / f"unmatched_targets_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["unmatched_rows"][0].keys())
                                    if result["unmatched_rows"] else ["split"])
            writer.writeheader()
            writer.writerows(result["unmatched_rows"])
        with (out_dir / f"per_group_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            flat = []
            for group in result["groups"]:
                row = {"scene_id": group["scene_id"], "station_id": group["station_id"],
                       "target_snr_db": group["target_snr_db"], "frames": group["frames"],
                       "visible_targets": group["visible_targets"], "detections": group["detections"],
                       "mismatch_frames": group["mismatch_frames"], "mismatch_abs": group["mismatch_abs"],
                       "matched_pairs": group["matched_pairs"],
                       "unmatched_detections": group["unmatched_detections"],
                       "unmatched_targets": group["unmatched_targets"],
                       "achieved_snr_mean_db": group["achieved_snr_mean_db"],
                       "achieved_max_abs_error_db": group["achieved_max_abs_error_db"],
                       "rd_local_maxima": group["rd_local_maxima"],
                       "nms_suppressed": group["nms_suppressed"],
                       "roi_rejected": group["roi_rejected"],
                       "fov_rejected": group["fov_rejected"],
                       "ghost_detections": group["ghost_detections"]}
                for metric in METRIC_KEYS:
                    for statistic in ("mae", "rmse", "median", "p95"):
                        row[f"{metric}_{statistic}"] = group[metric][statistic]
                        row[f"isolated_{metric}_{statistic}"] = group["isolated"][metric][statistic]
                        row[f"crowded_{metric}_{statistic}"] = group["crowded"][metric][statistic]
                row["isolated_pairs"] = group["isolated_pairs"]
                row["crowded_pairs"] = group["crowded_pairs"]
                row["unmatched_targets_crowded"] = group["unmatched_targets_crowded"]
                row["unmatched_targets_isolated"] = group["unmatched_targets_isolated"]
                flat.append(row)
            writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()) if flat else ["scene_id"])
            writer.writeheader()
            writer.writerows(flat)
        metrics["checks"][split] = {
            "achieved_snr_pass": all(group["achieved_pass"] for group in result["groups"]),
            "achieved_snr_max_abs_error_db": result["summary"]["achieved_max_abs_error_db"],
            "gradient_ratio_high_over_low": {
                subset: {metric: gradient["scopes"][f"overall|{subset}"][metric]["ratio_high_over_low"]
                         for metric in METRIC_KEYS}
                for subset in ("all", "isolated", "crowded")},
        }
    metrics["count_mismatch_note"] = ("GT count is used only here, after the detector, to report "
                                      "count consistency; it never enters the detector.")
    write_json(out_dir / "metrics.json", metrics)
    if results["train"] is not None:
        plot_outputs(out_dir, results["train"], metrics["gradient"]["train"])
    print(json.dumps({"train_summary": results["train"]["summary"] if results["train"] else None,
                      "val_summary": results["val"]["summary"] if results["val"] else None,
                      "gradient": metrics["checks"],
                      "runtime_s": {split: round(result["runtime_s"], 1)
                                    for split, result in results.items() if result is not None}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())