#!/usr/bin/env python3
"""Build Production ISAC Sensing Cache for train / val / test splits.

Chain:
  trajectories.csv GT state
    -> Frozen 3-BS Route-B ISAC (frontend.controlled_isac.automatum_frontend.sense_vehicle)
    -> [x_hat, y_hat, vx_hat, vy_hat]
    -> Production Sensing Cache (data/automatum_t_crossing/isac/<split>/sensing_cache.npz)

The cache covers all unique prediction-history states:
  (scene_id, frame, vehicle_id)
where frame = start_frame + 0...19, across all valid windows in samples.npz.
For each unique state, 5 SNR levels [-10, -5, 0, +5, +10] dB are evaluated.

Outputs per split:
  data/automatum_t_crossing/isac/<split>/sensing_cache.npz
  data/automatum_t_crossing/isac/<split>/sensing_manifest.json
Mirrored reports:
  reports/isac_production_cache/<split>_sensing_manifest.json

Usage:
  python tools/data_preprocessing/build_isac_production_cache.py [--split all|train|val|test]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontend.controlled_isac.automatum_frontend import STATE_FIELDS, sense_vehicle
from frontend.controlled_isac.automatum_measurement import (
    calibration_from_config,
    setup_from_config,
)

CONFIG_PATH = ROOT / "configs/automatum_controlled_isac.json"
OUT_BASE = ROOT / "data/automatum_t_crossing/isac"
REPORT_DIR = ROOT / "reports/isac_production_cache"

SNR_LEVELS = [-10.0, -5.0, 0.0, 5.0, 10.0]
CACHE_KEYS = [
    "scene_id",
    "frame",
    "timestamp",
    "vehicle_id",
    "snr_levels_db",
    "state_hat",
    "n_bs",
    "condition_ratio",
    "rank",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz_deterministic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in CACHE_KEYS:
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0
            info.internal_attr = 0
            info.flag_bits = 0
            with zf.open(info, "w") as fid:
                npy_format.write_array(fid, np.ascontiguousarray(arrays[key]), allow_pickle=False)


def write_json(payload: dict, path: Path) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def load_trajectories_lookup(path: Path) -> dict[tuple[int, int, int], tuple[float, float, float, float, float]]:
    """Parse trajectories.csv into mapping: (scene_id, frame, vehicle_id) -> (timestamp, x, y, vx, vy)."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    scene = table["scene_id"].astype(int)
    vehicle = table["vehicle_id"].astype(int)
    ts = table["timestamp"].astype(np.float64)
    frame = np.rint(ts * 29.97).astype(np.int64) // 3
    x = table["x"].astype(np.float64)
    y = table["y"].astype(np.float64)
    vx = table["vx"].astype(np.float64)
    vy = table["vy"].astype(np.float64)

    lookup = {}
    for i in range(len(scene)):
        lookup[(int(scene[i]), int(frame[i]), int(vehicle[i]))] = (
            float(ts[i]),
            float(x[i]),
            float(y[i]),
            float(vx[i]),
            float(vy[i]),
        )
    return lookup


def extract_unique_history_keys(samples_path: Path) -> tuple[list[tuple[int, int, int]], int, int]:
    """Extract sorted unique (scene_id, frame, vehicle_id) keys from prediction history windows."""
    samples = np.load(samples_path)
    keys: set[tuple[int, int, int]] = set()
    total_history_rows = 0
    num_samples = int(samples["scene_id"].size)

    for index in range(num_samples):
        scene_id = int(samples["scene_id"][index])
        start = int(samples["start_frame"][index])
        ids = samples["vehicle_ids"][index][samples["vehicle_mask"][index]].tolist()
        for offset in range(20):
            frame = start + offset
            for vehicle in ids:
                keys.add((scene_id, frame, int(vehicle)))
                total_history_rows += 1

    sorted_keys = sorted(keys)
    return sorted_keys, num_samples, total_history_rows


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def build_split_cache(
    split: str,
    config: dict,
    setup: dict,
    calibration: dict,
    git_commit: str,
    output_base: Path,
    report_base: Path,
) -> dict:
    print(f"\n==========================================")
    print(f"Building Production ISAC Cache: {split.upper()}")
    print(f"==========================================")

    splits_dir = ROOT / "data/automatum_t_crossing/splits" / split
    samples_path = splits_dir / "samples.npz"
    trajectories_path = splits_dir / "trajectories.csv"

    samples_sha = sha256_file(samples_path)
    trajectories_sha = sha256_file(trajectories_path)
    print(f"[{split}] trajectories.csv: {trajectories_path} (SHA256: {trajectories_sha[:16]}...)")
    print(f"[{split}] samples.npz:      {samples_path} (SHA256: {samples_sha[:16]}...)")

    sorted_keys, num_samples, total_history_rows = extract_unique_history_keys(samples_path)
    num_unique = len(sorted_keys)
    dedup_factor = total_history_rows / num_unique if num_unique > 0 else 0.0
    print(f"[{split}] Prediction samples: {num_samples:,}")
    print(f"[{split}] History rows before dedup: {total_history_rows:,}")
    print(f"[{split}] Unique history states:     {num_unique:,} (dedup factor: {dedup_factor:.2f}x)")

    lookup = load_trajectories_lookup(trajectories_path)
    missing_keys = [k for k in sorted_keys if k not in lookup]
    if missing_keys:
        raise ValueError(f"[{split}] {len(missing_keys)} unique keys not found in trajectories.csv!")

    # Pre-allocate arrays
    num_snrs = len(SNR_LEVELS)
    scene_id_arr = np.empty(num_unique, dtype=np.uint8)
    frame_arr = np.empty(num_unique, dtype=np.int32)
    timestamp_arr = np.empty(num_unique, dtype=np.float64)
    vehicle_id_arr = np.empty(num_unique, dtype=np.int32)

    snr_levels_arr = np.array(SNR_LEVELS, dtype=np.float32)
    state_hat_arr = np.empty((num_snrs, num_unique, 4), dtype=np.float32)
    n_bs_arr = np.empty((num_snrs, num_unique), dtype=np.uint8)
    condition_ratio_arr = np.empty((num_snrs, num_unique), dtype=np.float32)
    rank_arr = np.empty((num_snrs, num_unique), dtype=np.uint8)

    # For RMSE statistics
    gt_xy = np.empty((num_unique, 2), dtype=np.float64)
    gt_v = np.empty((num_unique, 2), dtype=np.float64)

    t0 = time.time()
    for i, (scene_id, frame, vehicle_id) in enumerate(sorted_keys):
        ts, x, y, vx, vy = lookup[(scene_id, frame, vehicle_id)]
        scene_id_arr[i] = scene_id
        frame_arr[i] = frame
        timestamp_arr[i] = ts
        vehicle_id_arr[i] = vehicle_id
        gt_xy[i, 0] = x
        gt_xy[i, 1] = y
        gt_v[i, 0] = vx
        gt_v[i, 1] = vy

        pos = (x, y)
        vel = (vx, vy)
        for s_idx, level in enumerate(SNR_LEVELS):
            est = sense_vehicle(scene_id, frame, vehicle_id, pos, vel, level, calibration, setup)
            state_hat_arr[s_idx, i, 0] = est["x_hat"]
            state_hat_arr[s_idx, i, 1] = est["y_hat"]
            state_hat_arr[s_idx, i, 2] = est["vx_hat"]
            state_hat_arr[s_idx, i, 3] = est["vy_hat"]
            n_bs_arr[s_idx, i] = est["n_bs"]
            condition_ratio_arr[s_idx, i] = est["condition_ratio"]
            rank_arr[s_idx, i] = est["rank"]

        if (i + 1) % 10000 == 0 or (i + 1) == num_unique:
            elapsed = time.time() - t0
            rate = (i + 1) * num_snrs / elapsed
            print(f"[{split}] Processed {i + 1:,} / {num_unique:,} states ({rate:.1f} senses/s)...")

    elapsed_total = time.time() - t0
    print(f"[{split}] Sensing completed in {elapsed_total:.2f} s ({num_unique * num_snrs / elapsed_total:.1f} senses/s)")

    # Finite check
    assert np.isfinite(state_hat_arr).all(), f"[{split}] non-finite values in state_hat!"
    assert np.isfinite(condition_ratio_arr).all(), f"[{split}] non-finite values in condition_ratio!"

    # Compute RMSE statistics per SNR
    metrics_by_snr = {}
    pos_rmses = []
    vel_rmses = []
    for s_idx, level in enumerate(SNR_LEVELS):
        dx = state_hat_arr[s_idx, :, 0].astype(np.float64) - gt_xy[:, 0]
        dy = state_hat_arr[s_idx, :, 1].astype(np.float64) - gt_xy[:, 1]
        dvx = state_hat_arr[s_idx, :, 2].astype(np.float64) - gt_v[:, 0]
        dvy = state_hat_arr[s_idx, :, 3].astype(np.float64) - gt_v[:, 1]

        pos_err = np.hypot(dx, dy)
        vel_err = np.hypot(dvx, dvy)

        pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
        vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))
        pos_mae = float(np.mean(pos_err))
        vel_mae = float(np.mean(vel_err))
        pos_p50 = float(np.median(pos_err))
        vel_p50 = float(np.median(vel_err))
        pos_p95 = float(np.percentile(pos_err, 95))
        vel_p95 = float(np.percentile(vel_err, 95))

        pos_rmses.append(pos_rmse)
        vel_rmses.append(vel_rmse)

        metrics_by_snr[f"{level:+g}dB"] = {
            "snr_db": float(level),
            "position_rmse_m": pos_rmse,
            "velocity_rmse_mps": vel_rmse,
            "position_mae_m": pos_mae,
            "velocity_mae_mps": vel_mae,
            "position_p50_m": pos_p50,
            "velocity_p50_mps": vel_p50,
            "position_p95_m": pos_p95,
            "velocity_p95_mps": vel_p95,
        }
        print(f"[{split}] SNR {level:+4.0f} dB: Pos RMSE = {pos_rmse:.4f} m | Vel RMSE = {vel_rmse:.4f} m/s")

    # Check monotonicity (+10 < +5 < 0 < -5 < -10 => pos_rmses descending as level ascends)
    pos_monotonic = all(pos_rmses[k] > pos_rmses[k + 1] for k in range(len(pos_rmses) - 1))
    vel_monotonic = all(vel_rmses[k] > vel_rmses[k + 1] for k in range(len(vel_rmses) - 1))
    print(f"[{split}] Monotonicity: Position = {pos_monotonic} | Velocity = {vel_monotonic}")

    cache_arrays = {
        "scene_id": scene_id_arr,
        "frame": frame_arr,
        "timestamp": timestamp_arr,
        "vehicle_id": vehicle_id_arr,
        "snr_levels_db": snr_levels_arr,
        "state_hat": state_hat_arr,
        "n_bs": n_bs_arr,
        "condition_ratio": condition_ratio_arr,
        "rank": rank_arr,
    }

    cache_dir = output_base / split
    cache_path = cache_dir / "sensing_cache.npz"
    write_npz_deterministic(cache_path, cache_arrays)
    cache_size = cache_path.stat().st_size
    cache_sha = sha256_file(cache_path)
    print(f"[{split}] Written: {cache_path} ({cache_size:,} bytes, SHA256: {cache_sha[:16]}...)")

    manifest = {
        "split": split,
        "isac_revision": config["revision"],
        "isac_status": config["status"],
        "source_git_commit": git_commit,
        "config_path": "configs/automatum_controlled_isac.json",
        "config_sha256": sha256_file(CONFIG_PATH),
        "trajectories_path": f"data/automatum_t_crossing/splits/{split}/trajectories.csv",
        "trajectories_sha256": trajectories_sha,
        "samples_path": f"data/automatum_t_crossing/splits/{split}/samples.npz",
        "samples_sha256": samples_sha,
        "prediction_sample_count": num_samples,
        "history_rows_before_dedup": total_history_rows,
        "unique_history_states": num_unique,
        "dedup_factor": round(dedup_factor, 4),
        "snr_levels_db": SNR_LEVELS,
        "state_fields": list(STATE_FIELDS),
        "state_hat_shape": list(state_hat_arr.shape),
        "dtype": "float32",
        "ordering_rule": "scene_id ascending, frame ascending, vehicle_id ascending",
        "cache_filename": "sensing_cache.npz",
        "cache_size_bytes": cache_size,
        "cache_sha256": cache_sha,
        "monotonicity": {
            "position_strictly_monotonic": pos_monotonic,
            "velocity_strictly_monotonic": vel_monotonic,
        },
        "metrics_by_snr": metrics_by_snr,
        "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = cache_dir / "sensing_manifest.json"
    write_json(manifest, manifest_path)
    print(f"[{split}] Written: {manifest_path}")

    # Mirror manifest to reports
    report_manifest_path = report_base / f"{split}_sensing_manifest.json"
    write_json(manifest, report_manifest_path)
    print(f"[{split}] Mirrored: {report_manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Production ISAC Sensing Cache")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--output-base", default=str(OUT_BASE))
    parser.add_argument("--report-base", default=str(REPORT_DIR))
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("revision") != "AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN" or config.get("status") != "FROZEN":
        raise ValueError("Config must be revision=AUTOMATUM-CONTROLLED-ISAC-V2-FROZEN, status=FROZEN")

    setup = setup_from_config(config)
    calibration = calibration_from_config(config)
    git_commit = get_git_commit()

    output_base = Path(args.output_base).resolve()
    report_base = Path(args.report_base).resolve()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    manifests = {}
    for split in splits:
        manifests[split] = build_split_cache(
            split, config, setup, calibration, git_commit, output_base, report_base
        )

    print("\n==========================================")
    print("ALL REQUESTED SPLITS BUILT SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()
