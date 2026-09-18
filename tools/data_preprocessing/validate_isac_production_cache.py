#!/usr/bin/env python3
"""Validate Production ISAC Sensing Cache for train / val / test splits.

Checks:
1. File existence and manifest SHA256 consistency
2. Cache schema (array keys, shapes, dtypes)
3. Finiteness (no NaN, no Inf)
4. Key ordering and uniqueness (strictly sorted, duplicate = 0)
5. Coverage against samples.npz prediction history (missing = 0, full sample window lookup)
6. Frontend equality against frozen automatum_frontend.sense_vehicle
7. SNR Monotonicity & RMSE sanity against frozen audit values
8. Determinism check (reproducible byte-for-byte SHA256)

Generates:
  reports/isac_production_cache/validation.json
  reports/isac_production_cache/production_cache_report.md

Usage:
  python tools/data_preprocessing/validate_isac_production_cache.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontend.controlled_isac.automatum_frontend import sense_vehicle
from frontend.controlled_isac.automatum_measurement import (
    calibration_from_config,
    setup_from_config,
)

CONFIG_PATH = ROOT / "configs/automatum_controlled_isac.json"
CACHE_BASE = ROOT / "data/automatum_t_crossing/isac"
REPORT_DIR = ROOT / "reports/isac_production_cache"

SNR_LEVELS = [-10.0, -5.0, 0.0, 5.0, 10.0]
EXPECTED_KEYS = [
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

FROZEN_AUDIT_RMSE = {
    "train": {
        "+10dB": {"pos": 0.1308, "vel": 0.1590},
        "+5dB":  {"pos": 0.2330, "vel": 0.2830},
        "+0dB":  {"pos": 0.4137, "vel": 0.5028},
        "-5dB":  {"pos": 0.7360, "vel": 0.8940},
        "-10dB": {"pos": 1.3090, "vel": 1.5908},
    },
    "val": {
        "+10dB": {"pos": 0.1430, "vel": 0.1750},
        "+5dB":  {"pos": 0.2550, "vel": 0.3110},
        "+0dB":  {"pos": 0.4530, "vel": 0.5530},
        "-5dB":  {"pos": 0.8060, "vel": 0.9830},
        "-10dB": {"pos": 1.4340, "vel": 1.7480},
    }
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectories_lookup(path: Path) -> dict[tuple[int, int, int], tuple[float, float, float, float, float]]:
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


def validate_split(
    split: str,
    config: dict,
    setup: dict,
    calibration: dict,
    cache_base: Path,
) -> dict:
    print(f"\n==========================================")
    print(f"Validating Production ISAC Cache: {split.upper()}")
    print(f"==========================================")

    split_dir = cache_base / split
    cache_path = split_dir / "sensing_cache.npz"
    manifest_path = split_dir / "sensing_manifest.json"

    # 1. Existence
    assert cache_path.exists(), f"[{split}] missing cache file: {cache_path}"
    assert manifest_path.exists(), f"[{split}] missing manifest: {manifest_path}"

    # 2. Manifest and Hash check
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_cache_sha = sha256_file(cache_path)
    assert actual_cache_sha == manifest["cache_sha256"], (
        f"[{split}] cache SHA256 mismatch! actual={actual_cache_sha}, manifest={manifest['cache_sha256']}"
    )
    print(f"[{split}] Cache SHA256 verified: {actual_cache_sha[:16]}...")

    # 3. Schema & Finiteness
    cache_data = np.load(cache_path)
    actual_keys = list(cache_data.keys())
    assert set(actual_keys) == set(EXPECTED_KEYS), f"[{split}] unexpected keys: {actual_keys}"

    M = cache_data["scene_id"].shape[0]
    print(f"[{split}] Cached state count M = {M:,}")

    # Shapes and dtypes
    assert cache_data["scene_id"].shape == (M,) and cache_data["scene_id"].dtype == np.uint8
    assert cache_data["frame"].shape == (M,) and cache_data["frame"].dtype == np.int32
    assert cache_data["timestamp"].shape == (M,) and cache_data["timestamp"].dtype == np.float64
    assert cache_data["vehicle_id"].shape == (M,) and cache_data["vehicle_id"].dtype == np.int32
    assert cache_data["snr_levels_db"].shape == (5,) and cache_data["snr_levels_db"].dtype == np.float32
    assert np.allclose(cache_data["snr_levels_db"], SNR_LEVELS)
    assert cache_data["state_hat"].shape == (5, M, 4) and cache_data["state_hat"].dtype == np.float32
    assert cache_data["n_bs"].shape == (5, M) and cache_data["n_bs"].dtype == np.uint8
    assert cache_data["condition_ratio"].shape == (5, M) and cache_data["condition_ratio"].dtype == np.float32
    assert cache_data["rank"].shape == (5, M) and cache_data["rank"].dtype == np.uint8

    # Finite
    non_finite_count = int((~np.isfinite(cache_data["state_hat"])).sum())
    assert non_finite_count == 0, f"[{split}] {non_finite_count} non-finite entries in state_hat!"
    assert np.isfinite(cache_data["condition_ratio"]).all()
    print(f"[{split}] Schema and finiteness checks: PASS (0 non-finite states)")

    # 4. Key ordering and uniqueness
    scene_arr = cache_data["scene_id"]
    frame_arr = cache_data["frame"]
    veh_arr = cache_data["vehicle_id"]

    cache_keys = [(int(scene_arr[i]), int(frame_arr[i]), int(veh_arr[i])) for i in range(M)]
    duplicate_count = len(cache_keys) - len(set(cache_keys))
    assert duplicate_count == 0, f"[{split}] {duplicate_count} duplicate keys in cache!"

    # Strictly sorted
    is_strictly_sorted = all(cache_keys[i] < cache_keys[i + 1] for i in range(M - 1))
    assert is_strictly_sorted, f"[{split}] cache keys are not strictly sorted in ascending order!"
    print(f"[{split}] Key ordering and uniqueness: PASS (strictly sorted, duplicate = 0)")

    # 5. Coverage against samples.npz
    samples_path = ROOT / "data/automatum_t_crossing/splits" / split / "samples.npz"
    samples = np.load(samples_path)
    num_samples = int(samples["scene_id"].size)

    key_to_idx = {k: i for i, k in enumerate(cache_keys)}
    missing_keys = set()
    total_lookup_checks = 0

    for s_idx in range(num_samples):
        sc_id = int(samples["scene_id"][s_idx])
        st_frame = int(samples["start_frame"][s_idx])
        active_ids = samples["vehicle_ids"][s_idx][samples["vehicle_mask"][s_idx]].tolist()
        for offset in range(20):
            fr = st_frame + offset
            for vid in active_ids:
                req_key = (sc_id, fr, int(vid))
                total_lookup_checks += 1
                if req_key not in key_to_idx:
                    missing_keys.add(req_key)

    assert len(missing_keys) == 0, f"[{split}] {len(missing_keys)} required keys missing from cache!"
    assert len(key_to_idx) == M
    print(f"[{split}] Coverage: PASS (missing = 0, verified {total_lookup_checks:,} sample window lookups)")

    # 6. Frontend Equality
    trajectories_path = ROOT / "data/automatum_t_crossing/splits" / split / "trajectories.csv"
    lookup = load_trajectories_lookup(trajectories_path)

    # Deterministically sample 200 states evenly spread across the split
    sample_indices = np.linspace(0, M - 1, min(200, M), dtype=int)
    max_pos_diff = 0.0
    max_vel_diff = 0.0

    for idx in sample_indices:
        sc_id, fr, vid = cache_keys[idx]
        ts, x, y, vx, vy = lookup[(sc_id, fr, vid)]
        pos = (x, y)
        vel = (vx, vy)
        for s_idx, level in enumerate(SNR_LEVELS):
            est = sense_vehicle(sc_id, fr, vid, pos, vel, level, calibration, setup)
            cached_x = float(cache_data["state_hat"][s_idx, idx, 0])
            cached_y = float(cache_data["state_hat"][s_idx, idx, 1])
            cached_vx = float(cache_data["state_hat"][s_idx, idx, 2])
            cached_vy = float(cache_data["state_hat"][s_idx, idx, 3])

            pos_diff = np.hypot(cached_x - float(np.float32(est["x_hat"])),
                                cached_y - float(np.float32(est["y_hat"])))
            vel_diff = np.hypot(cached_vx - float(np.float32(est["vx_hat"])),
                                cached_vy - float(np.float32(est["vy_hat"])))

            if pos_diff > max_pos_diff:
                max_pos_diff = pos_diff
            if vel_diff > max_vel_diff:
                max_vel_diff = vel_diff

    assert max_pos_diff < 1e-6, f"[{split}] max pos diff too large: {max_pos_diff}"
    assert max_vel_diff < 1e-6, f"[{split}] max vel diff too large: {max_vel_diff}"
    print(f"[{split}] Frontend equality: PASS (max pos diff={max_pos_diff:.2e} m, max vel diff={max_vel_diff:.2e} m/s)")

    # 7. SNR Monotonicity and RMSE check
    gt_xy = np.empty((M, 2), dtype=np.float64)
    gt_v = np.empty((M, 2), dtype=np.float64)
    for i, k in enumerate(cache_keys):
        _, x, y, vx, vy = lookup[k]
        gt_xy[i] = (x, y)
        gt_v[i] = (vx, vy)

    rmses = {}
    pos_rmses = []
    vel_rmses = []
    for s_idx, level in enumerate(SNR_LEVELS):
        dx = cache_data["state_hat"][s_idx, :, 0].astype(np.float64) - gt_xy[:, 0]
        dy = cache_data["state_hat"][s_idx, :, 1].astype(np.float64) - gt_xy[:, 1]
        dvx = cache_data["state_hat"][s_idx, :, 2].astype(np.float64) - gt_v[:, 0]
        dvy = cache_data["state_hat"][s_idx, :, 3].astype(np.float64) - gt_v[:, 1]

        p_rmse = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
        v_rmse = float(np.sqrt(np.mean(dvx ** 2 + dvy ** 2)))
        pos_rmses.append(p_rmse)
        vel_rmses.append(v_rmse)

        lbl = f"{level:+g}dB"
        rmses[lbl] = {"position_rmse": p_rmse, "velocity_rmse": v_rmse}
        print(f"[{split}] {lbl:>6s}: Pos RMSE = {p_rmse:.4f} m | Vel RMSE = {v_rmse:.4f} m/s")

    pos_monotonic = all(pos_rmses[k] > pos_rmses[k + 1] for k in range(len(pos_rmses) - 1))
    vel_monotonic = all(vel_rmses[k] > vel_rmses[k + 1] for k in range(len(vel_rmses) - 1))
    assert pos_monotonic, f"[{split}] position RMSE not strictly monotonic!"
    assert vel_monotonic, f"[{split}] velocity RMSE not strictly monotonic!"
    print(f"[{split}] Strict monotonicity across 5 SNRs: PASS")

    # Audit check against frozen reference if train/val
    if split in FROZEN_AUDIT_RMSE:
        for lbl, exp in FROZEN_AUDIT_RMSE[split].items():
            actual_pos = rmses[lbl]["position_rmse"]
            actual_vel = rmses[lbl]["velocity_rmse"]
            diff_pos = abs(actual_pos - exp["pos"])
            diff_vel = abs(actual_vel - exp["vel"])
            # Within 0.001 m/s (due to float32 vs float64 storage)
            assert diff_pos < 0.0015, f"[{split}] Pos RMSE at {lbl} deviates from frozen audit: {actual_pos} vs {exp['pos']}"
            assert diff_vel < 0.0015, f"[{split}] Vel RMSE at {lbl} deviates from frozen audit: {actual_vel} vs {exp['vel']}"
        print(f"[{split}] Concordance with frozen audit values: PASS")

    return {
        "split": split,
        "sample_count": num_samples,
        "history_rows": total_lookup_checks,
        "unique_states": M,
        "cache_sha256": actual_cache_sha,
        "cache_shape": list(cache_data["state_hat"].shape),
        "missing_keys": 0,
        "duplicate_keys": 0,
        "non_finite_states": 0,
        "max_position_diff": float(max_pos_diff),
        "max_velocity_diff": float(max_vel_diff),
        "monotonicity": {
            "position": pos_monotonic,
            "velocity": vel_monotonic,
        },
        "rmses": rmses,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Production ISAC Sensing Cache")
    parser.add_argument("--cache-base", default=str(CACHE_BASE))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    setup = setup_from_config(config)
    calibration = calibration_from_config(config)

    cache_base = Path(args.cache_base).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ["train", "val", "test"]:
        results[split] = validate_split(split, config, setup, calibration, cache_base)

    # 8. Determinism check: verify that rebuilt val split produces identical SHA256
    print("\n==========================================")
    print("Testing Deterministic Rebuild")
    print("==========================================")
    from tools.data_preprocessing.build_isac_production_cache import build_split_cache, get_git_commit
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cache_base = Path(tmp_dir) / "isac"
        tmp_report_base = Path(tmp_dir) / "reports"
        rebuilt_manifest = build_split_cache(
            "val", config, setup, calibration, get_git_commit(), tmp_cache_base, tmp_report_base
        )
        original_sha = results["val"]["cache_sha256"]
        rebuilt_sha = rebuilt_manifest["cache_sha256"]
        assert original_sha == rebuilt_sha, (
            f"Determinism failure! Original SHA: {original_sha}, Rebuilt SHA: {rebuilt_sha}"
        )
        print(f"Deterministic rebuild: PASS (SHA256 identical: {rebuilt_sha[:16]}...)")
        determinism_pass = True

    validation_summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "isac_revision": config["revision"],
        "isac_status": config["status"],
        "determinism_pass": determinism_pass,
        "splits": results,
        "overall_verdict": "PASS",
    }

    val_json_path = report_dir / "validation.json"
    val_json_path.write_text(json.dumps(validation_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWritten validation summary: {val_json_path}")

    # Generate Markdown report
    report_md_path = report_dir / "production_cache_report.md"
    md_lines = [
        "# Production ISAC Sensing Cache 构建与验证报告",
        "",
        f"- **构建时间 (UTC)**: `{validation_summary['timestamp_utc']}`",
        f"- **ISAC 版本**: `{config['revision']}` (`{config['status']}`)",
        f"- **确定性重建检验**: `{'PASS' if determinism_pass else 'FAIL'}`",
        f"- **全量验证结论**: `ALL PASS`",
        "",
        "---",
        "",
        "## 1. Split 规模与覆盖审计",
        "",
        "| Split | 预测样本数 (Windows) | History 行数 (未去重) | 唯一状态数 (Unique Keys) | 去重压缩比 | Cache 形状 | 缺失数 | 重复数 | 非有限值数 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in ["train", "val", "test"]:
        r = results[s]
        dedup = r["history_rows"] / r["unique_states"]
        md_lines.append(
            f"| `{s}` | {r['sample_count']:,} | {r['history_rows']:,} | {r['unique_states']:,} | "
            f"{dedup:.2f}x | `{r['cache_shape']}` | {r['missing_keys']} | {r['duplicate_keys']} | {r['non_finite_states']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. 五档 SNR 误差指标与严格单调性",
        "",
        "| Split | 指标 | +10 dB | +5 dB | 0 dB | -5 dB | -10 dB | 单调性 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in ["train", "val", "test"]:
        r = results[s]
        pos_str = " / ".join(f"{r['rmses'][f'{lvl:+g}dB']['position_rmse']:.4f}m" for lvl in [10.0, 5.0, 0.0, -5.0, -10.0])
        vel_str = " / ".join(f"{r['rmses'][f'{lvl:+g}dB']['velocity_rmse']:.4f}m/s" for lvl in [10.0, 5.0, 0.0, -5.0, -10.0])
        md_lines.append(
            f"| `{s}` | Position RMSE | "
            + " | ".join(f"{r['rmses'][f'{lvl:+g}dB']['position_rmse']:.4f} m" for lvl in [10.0, 5.0, 0.0, -5.0, -10.0])
            + f" | `{'PASS' if r['monotonicity']['position'] else 'FAIL'}` |"
        )
        md_lines.append(
            f"| `{s}` | Velocity RMSE | "
            + " | ".join(f"{r['rmses'][f'{lvl:+g}dB']['velocity_rmse']:.4f} m/s" for lvl in [10.0, 5.0, 0.0, -5.0, -10.0])
            + f" | `{'PASS' if r['monotonicity']['velocity'] else 'FAIL'}` |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Frontend 数值精确一致性 (Frontend Equality)",
        "",
        "在每个 split 均匀抽取 200 个状态重新调用 `frontend.controlled_isac.automatum_frontend.sense_vehicle`，与 cache 逐数值比对：",
        "",
        "| Split | 检验状态数 | 最大位置偏差 (m) | 最大速度偏差 (m/s) | 判定 |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in ["train", "val", "test"]:
        r = results[s]
        md_lines.append(f"| `{s}` | 200 | `{r['max_position_diff']:.2e}` | `{r['max_velocity_diff']:.2e}` | **PASS** |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Cache 文件与校验哈希",
        "",
        "| Split | 相对路径 | SHA256 |",
        "| :--- | :--- | :--- |",
    ])

    for s in ["train", "val", "test"]:
        r = results[s]
        md_lines.append(f"| `{s}` | `data/automatum_t_crossing/isac/{s}/sensing_cache.npz` | `{r['cache_sha256']}` |")

    md_lines.append("")
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Written markdown report: {report_md_path}")
    print("\nALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
