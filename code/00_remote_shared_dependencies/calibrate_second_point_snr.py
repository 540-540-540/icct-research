"""Estimate observation-noise scales from first-point localized SNR caches."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def robust_sigma(values: np.ndarray) -> float:
    centered = values - np.median(values)
    return float(1.4826 * np.median(np.abs(centered)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="logs")
    parser.add_argument("--snrs", default="5,10,15,20")
    parser.add_argument("--source-dt", type=float, default=0.05)
    parser.add_argument("--output", default="results/multitarget_snr/snr_calibration.json")
    args = parser.parse_args()
    requested = [int(value) for value in args.snrs.split(",")]
    rows = {}
    for snr in requested:
        matches = sorted(Path(args.cache_dir).glob(f"localized_cache_*_snr{snr}_*.npz"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one cache for {snr} dB, found: {matches}")
        path = matches[0]
        npz = np.load(path)
        recovered = np.concatenate([npz["train_recovered"], npz["val_recovered"]], axis=0).astype(np.float64)
        ground_truth = np.concatenate([npz["train_gt"], npz["val_gt"]], axis=0).astype(np.float64)
        error = recovered - ground_truth
        radial = np.linalg.norm(error, axis=-1)
        per_axis_rmse = float(np.sqrt(np.mean(error ** 2)))
        per_axis_robust = robust_sigma(error.reshape(-1))

        recovered_velocity = np.diff(recovered, axis=1) / args.source_dt
        truth_velocity = np.diff(ground_truth, axis=1) / args.source_dt
        velocity_error = recovered_velocity - truth_velocity
        velocity_per_axis_rmse = float(np.sqrt(np.mean(velocity_error ** 2)))
        velocity_per_axis_robust = robust_sigma(velocity_error.reshape(-1))
        rows[str(snr)] = {
            "cache": str(path),
            "trajectories": int(recovered.shape[0]),
            "frames_per_trajectory": int(recovered.shape[1]),
            "position_per_axis_rmse_m": per_axis_rmse,
            "position_per_axis_robust_sigma_m": per_axis_robust,
            "position_radial_mae_m": float(radial.mean()),
            "position_radial_median_m": float(np.median(radial)),
            "position_radial_p95_m": float(np.quantile(radial, 0.95)),
            "velocity_per_axis_rmse_mps": velocity_per_axis_rmse,
            "velocity_per_axis_robust_sigma_mps": velocity_per_axis_robust,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snr_db": requested,
        "source": "first-point RF localization caches",
        "source_dt_s": args.source_dt,
        "calibration": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
