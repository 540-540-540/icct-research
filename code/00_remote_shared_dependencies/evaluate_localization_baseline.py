"""
Classical multi-BS localization baseline on the current synthetic feature dataset.

Pipeline:
    per-BS complex distance/angle feature vectors
    -> matched-filter likelihood over 2D scene grid
    -> multi-BS posterior fusion
    -> position estimate
    -> temporal smoothing
    -> heading estimate from smoothed positions

Metrics:
    - frame-wise position MAE / RMSE
    - window-end anchor position MAE
    - window-end anchor heading MAE

This script is intended to answer:
    "Can the current dataset/features support a mature localization algorithm?"
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass

import numpy as np

try:
    from dataset_feature_config import LEGACY_FEATURE_CONFIG, load_feature_config
except ImportError:
    from .dataset_feature_config import LEGACY_FEATURE_CONFIG, load_feature_config

LIGHT_SPEED = 299792458.0
DELTA_F = 720e3
BS_POS = LEGACY_FEATURE_CONFIG.bs_positions
DT_DEFAULT = 0.05


@dataclass
class LocalizationMetrics:
    frame_pos_mae: float
    frame_pos_rmse: float
    anchor_pos_mae: float
    anchor_heading_mae_deg: float
    x_mae: float
    y_mae: float


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def split_complex_feature(feat: np.ndarray, feature_cfg):
    per_bs_dim = feature_cfg.per_bs_dim
    range_complex_len = feature_cfg.range_complex_len
    vel_complex_len = feature_cfg.vel_complex_len
    angle_complex_len = feature_cfg.angle_complex_len
    if feat.shape[-1] != per_bs_dim:
        raise ValueError(f"Expected feature dim {per_bs_dim}, got {feat.shape[-1]}")

    dist_re = feat[..., :range_complex_len]
    dist_im = feat[..., range_complex_len : 2 * range_complex_len]
    vel_re = feat[..., 2 * range_complex_len : 2 * range_complex_len + vel_complex_len]
    vel_im = feat[..., 2 * range_complex_len + vel_complex_len : 2 * range_complex_len + 2 * vel_complex_len]
    ang_start = 2 * range_complex_len + 2 * vel_complex_len
    ang_re = feat[..., ang_start : ang_start + angle_complex_len]
    ang_im = feat[..., ang_start + angle_complex_len : ang_start + 2 * angle_complex_len]

    dist_obs = dist_re + 1j * dist_im
    vel_obs = vel_re + 1j * vel_im
    ang_obs = ang_re + 1j * ang_im
    return dist_obs, vel_obs, ang_obs


def normalize_atoms(A: np.ndarray) -> np.ndarray:
    return A / np.clip(np.linalg.norm(A, axis=-1, keepdims=True), 1e-8, None)


def build_grid(grid_size: int):
    xs = np.linspace(0.0, 150.0, grid_size, dtype=np.float32)
    ys = np.linspace(0.0, 150.0, grid_size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    return pts


def precompute_atoms(grid_points: np.ndarray, feature_cfg, bs_positions: np.ndarray = BS_POS):
    kr = np.arange(feature_cfg.range_complex_len, dtype=np.float32)
    ka = np.arange(feature_cfg.angle_complex_len, dtype=np.float32)
    range_atoms = []
    angle_atoms = []
    for bs in bs_positions:
        rel = grid_points - bs
        r = np.linalg.norm(rel, axis=-1) + 1e-6
        phi = np.arctan2(rel[:, 1], rel[:, 0])
        range_atom = np.exp(-1j * 2.0 * np.pi * kr[None, :] * DELTA_F * 2.0 * r[:, None] / LIGHT_SPEED)
        angle_atom = np.exp(1j * np.pi * ka[None, :] * np.sin(phi)[:, None])
        range_atoms.append(normalize_atoms(range_atom))
        angle_atoms.append(normalize_atoms(angle_atom))
    return range_atoms, angle_atoms


def compute_frame_loglik(
    bs_obs: list[np.ndarray],
    range_atoms: list[np.ndarray],
    angle_atoms: list[np.ndarray],
    feature_cfg,
    range_weight: float,
    angle_weight: float,
    eps: float = 1e-8,
):
    num_points = range_atoms[0].shape[0]
    loglik = np.zeros(num_points, dtype=np.float32)
    for bs_idx in range(len(bs_obs)):
        dist_obs, _, ang_obs = split_complex_feature(bs_obs[bs_idx], feature_cfg)
        dist_obs = dist_obs / np.clip(np.linalg.norm(dist_obs), 1e-8, None)
        ang_obs = ang_obs / np.clip(np.linalg.norm(ang_obs), 1e-8, None)
        score_r = np.abs(range_atoms[bs_idx] @ np.conjugate(dist_obs)) ** 2
        score_a = np.abs(angle_atoms[bs_idx] @ np.conjugate(ang_obs)) ** 2
        loglik += range_weight * np.log(np.clip(score_r, eps, None))
        loglik += angle_weight * np.log(np.clip(score_a, eps, None))
    return loglik


def posterior_from_loglik(loglik: np.ndarray, posterior_temp: float):
    ll = loglik - np.max(loglik)
    prob = np.exp(ll / max(posterior_temp, 1e-6))
    prob = prob / np.clip(prob.sum(), 1e-8, None)
    return prob.astype(np.float32)


def posterior_mean_cov(grid_points: np.ndarray, prob: np.ndarray):
    mean = prob @ grid_points
    diff = grid_points - mean[None, :]
    cov = np.einsum("n,ni,nj->ij", prob, diff, diff, dtype=np.float64).astype(np.float32)
    return mean.astype(np.float32), cov


def estimate_positions_from_posteriors(
    traj_bs_feats: list[np.ndarray],
    grid_points: np.ndarray,
    range_atoms: list[np.ndarray],
    angle_atoms: list[np.ndarray],
    feature_cfg,
    range_weight: float,
    angle_weight: float,
    posterior_temp: float,
    use_expectation: bool,
    return_info: bool = False,
):
    T = traj_bs_feats[0].shape[0]
    pred = np.zeros((T, 2), dtype=np.float32)
    if return_info:
        posterior_entropy = np.zeros(T, dtype=np.float32)
        posterior_peak = np.zeros(T, dtype=np.float32)
        pos_cov = np.zeros((T, 2, 2), dtype=np.float32)
        pos_uncertainty = np.zeros(T, dtype=np.float32)
    for t in range(T):
        bs_obs = [traj_bs_feats[b][t] for b in range(len(traj_bs_feats))]
        loglik = compute_frame_loglik(bs_obs, range_atoms, angle_atoms, feature_cfg, range_weight, angle_weight)
        prob = posterior_from_loglik(loglik, posterior_temp)
        if use_expectation:
            pred[t] = prob @ grid_points
        else:
            pred[t] = grid_points[int(np.argmax(loglik))]
        if return_info:
            _, cov = posterior_mean_cov(grid_points, prob)
            posterior_entropy[t] = float(-np.sum(prob * np.log(np.clip(prob, 1e-8, None))))
            posterior_peak[t] = float(np.max(prob))
            pos_cov[t] = cov
            pos_uncertainty[t] = float(np.sqrt(max(0.0, cov[0, 0] + cov[1, 1])))
    if not return_info:
        return pred
    return {
        "position": pred,
        "posterior_entropy": posterior_entropy,
        "posterior_peak": posterior_peak,
        "position_cov": pos_cov,
        "position_uncertainty": pos_uncertainty,
    }


def moving_average_traj(traj: np.ndarray, kernel_size: int):
    if kernel_size <= 1:
        return traj.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    out = np.zeros_like(traj)
    for d in range(2):
        x = np.pad(traj[:, d], (pad, pad), mode="edge")
        out[:, d] = np.convolve(x, kernel, mode="valid")
    return out


def moving_average_signal(values: np.ndarray, kernel_size: int):
    if kernel_size <= 1:
        return values.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    x = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(x, kernel, mode="valid").astype(np.float32)


def heading_from_traj(traj: np.ndarray, dt: float):
    vel = np.zeros_like(traj, dtype=np.float32)
    if len(traj) >= 2:
        vel[1:] = (traj[1:] - traj[:-1]) / dt
        vel[0] = vel[1]
    hd = np.arctan2(vel[:, 1], vel[:, 0] + 1e-6)
    if len(traj) >= 2:
        hd[0] = hd[1]
        speed = np.linalg.norm(vel, axis=-1)
        for t in range(1, len(traj)):
            if speed[t] < 1e-4:
                hd[t] = hd[t - 1]
    return hd.astype(np.float32)


def evaluate_dataset(
    root_path: str,
    snr: int,
    grid_size: int,
    seq_len: int,
    eval_stride: int,
    range_weight: float,
    angle_weight: float,
    posterior_temp: float,
    use_expectation: bool,
    smooth_kernel: int,
    dt: float,
):
    data_dir = os.path.join(root_path, f"{snr}dB")
    gt_paths = sorted(glob.glob(os.path.join(data_dir, "gt_locs_*.npy")))
    if not gt_paths:
        raise FileNotFoundError(f"No gt_locs_*.npy found in {data_dir}")
    feature_cfg = load_feature_config(data_dir=data_dir)
    bs_pos = feature_cfg.bs_positions

    grid_points = build_grid(grid_size)
    range_atoms, angle_atoms = precompute_atoms(grid_points, feature_cfg, bs_pos)

    all_pred = []
    all_gt = []
    anchor_pos_err = []
    anchor_heading_err = []

    for gt_path in gt_paths:
        file_idx = os.path.splitext(os.path.basename(gt_path))[0].split("_")[-1]
        gt_locs = np.load(gt_path).astype(np.float32)
        bs_feats = [
            np.load(os.path.join(data_dir, f"bs{b + 1}_feat_{file_idx}.npy")).astype(np.float32)
            for b in range(feature_cfg.bs_count)
        ]

        for traj_idx in range(gt_locs.shape[0]):
            traj_feats = [bs_feats[b][traj_idx] for b in range(3)]
            pred = estimate_positions_from_posteriors(
                traj_feats,
                grid_points,
                range_atoms,
                angle_atoms,
                feature_cfg,
                range_weight=range_weight,
                angle_weight=angle_weight,
                posterior_temp=posterior_temp,
                use_expectation=use_expectation,
            )
            pred = np.clip(pred, 0.0, 150.0)
            pred = moving_average_traj(pred, kernel_size=smooth_kernel)

            gt = gt_locs[traj_idx]
            all_pred.append(pred)
            all_gt.append(gt)

            pred_heading = heading_from_traj(pred, dt=dt)
            gt_heading = heading_from_traj(gt, dt=dt)
            for st in range(0, max(1, len(gt) - seq_len + 1), eval_stride):
                end = st + seq_len - 1
                if end >= len(gt):
                    break
                anchor_pos_err.append(np.linalg.norm(pred[end] - gt[end]))
                anchor_heading_err.append(abs(wrap_angle(pred_heading[end] - gt_heading[end])) * 180.0 / np.pi)

    pred = np.stack(all_pred, axis=0)
    gt = np.stack(all_gt, axis=0)
    diff = pred - gt
    dist_err = np.linalg.norm(diff, axis=-1)
    return LocalizationMetrics(
        frame_pos_mae=float(np.mean(dist_err)),
        frame_pos_rmse=float(np.sqrt(np.mean(dist_err**2))),
        anchor_pos_mae=float(np.mean(anchor_pos_err)),
        anchor_heading_mae_deg=float(np.mean(anchor_heading_err)),
        x_mae=float(np.mean(np.abs(diff[..., 0]))),
        y_mae=float(np.mean(np.abs(diff[..., 1]))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--snr", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=61)
    parser.add_argument("--seq_len", type=int, default=72)
    parser.add_argument("--eval_stride", type=int, default=3)
    parser.add_argument("--range_weight", type=float, default=1.0)
    parser.add_argument("--angle_weight", type=float, default=1.0)
    parser.add_argument("--posterior_temp", type=float, default=0.25)
    parser.add_argument("--use_expectation", type=int, default=0)
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    args = parser.parse_args()

    metrics = evaluate_dataset(
        root_path=args.root_path,
        snr=args.snr,
        grid_size=args.grid_size,
        seq_len=args.seq_len,
        eval_stride=args.eval_stride,
        range_weight=args.range_weight,
        angle_weight=args.angle_weight,
        posterior_temp=args.posterior_temp,
        use_expectation=bool(args.use_expectation),
        smooth_kernel=args.smooth_kernel,
        dt=args.dt,
    )
    feature_cfg = load_feature_config(root_path=args.root_path, snr=args.snr)

    print("=" * 80)
    print(f"Localization baseline | root={args.root_path} | snr={args.snr}dB")
    print(
        f"grid={args.grid_size}x{args.grid_size} | range_w={args.range_weight} | angle_w={args.angle_weight} | "
        f"use_expectation={bool(args.use_expectation)} | smooth_kernel={args.smooth_kernel}"
    )
    print(
        "feature layout: "
        f"dist/vel/ang={feature_cfg.dist_dim}/{feature_cfg.vel_dim}/{feature_cfg.ang_dim} "
        f"(complex {feature_cfg.range_complex_len}/{feature_cfg.vel_complex_len}/{feature_cfg.angle_complex_len})"
    )
    print("-" * 80)
    print(f"Frame Position MAE      : {metrics.frame_pos_mae:.4f} m")
    print(f"Frame Position RMSE     : {metrics.frame_pos_rmse:.4f} m")
    print(f"Anchor Position MAE     : {metrics.anchor_pos_mae:.4f} m")
    print(f"Anchor Heading MAE      : {metrics.anchor_heading_mae_deg:.4f} deg")
    print(f"x MAE                   : {metrics.x_mae:.4f} m")
    print(f"y MAE                   : {metrics.y_mae:.4f} m")
    print("=" * 80)


if __name__ == "__main__":
    main()
