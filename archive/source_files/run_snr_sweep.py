"""
Train and compare predictors across different SNR (Signal-to-Noise Ratio) levels.

Pipeline:
    RF features (varying SNR) -> classical localization -> recovered history
    -> predictor (GPT2-soft, LSTM, TCN, Transformer) -> future coordinates

Goal:
    Demonstrate that our Discrete LLM Architecture is significantly more 
    robust to severe RF noise (e.g., 5dB, 0dB, -10dB) compared to 
    traditional continuous regression baselines.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2Model

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

from evaluate_localization_baseline import (
    build_grid,
    estimate_positions_from_posteriors,
    load_feature_config,
    moving_average_signal,
    moving_average_traj,
    precompute_atoms,
)
from motion_tokenizer_v7 import MotionTokenizerConfig, MotionTokenizerV7


def resolve_gpt2_source():
    here = Path(__file__).resolve().parent
    candidates = [
        Path("./gpt2"),
        here / "gpt2",
        Path("gpt2"),
    ]
    for path in candidates:
        if path.exists():
            return str(path), True
    return "gpt2", False


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score: float):
        improved = self.best_score is None or score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_constant_velocity_baseline(last_pos_norm, last_vel_norm, pred_len: int, dt: float):
    steps = torch.arange(1, pred_len + 1, device=last_pos_norm.device, dtype=last_pos_norm.dtype).view(1, pred_len, 1)
    return last_pos_norm.unsqueeze(1) + steps * (last_vel_norm.unsqueeze(1) * dt)


def weighted_smooth_l1_loss(pred, target, step_weights):
    point_loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    weights = step_weights.view(1, -1).to(pred.device, pred.dtype)
    return (point_loss * weights).mean()


def parse_report_steps(spec: str, pred_len: int):
    raw = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            step = int(item)
        except ValueError:
            continue
        if 1 <= step <= pred_len:
            raw.append(step)
    if not raw:
        raw = [1, min(5, pred_len), min(10, pred_len), min(15, pred_len), pred_len]
    return sorted(set(raw))


def build_motion_token_feature_table(tokenizer: MotionTokenizerV7) -> torch.Tensor:
    cfg = tokenizer.config
    token_f, token_l = tokenizer._token_tables(device=torch.device("cpu"), dtype=torch.float32)
    speed = torch.sqrt(token_f ** 2 + token_l ** 2)
    angle = torch.atan2(token_l, token_f)

    f_scale = max(abs(float(cfg.forward_min)), abs(float(cfg.forward_max)), 1e-6)
    l_scale = max(abs(float(cfg.lateral_min)), abs(float(cfg.lateral_max)), 1e-6)
    s_scale = max((f_scale ** 2 + l_scale ** 2) ** 0.5, 1e-6)

    token_ids = torch.arange(tokenizer.vocab_size, dtype=torch.long)
    f_idx, l_idx = tokenizer.token_bin_coords(token_ids)
    if cfg.forward_bins > 1:
        f_bin = 2.0 * (f_idx.float() / float(cfg.forward_bins - 1)) - 1.0
    else:
        f_bin = torch.zeros_like(token_f)
    if cfg.lateral_bins > 1:
        l_bin = 2.0 * (l_idx.float() / float(cfg.lateral_bins - 1)) - 1.0
    else:
        l_bin = torch.zeros_like(token_l)

    return torch.stack(
        [
            token_f / f_scale,
            token_l / l_scale,
            speed / s_scale,
            torch.sin(angle),
            torch.cos(angle),
            f_bin,
            l_bin,
        ],
        dim=-1,
    )


def list_split_file_ids(data_dir: str, split: str):
    file_ids = []
    for name in os.listdir(data_dir):
        if name.startswith("gt_locs_") and name.endswith(".npy"):
            try:
                file_ids.append(int(name[len("gt_locs_") : -4]))
            except ValueError:
                continue
    file_ids = sorted(file_ids)
    nfiles = len(file_ids)
    if nfiles == 0:
        raise FileNotFoundError(f"No files matching gt_locs_*.npy found in {data_dir}")
    if nfiles <= 5:
        splits = {
            "train": file_ids[0 : max(1, nfiles - 2)],
            "val": file_ids[max(1, nfiles - 2) : max(2, nfiles - 1)],
            "test": file_ids[max(2, nfiles - 1) : nfiles],
        }
    else:
        n_train = int(nfiles * 0.7)
        n_val = int(nfiles * 0.15)
        splits = {
            "train": file_ids[:n_train],
            "val": file_ids[n_train : n_train + n_val],
            "test": file_ids[n_train + n_val :],
        }
    return splits[split]


def normalize_transition_uncertainty(hist_uncertainty, q_low: float, q_high: float, eps: float = 1e-6):
    trans_uncertainty = 0.5 * (hist_uncertainty[:, :-1] + hist_uncertainty[:, 1:])
    if trans_uncertainty.numel() == 0:
        return trans_uncertainty, trans_uncertainty
    low = torch.quantile(trans_uncertainty, q_low, dim=1, keepdim=True)
    high = torch.quantile(trans_uncertainty, q_high, dim=1, keepdim=True)
    denom = (high - low).clamp_min(eps)
    norm_uncertainty = ((trans_uncertainty - low) / denom).clamp(0.0, 1.0)
    return trans_uncertainty, norm_uncertainty


def normalized_uncertainty_to_temperature(
    norm_uncertainty,
    base_temp: float,
    alpha: float,
    min_temp: float,
    max_temp: float,
    gamma: float,
):
    gamma = max(float(gamma), 1e-6)
    temperature = base_temp + alpha * norm_uncertainty.pow(gamma)
    return temperature.clamp(min=min_temp, max=max_temp)


def topk_renormalize_probs(probs, topk: int):
    if topk <= 0 or topk >= probs.shape[-1]:
        return probs
    values, indices = torch.topk(probs, k=topk, dim=-1)
    sparse = torch.zeros_like(probs)
    sparse.scatter_(-1, indices, values)
    denom = sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return sparse / denom


def build_sample_indices(num_traj: int, traj_len: int, seq_len: int, pred_len: int, stride: int):
    samples = []
    win = seq_len + pred_len
    for traj_idx in range(num_traj):
        for st in range(0, traj_len - win + 1, stride):
            samples.append((traj_idx, st))
    return np.asarray(samples, dtype=np.int32)


def localize_split(
    root_path: str,
    snr: int,
    split: str,
    seq_len: int,
    pred_len: int,
    stride: int,
    grid_size: int,
    smooth_kernel: int,
    range_weight: float,
    angle_weight: float,
    posterior_temp: float,
    use_expectation: bool,
    max_files: int,
    max_traj_per_file: int,
    log,
):
    data_dir = os.path.join(root_path, f"{snr}dB")
    feature_cfg = load_feature_config(data_dir=data_dir)
    bs_pos = feature_cfg.bs_positions
    file_ids = list_split_file_ids(data_dir, split)
    if max_files > 0:
        file_ids = file_ids[:max_files]

    grid_points = build_grid(grid_size)
    range_atoms, angle_atoms = precompute_atoms(grid_points, feature_cfg, bs_pos)

    recovered_list = []
    uncertainty_list = []
    gt_list = []
    total_frame_err = []
    total_anchor_err = []

    t0 = time.time()
    for file_pos, file_id in enumerate(file_ids, start=1):
        gt_locs = np.load(os.path.join(data_dir, f"gt_locs_{file_id}.npy")).astype(np.float32)
        bs_feats = [
            np.load(os.path.join(data_dir, f"bs{b + 1}_feat_{file_id}.npy")).astype(np.float32)
            for b in range(feature_cfg.bs_count)
        ]
        num_traj = gt_locs.shape[0]
        if max_traj_per_file > 0:
            num_traj = min(num_traj, max_traj_per_file)

        file_frame_err = []
        file_anchor_err = []
        file_recovered = np.zeros((num_traj, gt_locs.shape[1], 2), dtype=np.float32)
        file_uncertainty = np.zeros((num_traj, gt_locs.shape[1]), dtype=np.float32)
        file_gt = gt_locs[:num_traj].copy()

        for traj_idx in range(num_traj):
            traj_feats = [bs_feats[b][traj_idx] for b in range(feature_cfg.bs_count)]
            loc_info = estimate_positions_from_posteriors(
                traj_feats,
                grid_points,
                range_atoms,
                angle_atoms,
                feature_cfg,
                range_weight=range_weight,
                angle_weight=angle_weight,
                posterior_temp=posterior_temp,
                use_expectation=use_expectation,
                return_info=True,
            )
            pred = loc_info["position"]
            pred = np.clip(pred, 0.0, 150.0)
            pred = moving_average_traj(pred, kernel_size=smooth_kernel)
            uncertainty = moving_average_signal(loc_info["position_uncertainty"], kernel_size=smooth_kernel)

            gt = file_gt[traj_idx]
            file_recovered[traj_idx] = pred
            file_uncertainty[traj_idx] = uncertainty
            file_frame_err.append(np.linalg.norm(pred - gt, axis=-1).mean())
            file_anchor_err.append(np.linalg.norm(pred[seq_len - 1] - gt[seq_len - 1]))

        recovered_list.append(file_recovered)
        uncertainty_list.append(file_uncertainty)
        gt_list.append(file_gt)
        total_frame_err.extend(file_frame_err)
        total_anchor_err.extend(file_anchor_err)
        
    recovered_full = np.concatenate(recovered_list, axis=0)
    uncertainty_full = np.concatenate(uncertainty_list, axis=0)
    gt_full = np.concatenate(gt_list, axis=0)
    sample_indices = build_sample_indices(recovered_full.shape[0], recovered_full.shape[1], seq_len, pred_len, stride)

    return {
        "recovered": recovered_full,
        "uncertainty": uncertainty_full,
        "gt": gt_full,
        "samples": sample_indices,
        "frame_pos_mae": float(np.mean(total_frame_err)),
        "anchor_pos_mae": float(np.mean(total_anchor_err)),
    }


def default_cache_path(args):
    root_tag = os.path.basename(os.path.abspath(args.root_path.rstrip("/\\")))
    # 💡 缓存在名字里天生带有 SNR 标识，绝对不会互相污染
    return os.path.join(
        "./logs",
        f"localized_cache_{root_tag}_snr{args.snr}_g{args.grid_size}_s{args.smooth_kernel}_sl{args.seq_len}_pl{args.pred_len}.npz",
    )


def load_or_build_cache(args, log):
    cache_path = args.cache_path.strip() if args.cache_path else default_cache_path(args)
    if args.use_cache and os.path.exists(cache_path):
        npz_obj = np.load(cache_path)
        try:
            data = {
                "train_recovered": npz_obj["train_recovered"].astype(np.float32),
                "train_uncertainty": npz_obj["train_uncertainty"].astype(np.float32),
                "train_gt": npz_obj["train_gt"].astype(np.float32),
                "train_samples": npz_obj["train_samples"].astype(np.int32),
                "val_recovered": npz_obj["val_recovered"].astype(np.float32),
                "val_uncertainty": npz_obj["val_uncertainty"].astype(np.float32),
                "val_gt": npz_obj["val_gt"].astype(np.float32),
                "val_samples": npz_obj["val_samples"].astype(np.int32),
            }
            log(f"[Cache] loaded localized trajectories from {cache_path}")
            return data, cache_path
        finally:
            npz_obj.close()

    train_data = localize_split(
        root_path=args.root_path, snr=args.snr, split="train", seq_len=args.seq_len, pred_len=args.pred_len,
        stride=args.train_stride, grid_size=args.grid_size, smooth_kernel=args.smooth_kernel,
        range_weight=args.range_weight, angle_weight=args.angle_weight, posterior_temp=args.posterior_temp,
        use_expectation=bool(args.use_expectation), max_files=args.max_train_files, max_traj_per_file=args.max_traj_per_file, log=log,
    )
    val_data = localize_split(
        root_path=args.root_path, snr=args.snr, split="val", seq_len=args.seq_len, pred_len=args.pred_len,
        stride=args.eval_stride, grid_size=args.grid_size, smooth_kernel=args.smooth_kernel,
        range_weight=args.range_weight, angle_weight=args.angle_weight, posterior_temp=args.posterior_temp,
        use_expectation=bool(args.use_expectation), max_files=args.max_val_files, max_traj_per_file=args.max_traj_per_file, log=log,
    )
    data = {
        "train_recovered": train_data["recovered"], "train_uncertainty": train_data["uncertainty"],
        "train_gt": train_data["gt"], "train_samples": train_data["samples"],
        "val_recovered": val_data["recovered"], "val_uncertainty": val_data["uncertainty"],
        "val_gt": val_data["gt"], "val_samples": val_data["samples"],
    }
    if args.use_cache:
        np.savez_compressed(cache_path, **data)
        log(f"[Cache] saved localized trajectories to {cache_path}")
    return data, cache_path


class LocalizedWindowDataset(Dataset):
    def __init__(
        self, recovered_full, uncertainty_full, gt_full, samples, seq_len, pred_len,
        pos_mean, pos_scale, unc_mean, unc_scale, input_mode="state_unc", dt=0.05,
    ):
        self.recovered_full = recovered_full
        self.uncertainty_full = uncertainty_full
        self.gt_full = gt_full
        self.samples = samples
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pos_mean = pos_mean.astype(np.float32)
        self.pos_scale = pos_scale.astype(np.float32)
        self.unc_mean = np.float32(unc_mean)
        self.unc_scale = np.float32(max(float(unc_scale), 1e-6))
        self.input_mode = input_mode
        self.dt = dt

    def __len__(self):
        return len(self.samples)

    def _normalize_pos(self, x): return (x - self.pos_mean) / self.pos_scale
    def _normalize_unc(self, x): return (x - self.unc_mean) / self.unc_scale

    def __getitem__(self, index):
        traj_idx, st = self.samples[index]
        se = st + self.seq_len
        re = se + self.pred_len

        hist_real = self.recovered_full[traj_idx, st:se].astype(np.float32)
        hist_uncertainty = self.uncertainty_full[traj_idx, st:se].astype(np.float32)
        future_real = self.gt_full[traj_idx, se:re].astype(np.float32)

        hist_norm = self._normalize_pos(hist_real)
        future_norm = self._normalize_pos(future_real)
        hist_unc_norm = self._normalize_unc(hist_uncertainty).reshape(-1, 1).astype(np.float32)

        vel = np.zeros_like(hist_norm, dtype=np.float32)
        if hist_norm.shape[0] >= 2:
            vel[1:] = (hist_norm[1:] - hist_norm[:-1]) / self.dt
            vel[0] = vel[1]

        if self.input_mode == "coord":
            x_hist = hist_norm
        elif self.input_mode == "state":
            x_hist = np.concatenate([hist_norm, vel], axis=-1)
        elif self.input_mode == "state_unc":
            x_hist = np.concatenate([hist_norm, vel, hist_unc_norm], axis=-1)
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        return {
            "hist_real": torch.from_numpy(hist_real),
            "hist_uncertainty": torch.from_numpy(hist_uncertainty),
            "future_real": torch.from_numpy(future_real),
            "hist_input": torch.from_numpy(x_hist),
            "future_norm": torch.from_numpy(future_norm),
        }


class BaseContinuousForecaster(nn.Module):
    def _format_future_inputs(self, x_hist, future_norm, dt):
        B, T_p, _ = future_norm.shape
        out = torch.zeros(B, T_p, self.input_dim, device=future_norm.device, dtype=future_norm.dtype)
        out[:, :, :2] = future_norm
        if self.input_dim >= 4:
            vels = torch.zeros_like(future_norm)
            vels[:, 0] = (future_norm[:, 0] - x_hist[:, -1, :2]) / dt
            if T_p > 1:
                vels[:, 1:] = (future_norm[:, 1:] - future_norm[:, :-1]) / dt
            out[:, :, 2:4] = vels
        return out

    def _format_single_step(self, last_x, curr_coord, dt):
        B = last_x.shape[0]
        out = torch.zeros(B, 1, self.input_dim, device=last_x.device, dtype=last_x.dtype)
        out[:, 0, :2] = curr_coord[:, 0]
        if self.input_dim >= 4:
            out[:, 0, 2:4] = (curr_coord[:, 0] - last_x[:, :2]) / dt
        return out


class LSTMForecaster(BaseContinuousForecaster):
    def __init__(self, input_dim, pred_len, hidden_dim=256, n_layers=2, dropout=0.1, dt=0.05):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.dt = dt
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=False, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2),
        )
        nn.init.zeros_(self.regression_head[-1].weight)
        nn.init.zeros_(self.regression_head[-1].bias)

    def forward(self, x_hist, future_norm=None, **kwargs):
        last_pos = x_hist[:, -1, :2]
        last_vel = x_hist[:, -1, 2:4] if x_hist.shape[-1] >= 4 else torch.zeros_like(last_pos)
        baseline = build_constant_velocity_baseline(last_pos, last_vel, self.pred_len, self.dt)

        if future_norm is not None:
            future_x = self._format_future_inputs(x_hist, future_norm, self.dt)
            inputs = torch.cat([x_hist, future_x[:, :-1]], dim=1)
            out, _ = self.encoder(inputs)
            offset = self.regression_head(self.norm(out[:, -self.pred_len:]))
            return baseline + offset
        else:
            current_x = x_hist
            pred_offsets = []
            for t in range(self.pred_len):
                out, _ = self.encoder(current_x)
                offset = self.regression_head(self.norm(out[:, -1:]))
                pred_offsets.append(offset)
                pred_coord = baseline[:, t:t+1] + offset
                next_x = self._format_single_step(current_x[:, -1], pred_coord, self.dt)
                current_x = torch.cat([current_x, next_x], dim=1)
            return baseline + torch.cat(pred_offsets, dim=1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(BaseContinuousForecaster):
    def __init__(self, input_dim, pred_len, num_channels=[64, 128, 256], kernel_size=3, dropout=0.1, dt=0.05):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.dt = dt
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.regression_head = nn.Sequential(nn.Linear(num_channels[-1], 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))
        nn.init.zeros_(self.regression_head[-1].weight)
        nn.init.zeros_(self.regression_head[-1].bias)

    def forward(self, x_hist, future_norm=None, **kwargs):
        last_pos = x_hist[:, -1, :2]
        last_vel = x_hist[:, -1, 2:4] if x_hist.shape[-1] >= 4 else torch.zeros_like(last_pos)
        baseline = build_constant_velocity_baseline(last_pos, last_vel, self.pred_len, self.dt)

        if future_norm is not None:
            future_x = self._format_future_inputs(x_hist, future_norm, self.dt)
            inputs = torch.cat([x_hist, future_x[:, :-1]], dim=1)
            inputs_t = inputs.transpose(1, 2)
            out = self.network(inputs_t) 
            out = out.transpose(1, 2) 
            offset = self.regression_head(out[:, -self.pred_len:])
            return baseline + offset
        else:
            current_x = x_hist
            pred_offsets = []
            for t in range(self.pred_len):
                inputs_t = current_x.transpose(1, 2)
                out = self.network(inputs_t)
                out = out.transpose(1, 2)
                offset = self.regression_head(out[:, -1:])
                pred_offsets.append(offset)
                pred_coord = baseline[:, t:t+1] + offset
                next_x = self._format_single_step(current_x[:, -1], pred_coord, self.dt)
                current_x = torch.cat([current_x, next_x], dim=1)
            return baseline + torch.cat(pred_offsets, dim=1)


class TransformerForecaster(BaseContinuousForecaster):
    def __init__(self, input_dim, pred_len, d_model=768, n_heads=12, n_layers=6, dropout=0.1, dt=0.05):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.dt = dt
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.time_embed = nn.Embedding(512, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2),
        )
        nn.init.zeros_(self.regression_head[-1].weight)
        nn.init.zeros_(self.regression_head[-1].bias)

    def _encode(self, state_seq):
        _, seq_len, _ = state_seq.shape
        x = self.input_proj(state_seq) + self.time_embed(torch.arange(seq_len, device=state_seq.device).view(1, seq_len))
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=state_seq.device), diagonal=1)
        return self.encoder(x, mask=mask)

    def forward(self, x_hist, future_norm=None, **kwargs):
        last_pos = x_hist[:, -1, :2]
        last_vel = x_hist[:, -1, 2:4] if x_hist.shape[-1] >= 4 else torch.zeros_like(last_pos)
        baseline = build_constant_velocity_baseline(last_pos, last_vel, self.pred_len, self.dt)

        if future_norm is not None:
            future_x = self._format_future_inputs(x_hist, future_norm, self.dt)
            inputs = torch.cat([x_hist, future_x[:, :-1]], dim=1)
            enc = self._encode(inputs)
            offset = self.regression_head(enc[:, -self.pred_len:])
            return baseline + offset
        else:
            current_x = x_hist
            pred_offsets = []
            for t in range(self.pred_len):
                enc = self._encode(current_x)
                offset = self.regression_head(enc[:, -1:])
                pred_offsets.append(offset)
                pred_coord = baseline[:, t:t+1] + offset
                next_x = self._format_single_step(current_x[:, -1], pred_coord, self.dt)
                current_x = torch.cat([current_x, next_x], dim=1)
            return baseline + torch.cat(pred_offsets, dim=1)


class TrajectoryRefiner(nn.Module):
    def __init__(self, d_llm, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.conv1d = nn.Conv1d(in_channels=d_llm + 2, out_channels=32, kernel_size=3, padding=1)
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(32, 2)
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, coarse_traj, llm_hidden):
        x = torch.cat([coarse_traj, llm_hidden], dim=-1)
        x = x.transpose(1, 2) 
        feat = self.conv1d(x)
        feat = feat.transpose(1, 2) 
        global_delta = 0.5 * torch.tanh(self.out_proj(feat)) 
        return coarse_traj + global_delta


class GPT2ARForecaster(BaseContinuousForecaster):
    def __init__(
        self, vocab_size, token_feature_table, pred_len, input_dim, token_f_m, token_l_m, 
        d_llm=768, llm_layers=6, dropout=0.1, finetune_mode="lora", unfreeze_last_n=2, use_lora=True, lora_r=16,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.d_llm = d_llm
        self.vocab_size = vocab_size
        self.d_gru = 128
        
        self.register_buffer("token_feature_table", token_feature_table.float())
        self.register_buffer("token_f_m", token_f_m.float())
        self.register_buffer("token_l_m", token_l_m.float())
        
        self.token_embed = nn.Embedding(vocab_size, d_llm)
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(token_feature_table.shape[-1]),
            nn.Linear(token_feature_table.shape[-1], d_llm),
            nn.Tanh(),
        )
        self.history_adapter = nn.Sequential(
            nn.LayerNorm(d_llm), nn.Linear(d_llm, d_llm), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_llm, d_llm),
        )

        model_source, local_only = resolve_gpt2_source()
        config = GPT2Config.from_pretrained(model_source, local_files_only=local_only)
        config.num_hidden_layers = llm_layers
        self.gpt2 = GPT2Model.from_pretrained(model_source, config=config, local_files_only=local_only)

        with torch.no_grad():
            pretrained_wte = self.gpt2.wte.weight.detach()
            idx = torch.linspace(0, pretrained_wte.shape[0] - 1, steps=vocab_size, dtype=torch.float32).round().long()
            self.token_embed.weight.copy_(0.25 * pretrained_wte.index_select(0, idx))
            nn.init.xavier_uniform_(self.semantic_proj[1].weight, gain=0.5)
            nn.init.zeros_(self.semantic_proj[1].bias)

        finetune_mode = str(finetune_mode).lower()
        for p in self.gpt2.parameters(): p.requires_grad = False
        if finetune_mode == "full":
            for p in self.gpt2.parameters(): p.requires_grad = True
        elif finetune_mode == "top":
            for block in getattr(self.gpt2, "h", [])[-max(int(unfreeze_last_n), 0):]:
                for p in block.parameters(): p.requires_grad = True
            if hasattr(self.gpt2, "ln_f"):
                for p in self.gpt2.ln_f.parameters(): p.requires_grad = True
        elif finetune_mode == "lora" and use_lora and get_peft_model is not None and LoraConfig is not None:
            self.gpt2 = get_peft_model(self.gpt2, LoraConfig(r=lora_r, lora_alpha=lora_r*2, target_modules=["c_attn", "c_proj"], lora_dropout=0.1))

        self.token_cls_head = nn.Sequential(nn.LayerNorm(d_llm), nn.Linear(d_llm, vocab_size))
        self.kinematic_encoder = nn.GRU(input_size=input_dim, hidden_size=self.d_gru, num_layers=1, batch_first=True)
        
        self.coord_refine_head = nn.Sequential(
            nn.Linear(d_llm + self.d_gru, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2),
        )
        nn.init.zeros_(self.coord_refine_head[-1].weight)
        nn.init.zeros_(self.coord_refine_head[-1].bias)

        self.global_refiner = TrajectoryRefiner(d_llm=d_llm, pred_len=pred_len)

    def _combined_token_table(self):
        return F.layer_norm(self.token_embed.weight + self.semantic_proj(self.token_feature_table), (self.d_llm,))

    def history_embed_from_ids(self, token_ids): return self._combined_token_table()[token_ids]
    def history_embed_from_probs(self, token_probs): return token_probs @ self._combined_token_table()

    def forward(self, hist_embeds, x_hist, hist_real, pos_mean, pos_scale, future_ids=None, dt=0.05):
        _, gru_hidden = self.kinematic_encoder(x_hist)
        h_kinematic = gru_hidden[-1].unsqueeze(1)

        current_embeds = hist_embeds + self.history_adapter(hist_embeds)
        
        current_pos_real = hist_real[:, -1, :2].clone()
        current_heading = torch.atan2(
            hist_real[:, -1, 1] - hist_real[:, -2, 1], 
            hist_real[:, -1, 0] - hist_real[:, -2, 0]
        )
        
        if future_ids is not None:
            future_token_embeds = self.history_embed_from_ids(future_ids)
            future_embeds = future_token_embeds + self.history_adapter(future_token_embeds)
            full_embeds = torch.cat([current_embeds, future_embeds[:, :-1]], dim=1)
            
            hidden_tf = self.gpt2(inputs_embeds=full_embeds).last_hidden_state[:, -self.pred_len:, :]
            logits_tf = self.token_cls_head(hidden_tf)  
            
            pred_coords_real = []
            hidden_list = []
            current_embeds_ar = current_embeds
            current_pos_real_ar = current_pos_real.clone()
            current_heading_ar = current_heading.clone()
            
            outputs = self.gpt2(inputs_embeds=current_embeds_ar, use_cache=True)
            hidden_t = outputs.last_hidden_state[:, -1:, :]
            past_kv = outputs.past_key_values
            
            for t in range(self.pred_len):
                hidden_list.append(hidden_t)
                logit_t = self.token_cls_head(hidden_t)
                
                fused_hidden_t = torch.cat([hidden_t, h_kinematic], dim=-1)
                raw_res = self.coord_refine_head(fused_hidden_t)
                res_f = 0.5 * torch.tanh(raw_res[..., 0]).squeeze(1) 
                res_l = 0.5 * torch.tanh(raw_res[..., 1]).squeeze(1) 
                
                probs_t = F.softmax(logit_t, dim=-1).squeeze(1)
                expected_f = (probs_t * self.token_f_m).sum(dim=-1)
                expected_l = (probs_t * self.token_l_m).sum(dim=-1)
                
                cos_h = torch.cos(current_heading_ar)
                sin_h = torch.sin(current_heading_ar)
                
                coarse_dx = expected_f * cos_h - expected_l * sin_h
                coarse_dy = expected_f * sin_h + expected_l * cos_h
                res_dx = res_f * cos_h - res_l * sin_h
                res_dy = res_f * sin_h + res_l * cos_h
                
                current_pos_real_ar = current_pos_real_ar + torch.stack([coarse_dx + res_dx, coarse_dy + res_dy], dim=-1)
                pred_coords_real.append(current_pos_real_ar)
                
                step_norm = torch.hypot(coarse_dx, coarse_dy)
                mask = step_norm > 0.05
                current_heading_ar = torch.where(mask, torch.atan2(coarse_dy, coarse_dx), current_heading_ar)
                
                prob_ar = F.softmax(logit_t, dim=-1)
                next_token_embed = self.history_embed_from_probs(prob_ar)
                next_embed = next_token_embed + self.history_adapter(next_token_embed)
                
                if t < self.pred_len - 1:
                    outputs = self.gpt2(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    hidden_t = outputs.last_hidden_state
                    past_kv = outputs.past_key_values
                
            coarse_real_tensor = torch.stack(pred_coords_real, dim=1)
            full_hidden_tensor = torch.cat(hidden_list, dim=1)
            coarse_norm_tensor = (coarse_real_tensor - pos_mean) / pos_scale
            final_norm_tensor = self.global_refiner(coarse_norm_tensor, full_hidden_tensor)
            
            return logits_tf, final_norm_tensor
            
        else:
            pred_logits, pred_coords_real = [], []
            hidden_list = []
            
            outputs = self.gpt2(inputs_embeds=current_embeds, use_cache=True)
            hidden_t = outputs.last_hidden_state[:, -1:, :]
            past_kv = outputs.past_key_values
            
            for t in range(self.pred_len):
                hidden_list.append(hidden_t)
                logit_t = self.token_cls_head(hidden_t)
                pred_logits.append(logit_t)
                
                fused_hidden_t = torch.cat([hidden_t, h_kinematic], dim=-1)
                raw_res = self.coord_refine_head(fused_hidden_t)
                res_f = 0.5 * torch.tanh(raw_res[..., 0]).squeeze(1) 
                res_l = 0.5 * torch.tanh(raw_res[..., 1]).squeeze(1) 
                
                probs_t = F.softmax(logit_t, dim=-1).squeeze(1)
                expected_f = (probs_t * self.token_f_m).sum(dim=-1)
                expected_l = (probs_t * self.token_l_m).sum(dim=-1)
                
                cos_h = torch.cos(current_heading)
                sin_h = torch.sin(current_heading)
                
                coarse_dx = expected_f * cos_h - expected_l * sin_h
                coarse_dy = expected_f * sin_h + expected_l * cos_h
                res_dx = res_f * cos_h - res_l * sin_h
                res_dy = res_f * sin_h + res_l * cos_h
                
                current_pos_real = current_pos_real + torch.stack([coarse_dx + res_dx, coarse_dy + res_dy], dim=-1)
                pred_coords_real.append(current_pos_real.unsqueeze(1))
                
                step_norm = torch.hypot(coarse_dx, coarse_dy)
                mask = step_norm > 0.05
                current_heading = torch.where(mask, torch.atan2(coarse_dy, coarse_dx), current_heading)
                
                prob_ar = F.softmax(logit_t, dim=-1)
                next_token_embed = self.history_embed_from_probs(prob_ar)
                next_embed = next_token_embed + self.history_adapter(next_token_embed)
                
                if t < self.pred_len - 1:
                    outputs = self.gpt2(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    hidden_t = outputs.last_hidden_state
                    past_kv = outputs.past_key_values

            coarse_real_tensor = torch.cat(pred_coords_real, dim=1)
            full_hidden_tensor = torch.cat(hidden_list, dim=1)
            coarse_norm_tensor = (coarse_real_tensor - pos_mean) / pos_scale
            final_norm_tensor = self.global_refiner(coarse_norm_tensor, full_hidden_tensor)
            
            return torch.cat(pred_logits, dim=1), final_norm_tensor


def build_model(model_name: str, args, input_dim: int):
    if model_name == "lstm": 
        return LSTMForecaster(input_dim=input_dim, pred_len=args.pred_len, hidden_dim=256, n_layers=2, dropout=args.dropout, dt=args.dt)
    if model_name == "tcn": 
        return TCNForecaster(input_dim=input_dim, pred_len=args.pred_len, dropout=args.dropout, dt=args.dt)
    if model_name == "transformer": 
        return TransformerForecaster(input_dim=input_dim, pred_len=args.pred_len, d_model=768, n_heads=12, n_layers=6, dropout=args.dropout, dt=args.dt)
        
    if model_name in {"gpt2_soft"}:
        tokenizer_cfg = MotionTokenizerConfig(dt=args.dt)
        tokenizer = MotionTokenizerV7(tokenizer_cfg)
        token_f_m, token_l_m = tokenizer._token_tables(device=torch.device("cpu"), dtype=torch.float32)
        return GPT2ARForecaster(
            vocab_size=tokenizer.vocab_size,
            token_feature_table=build_motion_token_feature_table(tokenizer),
            pred_len=args.pred_len,
            input_dim=input_dim,
            token_f_m=token_f_m,
            token_l_m=token_l_m,
            d_llm=args.llm_dim,
            llm_layers=args.llm_layers,
            dropout=args.dropout,
            finetune_mode=args.gpt2_finetune_mode,
            unfreeze_last_n=args.gpt2_unfreeze_last_n,
            use_lora=bool(args.use_lora),
            lora_r=args.lora_r,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(model, model_name: str, args):
    if not model_name.startswith("gpt2"): 
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    gpt2_params, new_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "gpt2" in name: gpt2_params.append(param)
        else: new_params.append(param)
    groups = []
    if gpt2_params: groups.append({"params": gpt2_params, "lr": args.lr_gpt2})
    if new_params: groups.append({"params": new_params, "lr": args.lr})
    return optim.AdamW(groups, weight_decay=args.weight_decay)


def compute_window_turn_score(locs, dt=0.05):
    dx = np.diff(locs[:, 0]); dy = np.diff(locs[:, 1])
    speeds = np.sqrt(dx**2 + dy**2) / dt
    mask = speeds > 0.5
    if mask.sum() < 3: return 0.0
    headings = np.arctan2(dy[mask], dx[mask])
    dh = np.diff(headings)
    return float(np.degrees(np.sum(np.abs(np.arctan2(np.sin(dh), np.cos(dh))))))


def summarize_grouped_metrics(sample_ades, sample_fdes, turn_scores, straight_thr: float, turn_thr: float):
    ade_arr = np.asarray(sample_ades, dtype=np.float32)
    fde_arr = np.asarray(sample_fdes, dtype=np.float32)
    turn_arr = np.asarray(turn_scores, dtype=np.float32)
    straight_mask = turn_arr < straight_thr
    turn_mask = turn_arr >= turn_thr
    def group_stats(mask):
        if mask.sum() == 0: return float("nan"), float("nan"), 0
        return float(ade_arr[mask].mean()), float(fde_arr[mask].mean()), int(mask.sum())
    straight_ade, straight_fde, straight_count = group_stats(straight_mask)
    turn_ade, turn_fde, turn_count = group_stats(turn_mask)
    return {
        "straight_ade": straight_ade, "straight_fde": straight_fde, "straight_count": straight_count,
        "turn_ade": turn_ade, "turn_fde": turn_fde, "turn_count": turn_count,
    }


def transition_uncertainty_to_temperature(hist_uncertainty, args):
    trans_uncertainty, norm_uncertainty = normalize_transition_uncertainty(hist_uncertainty, q_low=args.soft_uncertainty_q_low, q_high=args.soft_uncertainty_q_high)
    temperature = normalized_uncertainty_to_temperature(norm_uncertainty, base_temp=args.soft_token_base_temp, alpha=args.soft_token_alpha, min_temp=args.soft_token_min_temp, max_temp=args.soft_token_max_temp, gamma=args.soft_token_gamma)
    return trans_uncertainty, temperature


def prepare_gpt2_token_batch(batch, tokenizer: MotionTokenizerV7, dt: float, device, model_name: str, model, args):
    hist_real = batch["hist_real"].to(device)
    hist_uncertainty = batch["hist_uncertainty"].to(device)
    hist_state = tokenizer.build_state_sequence(hist_real, dt=dt)
    hist_ids = tokenizer.motion_token_ids_from_state_sequence(hist_state)[:, : hist_real.shape[1] - 1]

    if model_name == "gpt2_hard":
        hist_embeds = model.history_embed_from_ids(hist_ids.long())
        hist_temperature = torch.full_like(hist_ids, fill_value=args.soft_token_base_temp, dtype=torch.float32)
    else:
        _, hist_temperature = transition_uncertainty_to_temperature(hist_uncertainty, args)
        hist_probs = tokenizer.motion_soft_targets_from_state_sequence(hist_state, temperature=hist_temperature)
        hist_embeds = model.history_embed_from_probs(topk_renormalize_probs(hist_probs, args.soft_token_topk))

    return {"hist_embeds": hist_embeds, "hist_temperature": hist_temperature}


@torch.no_grad()
def evaluate_model_once(model_name, model, val_loader, pos_mean_t, pos_scale_t, device, args):
    model.eval()
    is_gpt2_token = model_name.startswith("gpt2")
    tokenizer = MotionTokenizerV7(MotionTokenizerConfig(dt=args.dt)) if is_gpt2_token else None

    sample_ades, sample_fdes, turn_scores, token_temps, token_accs = [], [], [], [], []
    step_err_sums = {step: 0.0 for step in args.report_steps_list}
    step_err_counts = {step: 0 for step in args.report_steps_list}

    for batch in val_loader:
        x_hist = batch["hist_input"].to(device)
        y_future = batch["future_norm"].to(device)
        
        if is_gpt2_token:
            gpt_batch = prepare_gpt2_token_batch(batch, tokenizer, args.dt, device, model_name, model, args)
            logits, pred_norm = model(
                gpt_batch["hist_embeds"], x_hist, batch["hist_real"].to(device), pos_mean_t, pos_scale_t, future_ids=None, dt=args.dt
            )
            token_temps.append(gpt_batch["hist_temperature"].mean().item())
            
            full_real = torch.cat([batch["hist_real"][:, -1:].to(device), batch["future_real"].to(device)], dim=1)
            future_ids = tokenizer.motion_token_ids_from_state_sequence(tokenizer.build_state_sequence(full_real, dt=args.dt)).long()[:, :args.pred_len]
            token_accs.append((logits.argmax(dim=-1) == future_ids).float().mean().item())
        else:
            pred_norm = model(x_hist, future_norm=None)

        pred_real = pred_norm * pos_scale_t + pos_mean_t
        true_real = y_future * pos_scale_t + pos_mean_t
        err = torch.sqrt(torch.sum((pred_real - true_real) ** 2, dim=-1))
        
        sample_ades.extend(err.mean(dim=1).detach().cpu().tolist())
        sample_fdes.extend(err[:, -1].detach().cpu().tolist())
        turn_scores.extend(compute_window_turn_score(locs, dt=args.dt) for locs in true_real.detach().cpu().numpy())

        for step in args.report_steps_list:
            step_err_sums[step] += err[:, step - 1].sum().item()
            step_err_counts[step] += err.shape[0]

    return {
        "best_val_ade": float(np.mean(sample_ades)), "best_val_fde": float(np.mean(sample_fdes)),
        "val_token_acc": float(np.mean(token_accs)) if token_accs else np.nan,
        "val_token_temp": float(np.mean(token_temps)) if token_temps else np.nan,
        **{f"step{step}_ade": (step_err_sums[step] / max(step_err_counts[step], 1)) for step in args.report_steps_list},
        **summarize_grouped_metrics(sample_ades, sample_fdes, turn_scores, args.straight_turn_score_thr, args.turn_score_thr)
    }


def train_one_model(model_name, args, train_ds, val_ds, pos_mean_t, pos_scale_t, log, epoch_writer, ckpt_root, device):
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=pin)

    input_dim = {"coord": 2, "state": 4, "state_unc": 5}[args.input_mode]
    model = build_model(model_name, args, input_dim).to(device)
    optimizer = build_optimizer(model, model_name, args)
    
    is_gpt2_token = model_name.startswith("gpt2")
    tokenizer = MotionTokenizerV7(MotionTokenizerConfig(dt=args.dt)) if is_gpt2_token else None
    early = EarlyStopping(patience=args.patience)
    step_weights = torch.linspace(args.future_loss_min_weight, args.future_loss_max_weight, steps=args.pred_len, device=device, dtype=pos_mean_t.dtype)

    model_snr_folder = f"{model_name}_snr{args.snr}"
    best_path = os.path.join(ckpt_root, model_snr_folder, "checkpoint.pth")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    
    log(f"\n[{model_snr_folder}] Params: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.train_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses, train_ades, train_token_accs, train_token_temps = [], [], [], []
        train_loss_cls_list, train_loss_reg_list = [], []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x_hist = batch["hist_input"].to(device)
            y_future = batch["future_norm"].to(device)

            if is_gpt2_token:
                gpt_batch = prepare_gpt2_token_batch(batch, tokenizer, args.dt, device, model_name, model, args)
                full_real = torch.cat([batch["hist_real"][:, -1:].to(device), batch["future_real"].to(device)], dim=1)
                future_ids = tokenizer.motion_token_ids_from_state_sequence(tokenizer.build_state_sequence(full_real, dt=args.dt)).long()[:, :args.pred_len]
                
                logits, pred = model(
                    gpt_batch["hist_embeds"], x_hist, batch["hist_real"].to(device), pos_mean_t, pos_scale_t, future_ids=future_ids, dt=args.dt
                )
                
                loss_cls = (F.cross_entropy(logits.view(-1, tokenizer.vocab_size), future_ids.view(-1), reduction="none").view(logits.shape[0], args.pred_len) * step_weights).mean()
                loss_reg = weighted_smooth_l1_loss(pred, y_future, step_weights)
                loss = loss_cls + args.refine_weight * loss_reg
                
                train_loss_cls_list.append(loss_cls.item())
                train_loss_reg_list.append(loss_reg.item())
                train_token_accs.append((logits.argmax(dim=-1) == future_ids).float().mean().item())
                train_token_temps.append(gpt_batch["hist_temperature"].mean().item())
            else:
                pred = model(x_hist, future_norm=y_future)
                loss = weighted_smooth_l1_loss(pred, y_future, step_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            err = torch.sqrt(torch.sum(((pred.detach() * pos_scale_t + pos_mean_t) - (y_future * pos_scale_t + pos_mean_t)) ** 2, dim=-1))
            train_ades.append(err.mean().item())

        model.eval()
        val_losses, val_ades, val_fdes, val_token_accs, val_token_temps = [], [], [], [], []
        val_step_err_sums = {s: 0.0 for s in args.report_steps_list}
        val_step_err_counts = {s: 0 for s in args.report_steps_list}

        with torch.no_grad():
            for batch in val_loader:
                x_hist = batch["hist_input"].to(device)
                y_future = batch["future_norm"].to(device)

                if is_gpt2_token:
                    gpt_batch = prepare_gpt2_token_batch(batch, tokenizer, args.dt, device, model_name, model, args)
                    logits, pred = model(
                        gpt_batch["hist_embeds"], x_hist, batch["hist_real"].to(device), pos_mean_t, pos_scale_t, future_ids=None, dt=args.dt
                    )
                    
                    full_real = torch.cat([batch["hist_real"][:, -1:].to(device), batch["future_real"].to(device)], dim=1)
                    future_ids = tokenizer.motion_token_ids_from_state_sequence(tokenizer.build_state_sequence(full_real, dt=args.dt)).long()[:, :args.pred_len]
                    
                    loss_cls = (F.cross_entropy(logits.view(-1, tokenizer.vocab_size), future_ids.view(-1), reduction="none").view(logits.shape[0], args.pred_len) * step_weights).mean()
                    loss_reg = weighted_smooth_l1_loss(pred, y_future, step_weights)
                    loss = loss_cls + args.refine_weight * loss_reg
                    val_token_accs.append((logits.argmax(dim=-1) == future_ids).float().mean().item())
                    val_token_temps.append(gpt_batch["hist_temperature"].mean().item())
                else:
                    pred = model(x_hist, future_norm=None)
                    loss = weighted_smooth_l1_loss(pred, y_future, step_weights)

                val_losses.append(loss.item())
                err = torch.sqrt(torch.sum(((pred * pos_scale_t + pos_mean_t) - (y_future * pos_scale_t + pos_mean_t)) ** 2, dim=-1))
                val_ades.append(err.mean(dim=1).mean().item())
                val_fdes.append(err[:, -1].mean().item())

                for step in args.report_steps_list:
                    val_step_err_sums[step] += err[:, step - 1].sum().item()
                    val_step_err_counts[step] += err.shape[0]

        improved = early.step(float(np.mean(val_ades)))
        if improved: torch.save(model.state_dict(), best_path)

        val_step_metrics = {f"step{s}_ade": (val_step_err_sums[s] / max(val_step_err_counts[s], 1)) for s in args.report_steps_list}
        epoch_writer.writerow({
            "model": model_snr_folder, "epoch": epoch,
            "train_loss": float(np.mean(train_losses)), "train_ade": float(np.mean(train_ades)),
            "val_loss": float(np.mean(val_losses)), "val_ade": float(np.mean(val_ades)), "val_fde": float(np.mean(val_fdes)),
            "best_val_ade": early.best_score,
            "train_token_acc": float(np.mean(train_token_accs)) if train_token_accs else np.nan,
            "val_token_acc": float(np.mean(val_token_accs)) if val_token_accs else np.nan,
            "train_token_temp": float(np.mean(train_token_temps)) if train_token_temps else np.nan,
            "val_token_temp": float(np.mean(val_token_temps)) if val_token_temps else np.nan,
            **val_step_metrics
        })

        step_msg = " ".join(f"S{step}={val_step_metrics[f'step{step}_ade']:.3f}m" for step in args.report_steps_list)
        msg = (f"[{model_snr_folder}] Ep {epoch:3d}/{args.train_epochs} [{time.time() - t0:.0f}s] "
               f"Loss={np.mean(train_losses):.4f}/{np.mean(val_losses):.4f} | "
               f"Train_ADE={np.mean(train_ades):.3f}m | Val_ADE={np.mean(val_ades):.3f}m FDE={np.mean(val_fdes):.3f}m "
               f"(best={early.best_score:.3f}) | {step_msg}")
        if is_gpt2_token: 
            msg += f" | L_cls={np.mean(train_loss_cls_list):.3f} L_reg={np.mean(train_loss_reg_list):.4f} | Acc={np.mean(train_token_accs):.2%}/{np.mean(val_token_accs):.2%}"
        log(msg)

        if early.early_stop:
            log(f"[{model_snr_folder}] Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
    final_eval = evaluate_model_once(model_name, model, val_loader, pos_mean_t, pos_scale_t, device, args)
    log(f"[{model_snr_folder}] Best-checkpoint eval | ADE={final_eval['best_val_ade']:.3f} | Turn={final_eval['turn_ade']:.3f}")
    return {"model": model_snr_folder, "checkpoint": best_path, **final_eval}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--snr", type=int, default=5)
    parser.add_argument("--models", type=str, default="gpt2_soft,lstm,tcn,transformer")
    parser.add_argument("--input_mode", type=str, default="state_unc", choices=["coord", "state", "state_unc"])
    parser.add_argument("--seq_len", type=int, default=72)
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=61)
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--cache_path", type=str, default="")
    
    # 💡 补回的特征提取与定位超参数 (与10dB生成完全一致)
    parser.add_argument("--range_weight", type=float, default=1.0)
    parser.add_argument("--angle_weight", type=float, default=1.0)
    parser.add_argument("--posterior_temp", type=float, default=0.25)
    parser.add_argument("--use_expectation", type=int, default=0)
    parser.add_argument("--train_stride", type=int, default=4)
    parser.add_argument("--eval_stride", type=int, default=3)
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_val_files", type=int, default=0)
    parser.add_argument("--max_traj_per_file", type=int, default=0)
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_gpt2", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--llm_layers", type=int, default=6)
    parser.add_argument("--gpt2_finetune_mode", type=str, default="top")
    parser.add_argument("--gpt2_unfreeze_last_n", type=int, default=2)
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--soft_token_base_temp", type=float, default=0.18)
    parser.add_argument("--soft_token_alpha", type=float, default=0.12)
    parser.add_argument("--soft_token_min_temp", type=float, default=0.12)
    parser.add_argument("--soft_token_max_temp", type=float, default=1.00)
    parser.add_argument("--soft_token_gamma", type=float, default=2.0)
    parser.add_argument("--soft_uncertainty_q_low", type=float, default=0.2)
    parser.add_argument("--soft_uncertainty_q_high", type=float, default=0.8)
    parser.add_argument("--soft_token_topk", type=int, default=9)
    parser.add_argument("--future_loss_min_weight", type=float, default=0.5)
    parser.add_argument("--future_loss_max_weight", type=float, default=1.0)
    parser.add_argument("--report_steps", type=str, default="1,5,10,15,20")
    parser.add_argument("--straight_turn_score_thr", type=float, default=6.0)
    parser.add_argument("--turn_score_thr", type=float, default=20.0)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--refine_weight", type=float, default=10.0)
    args, _ = parser.parse_known_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.report_steps_list = parse_report_steps(args.report_steps, args.pred_len)

    model_names = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    model_tag = "-".join(model_names) if model_names else "nomodel"
    model_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_tag)

    os.makedirs("./logs", exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)
    
    run_tag = f"{model_tag}_snr{args.snr}_sl{args.seq_len}_pl{args.pred_len}_SWEEP"
    log_path = os.path.join("./logs", f"localized_predictor_{run_tag}.txt")
    epoch_csv_path = os.path.join("./logs", f"localized_predictor_{run_tag}.csv")
    summary_csv_path = os.path.join("./logs", f"localized_summary_{run_tag}.csv")
    ckpt_root = os.path.join(args.checkpoints, f"localized_{run_tag}")

    with open(log_path, "w", encoding="utf-8") as logf, open(epoch_csv_path, "w", newline="", encoding="utf-8") as epoch_f:
        def log(msg): print(msg); logf.write(msg + "\n"); logf.flush()
        epoch_writer = csv.DictWriter(epoch_f, fieldnames=["model", "epoch", "train_loss", "train_ade", "val_loss", "val_ade", "val_fde", "best_val_ade", "train_token_acc", "val_token_acc", "train_token_temp", "val_token_temp", *[f"step{s}_ade" for s in args.report_steps_list]])
        epoch_writer.writeheader()

        log("=" * 80)
        log(f"📡 SNR ROBUSTNESS SWEEP | Target SNR: {args.snr}dB")
        
        cache_data, _ = load_or_build_cache(args, log)
        train_gt = cache_data["train_gt"]
        train_mean = train_gt.reshape(-1, 2).mean(axis=0).astype(np.float32)
        train_scale = np.clip(train_gt.reshape(-1, 2).std(axis=0).astype(np.float32), 1e-6, None)
        pos_mean_t = torch.as_tensor(train_mean, dtype=torch.float32, device=device).view(1, 1, 2)
        pos_scale_t = torch.as_tensor(train_scale, dtype=torch.float32, device=device).view(1, 1, 2)

        train_ds = LocalizedWindowDataset(cache_data["train_recovered"], cache_data["train_uncertainty"], cache_data["train_gt"], cache_data["train_samples"], args.seq_len, args.pred_len, train_mean, train_scale, cache_data["train_uncertainty"].mean(), cache_data["train_uncertainty"].std(), args.input_mode, args.dt)
        val_ds = LocalizedWindowDataset(cache_data["val_recovered"], cache_data["val_uncertainty"], cache_data["val_gt"], cache_data["val_samples"], args.seq_len, args.pred_len, train_mean, train_scale, cache_data["train_uncertainty"].mean(), cache_data["train_uncertainty"].std(), args.input_mode, args.dt)

        summaries = [train_one_model(m, args, train_ds, val_ds, pos_mean_t, pos_scale_t, log, epoch_writer, ckpt_root, device) for m in model_names]

        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=summaries[0].keys()).writerows(summaries)

if __name__ == "__main__":
    main()