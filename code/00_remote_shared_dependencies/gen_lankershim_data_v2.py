"""
Vector-feature Lankershim generator for multi-BS ISAC trajectory prediction.

Each BS observation at each step is represented only by feature vectors:
  - distance feature vector
  - radial-velocity feature vector
  - angle feature vector

The vectors are generated from the target geometry using symbol-level style
phase expressions inspired by the ISAC cooperative sensing papers.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c as LIGHT_SPEED
from scipy.interpolate import interp1d

FT_TO_M = 0.3048
AREA_SIZE = 150.0
AREA_CENTER = np.array([AREA_SIZE / 2, AREA_SIZE / 2], dtype=np.float32)
MARGIN = 5.0
DT_DEFAULT = 0.05
TURN_NET_HEADING_DEG = 35.0
TURN_LOCAL_THC_DEG = 45.0
TURN_GLOBAL_THC_DEG = 90.0
TURN_SMOOTH_WINDOW = 9
TURN_LOCAL_WINDOW_SEC = 2.0

FC = 28e9
DELTA_F = 720e3
TSYM = 1.0 / DELTA_F
BS_POS = np.array([[0.0, 0.0], [150.0, 0.0], [75.0, 150.0]], dtype=np.float32)

LEGACY_RANGE_COMPLEX_LEN = 8
LEGACY_VEL_COMPLEX_LEN = 8
LEGACY_ANGLE_COMPLEX_LEN = 4


@dataclass(frozen=True)
class FeatureSpec:
    range_complex_len: int
    vel_complex_len: int
    angle_complex_len: int

    @property
    def per_bs_dim(self):
        return 2 * (self.range_complex_len + self.vel_complex_len + self.angle_complex_len)

    @property
    def rd_map_shape(self):
        return (2, self.range_complex_len, self.vel_complex_len)

    @property
    def ra_map_shape(self):
        return (2, self.range_complex_len, self.angle_complex_len)

    @property
    def dist_dim(self):
        return 2 * self.range_complex_len

    @property
    def vel_dim(self):
        return 2 * self.vel_complex_len

    @property
    def ang_dim(self):
        return 2 * self.angle_complex_len


def resolve_feature_spec(args):
    presets = {
        "legacy": FeatureSpec(
            range_complex_len=LEGACY_RANGE_COMPLEX_LEN,
            vel_complex_len=LEGACY_VEL_COMPLEX_LEN,
            angle_complex_len=LEGACY_ANGLE_COMPLEX_LEN,
        ),
        "localize": FeatureSpec(
            range_complex_len=32,
            vel_complex_len=16,
            angle_complex_len=16,
        ),
    }
    spec = presets[args.feature_preset]
    range_len = args.range_complex_len if args.range_complex_len is not None else spec.range_complex_len
    vel_len = args.vel_complex_len if args.vel_complex_len is not None else spec.vel_complex_len
    angle_len = args.angle_complex_len if args.angle_complex_len is not None else spec.angle_complex_len
    return FeatureSpec(
        range_complex_len=int(range_len),
        vel_complex_len=int(vel_len),
        angle_complex_len=int(angle_len),
    )


def atomic_save_npy(path, array):
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        size_mib = np.asarray(array).nbytes / (1024.0 * 1024.0)
        raise OSError(
            f"Atomic save failed for '{path}' (~{size_mib:.2f} MiB). "
            f"This usually indicates disk-full/quota pressure or an unstable filesystem write. "
            f"Original error: {exc}"
        ) from exc


def atomic_save_npz(path, **arrays):
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise OSError(
            f"Atomic save failed for '{path}'. "
            f"This usually indicates disk-full/quota pressure or an unstable filesystem write. "
            f"Original error: {exc}"
        ) from exc


def save_feature_config(out_dir, feature_spec: FeatureSpec, dt):
    atomic_save_npz(
        os.path.join(out_dir, "feature_config.npz"),
        bs_count=np.int32(BS_POS.shape[0]),
        range_complex_len=np.int32(feature_spec.range_complex_len),
        vel_complex_len=np.int32(feature_spec.vel_complex_len),
        angle_complex_len=np.int32(feature_spec.angle_complex_len),
        per_bs_dim=np.int32(feature_spec.per_bs_dim),
        dist_dim=np.int32(feature_spec.dist_dim),
        vel_dim=np.int32(feature_spec.vel_dim),
        ang_dim=np.int32(feature_spec.ang_dim),
        rd_map_shape=np.asarray(feature_spec.rd_map_shape, dtype=np.int32),
        ra_map_shape=np.asarray(feature_spec.ra_map_shape, dtype=np.int32),
        dt=np.float32(dt),
        fc=np.float32(FC),
        delta_f=np.float32(DELTA_F),
        bs_positions=BS_POS.astype(np.float32),
        measurement_dim=np.int32(5),
    )


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def dominant_value(series, default=-1):
    mode = series.mode(dropna=True)
    if not mode.empty:
        return int(mode.iat[0])
    if len(series) == 0:
        return default
    value = series.iloc[0]
    return int(value) if pd.notna(value) else default


def compute_thc(traj, dt=DT_DEFAULT):
    dx = np.diff(traj[:, 0])
    dy = np.diff(traj[:, 1])
    speeds = np.sqrt(dx**2 + dy**2) / dt
    mask = speeds > 0.5
    if mask.sum() < 3:
        return 0.0
    headings = np.arctan2(dy[mask], dx[mask])
    d_heading = wrap_angle(np.diff(headings))
    return float(np.degrees(np.sum(np.abs(d_heading))))


def smooth_trajectory(traj, window=TURN_SMOOTH_WINDOW):
    traj = np.asarray(traj, dtype=np.float32)
    if window <= 1 or len(traj) < window:
        return traj.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    x = np.pad(traj[:, 0], (pad, pad), mode="edge")
    y = np.pad(traj[:, 1], (pad, pad), mode="edge")
    xs = np.convolve(x, kernel, mode="valid")
    ys = np.convolve(y, kernel, mode="valid")
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def compute_turn_metrics(traj, dt=DT_DEFAULT):
    traj_s = smooth_trajectory(traj)
    vel = trajectory_velocity(traj_s, dt=dt)
    speeds = np.sqrt(np.sum(vel**2, axis=-1))
    mask = speeds > 0.5
    if mask.sum() < 6:
        return {
            "global_thc_deg": 0.0,
            "local_thc_deg": 0.0,
            "net_heading_deg": 0.0,
        }

    headings = np.arctan2(vel[mask, 1], vel[mask, 0])
    d_heading = wrap_angle(np.diff(headings))
    abs_dh = np.abs(d_heading)
    global_thc_deg = float(np.degrees(np.sum(abs_dh)))

    local_steps = max(3, int(TURN_LOCAL_WINDOW_SEC / dt))
    if len(abs_dh) == 0:
        local_thc_deg = 0.0
    elif len(abs_dh) <= local_steps:
        local_thc_deg = float(np.degrees(np.sum(abs_dh)))
    else:
        prefix = np.concatenate([[0.0], np.cumsum(abs_dh)])
        window_sum = prefix[local_steps:] - prefix[:-local_steps]
        local_thc_deg = float(np.degrees(window_sum.max()))

    edge = min(8, max(2, len(headings) // 6))
    start_heading = np.mean(headings[:edge])
    end_heading = np.mean(headings[-edge:])
    net_heading_deg = float(np.degrees(np.abs(wrap_angle(end_heading - start_heading))))
    return {
        "global_thc_deg": global_thc_deg,
        "local_thc_deg": local_thc_deg,
        "net_heading_deg": net_heading_deg,
    }


def infer_turn_label(traj, dt=DT_DEFAULT):
    metrics = compute_turn_metrics(traj, dt=dt)
    is_turn = int(
        (metrics["net_heading_deg"] >= TURN_NET_HEADING_DEG and metrics["local_thc_deg"] >= TURN_LOCAL_THC_DEG)
        or metrics["global_thc_deg"] >= TURN_GLOBAL_THC_DEG
    )
    return metrics, is_turn


def rotate_trajectory(traj, angle_deg, center=AREA_CENTER):
    theta = np.radians(angle_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    shifted = traj - center
    return (rot @ shifted.T).T + center


def mirror_trajectory_x(traj):
    mirrored = traj.copy()
    mirrored[:, 0] = AREA_SIZE - mirrored[:, 0]
    return mirrored


def is_in_area(traj, margin=MARGIN):
    return (
        traj[:, 0].min() >= margin
        and traj[:, 0].max() <= AREA_SIZE - margin
        and traj[:, 1].min() >= margin
        and traj[:, 1].max() <= AREA_SIZE - margin
    )


def ensure_traj_length(traj, T):
    """
    Resample a trajectory to exactly T steps.
    """
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == T:
        return traj
    if len(traj) == 0:
        return np.zeros((T, 2), dtype=np.float32)
    if len(traj) == 1:
        return np.repeat(traj[:1], T, axis=0).astype(np.float32)

    src_idx = np.linspace(0.0, 1.0, len(traj), dtype=np.float32)
    dst_idx = np.linspace(0.0, 1.0, T, dtype=np.float32)
    x = np.interp(dst_idx, src_idx, traj[:, 0])
    y = np.interp(dst_idx, src_idx, traj[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def load_and_preprocess(csv_path, T=300, dt_target=DT_DEFAULT):
    df = pd.read_csv(csv_path, sep=r",|\t", engine="python")

    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if "vehicle_id" in key:
            col_map["vid"] = col
        elif "frame_id" in key:
            col_map["fid"] = col
        elif "local_x" in key:
            col_map["x"] = col
        elif "local_y" in key:
            col_map["y"] = col
        elif key == "movement":
            col_map["movement"] = col
        elif key == "int_id":
            col_map["int_id"] = col

    df = df.sort_values([col_map["vid"], col_map["fid"]])
    groups = df.groupby(col_map["vid"])
    sample_fids = df[col_map["fid"]].diff().dropna()
    frame_step = sample_fids[sample_fids > 0].median()
    dt_original = frame_step * 0.1
    min_frames = int((T * dt_target) / dt_original) + 10

    records = []
    skipped = 0
    for _, grp in groups:
        grp = grp.sort_values(col_map["fid"])
        if len(grp) < min_frames:
            skipped += 1
            continue

        x_m = grp[col_map["x"]].to_numpy(dtype=float) * FT_TO_M
        y_m = grp[col_map["y"]].to_numpy(dtype=float) * FT_TO_M
        fids = grp[col_map["fid"]].to_numpy(dtype=float)
        t_orig = (fids - fids[0]) * (dt_original / frame_step)

        _, unique_idx = np.unique(t_orig, return_index=True)
        t_orig = t_orig[unique_idx]
        x_m = x_m[unique_idx]
        y_m = y_m[unique_idx]
        if len(t_orig) < 3:
            skipped += 1
            continue

        t_new = np.arange(0, t_orig[-1], dt_target)
        if len(t_new) < T:
            skipped += 1
            continue
        t_new = t_new[:T]

        fx = interp1d(t_orig, x_m, kind="linear", fill_value="extrapolate")
        fy = interp1d(t_orig, y_m, kind="linear", fill_value="extrapolate")
        traj = np.stack([fx(t_new), fy(t_new)], axis=-1).astype(np.float32)
        if np.isnan(traj).any():
            skipped += 1
            continue

        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        speeds = np.sqrt(dx**2 + dy**2) / dt_target
        if speeds.mean() < 2.0:
            skipped += 1
            continue
        if np.linalg.norm(traj[-1] - traj[0]) < 20.0:
            skipped += 1
            continue

        x_span = traj[:, 0].max() - traj[:, 0].min()
        y_span = traj[:, 1].max() - traj[:, 1].min()
        if x_span > 140.0 or y_span > 140.0:
            skipped += 1
            continue

        shifted = traj.copy()
        shifted[:, 0] += (AREA_SIZE - x_span) / 2.0 - traj[:, 0].min()
        shifted[:, 1] += (AREA_SIZE - y_span) / 2.0 - traj[:, 1].min()
        jitter_x = np.random.uniform(
            -min(20.0, (AREA_SIZE - x_span) / 2.0 - 5.0),
            min(20.0, (AREA_SIZE - x_span) / 2.0 - 5.0),
        )
        jitter_y = np.random.uniform(
            -min(20.0, (AREA_SIZE - y_span) / 2.0 - 5.0),
            min(20.0, (AREA_SIZE - y_span) / 2.0 - 5.0),
        )
        shifted[:, 0] = np.clip(shifted[:, 0] + jitter_x, 2.0, 148.0)
        shifted[:, 1] = np.clip(shifted[:, 1] + jitter_y, 2.0, 148.0)

        metrics, is_turn = infer_turn_label(shifted, dt=dt_target)
        records.append(
            {
                "traj": shifted.astype(np.float32),
                "movement": dominant_value(grp[col_map["movement"]]) if "movement" in col_map else -1,
                "int_id": dominant_value(grp[col_map["int_id"]]) if "int_id" in col_map else -1,
                "thc": metrics["global_thc_deg"],
                "local_thc": metrics["local_thc_deg"],
                "net_heading_deg": metrics["net_heading_deg"],
                "is_turn": is_turn,
                "is_synth": 0,
            }
        )

    print(f"  extracted valid trajectories: {len(records)} (skipped {skipped})")
    return records


def clone_record(record, traj):
    copied = dict(record)
    copied["traj"] = traj.astype(np.float32)
    metrics, is_turn = infer_turn_label(copied["traj"])
    copied["thc"] = metrics["global_thc_deg"]
    copied["local_thc"] = metrics["local_thc_deg"]
    copied["net_heading_deg"] = metrics["net_heading_deg"]
    copied["is_turn"] = is_turn
    return copied


def augment_with_rotations(records):
    augmented = list(records)
    added = 0
    for record in records:
        score = record.get("net_heading_deg", record["thc"])
        if score < 20.0:
            angles = (-30, 30, -45, 45)
        elif score < 60.0:
            angles = (-20, 20, 35)
        else:
            angles = (-15, 15)

        for angle in angles:
            rotated = rotate_trajectory(record["traj"], angle)
            if is_in_area(rotated):
                augmented.append(clone_record(record, rotated))
                added += 1

        mirrored = mirror_trajectory_x(record["traj"])
        if is_in_area(mirrored):
            augmented.append(clone_record(record, mirrored))
            added += 1

    print(f"  geometric augmentation: {len(records)} -> {len(augmented)} (+{added})")
    return augmented


def synthesize_turns(n_synth=240, T=300, dt=DT_DEFAULT, seed=42):
    rng = np.random.RandomState(seed)
    results = []
    for _ in range(n_synth):
        mode = rng.choice(["arc", "s_curve", "right_angle"])
        speed = rng.uniform(5.0, 14.0)

        if mode == "arc":
            radius = rng.uniform(25.0, 65.0)
            span_deg = rng.uniform(45.0, 120.0)
            n_steps = min(T, max(20, int(np.radians(span_deg) * radius / (speed * dt))))
            angles = np.linspace(0.0, np.radians(span_deg), n_steps)
            start_heading = rng.uniform(0.0, 2.0 * np.pi)
            x = radius * np.cos(angles + start_heading)
            y = radius * np.sin(angles + start_heading)
        elif mode == "s_curve":
            t_arr = np.arange(T) * dt
            freq = rng.uniform(0.25, 0.55)
            amp = rng.uniform(12.0, 28.0)
            heading = rng.uniform(0.0, 2.0 * np.pi)
            progress = speed * t_arr
            x = progress * np.cos(heading) + amp * np.sin(freq * t_arr) * np.sin(heading)
            y = progress * np.sin(heading) + amp * np.sin(freq * t_arr) * np.cos(heading)
        else:
            straight_steps = rng.randint(50, 120)
            turn_radius = rng.uniform(8.0, 18.0)
            turn_dir = rng.choice([-1.0, 1.0])
            entry_heading = rng.uniform(0.0, 2.0 * np.pi)
            pts = []
            for idx in range(straight_steps):
                dist = idx * dt * speed
                pts.append([dist * np.cos(entry_heading), dist * np.sin(entry_heading)])
            n_turn = max(16, int((np.pi / 2.0) * turn_radius / (speed * dt)))
            last = np.array(pts[-1], dtype=np.float32)
            for idx in range(n_turn):
                angle = idx / n_turn * (np.pi / 2.0) * turn_dir
                pts.append(
                    last
                    + turn_radius
                    * np.array(
                        [
                            np.sin(angle) * np.cos(entry_heading)
                            - (1.0 - np.cos(angle)) * np.sin(entry_heading) * turn_dir,
                            np.sin(angle) * np.sin(entry_heading)
                            + (1.0 - np.cos(angle)) * np.cos(entry_heading) * turn_dir,
                        ],
                        dtype=np.float32,
                    )
                )
            new_heading = entry_heading + np.pi / 2.0 * turn_dir
            last = np.array(pts[-1], dtype=np.float32)
            while len(pts) < T:
                step_idx = len(pts) - straight_steps - n_turn + 1
                pts.append(
                    last
                    + np.array(
                        [step_idx * dt * speed * np.cos(new_heading), step_idx * dt * speed * np.sin(new_heading)],
                        dtype=np.float32,
                    )
                )
            pts = np.array(pts[:T], dtype=np.float32)
            x = pts[:, 0]
            y = pts[:, 1]

        traj = np.stack([x, y], axis=-1).astype(np.float32)
        traj = ensure_traj_length(traj, T)
        traj[:, 0] += AREA_CENTER[0] - traj[:, 0].mean()
        traj[:, 1] += AREA_CENTER[1] - traj[:, 1].mean()
        if not is_in_area(traj):
            continue
        metrics, is_turn = infer_turn_label(traj, dt=dt)
        if not is_turn:
            continue
        results.append(
            {
                "traj": traj,
                "movement": -1,
                "int_id": -1,
                "thc": metrics["global_thc_deg"],
                "local_thc": metrics["local_thc_deg"],
                "net_heading_deg": metrics["net_heading_deg"],
                "is_turn": 1,
                "is_synth": 1,
            }
        )
    print(f"  synthesized turn trajectories: {len(results)}/{n_synth}")
    return results


def balance_turn_ratio(records, target_turn_ratio, rng):
    if not records:
        return records
    turn_records = [record for record in records if record["is_turn"] == 1]
    straight_records = [record for record in records if record["is_turn"] == 0]
    if not turn_records or not straight_records:
        return records

    ratio = len(turn_records) / len(records)
    if np.isclose(ratio, target_turn_ratio, atol=0.01):
        return records

    desired_turn = int(np.round((target_turn_ratio / max(1e-6, 1.0 - target_turn_ratio)) * len(straight_records)))
    desired_turn = max(1, desired_turn)

    if len(turn_records) > desired_turn:
        keep_idx = rng.choice(len(turn_records), size=desired_turn, replace=False)
        kept_turns = [turn_records[idx] for idx in keep_idx]
        balanced = list(straight_records) + kept_turns
        rng.shuffle(balanced)
        print(
            f"  downsampling turn trajectories: {len(turn_records)} -> {len(kept_turns)} "
            f"to reach turn ratio ~{target_turn_ratio:.2f}"
        )
        return balanced

    needed = desired_turn - len(turn_records)
    print(f"  duplicating {needed} turn trajectories to reach turn ratio ~{target_turn_ratio:.2f}")
    balanced = list(records)
    for _ in range(max(0, needed)):
        src = turn_records[rng.randint(len(turn_records))]
        jitter = rng.normal(scale=0.25, size=src["traj"].shape).astype(np.float32)
        aug = np.clip(src["traj"] + jitter, 2.0, 148.0)
        balanced.append(clone_record(src, aug))
    return balanced


def cap_total_records(records, max_total_traj, rng):
    if max_total_traj is None or max_total_traj <= 0 or len(records) <= max_total_traj:
        return records

    turn_records = [record for record in records if record["is_turn"] == 1]
    straight_records = [record for record in records if record["is_turn"] == 0]

    if not turn_records or not straight_records:
        keep_idx = rng.choice(len(records), size=max_total_traj, replace=False)
        capped = [records[idx] for idx in keep_idx]
        rng.shuffle(capped)
        print(f"  capping total trajectories: {len(records)} -> {len(capped)}")
        return capped

    turn_ratio = len(turn_records) / len(records)
    keep_turn = int(round(max_total_traj * turn_ratio))
    keep_turn = min(max(1, keep_turn), len(turn_records))
    keep_straight = max_total_traj - keep_turn
    keep_straight = min(max(1, keep_straight), len(straight_records))

    total_kept = keep_turn + keep_straight
    if total_kept < max_total_traj:
        extra = min(max_total_traj - total_kept, len(turn_records) - keep_turn)
        keep_turn += max(0, extra)
        total_kept = keep_turn + keep_straight
    if total_kept < max_total_traj:
        extra = min(max_total_traj - total_kept, len(straight_records) - keep_straight)
        keep_straight += max(0, extra)

    turn_idx = rng.choice(len(turn_records), size=keep_turn, replace=False)
    straight_idx = rng.choice(len(straight_records), size=keep_straight, replace=False)
    capped = [turn_records[idx] for idx in turn_idx] + [straight_records[idx] for idx in straight_idx]
    rng.shuffle(capped)
    print(
        f"  capping total trajectories: {len(records)} -> {len(capped)} "
        f"(straight={keep_straight}, turn={keep_turn})"
    )
    return capped


def rician_fading(T, rng, k_factor=6.0, alpha_fade=0.95):
    nu = np.sqrt(k_factor / (k_factor + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (k_factor + 1.0)))
    real = np.zeros(T, dtype=np.float32)
    imag = np.zeros(T, dtype=np.float32)
    real[0] = nu + sigma * rng.randn()
    imag[0] = sigma * rng.randn()
    for t in range(1, T):
        real[t] = alpha_fade * real[t - 1] + (1.0 - alpha_fade) * (nu + sigma * rng.randn())
        imag[t] = alpha_fade * imag[t - 1] + (1.0 - alpha_fade) * sigma * rng.randn()
    return np.sqrt(real**2 + imag**2).astype(np.float32)


def complex_to_real(vec):
    return np.concatenate([vec.real.astype(np.float32), vec.imag.astype(np.float32)], axis=-1)


def complex_map_to_channels(mat):
    return np.stack([mat.real.astype(np.float32), mat.imag.astype(np.float32)], axis=0)


def trajectory_velocity(traj, dt=DT_DEFAULT):
    vel = np.zeros_like(traj, dtype=np.float32)
    vel[1:] = (traj[1:] - traj[:-1]) / dt
    vel[0] = vel[1] if len(traj) > 1 else 0.0
    return vel


def noise_scale_from_snr(snr_db):
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(1.0 / np.sqrt(max(snr_linear, 1e-3)))


def gen_bs_observations(traj, bs_pos, snr_db, rng, dt=DT_DEFAULT, save_rf_maps=True, feature_spec=None):
    feature_spec = feature_spec or FeatureSpec(
        range_complex_len=LEGACY_RANGE_COMPLEX_LEN,
        vel_complex_len=LEGACY_VEL_COMPLEX_LEN,
        angle_complex_len=LEGACY_ANGLE_COMPLEX_LEN,
    )
    vel = trajectory_velocity(traj, dt=dt)
    rel = traj - bs_pos.reshape(1, 2)
    true_r = np.sqrt(np.sum(rel**2, axis=-1)) + 1e-6
    true_phi = np.arctan2(rel[:, 1], rel[:, 0])
    true_vr = np.sum(rel * vel, axis=-1) / true_r

    fade = rician_fading(len(traj), rng)
    noise_scale = noise_scale_from_snr(snr_db)
    feats = np.zeros((len(traj), feature_spec.per_bs_dim), dtype=np.float32)
    meas = np.zeros((len(traj), 5), dtype=np.float32)
    rd_maps = None
    ra_maps = None
    if save_rf_maps:
        rd_maps = np.zeros((len(traj),) + feature_spec.rd_map_shape, dtype=np.float32)
        ra_maps = np.zeros((len(traj),) + feature_spec.ra_map_shape, dtype=np.float32)

    dist_idx = np.arange(feature_spec.range_complex_len, dtype=np.float32)
    vel_idx = np.arange(feature_spec.vel_complex_len, dtype=np.float32)
    ang_idx = np.arange(feature_spec.angle_complex_len, dtype=np.float32)

    for t in range(len(traj)):
        amp = fade[t] * np.exp(-true_r[t] / 220.0)

        dist_vec = amp * np.exp(-1j * 2.0 * np.pi * dist_idx * DELTA_F * 2.0 * true_r[t] / LIGHT_SPEED)
        vel_vec = amp * np.exp(1j * 2.0 * np.pi * FC * 2.0 * true_vr[t] * vel_idx * TSYM / LIGHT_SPEED)
        ang_vec = amp * np.exp(1j * np.pi * ang_idx * np.sin(true_phi[t]))

        dist_noise = noise_scale * (
            rng.randn(feature_spec.range_complex_len) + 1j * rng.randn(feature_spec.range_complex_len)
        ) / np.sqrt(2.0)
        vel_noise = noise_scale * (
            rng.randn(feature_spec.vel_complex_len) + 1j * rng.randn(feature_spec.vel_complex_len)
        ) / np.sqrt(2.0)
        ang_noise = noise_scale * (
            rng.randn(feature_spec.angle_complex_len) + 1j * rng.randn(feature_spec.angle_complex_len)
        ) / np.sqrt(2.0)
        dist_obs = dist_vec + dist_noise
        vel_obs = vel_vec + vel_noise
        ang_obs = ang_vec + ang_noise

        feat = np.concatenate(
            [
                complex_to_real(dist_obs),
                complex_to_real(vel_obs),
                complex_to_real(ang_obs),
            ],
            axis=-1,
        )
        feats[t] = feat.astype(np.float32)
        meas[t] = np.array(
            [
                true_r[t],
                true_vr[t],
                true_phi[t],
                np.sin(true_phi[t]),
                np.cos(true_phi[t]),
            ],
            dtype=np.float32,
        )
        if save_rf_maps:
            rd_maps[t] = complex_map_to_channels(np.outer(dist_obs, np.conjugate(vel_obs)))
            ra_maps[t] = complex_map_to_channels(np.outer(dist_obs, np.conjugate(ang_obs)))
    return feats, rd_maps, ra_maps, meas


def gen_one_file(file_idx, records, T, snr_db, out_dir, rng, dt=DT_DEFAULT, save_rf_maps=True, feature_spec=None, save_measurements=True):
    feature_spec = feature_spec or FeatureSpec(
        range_complex_len=LEGACY_RANGE_COMPLEX_LEN,
        vel_complex_len=LEGACY_VEL_COMPLEX_LEN,
        angle_complex_len=LEGACY_ANGLE_COMPLEX_LEN,
    )
    n_traj = len(records)
    all_locs = np.zeros((n_traj, T, 2), dtype=np.float32)
    all_feats = [np.zeros((n_traj, T, feature_spec.per_bs_dim), dtype=np.float32) for _ in range(3)]
    all_meas = [np.zeros((n_traj, T, 5), dtype=np.float32) for _ in range(3)] if save_measurements else None
    all_rd_maps = [np.zeros((n_traj, T) + feature_spec.rd_map_shape, dtype=np.float32) for _ in range(3)] if save_rf_maps else None
    all_ra_maps = [np.zeros((n_traj, T) + feature_spec.ra_map_shape, dtype=np.float32) for _ in range(3)] if save_rf_maps else None

    thc = np.zeros(n_traj, dtype=np.float32)
    local_thc = np.zeros(n_traj, dtype=np.float32)
    net_heading_deg = np.zeros(n_traj, dtype=np.float32)
    is_turn = np.zeros(n_traj, dtype=np.int32)
    is_synth = np.zeros(n_traj, dtype=np.int32)
    movement = np.full(n_traj, -1, dtype=np.int32)
    int_id = np.full(n_traj, -1, dtype=np.int32)

    for idx, record in enumerate(records):
        traj = ensure_traj_length(record["traj"], T)
        all_locs[idx] = traj
        thc[idx] = record["thc"]
        local_thc[idx] = record.get("local_thc", 0.0)
        net_heading_deg[idx] = record.get("net_heading_deg", 0.0)
        is_turn[idx] = record["is_turn"]
        is_synth[idx] = record["is_synth"]
        movement[idx] = record["movement"]
        int_id[idx] = record["int_id"]

        for bs_idx in range(3):
            feats, rd_maps, ra_maps, meas = gen_bs_observations(
                traj,
                BS_POS[bs_idx],
                snr_db,
                rng,
                dt=dt,
                save_rf_maps=save_rf_maps,
                feature_spec=feature_spec,
            )
            all_feats[bs_idx][idx] = feats
            if save_measurements:
                all_meas[bs_idx][idx] = meas
            if save_rf_maps:
                all_rd_maps[bs_idx][idx] = rd_maps
                all_ra_maps[bs_idx][idx] = ra_maps

    os.makedirs(out_dir, exist_ok=True)
    atomic_save_npy(os.path.join(out_dir, f"gt_locs_{file_idx}.npy"), all_locs)
    for bs_idx in range(3):
        atomic_save_npy(os.path.join(out_dir, f"bs{bs_idx + 1}_feat_{file_idx}.npy"), all_feats[bs_idx])
        if save_measurements:
            atomic_save_npy(os.path.join(out_dir, f"bs{bs_idx + 1}_meas_{file_idx}.npy"), all_meas[bs_idx])
        if save_rf_maps:
            atomic_save_npy(os.path.join(out_dir, f"bs{bs_idx + 1}_rd_map_{file_idx}.npy"), all_rd_maps[bs_idx])
            atomic_save_npy(os.path.join(out_dir, f"bs{bs_idx + 1}_ra_map_{file_idx}.npy"), all_ra_maps[bs_idx])
    atomic_save_npz(
        os.path.join(out_dir, f"meta_{file_idx}.npz"),
        thc=thc,
        local_thc=local_thc,
        net_heading_deg=net_heading_deg,
        is_turn=is_turn,
        is_synth=is_synth,
        movement=movement,
        int_id=int_id,
        bs_positions=BS_POS.astype(np.float32),
    )
    print(f"    file {file_idx}: {n_traj} trajs (SNR={snr_db}dB)")


def summarize_distribution(records, title):
    if not records:
        print(f"  {title}: empty")
        return
    n_turn = sum(record["is_turn"] for record in records)
    print(
        f"  {title}: total={len(records)}, "
        f"straight={len(records) - n_turn} ({(len(records) - n_turn) / len(records):.0%}), "
        f"turn={n_turn} ({n_turn / len(records):.0%})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dataset_lankershim_v2/")
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--traj_per_file", type=int, default=64)
    parser.add_argument("--snr_list", type=str, default="-10,-5,0,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_synth_turns", type=int, default=240)
    parser.add_argument("--target_turn_ratio", type=float, default=0.40)
    parser.add_argument("--max_total_traj", type=int, default=None)
    parser.add_argument("--disable_geom_aug", type=int, default=0)
    parser.add_argument("--save_rf_maps", type=int, default=1)
    parser.add_argument("--save_measurements", type=int, default=1)
    parser.add_argument("--feature_preset", type=str, default="legacy", choices=["legacy", "localize"])
    parser.add_argument("--range_complex_len", type=int, default=None)
    parser.add_argument("--vel_complex_len", type=int, default=None)
    parser.add_argument("--angle_complex_len", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    snr_list = [int(item) for item in args.snr_list.split(",")]
    feature_spec = resolve_feature_spec(args)

    print("=" * 60)
    print("  Lankershim v2: vector-feature sensing observations")
    print(f"  T={args.T}, dt={args.dt:.2f}s, SNR={snr_list}")
    print(
        f"  feature preset={args.feature_preset}, "
        f"range/vel/angle complex lens="
        f"{feature_spec.range_complex_len}/{feature_spec.vel_complex_len}/{feature_spec.angle_complex_len}"
    )
    print("=" * 60)

    records = load_and_preprocess(args.csv_path, T=args.T, dt_target=args.dt)
    summarize_distribution(records, "raw")

    if bool(args.disable_geom_aug):
        print("  geometric augmentation: disabled")
    else:
        records = augment_with_rotations(records)
    if args.n_synth_turns > 0:
        records.extend(synthesize_turns(args.n_synth_turns, T=args.T, dt=args.dt, seed=args.seed))
    else:
        print("  synthesized turn trajectories: disabled")
    records = balance_turn_ratio(records, args.target_turn_ratio, rng)
    records = cap_total_records(records, args.max_total_traj, rng)
    summarize_distribution(records, "balanced")

    rng.shuffle(records)
    n_total = len(records)
    n_per_file = args.traj_per_file
    n_files = int(np.ceil(n_total / n_per_file))
    print(f"  final pack: {n_total} trajectories -> {n_files} files x {n_per_file}")

    for snr in snr_list:
        out_dir = os.path.join(args.output_dir, f"{snr}dB")
        print(f"\n=== SNR={snr}dB ===")
        os.makedirs(out_dir, exist_ok=True)
        save_feature_config(out_dir, feature_spec, dt=args.dt)
        for file_idx in range(n_files):
            start = file_idx * n_per_file
            end = min(start + n_per_file, n_total)
            batch = list(records[start:end])
            while len(batch) < n_per_file:
                batch.append(records[rng.randint(n_total)])
            gen_one_file(
                file_idx,
                batch,
                args.T,
                snr,
                out_dir,
                rng,
                dt=args.dt,
                save_rf_maps=bool(args.save_rf_maps),
                feature_spec=feature_spec,
                save_measurements=bool(args.save_measurements),
            )

    print(f"\nDone. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
