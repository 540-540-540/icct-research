#!/usr/bin/env python3
"""Fixed-origin, per-target SinD views. Train/val only; old data stay untouched.

Every complete-future target gets its own history-selected neighborhood. Neighbors
need only a complete history. Source GT-history selection and source smoothing are
inherited audit conditions, not a claim of online causal availability.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frontend.controlled_isac.sind_frontend import sense_vehicle
from frontend.controlled_isac.automatum_measurement import (
    setup_from_config, calibration_from_config,
)

HIST, FUT, MAXN = 20, 20, 8
REVISION = "SIND-TARGET-VIEWS-V1-FIXED-ORIGIN"


def target_neighbors(ids, states, target):
    """Original focal-neighbor ordering, applied independently to this target."""
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids) or not 0 <= target < len(ids):
        raise ValueError("unique candidate IDs and a valid target are required")
    states = np.asarray(states, dtype=float)
    dp, dv = states[:, :2] - states[target, :2], states[:, 2:] - states[target, 2:]
    distance = np.linalg.norm(dp, axis=1)
    dot, vv = (dp * dv).sum(axis=1), (dv * dv).sum(axis=1)
    closing = np.maximum(0.0, -dot / np.maximum(distance, 1e-9))
    tcpa = np.where(vv > 1e-9, -dot / np.maximum(vv, 1e-9), np.inf)
    valid = (tcpa > 0) & (tcpa <= 4)
    dcpa = np.where(valid, np.linalg.norm(dp + dv * np.where(valid, tcpa, 0)[:, None], axis=1), np.inf)
    active = (distance <= 30) & ((closing > .5) | (valid & (dcpa <= 10)))
    other = [j for j in range(len(ids)) if j != target]
    other.sort(key=lambda j: (0 if active[j] else 1,
                             float(dcpa[j]) if np.isfinite(dcpa[j]) else 1e9,
                             -float(closing[j]), float(distance[j]), int(ids[j])))
    return np.asarray([target] + other[:MAXN - 1], dtype=np.int32)


def eligible_windows(first, last, start):
    history = (first <= start) & (last >= start + HIST - 1)
    target = history & (last >= start + HIST + FUT - 1)
    return np.flatnonzero(history), np.flatnonzero(target)


def source_identity(paths):
    return {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size,
                                     "mtime_ns": p.stat().st_mtime_ns} for p in paths}


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def verify_sensing_parity(keys, source_states, lookup, old_states, setup, calibration):
    """Refuse mixing old/new cache states if the current sensor protocol drifted."""
    overlap = [i for i, key in enumerate(keys) if key in lookup]
    if not overlap:
        raise RuntimeError("no overlapping frozen sensing states available for protocol parity")
    chosen = [overlap[i] for i in np.unique(np.linspace(0, len(overlap) - 1, min(128, len(overlap)), dtype=int))]
    maximum = 0.0
    for i in chosen:
        scene, frame, vid = keys[i]
        estimate = sense_vehicle(scene, frame, vid, source_states[i, :2], source_states[i, 2:], 0.0, calibration, setup)
        fresh = np.asarray([estimate[k] for k in ("x_hat", "y_hat", "vx_hat", "vy_hat")], np.float32)
        cached = old_states[lookup[keys[i]]]
        maximum = max(maximum, float(np.max(np.abs(fresh - cached))))
        if not np.array_equal(fresh, cached):
            raise RuntimeError(f"current sensing differs from frozen cache at {keys[i]}: max_abs={maximum}; refusing a mixed cache")
    return {"checked": len(chosen), "max_abs_difference": maximum, "bitwise_equal": True}


def build(split, out_root):
    if split not in ("train", "val"):
        raise ValueError("Only train/val are permitted; test remains closed")
    base = ROOT / "data/sind/splits" / split
    samples, trajectories = base / "samples.npz", base / "trajectories.csv"
    old_cache = ROOT / "data/sind/isac" / split / "sensing_cache.npz"
    config_path = ROOT / "configs/sind_controlled_isac.json"
    identity = source_identity([samples, trajectories, old_cache, config_path])
    out = out_root / split
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = out / ".BUILD_COMPLETE"
        if marker.exists():
            previous = json.loads((out / "manifest.json").read_text())
            if previous.get("revision") != REVISION or previous.get("source_identity") != identity:
                raise RuntimeError(f"{out}: completed build differs; use a new output directory")
            for name, size in previous["output_bytes"].items():
                if not (out / name).is_file() or (out / name).stat().st_size != size:
                    raise RuntimeError(f"{out}: completed artifact missing or size changed: {name}")
            print(f"{split}: completed matching build exists", flush=True)
            return previous
        return _build(split, out, samples, trajectories, old_cache, config_path, identity)


def _build(split, out, samples, trajectories, old_cache, config_path, identity):
    started = time.monotonic()
    with np.load(samples, allow_pickle=False) as old:
        origin_scene = old["scene_id"].copy()
        origin_start = old["start_frame"].copy()
        timestamps = old["history_timestamp"].copy()
    origins = list(zip(origin_scene.tolist(), origin_start.tolist()))
    if len(set(origins)) != len(origins):
        raise ValueError("source origin keys are not unique")
    d = pd.read_csv(trajectories).sort_values(["scene_id", "vehicle_id", "frame_id"]).reset_index(drop=True)
    gt = d[["x", "y", "vx", "vy"]].to_numpy(np.float32)
    if not np.isfinite(gt).all() or d.duplicated(["scene_id", "vehicle_id", "frame_id"]).any():
        raise ValueError("nonfinite or duplicate trajectory states")
    tracks = {}
    for (scene, vid), group in d.groupby(["scene_id", "vehicle_id"], sort=False):
        frames = group.frame_id.to_numpy()
        if not np.all(np.diff(frames) == 1):
            raise ValueError(f"noncontiguous trajectory: {scene}, {vid}")
        tracks.setdefault(int(scene), []).append((int(vid), int(frames[0]), int(frames[-1]), int(group.index[0])))
    tracks = {scene: np.asarray(rows, dtype=np.int64) for scene, rows in tracks.items()}
    context_histories, patch_history_rows, patch_ids, future_rows = [], [], [], []
    target_origins = []
    origin_context_counts, origin_target_counts = [], []
    history_only_context_occurrences = 0
    for oi, (scene, start) in enumerate(origins):
        tr = tracks[scene]
        hi, ti = eligible_windows(tr[:, 1], tr[:, 2], start)
        if len(ti) == 0:
            raise ValueError(f"accepted origin has no complete-future targets: {(scene, start)}")
        selected = tr[hi]
        rows = selected[:, 3] + start - selected[:, 1]
        context_base = len(context_histories)
        context_histories.extend(rows[:, None] + np.arange(HIST)[None, :])
        expected_ts = d.timestamp.to_numpy()[rows[0]:rows[0] + HIST]
        if not np.allclose(expected_ts, timestamps[oi], rtol=0, atol=1e-9):
            raise ValueError(f"source timestamps differ at {(scene, start)}")
        end = gt[rows + HIST - 1]
        origin_context_counts.append(len(hi))
        origin_target_counts.append(len(ti))
        for target_track in ti:
            target = int(np.searchsorted(hi, target_track))
            pick = target_neighbors(selected[:, 0], end, target)
            ids = np.full(MAXN, -1, np.int32)
            indices = np.full(MAXN, -1, np.int32)
            ids[:len(pick)] = selected[pick, 0]
            indices[:len(pick)] = context_base + pick
            patch_ids.append(ids)
            patch_history_rows.append(indices)
            future_rows.append(rows[target] + HIST + np.arange(FUT))
            target_origins.append(oi)
            history_only_context_occurrences += int(np.sum(selected[pick, 2] < start + HIST + FUT - 1))
        if (oi + 1) % 5000 == 0:
            print(f"{split}: indexed {oi + 1}/{len(origins)} origins", flush=True)
    history_gt_rows = np.asarray(context_histories, np.int32)
    needed = np.unique(history_gt_rows)
    raw_to_sensing = np.full(len(d), -1, np.int32)
    raw_to_sensing[needed] = np.arange(len(needed), dtype=np.int32)
    history_indices = raw_to_sensing[history_gt_rows]
    needed_data = d.iloc[needed]
    keys = list(zip(needed_data.scene_id.astype(int), needed_data.frame_id.astype(int), needed_data.vehicle_id.astype(int)))
    with np.load(old_cache, allow_pickle=False) as cache:
        snr = np.flatnonzero(np.isclose(cache["snr_levels_db"], 0.0))
        if len(snr) != 1:
            raise ValueError("old sensing cache must contain exactly one 0 dB level")
        old_states = cache["state_hat"][int(snr[0])]
        lookup = {tuple(map(int, k)): i for i, k in enumerate(zip(cache["scene_id"], cache["frame"], cache["vehicle_id"]))}
    state_hat = np.empty((len(keys), 4), np.float32)
    missing = []
    for i, key in enumerate(keys):
        old_index = lookup.get(key)
        if old_index is None:
            missing.append(i)
        else:
            state_hat[i] = old_states[old_index]
    cfg = json.loads(config_path.read_text())
    setup, calibration = setup_from_config(cfg), calibration_from_config(cfg)
    # Use CSV float64 values, exactly as the frozen sensing builder does.
    source_states = needed_data[["x", "y", "vx", "vy"]].to_numpy(float)
    parity = verify_sensing_parity(keys, source_states, lookup, old_states, setup, calibration)
    print(f"{split}: frozen/current sensing parity {parity}", flush=True)
    sensing_start = time.monotonic()
    for completed, i in enumerate(missing, 1):
        scene, frame, vid = keys[i]
        estimate = sense_vehicle(scene, frame, vid, source_states[i, :2], source_states[i, 2:], 0.0, calibration, setup)
        state_hat[i] = [estimate[k] for k in ("x_hat", "y_hat", "vx_hat", "vy_hat")]
        if completed == min(100, len(missing)) or completed % 10000 == 0 or completed == len(missing):
            elapsed = time.monotonic() - sensing_start
            print(f"{split}: missing sensing {completed}/{len(missing)}, elapsed={elapsed:.2f}s, remaining_estimate={elapsed / completed * (len(missing) - completed):.1f}s", flush=True)
    if not np.isfinite(state_hat).all():
        raise ValueError("nonfinite sensed states")
    origin_index = np.asarray(target_origins, np.int32)
    vehicle_ids = np.asarray(patch_ids, np.int32)
    future_indices = np.asarray(future_rows, np.int32)
    patch_history_rows = np.asarray(patch_history_rows, np.int32)
    targets = len(origin_index)
    unique_targets = {(int(origin_scene[o]), int(origin_start[o]), int(ids[0])) for o, ids in zip(origin_index, vehicle_ids)}
    if len(unique_targets) != targets or targets != sum(origin_target_counts):
        raise ValueError("missing or repeated target-origin")
    atomic_npz(out / "views.npz", origin_scene=origin_scene, origin_start=origin_start,
               history_timestamp=timestamps, origin_index=origin_index, vehicle_ids=vehicle_ids,
               history_rows=patch_history_rows, history_indices=history_indices,
               future_indices=future_indices,
               origin_target_count=np.asarray(origin_target_counts, np.int32),
               origin_context_count=np.asarray(origin_context_counts, np.int32))
    atomic_npz(out / "state_pool.npz", state_hat=state_hat, state_gt=gt,
               sensing_gt_rows=needed.astype(np.int32),
               scene_id=d.scene_id.to_numpy(np.int32), frame=d.frame_id.to_numpy(np.int32),
               vehicle_id=d.vehicle_id.to_numpy(np.int32), timestamp=d.timestamp.to_numpy(float))
    manifest = {
        "revision": REVISION, "status": "COMPLETE", "split": split, "test_used": False,
        "source_identity": identity, "snr_db": 0.0, "history_length": HIST,
        "prediction_length": FUT, "max_vehicles": MAXN,
        "selection": "GT last-history-state ranking; inherited fixed accepted origins; source smoothing not proven online causal",
        "context_eligibility": "20 complete historical frames, independent of future availability",
        "target_eligibility": "20 historical plus 20 future frames within this existing guarded split",
        "target_slot": 0, "label_slots": [0], "origins": len(origins), "targets": targets,
        "context_windows": len(context_histories), "historical_only_neighbor_occurrences": history_only_context_occurrences,
        "target_unique": True, "all_eligible_targets_covered": True,
        "unique_history_states": len(needed), "cache_reused_states": len(keys) - len(missing),
        "cache_generated_states": len(missing), "cache_coverage": 1.0,
        "sensing_protocol_parity": parity,
        "gt_pool_rows": len(d), "metric_primary": "origin-macro over each origin's unique target predictions",
        "metric_secondary": "target-window-macro",
        "build_seconds": time.monotonic() - started,
        "output_bytes": {name: (out / name).stat().st_size for name in ("views.npz", "state_pool.npz")},
    }
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / ".BUILD_COMPLETE", {"revision": REVISION, "status": "COMPLETE", "test_used": False})
    print(json.dumps(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/sind/target_views_v1")
    args = parser.parse_args()
    for split in (("train", "val") if args.split == "all" else (args.split,)):
        build(split, args.out_root.resolve())


if __name__ == "__main__":
    main()
