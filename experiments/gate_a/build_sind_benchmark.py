from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate_a.build_benchmark import neighbor_order, sha256, write_json
from experiments.gate_a.data import ego_transform, pair_metrics
from frontend.controlled_isac.sind_frontend import sense_vehicle
from frontend.controlled_isac.automatum_measurement import calibration_from_config, setup_from_config


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split_metadata() -> tuple[dict[str, set[tuple[int, int]]], dict[str, dict[int, tuple[int, int]]]]:
    vehicles, bounds = {}, {}
    for split in ("train", "val", "test"):
        path = ROOT / "data/sind/splits" / split / "trajectories.csv"
        meta = pd.read_csv(path, usecols=["scene_id", "vehicle_id", "frame_id"])
        vehicles[split] = set(zip(meta.scene_id.astype(int), meta.vehicle_id.astype(int)))
        bounds[split] = {int(scene): (int(group.frame_id.min()), int(group.frame_id.max()))
                         for scene, group in meta.groupby("scene_id")}
    return vehicles, bounds


def sensed_lookup(split: str) -> dict[tuple[int, int, int], np.ndarray]:
    path = ROOT / "data/sind/target_views_v1" / split / "state_pool.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        rows = z["sensing_gt_rows"]
        keys = zip(z["scene_id"][rows], z["frame"][rows], z["vehicle_id"][rows])
        return {tuple(map(int, key)): value for key, value in zip(keys, z["state_hat"])}


def sense_rows(frame: pd.DataFrame, split: str, cfg: dict) -> tuple[np.ndarray, dict]:
    frozen = sensed_lookup(split)
    sensing_cfg_path = ROOT / "configs/sind_controlled_isac.json"
    sensing_cfg = json.loads(sensing_cfg_path.read_text())
    setup, calibration = setup_from_config(sensing_cfg), calibration_from_config(sensing_cfg)
    out = np.empty((len(frame), 4), np.float32)
    reused = generated = 0
    bs = Counter()
    for i, row in enumerate(frame.itertuples(index=False)):
        key = (int(row.scene_id), int(row.frame_id), int(row.vehicle_id))
        cached = frozen.get(key)
        if cached is not None:
            out[i] = cached
            reused += 1
        else:
            estimate = sense_vehicle(key[0], key[1], key[2], [row.x, row.y], [row.vx, row.vy],
                                     cfg["snr_db"], calibration, setup)
            out[i] = [estimate[k] for k in ("x_hat", "y_hat", "vx_hat", "vy_hat")]
            bs[int(estimate["n_bs"])] += 1
            generated += 1
        if (i + 1) % 100000 == 0:
            print(f"{split}: sensed/reused {i + 1}/{len(frame)} rows", flush=True)
    if not np.isfinite(out).all():
        raise RuntimeError(f"{split}: nonfinite sensed history")
    return out, {"reused": reused, "generated": generated,
                 "generated_visible_bs": {str(k): int(v) for k, v in bs.items()},
                 "config_sha256": config_hash(sensing_cfg_path)}


def tracks_from_frame(frame: pd.DataFrame) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    tracks = {}
    for (scene, vehicle), group in frame.groupby(["scene_id", "vehicle_id"], sort=False):
        group = group.sort_values("frame_id")
        tracks[(int(scene), int(vehicle))] = {
            "frame": group.frame_id.to_numpy(np.int32),
            "timestamp": group.timestamp.to_numpy(np.float64),
            "state": group[["x", "y", "vx", "vy"]].to_numpy(np.float32),
            "sensed": group[["x_hat", "y_hat", "vx_hat", "vy_hat"]].to_numpy(np.float32),
        }
    return tracks


def history_slice(track: dict, t0: int, history: int) -> np.ndarray | None:
    frames = track["frame"]
    end = int(np.searchsorted(frames, t0))
    if end >= len(frames) or frames[end] != t0:
        return None
    start = end - history + 1
    if start < 0 or not np.array_equal(frames[start:end + 1], np.arange(t0 - history + 1, t0 + 1)):
        return None
    return np.arange(start, end + 1)


def build_records(frame: pd.DataFrame, cfg: dict, split: str) -> tuple[list[dict], dict]:
    tracks = tracks_from_frame(frame)
    by_scene = {}
    for key in tracks:
        by_scene.setdefault(key[0], []).append(key)
    records, origin_count = [], 0
    dt = float(cfg["dt_s"])
    for scene, keys in sorted(by_scene.items()):
        lo = int(frame.loc[frame.scene_id == scene, "frame_id"].min())
        hi = int(frame.loc[frame.scene_id == scene, "frame_id"].max())
        first_t0 = lo + cfg["history_steps"] - 1
        first_t0 += (-first_t0 + cfg["history_steps"] - 1) % cfg["origin_stride_steps"]
        for t0 in range(first_t0, hi - cfg["future_steps"] + 1, cfg["origin_stride_steps"]):
            candidates = []
            for key in keys:
                rows = history_slice(tracks[key], t0, cfg["history_steps"])
                if rows is not None:
                    candidates.append((key, rows))
            if not candidates:
                continue
            origin_count += 1
            end_state = np.stack([tracks[key]["sensed"][rows[-1]] for key, rows in candidates])
            metrics = pair_metrics(end_state, cfg)
            ids = np.asarray([key[1] for key, _ in candidates], np.int32)
            for target, (key, target_rows) in enumerate(candidates):
                ordered = neighbor_order(target, metrics, ids)
                nodes = [target] + ordered
                history = np.stack([tracks[candidates[j][0]]["sensed"][candidates[j][1]] for j in nodes], axis=1)
                future = np.zeros((cfg["future_steps"], 2), np.float32)
                future_mask = np.zeros(cfg["future_steps"], bool)
                target_track = tracks[key]
                pos = int(target_rows[-1])
                for step in range(1, cfg["future_steps"] + 1):
                    q = pos + step
                    if q < len(target_track["frame"]) and target_track["frame"][q] == t0 + step:
                        future[step - 1] = target_track["state"][q, :2]
                        future_mask[step - 1] = True
                history, future, _, _ = ego_transform(history, future, future_mask)
                near = metrics["near"][target]
                valid_cpa = near & (metrics["tcpa"][target] > 0) & (metrics["tcpa"][target] <= cfg["tcpa_s"])
                records.append({
                    "history": history.astype(np.float32), "future": future,
                    "future_mask": future_mask, "scene_id": scene, "start_frame": t0,
                    "origin_id": scene * 100000 + t0, "target_vehicle_id": int(key[1]),
                    "source_group": scene * 1000000 + int(key[1]),
                    "n_context": int(near.sum()), "k": int(metrics["edge"][target].sum()),
                    "has_cpa": bool(metrics["cpa"][target].any()),
                    "has_following": bool(metrics["following"][target].any()),
                    "max_closing": float(metrics["closing"][target, near].max()) if near.any() else 0.0,
                    "min_dcpa": float(metrics["dcpa"][target, valid_cpa].min()) if valid_cpa.any() else 999.0,
                    "dt_s": dt,
                })
        print(f"{split}: scene {scene}, origins={origin_count}, targets={len(records)}", flush=True)
    return records, {"origins": origin_count, "tracks": len(tracks)}


def pack(path: Path, records: list[dict], max_nodes: int):
    n = len(records)
    history = np.zeros((n, 20, max_nodes, 4), np.float32)
    node_mask = np.zeros((n, max_nodes), bool)
    future = np.zeros((n, 40, 2), np.float32)
    future_mask = np.zeros((n, 40), bool)
    scalar_names = ("scene_id", "start_frame", "origin_id", "target_vehicle_id", "source_group",
                    "n_context", "k", "has_cpa", "has_following", "max_closing", "min_dcpa")
    scalars = {name: [] for name in scalar_names}
    for i, record in enumerate(records):
        nodes = record["history"].shape[1]
        history[i, :, :nodes] = record["history"]
        node_mask[i, :nodes] = True
        future[i], future_mask[i] = record["future"], record["future_mask"]
        for name in scalar_names:
            scalars[name].append(record[name])
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, history_state=history, node_mask=node_mask, future_xy=future,
                        future_mask=future_mask, **{k: np.asarray(v) for k, v in scalars.items()})
    os.replace(tmp, path)


def stats(records: list[dict]) -> dict:
    contexts = np.asarray([r["n_context"] for r in records])
    return {
        "samples": len(records), "origins": len({r["origin_id"] for r in records}),
        "targets": len({(r["scene_id"], r["start_frame"], r["target_vehicle_id"]) for r in records}),
        "ic": int(sum(r["k"] >= 1 for r in records)),
        "k_ge_2": int(sum(r["k"] >= 2 for r in records)),
        "future2_endpoint": int(sum(r["future_mask"][19] for r in records)),
        "future4_endpoint": int(sum(r["future_mask"][39] for r in records)),
        "future4_complete": int(sum(r["future_mask"].all() for r in records)),
        "n_gt_8": int(np.sum(contexts + 1 > 8)),
        "neighbor_quantiles": {str(q): float(np.quantile(contexts, q)) for q in (0, .1, .5, .9, 1)},
        "by_city": {str(scene): int(sum(r["scene_id"] == scene for r in records)) for scene in (0, 1)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_a_sind_ic4.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = json.loads(config_path.read_text())
    out = ROOT / (args.output or cfg["cache"])
    out.mkdir(parents=True, exist_ok=True)
    vehicles, old_bounds = load_split_metadata()
    overlap = {f"{a}_{b}": len(vehicles[a] & vehicles[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise RuntimeError(f"physical vehicle split overlap: {overlap}")
    all_records, sensing, effective_bounds = {}, {}, {}
    for split in ("train", "val"):
        frame = pd.read_csv(ROOT / "data/sind/splits" / split / "trajectories.csv")
        if split == "val":
            keep = np.zeros(len(frame), bool)
            for scene, group in frame.groupby("scene_id"):
                keep[group.index] = group.frame_id >= int(group.frame_id.min()) + 20
            frame = frame[keep].copy()
        effective_bounds[split] = {str(scene): [int(group.frame_id.min()), int(group.frame_id.max())]
                                   for scene, group in frame.groupby("scene_id")}
        sensed, sensing[split] = sense_rows(frame, split, cfg)
        frame[["x_hat", "y_hat", "vx_hat", "vy_hat"]] = sensed
        all_records[split], _ = build_records(frame, cfg, split)
    max_nodes = max(r["history"].shape[1] for records in all_records.values() for r in records)
    for split, records in all_records.items():
        pack(out / f"{split}.npz", records, max_nodes)
    groups = {split: {r["source_group"] for r in records} for split, records in all_records.items()}
    if groups["train"] & groups["val"]:
        raise RuntimeError("train/validation source-group overlap")
    raw_files = {city: ROOT / "data/sind/raw" / city / "Veh_smoothed_tracks.csv"
                 for city in ("changchun", "xian")}
    raw_hashes = {city: sha256(path) for city, path in raw_files.items()}
    if raw_hashes != cfg["source_sha256"]:
        raise RuntimeError(f"canonical source hash mismatch: {raw_hashes}")
    manifest = {
        "status": "COMPLETE", "revision": cfg["revision"], "dataset": cfg["dataset"],
        "history_source": "official SinD canonical Veh_smoothed_tracks.csv states through frozen 3-BS sensing",
        "history_causality_scope": "official smoothed canonical states; no claim of raw-video causal reconstruction",
        "future_dependency": "future canonical x/y only populate target future_xy and future_mask",
        "membership_dependency": "20-frame history only; future availability never selects target or neighbor",
        "test_materialized": False,
        "test_access": "vehicle/frame metadata only for physical-overlap audit; no test coordinates or labels loaded",
        "raw_sha256": raw_hashes, "sensing": sensing, "snr_db": cfg["snr_db"],
        "history_steps": cfg["history_steps"], "future_steps": cfg["future_steps"],
        "guard_frames": cfg["guard_frames"], "old_split_bounds": old_bounds,
        "effective_bounds": effective_bounds, "physical_vehicle_overlap": overlap,
        "max_nodes": max_nodes, "splits": {split: stats(records) for split, records in all_records.items()},
        "thresholds": {k: cfg[k] for k in ("radius_m", "closing_mps", "tcpa_s", "dcpa_m",
                                                     "following_heading_deg", "following_headway_s",
                                                     "following_lateral_m", "minimum_speed_mps")},
        "files": {f"{split}.npz": {"bytes": (out / f"{split}.npz").stat().st_size,
                                        "sha256": sha256(out / f"{split}.npz")}
                  for split in ("train", "val")},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
