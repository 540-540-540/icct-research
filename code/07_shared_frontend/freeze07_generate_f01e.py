"""SENS-FREEZE-07 F01-E generator: frozen B64 production chain over the A01 episodes.

Stage E1 (``generate``): one process per SNR writes data/f01e/sequences/snr_{s}/episode_{i:03d}.npz
plus per-episode diagnostics, streaming frame by frame (no RD maps or raw Y cached).
Stage E2 (``pack``): reads the sequences and writes inputs/{split}_{snr}.npz, labels/{split}.npz
and metadata/{split}.json under the frozen F01-D container schema.

Slot semantics follow the frozen shared frontend: anonymous sticky tracker slots 0..7. Future
labels are built by offline C-domain slot<->source-vehicle alignment over the 20 history frames
(cost = mean position distance, >=3 common frames, Hungarian, 5 m gate); GT never enters sensing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

CONFIG_PATH = ROOT / "configs/shared_frontend.json"
SEQUENCE_FRAMES = 199
DT_NS = 100_000_000
KEYS = ("state_hat", "track_exists", "detected", "timestamp")


def snr_name(snr: float) -> str:
    return "snr_" + str(int(snr)).replace("-", "m")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_npz(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(tmp, path)


def sequence_path(out: Path, snr: float, index: int) -> Path:
    return out / "sequences" / snr_name(snr) / f"episode_{index:03d}.npz"


def load_sequence(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(KEYS):
            raise ValueError(f"Unexpected sequence fields: {path}")
        result = {key: archive[key] for key in KEYS}
    if result["state_hat"].shape != (SEQUENCE_FRAMES, 8, 4) or result["state_hat"].dtype != np.float32:
        raise ValueError(f"Invalid state array: {path}")
    for key in ("track_exists", "detected"):
        if result[key].shape != (SEQUENCE_FRAMES, 8) or result[key].dtype != np.bool_:
            raise ValueError(f"Invalid {key}: {path}")
    if result["timestamp"].shape != (SEQUENCE_FRAMES,) or result["timestamp"].dtype != np.float64:
        raise ValueError(f"Invalid timestamp: {path}")
    if np.any(result["detected"] & ~result["track_exists"]):
        raise ValueError(f"Detection outside existence: {path}")
    return result


def generate(args) -> None:
    from frontend.echo_source import SourceEpisodes
    from frontend.fusion.association import fuse_frame
    from frontend.sensing import detector
    from frontend.tracking.cv_kf import CvKalmanTracker

    out = Path(args.out)
    config = _common.load_frontend_config()
    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    geometry = _common.load_geometry_config()
    waveform, array = _common.build_objects(config)
    stations, boresights = geometry["stations"], geometry["boresights"]
    height = float(geometry["height_difference_m"])
    multiplier = float(config["detector"]["cfar_threshold_multiplier"])
    lut = _common.load_production_lut(config)
    if lut is None:
        raise SystemExit("production covariance LUT missing")
    resource = _common.load_resource(config)
    source = SourceEpisodes()
    device = "cuda:0"
    requested = set(args.split_list) if args.split_list else None
    episode_indices = [index for index, episode in enumerate(source.episodes)
                       if requested is None or episode["split"] in requested]
    if args.episode_limit:
        episode_indices = episode_indices[:args.episode_limit]

    generated = 0
    reused = 0
    seconds = 0.0
    frames = 0
    for index in episode_indices:
        episode = source.episodes[index]
        for snr in args.snr_list:
            path = sequence_path(out, snr, index)
            diagnostic_path = out / "diagnostics" / snr_name(snr) / f"episode_{index:03d}.json"
            if path.exists() and diagnostic_path.exists() and not args.force:
                report = json.loads(diagnostic_path.read_text())
                cached = load_sequence(path)
                deadlines = int(episode["start_ms"]) + np.arange(1, SEQUENCE_FRAMES + 1) * 100
                if report.get("config_sha256") == config_hash and report.get("complete") is True \
                        and not report.get("failures") and np.array_equal(
                            np.rint(cached["timestamp"] * 1000).astype(np.int64), deadlines):
                    reused += 1
                    continue
                raise ValueError(f"Stale F01-E sequence: {path}")
            deadlines = int(episode["start_ms"]) + np.arange(1, SEQUENCE_FRAMES + 1, dtype=np.int64) * 100
            result = {"state_hat": np.zeros((SEQUENCE_FRAMES, 8, 4), np.float32),
                      "track_exists": np.zeros((SEQUENCE_FRAMES, 8), bool),
                      "detected": np.zeros((SEQUENCE_FRAMES, 8), bool),
                      "timestamp": deadlines.astype(np.float64) / 1000.0}
            report = {"episode_index": index, "episode_id": episode["episode_id"],
                      "split": episode["split"], "snr_db": snr, "config_sha256": config_hash,
                      "detector_sha256": hashlib.sha256(
                          (ROOT / "frontend/sensing/detector.py").read_bytes()).hexdigest(),
                      "waveform_sha256": hashlib.sha256(
                          (ROOT / "frontend/sensing/waveform.py").read_bytes()).hexdigest(),
                      "calibration": {"cfar_multiplier": multiplier,
                                      "covariance_inflated": bool(lut.get("covariance_inflation", 1.0) > 1.0),
                                      "tracker_q_a": float(config["tracker"]["q_a_m2_s3"])},
                      "frames": SEQUENCE_FRAMES, "complete": False, "targets_written": 0,
                      "confirmed_frames": 0, "detected_frames": 0, "max_confirmed": 0,
                      "failures": []}
            tracker = CvKalmanTracker(config)
            started = time.perf_counter()
            for frame, deadline in enumerate(deadlines):
                try:
                    states, slots, used = source.at_time(episode, int(deadline) * 1_000_000)
                    keys = [int(episode["source_keys"][slot]) for slot in slots]
                    if len(states):
                        positions = states[:, :2].tolist()
                        velocities = states[:, 2:].tolist()
                    else:
                        positions, velocities = [], []
                    echo = _common.synthesize(positions, velocities, keys, stations, boresights, waveform,
                                              array, config, snr, index, frame, device, noise=True,
                                              height_m=height)
                    detections_by_bs = {}
                    for bs in range(3):
                        maps = detector.compute_maps(echo["Y"][bs], echo["X"][bs], waveform, array,
                                                     config["detector"], resource=resource)
                        detections, _, _ = detector.detect_from_maps(
                            maps, bs, int(deadline) * 1_000_000, stations[bs], float(boresights[bs]),
                            config, array, multiplier=multiplier, covariance_lut=lut, height=height)
                        detections_by_bs[bs] = detections
                    observations, _ = fuse_frame(detections_by_bs, config)
                    records = tracker.step(observations, int(deadline) * 1_000_000)
                    confirmed = 0
                    for record in records:
                        slot = int(record["slot"])
                        if not record["track_exists"] or not 0 <= slot < 8:
                            continue
                        confirmed += 1
                        result["state_hat"][frame, slot] = np.asarray(record["state_hat"], np.float32)
                        result["track_exists"][frame, slot] = True
                        result["detected"][frame, slot] = bool(record["detected"])
                        report["targets_written"] += 1
                    report["confirmed_frames"] += confirmed
                    report["detected_frames"] += int(result["detected"][frame].sum())
                    report["max_confirmed"] = max(report["max_confirmed"], confirmed)
                except Exception as error:  # noqa: BLE001
                    report["failures"].append({"frame_index": frame, "deadline_ms": int(deadline),
                                               "error": repr(error)})
                    write_json(diagnostic_path, report)
                    raise RuntimeError(f"F01-E episode {index}, SNR {snr}, frame {frame} failed") from error
            seconds += time.perf_counter() - started
            frames += SEQUENCE_FRAMES
            report["complete"] = True
            write_npz(path, result)
            write_json(diagnostic_path, report)
            generated += 1
            print(json.dumps({"episode": index, "split": episode["split"], "snr": snr,
                              "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    if args.dry_run:
        summary = {"mode": "dry_run", "episodes": len(episode_indices), "snr_list": args.snr_list,
                   "frames_generated": frames, "seconds": seconds,
                   "seconds_per_frame": seconds / max(frames, 1), "generated": generated,
                   "reused": reused, "out": str(out)}
        dry_path = Path(args.dry_out) if args.dry_out else ROOT / "reports/f01e/freeze_07/dry_run_raw.json"
        write_json(dry_path, summary)
        print(json.dumps(summary, indent=1))


def history_indices(sequence: dict, origin_ms: int) -> np.ndarray:
    times = np.rint(sequence["timestamp"] * 1000).astype(np.int64)
    wanted = int(origin_ms) + np.arange(-19, 1, dtype=np.int64) * 100
    indices = np.searchsorted(times, wanted)
    if np.any(indices >= len(times)) or not np.array_equal(times[indices], wanted):
        raise ValueError(f"Origin {origin_ms} lacks a complete 20-frame timestamp grid")
    return indices


def gt_track_xy(track, times_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(track["time_ms"], times_ms)
    in_range = positions < len(track)
    safe = np.minimum(positions, max(0, len(track) - 1))
    exact = in_range & (track["time_ms"][safe] == times_ms)
    xy = np.stack([track["x"][safe], track["y"][safe]], axis=-1).astype(np.float64)
    xy[~exact] = np.nan
    return xy, exact


def origin_eligibility(sequence: dict, origin_ms: int) -> tuple:
    """Origin-safe eligibility: the origin frame must be alive and the contiguous alive run ending
    at the origin must cover at least 3 history frames (never a sum across separate track lives)."""
    indices = history_indices(sequence, origin_ms)
    exists = sequence["track_exists"][indices]
    segments = {}
    for slot in range(8):
        if not exists[19, slot]:
            continue
        length = 0
        for frame in range(19, -1, -1):
            if exists[frame, slot]:
                length += 1
            else:
                break
        if length >= 3:
            segments[slot] = {"segment_start_history_index": int(20 - length),
                              "contiguous_alive_length": int(length)}
    return indices, exists, segments


def align_window(sequence: dict, episode: dict, origin_ms: int, source, gate_m: float = 5.0) -> list:
    """Origin-safe C-domain slot <-> source-vehicle alignment (SENS-FREEZE-09).

    Only origin-eligible slots enter the Hungarian assignment, and each slot's cost uses only its
    current contiguous alive segment (frames from ``segment_start_history_index`` to the origin).
    States from an earlier life of the same slot never participate; identical cost/gate rules as
    the frozen matcher (mean Euclidean xy distance, >=3 common frames, 5 m gate).
    """
    from scipy.optimize import linear_sum_assignment

    indices, exists, segments = origin_eligibility(sequence, origin_ms)
    if not segments:
        return []
    times = (int(origin_ms) + np.arange(-19, 1, dtype=np.int64) * 100).astype(np.int64)
    states = sequence["state_hat"][indices]
    vehicles = [int(key) for key in episode["source_keys"]]
    gt_xy = np.full((len(vehicles), 20, 2), np.nan)
    for vehicle_index, key in enumerate(vehicles):
        gt_xy[vehicle_index], _ = gt_track_xy(source.tracks[key], times)
    slots = sorted(segments)
    cost = np.full((len(slots), len(vehicles)), 1e9)
    common = np.zeros((len(slots), len(vehicles)), np.int64)
    for row, slot in enumerate(slots):
        start = segments[slot]["segment_start_history_index"]
        for vehicle_index in range(len(vehicles)):
            mask = exists[start:, slot] & np.isfinite(gt_xy[vehicle_index, start:, 0])
            common[row, vehicle_index] = int(mask.sum())
            if common[row, vehicle_index] >= 3:
                distance = np.linalg.norm(states[start:, slot, :2][mask] - gt_xy[vehicle_index, start:][mask],
                                          axis=1)
                cost[row, vehicle_index] = float(distance.mean())
    small = np.where(cost <= gate_m, cost, 1e9)
    rows, columns = linear_sum_assignment(small) if small.size else ([], [])
    alignment = []
    for row, vehicle_index in zip(np.asarray(rows).tolist(), np.asarray(columns).tolist()):
        if small[row, vehicle_index] < 1e9:
            slot = slots[row]
            alignment.append({"slot": int(slot), "source_key": int(vehicles[vehicle_index]),
                              "source_slot": int(vehicle_index),
                              "common_frames": int(common[row, vehicle_index]),
                              "cost_m": float(cost[row, vehicle_index]),
                              "segment_start_history_index":
                                  segments[slot]["segment_start_history_index"],
                              "contiguous_alive_length": segments[slot]["contiguous_alive_length"]})
    return alignment


def aligned_labels(source, index: int, origins_ms: list, alignment_per_window: list) -> dict:
    """Raw future xy sidecar per window under a slot<->vehicle alignment.

    The slot dimension is the output sticky slot; a slot's label row is filled only where that
    slot is aligned to a source vehicle in that window and the raw source sample exists.
    """
    episode = source.episodes[index]
    origins = np.asarray(origins_ms, np.int64)
    future = np.zeros((len(origins), 20, 8, 2), np.float32)
    valid = np.zeros((len(origins), 20, 8), bool)
    for window, entries in enumerate(alignment_per_window):
        for entry in entries:
            slot = int(entry["slot"])
            key = int(entry["source_key"])
            times = origins[window] + np.arange(1, 21, dtype=np.int64) * 100
            xy, exact = gt_track_xy(source.tracks[key], times)
            usable = exact & np.isfinite(xy[:, 0])
            future[window, usable, slot] = xy[usable].astype(np.float32)
            valid[window, :, slot] = usable
    return {"future_position": future, "label_valid": valid}


def alignment_map(entries: list) -> dict:
    return {int(entry["slot"]): int(entry["source_key"]) for entry in entries}


def pack(args) -> None:
    from frontend.echo_source import SourceEpisodes

    out = Path(args.out)
    config = _common.load_frontend_config()
    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    allowed_provenance = set(getattr(args, "allowed_provenance_hashes", None) or []) | {config_hash}
    source = SourceEpisodes()
    reference_snr = max(args.snr_list)
    totals = {}
    for split in dict.fromkeys(episode["split"] for episode in source.episodes):
        if args.split_list and split not in args.split_list:
            continue
        episode_indices = [index for index, episode in enumerate(source.episodes)
                           if episode["split"] == split]
        if args.episode_limit:
            episode_indices = episode_indices[:args.episode_limit]
        metadata = []
        label_parts = {snr: [] for snr in args.snr_list}
        packed = {snr: {key: [] for key in KEYS} for snr in args.snr_list}
        empty = {snr: 0 for snr in args.snr_list}
        alignment_mismatch = {snr: 0 for snr in args.snr_list}
        alignment_windows = 0
        for index in episode_indices:
            episode = source.episodes[index]
            origins = episode["prediction_grid_ms"]
            sequences = {}
            for snr in args.snr_list:
                path = sequence_path(out, snr, index)
                diagnostic_path = out / "diagnostics" / snr_name(snr) / f"episode_{index:03d}.json"
                if not (path.exists() and diagnostic_path.exists()):
                    raise ValueError(f"Missing F01-E sequence: {path}")
                report = json.loads(diagnostic_path.read_text())
                if report.get("config_sha256") not in allowed_provenance or report.get("complete") is not True \
                        or report.get("failures"):
                    raise ValueError(f"Invalid F01-E sequence provenance: {path}")
                sequences[snr] = load_sequence(path)
            alignments = {snr: [align_window(sequences[snr], episode, int(origin), source)
                                for origin in origins] for snr in args.snr_list}
            reference = alignments[reference_snr]
            for window in range(len(origins)):
                alignment_windows += 1
                reference_map = alignment_map(reference[window])
                for snr in args.snr_list:
                    if alignment_map(alignments[snr][window]) != reference_map:
                        alignment_mismatch[snr] += 1
            for snr in args.snr_list:
                label_parts[snr].append(aligned_labels(source, index, origins, alignments[snr]))
            base_sample = len(metadata)
            for window, origin in enumerate(origins):
                for snr in args.snr_list:
                    sequence = sequences[snr]
                    indices = history_indices(sequence, int(origin))
                    state = sequence["state_hat"][indices].copy()
                    exists = sequence["track_exists"][indices].copy()
                    detected = sequence["detected"][indices].copy()
                    detected &= exists
                    state[~exists] = 0
                    empty[snr] += int(not exists.any())
                    packed[snr]["state_hat"].append(state)
                    packed[snr]["track_exists"].append(exists)
                    packed[snr]["detected"].append(detected)
                    packed[snr]["timestamp"].append(sequence["timestamp"][indices].copy())
                metadata.append(dict(
                    sample_index=base_sample + window, episode_index=index, origin_ms=int(origin),
                    source_keys=episode["source_keys"], episode_id=episode["episode_id"],
                    source_block=episode["source_block"], source_groups=episode["source_groups"],
                    label_alignment_snr=reference_snr, label_rule="origin-safe contiguous segment",
                    slot_alignment=reference[window],
                    slot_alignment_by_snr={str(snr): alignment_map(alignments[snr][window])
                                           for snr in args.snr_list},
                    slot_alignment_detail_by_snr={str(snr): alignments[snr][window]
                                                  for snr in args.snr_list}))
        for snr in args.snr_list:
            labels = {key: np.concatenate([part[key] for part in label_parts[snr]])
                      for key in ("future_position", "label_valid")}
            write_npz(out / "labels" / f"{split}_{snr_name(snr)}.npz", labels)
            if snr == reference_snr:
                write_npz(out / "labels" / f"{split}.npz", labels)
        for snr in args.snr_list:
            write_npz(out / "inputs" / f"{split}_{snr_name(snr)}.npz",
                      {key: np.stack(packed[snr][key]) for key in KEYS})
        write_json(out / "metadata" / f"{split}.json",
                   dict(split=split, config_sha256=config_hash,
                        revision=json.loads(CONFIG_PATH.read_text()).get("revision"),
                        source="Frozen A01 prediction_grid_ms; B64 production chain",
                        schema="F01-D container schema; anonymous sticky slots with origin-safe offline alignment",
                        label_files={snr_name(snr): f"labels/{split}_{snr_name(snr)}.npz"
                                     for snr in args.snr_list},
                        label_reference_file=f"labels/{split}.npz (aligned to {reference_snr} dB)",
                        label_alignment_snr=reference_snr,
                        label_rule="labels only for origin-eligible slots (origin alive and contiguous "
                                   "alive run >= 3) matched by the origin-segment matcher",
                        contract_note="F01-D used identity-fixed source slots; B64 sticky slots drift per "
                                      "SNR, so each SNR has its own aligned label file plus the reference file",
                        alignment_mismatch_windows={snr: alignment_mismatch[snr] for snr in args.snr_list},
                        samples=metadata))
        totals[split] = dict(samples=len(metadata), episodes=len(episode_indices), all_input_unavailable=empty,
                             alignment_windows=alignment_windows,
                             alignment_mismatch_windows={snr: alignment_mismatch[snr]
                                                         for snr in args.snr_list})
        print(json.dumps({split: totals[split]}, ensure_ascii=False), flush=True)
    print(json.dumps({"totals": totals}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "pack"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--split-list", nargs="*", default=None)
    parser.add_argument("--out", default=str(ROOT / "data/f01e"))
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-out", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.action == "generate":
        generate(args)
    else:
        pack(args)


if __name__ == "__main__":
    main()