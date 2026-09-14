"""F01-D: frozen symbol estimator sequences and causal predictor packing."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
import time
import numpy as np
from .echo_source import SourceEpisodes, ROOT
from .run_symbol_frontend import estimate_frame

DATA = ROOT / "data/f01d"
CONFIG = ROOT / "configs/symbol_frontend.json"
KEYS = ("state_hat", "track_exists", "detected", "timestamp")
_SOURCE = _CONFIG = _HASH = None


def snr_name(snr):
    return "snr_" + str(int(snr)).replace("-", "m")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_npz(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(tmp, path)


def init_worker():
    global _SOURCE, _CONFIG, _HASH
    _SOURCE = SourceEpisodes()
    raw = CONFIG.read_bytes()
    _CONFIG = json.loads(raw)
    _HASH = hashlib.sha256(raw).hexdigest()
    if _CONFIG["revision"] != "A03-symbol-v4-cuda-stable-solver" or _CONFIG["snr_db"] != [5, 10, 15, 20]:
        raise ValueError("Expected frozen A03 symbol configuration")


def sequence_path(index, snr, data=DATA):
    return Path(data) / "sequences" / snr_name(snr) / f"episode_{index:03d}.npz"


def load_sequence(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(KEYS):
            raise ValueError(f"Unexpected sequence fields: {path}")
        result = {k: archive[k] for k in KEYS}
    if result["state_hat"].shape != (199, 8, 4) or result["state_hat"].dtype != np.float32:
        raise ValueError(f"Invalid state array: {path}")
    for key in ("track_exists", "detected"):
        if result[key].shape != (199, 8) or result[key].dtype != np.bool_:
            raise ValueError(f"Invalid {key}: {path}")
    if result["timestamp"].shape != (199,) or result["timestamp"].dtype != np.float64:
        raise ValueError(f"Invalid timestamp: {path}")
    if not np.isfinite(result["state_hat"]).all() or not np.isfinite(result["timestamp"]).all():
        raise ValueError(f"Nonfinite cached sequence: {path}")
    if np.any(result["detected"] & ~result["track_exists"]):
        raise ValueError(f"Detection outside existence: {path}")
    return result


def generate_episode(index):
    source, config = _SOURCE, _CONFIG
    ep = source.episodes[index]
    deadlines = int(ep["start_ms"]) + np.arange(1, 200, dtype=np.int64) * 100
    if len(ep["source_keys"]) > 8:
        raise ValueError("Episode has more than eight frozen slots")
    started = time.perf_counter()
    counts = []
    for snr in config["snr_db"]:
        path = sequence_path(index, snr)
        diagnostic = DATA / "diagnostics" / snr_name(snr) / f"episode_{index:03d}.json"
        if path.exists() and diagnostic.exists():
            old = json.loads(diagnostic.read_text())
            cached = load_sequence(path)
            if old.get("config_sha256") != _HASH or not np.array_equal(np.rint(cached["timestamp"] * 1000).astype(np.int64), deadlines):
                raise ValueError(f"Stale cache: {path}")
            if old.get("complete") is not True:
                raise ValueError(f"Incomplete cache: {path}")
            counts.append({"snr_db": snr, "resumed": True})
            continue
        result = dict(state_hat=np.zeros((199, 8, 4), np.float32),
                      track_exists=np.zeros((199, 8), bool), detected=np.zeros((199, 8), bool),
                      timestamp=deadlines.astype(np.float64) / 1000)
        report = dict(episode_index=index, split=ep["split"], snr_db=snr,
                      revision=config["revision"], config_sha256=_HASH, complete=False,
                      frames=199, outputs=0, diagnostic_counts={}, failures=[])
        for frame, deadline in enumerate(deadlines):
            try:
                observation, _ = estimate_frame(source, index, int(deadline), snr, config)
                for target in observation["targets"]:
                    slot = int(target["key"])
                    value = np.asarray(target["state_hat"], np.float32)
                    if not 0 <= slot < len(ep["source_keys"]) or value.shape != (4,) or not np.isfinite(value).all():
                        raise ValueError("Invalid estimator output")
                    result["state_hat"][frame, slot] = value
                    result["track_exists"][frame, slot] = bool(target["exists"])
                    result["detected"][frame, slot] = bool(target["detected"])
                    report["outputs"] += 1
                    for key, val in target.get("diagnostics", {}).items():
                        if isinstance(val, (bool, np.bool_)):
                            report["diagnostic_counts"][key] = report["diagnostic_counts"].get(key, 0) + int(val)
            except Exception as error:
                report["failures"].append(dict(frame_index=frame, deadline_ms=int(deadline), error=repr(error)))
                write_json(diagnostic, report)
                raise RuntimeError(f"Episode {index}, SNR {snr}, frame {frame} failed; diagnostic recorded") from error
        report["complete"] = True
        write_npz(path, result)
        write_json(diagnostic, report)
        counts.append({"snr_db": snr, "outputs": report["outputs"]})
    result = dict(episode_index=index, seconds=time.perf_counter()-started, conditions=counts)
    print(json.dumps(result), flush=True)
    return result


def generate(workers=4, indices=None):
    init_worker()
    indices = list(range(len(_SOURCE.episodes))) if indices is None else list(indices)
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(_SOURCE.episodes) for i in indices):
        raise ValueError("Invalid or duplicate episode indices")
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        results = list(pool.map(generate_episode, indices))
    write_json(ROOT / "reports/f01d/generation.json", dict(config_sha256=_HASH, episodes=results, complete=True))


def pack_history(sequence, origin_ms):
    """Pure causal transformation of an estimated cache; never reads source rows."""
    times = np.rint(sequence["timestamp"] * 1000).astype(np.int64)
    wanted = int(origin_ms) + np.arange(-19, 1, dtype=np.int64) * 100
    indices = np.searchsorted(times, wanted)
    if np.any(indices >= len(times)) or not np.array_equal(times[indices], wanted):
        raise ValueError("Origin lacks a complete 20-frame timestamp grid")
    state = sequence["state_hat"][indices].copy()
    exists = sequence["track_exists"][indices].copy()
    detected = sequence["detected"][indices].copy()
    # Preserve historical nodes as graph context; the loader computes origin eligibility.
    detected &= exists
    state[~exists] = 0
    return dict(state_hat=state, track_exists=exists, detected=detected,
                timestamp=sequence["timestamp"][indices].copy())


def build_labels(source, index, origins_ms, tracks=None):
    """Raw future xy sidecar; exact source sample only, with an independent mask."""
    tracks = source.tracks if tracks is None else tracks
    ep = source.episodes[index]
    future = np.zeros((len(origins_ms), 20, 8, 2), np.float32)
    valid = np.zeros((len(origins_ms), 20, 8), bool)
    times = np.asarray(origins_ms, np.int64)[:, None] + np.arange(1, 21) * 100
    for slot, key in enumerate(ep["source_keys"]):
        track = tracks[key]
        positions = np.searchsorted(track["time_ms"], times)
        in_range = positions < len(track)
        safe = np.minimum(positions, max(0, len(track)-1))
        if not len(track):
            continue
        good = in_range & (track["time_ms"][safe] == times)
        xy = np.stack([track["x"][safe], track["y"][safe]], axis=-1)
        if not np.isfinite(xy[good]).all():
            raise ValueError("Nonfinite raw future position")
        future[:, :, slot][good] = xy[good]
        valid[:, :, slot] = good
    return dict(future_position=future, label_valid=valid)


def pack(data=DATA):
    init_worker()
    data = Path(data)
    totals = {}
    for split in dict.fromkeys(ep["split"] for ep in _SOURCE.episodes):
        episode_indices = [i for i, ep in enumerate(_SOURCE.episodes) if ep["split"] == split]
        metadata = []
        label_parts = []
        packed = {snr: {key: [] for key in KEYS} for snr in _CONFIG["snr_db"]}
        empty = {snr: 0 for snr in _CONFIG["snr_db"]}
        for index in episode_indices:
            ep = _SOURCE.episodes[index]
            origins = ep["prediction_grid_ms"]
            label_parts.append(build_labels(_SOURCE, index, origins))
            base_sample = len(metadata)
            metadata.extend(dict(sample_index=base_sample+j, episode_index=index,
                                 origin_ms=int(origin), source_keys=ep["source_keys"],
                                 episode_id=ep["episode_id"], source_block=ep["source_block"],
                                 source_groups=ep["source_groups"],
                                 track_keys=list(range(len(ep["source_keys"])))) for j, origin in enumerate(origins))
            for snr in _CONFIG["snr_db"]:
                path = sequence_path(index, snr, data)
                diagnostic = data / "diagnostics" / snr_name(snr) / f"episode_{index:03d}.json"
                report = json.loads(diagnostic.read_text())
                if report.get("config_sha256") != _HASH or report.get("complete") is not True or report.get("failures"):
                    raise ValueError(f"Invalid sequence provenance: {path}")
                sequence = load_sequence(path)
                for origin in origins:
                    sample = pack_history(sequence, origin)
                    empty[snr] += int(not sample["track_exists"].any())
                    for key in KEYS:
                        packed[snr][key].append(sample[key])
        labels = {key: np.concatenate([part[key] for part in label_parts]) for key in ("future_position", "label_valid")}
        write_npz(data / "labels" / f"{split}.npz", labels)
        for snr in _CONFIG["snr_db"]:
            write_npz(data / "inputs" / f"{split}_{snr_name(snr)}.npz",
                      {key: np.stack(packed[snr][key]) for key in KEYS})
        write_json(data / "metadata" / f"{split}.json", dict(split=split, config_sha256=_HASH,
                   revision=_CONFIG["revision"], samples=metadata, source="Frozen A01 prediction_grid_ms"))
        totals[split] = dict(samples=len(metadata), episodes=len(episode_indices), all_input_unavailable=empty)
    write_json(ROOT / "reports/f01d/packing.json", dict(complete=True, config_sha256=_HASH,
               splits=totals, input_keys=list(KEYS), label_keys=["future_position", "label_valid"],
               rule="Fixed A01 origins and source slots; no SNR-dependent or future-label sample selection; no normalization"))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "pack"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, nargs="*")
    args = parser.parse_args()
    if args.action == "generate":
        raise RuntimeError("CPU generation disabled by user; use python -m frontend.generate_symbol_gpu")
    else:
        pack()
