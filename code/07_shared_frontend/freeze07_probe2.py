"""SENS-FREEZE-07 probe 2: F01-D sizes/origins per split + GPU state (read-only)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/dell/YrM/ICCT")
sys.path.insert(0, str(ROOT))


def main() -> None:
    report = {}
    data = ROOT / "data/f01d"
    for path in sorted((data / "inputs").glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            report[f"inputs/{path.name}"] = {"samples": int(f["state_hat"].shape[0])}
    for path in sorted((data / "labels").glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            report[f"labels/{path.name}"] = {"samples": int(f["future_position"].shape[0])}
    sequences = sorted((data / "sequences" / "snr_5").glob("episode_*.npz"))
    if sequences:
        with np.load(sequences[0], allow_pickle=False) as f:
            report["sequence_probe"] = {
                "file": sequences[0].name,
                "keys": list(f.files),
                "timestamp_first_ms": float(f["timestamp"][0] * 1000),
                "timestamp_last_ms": float(f["timestamp"][-1] * 1000),
                "frames": int(f["timestamp"].shape[0]),
                "positive_track_fraction": float(f["track_exists"].mean()),
                "detected_fraction": float(f["detected"].mean())}
    metadata = json.loads((data / "metadata" / "V_confirm.json").read_text())
    report["metadata_probe"] = {"keys": list(metadata.keys()),
                                "sample_keys": list(metadata["samples"][0].keys()),
                                "samples": len(metadata["samples"])}
    episodes = [json.loads(line) for line in (ROOT / "reports/f01a/episodes.jsonl").read_text().splitlines()]
    by_split = {}
    for episode in episodes:
        origins = episode["prediction_grid_ms"]
        entry = by_split.setdefault(episode["split"], {"episodes": 0, "origins_total": 0,
                                                       "origins_per_episode": set(),
                                                       "first_origin_offset_ms": set(),
                                                       "stride_ms": set()})
        entry["episodes"] += 1
        entry["origins_total"] += len(origins)
        entry["origins_per_episode"].add(len(origins))
        if origins:
            entry["first_origin_offset_ms"].add(int(origins[0]) - int(episode["start_ms"]))
            entry["stride_ms"].update(int(b - a) for a, b in zip(origins, origins[1:]))
    report["origins_by_split"] = {split: {key: (sorted(value) if isinstance(value, set) else value)
                                          for key, value in entry.items()}
                                  for split, entry in by_split.items()}
    print(json.dumps(report, ensure_ascii=False, default=str)[:1800])


if __name__ == "__main__":
    main()