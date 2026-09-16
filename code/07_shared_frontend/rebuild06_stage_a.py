"""SENS-SNR-REBUILD-06 Stage A1: real A01 train Doppler separability (server-only SourceEpisodes).

Reads only the A01 train split through frontend.echo_source.SourceEpisodes. If that module or its
data is unavailable this script fails; it never falls back to a raw-CSV proxy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402
import probe_contiguous_bursts as r3  # noqa: E402

OUT = ROOT / "reports/f01e/snr_rebuild_06"
ORIGIN_STRIDE = 4
DENSITY = {"low": (1, 3), "mid": (4, 6), "high": (7, 8)}


def density_bucket(count: int) -> str:
    for name, (low, high) in DENSITY.items():
        if low <= count <= high:
            return name
    return "high"


def main() -> None:
    from frontend.echo_source import SourceEpisodes

    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    waveform, _array = _common.build_objects(config)
    stations = geometry["stations"]
    boresights = geometry["boresights"]
    height = float(geometry["height_difference_m"])
    source = SourceEpisodes()
    train = [episode for episode in source.episodes if episode["split"] == "train"]
    snapshots, buckets = [], {"low": [], "mid": [], "high": []}
    for episode in train:
        origins = list(episode.get("prediction_grid_ms", []))
        for origin in origins[::ORIGIN_STRIDE]:
            states, _slots, _used = source.at_time(episode, int(origin) * 1_000_000)
            if len(states) < 2:
                continue
            entry = (states[:, :2].tolist(), states[:, 2:].tolist())
            snapshots.append(entry)
            buckets[density_bucket(len(states))].append(entry)
    stats = r3.separability_stats(snapshots, stations, boresights, height, waveform)
    stats["by_density"] = {
        name: r3.separability_stats(entries, stations, boresights, height, waveform)
        for name, entries in buckets.items()}
    stats.update({
        "stage": "SENS-SNR-REBUILD-06 Stage A1",
        "data_source": "A01 SourceEpisodes (train split only)",
        "train_episodes": len(train),
        "origin_stride": ORIGIN_STRIDE,
        "snapshots": len(snapshots),
        "snapshots_by_density": {name: len(entries) for name, entries in buckets.items()},
    })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train_doppler_separability.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({
        "snapshots": stats["snapshots"], "train_episodes": len(train),
        "pair_count_total": stats["pair_count_total"],
        "spatially_close_pair_count": stats["spatially_close_pair_count"],
        "close_pair_dvr": stats["close_pair_dvr"],
        "E0_band_B64": stats["per_model"]["B64"],
    }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()