"""SENS-FREEZE-07 Stage F: train-only dataset-level quality audit of the F01-E cache.

Reads only the train split. GT is used exclusively for offline evaluation (C domain); nothing is
fed back into sensing. Writes reports/f01e/freeze_07/train_quality_audit.json.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402
import freeze07_generate_f01e as generator  # noqa: E402

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/freeze_07/train_quality_audit.json"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
JUMP_LIMIT_M = 20.0


def percentile(values, fraction):
    return float(np.percentile(values, fraction * 100)) if len(values) else None


def main() -> None:
    from frontend.echo_source import SourceEpisodes

    source = SourceEpisodes()
    metadata = json.loads((DATA / "metadata" / "train.json").read_text())
    samples = metadata["samples"]
    by_episode = defaultdict(list)
    for sample in samples:
        by_episode[sample["episode_index"]].append(sample)
    report = {"stage": "SENS-FREEZE-07 Stage F", "split": "train", "snr_levels": SNR_LIST,
              "windows": len(samples), "episodes": len(by_episode), "by_snr": {},
              "paired_snr_check": {}, "sticky_slot_continuity_violations": 0}
    error_pool = {snr: {"position": [], "velocity": [], "by_density": defaultdict(lambda: {"position": [], "velocity": []}),
                        "coast": 0, "alive": 0, "detected": 0, "occupancy": [],
                        "problematic": 0, "windows": 0}
                  for snr in SNR_LIST}
    for snr in SNR_LIST:
        sequences = {index: generator.load_sequence(generator.sequence_path(DATA, snr, index))
                     for index in by_episode}
        for index, episode_samples in by_episode.items():
            episode = source.episodes[index]
            sequence = sequences[index]
            density = "low" if len(episode["source_keys"]) <= 3 else (
                "mid" if len(episode["source_keys"]) <= 6 else "high")
            for sample in episode_samples:
                origin = int(sample["origin_ms"])
                indices = generator.history_indices(sequence, origin)
                exists = sequence["track_exists"][indices]
                detected = sequence["detected"][indices]
                states = sequence["state_hat"][indices]
                pool = error_pool[snr]
                pool["windows"] += 1
                pool["occupancy"].append(int(exists[-1].sum()))
                pool["alive"] += int(exists.sum())
                pool["detected"] += int(detected.sum())
                pool["coast"] += int((exists & ~detected).sum())
                violations = 0
                for slot in range(8):
                    for frame in range(1, 20):
                        if exists[frame - 1, slot] and exists[frame, slot]:
                            jump = float(np.linalg.norm(states[frame, slot, :2] - states[frame - 1, slot, :2]))
                            if jump > JUMP_LIMIT_M:
                                violations += 1
                report["sticky_slot_continuity_violations"] += violations
                alignment = generator.align_window(sequence, episode, origin, source)
                window_position = []
                for entry in alignment:
                    slot = entry["slot"]
                    key = entry["source_key"]
                    times = origin + np.arange(-19, 1, dtype=np.int64) * 100
                    xy, exact = generator.gt_track_xy(source.tracks[int(key)], times)
                    for frame in range(20):
                        if not exists[frame, slot] or not exact[frame]:
                            continue
                        position = float(np.linalg.norm(states[frame, slot, :2] - xy[frame]))
                        pool["position"].append(position)
                        pool["by_density"][density]["position"].append(position)
                        track = source.tracks[int(key)]
                        row = np.searchsorted(track["time_ms"], times[frame])
                        if row < len(track) and track["time_ms"][row] == times[frame]:
                            velocity = float(np.linalg.norm(states[frame, slot, 2:]
                                                            - np.array([track["vx"][row], track["vy"][row]])))
                            pool["velocity"].append(velocity)
                            pool["by_density"][density]["velocity"].append(velocity)
                        window_position.append(position)
                coast_ratio = float((exists & ~detected).sum() / max(exists.sum(), 1))
                p90 = float(np.percentile(window_position, 90)) if window_position else 0.0
                if p90 > 1.0 or coast_ratio > 0.30 or violations > 0:
                    pool["problematic"] += 1
        position = np.asarray(error_pool[snr]["position"])
        velocity = np.asarray(error_pool[snr]["velocity"])
        report["by_snr"][str(snr)] = {
            "windows": error_pool[snr]["windows"],
            "occupancy_per_window": {"p10": percentile(error_pool[snr]["occupancy"], 0.10),
                                     "median": percentile(error_pool[snr]["occupancy"], 0.50),
                                     "p90": percentile(error_pool[snr]["occupancy"], 0.90),
                                     "max": int(max(error_pool[snr]["occupancy"])) if error_pool[snr]["occupancy"] else 0},
            "track_exists_ratio": error_pool[snr]["alive"] / max(error_pool[snr]["windows"] * 20 * 8, 1),
            "detected_ratio": error_pool[snr]["detected"] / max(error_pool[snr]["windows"] * 20 * 8, 1),
            "coast_ratio": error_pool[snr]["coast"] / max(error_pool[snr]["alive"], 1),
            "confirmed_state_ratio": error_pool[snr]["alive"] / max(error_pool[snr]["windows"] * 20 * 8, 1),
            "position_error_m": {"samples": int(len(position)), "p50": percentile(position, 0.50),
                                 "p90": percentile(position, 0.90), "p95": percentile(position, 0.95),
                                 "p99": percentile(position, 0.99),
                                 "fraction_gt_0p5m": float((position > 0.5).mean()) if len(position) else None,
                                 "fraction_gt_1m": float((position > 1.0).mean()) if len(position) else None,
                                 "fraction_gt_2m": float((position > 2.0).mean()) if len(position) else None},
            "velocity_error_mps": {"samples": int(len(velocity)), "p50": percentile(velocity, 0.50),
                                   "p90": percentile(velocity, 0.90), "p95": percentile(velocity, 0.95),
                                   "p99": percentile(velocity, 0.99),
                                   "fraction_gt_2mps": float((velocity > 2.0).mean()) if len(velocity) else None,
                                   "fraction_gt_4mps": float((velocity > 4.0).mean()) if len(velocity) else None},
            "problematic_window_fraction": error_pool[snr]["problematic"] / max(error_pool[snr]["windows"], 1),
            "by_density": {density: {
                "position_p50": percentile(np.asarray(values["position"]), 0.50),
                "position_p90": percentile(np.asarray(values["position"]), 0.90),
                "velocity_p50": percentile(np.asarray(values["velocity"]), 0.50),
                "velocity_p90": percentile(np.asarray(values["velocity"]), 0.90),
                "samples": len(values["position"])}
                for density, values in error_pool[snr]["by_density"].items()},
        }
    all_position = np.concatenate([np.asarray(error_pool[snr]["position"]) for snr in SNR_LIST])
    all_velocity = np.concatenate([np.asarray(error_pool[snr]["velocity"]) for snr in SNR_LIST])
    report["overall"] = {
        "position_error_m": {"p50": percentile(all_position, 0.50), "p90": percentile(all_position, 0.90),
                             "p95": percentile(all_position, 0.95), "p99": percentile(all_position, 0.99)},
        "velocity_error_mps": {"p50": percentile(all_velocity, 0.50), "p90": percentile(all_velocity, 0.90),
                               "p95": percentile(all_velocity, 0.95), "p99": percentile(all_velocity, 0.99)},
        "problematic_window_fraction_overall": float(np.mean([report["by_snr"][str(snr)][
            "problematic_window_fraction"] for snr in SNR_LIST])),
    }
    paired = {}
    file_hashes = {}
    for snr in SNR_LIST:
        path = DATA / "inputs" / f"train_{generator.snr_name(snr)}.npz"
        with np.load(path, allow_pickle=False) as payload:
            file_hashes[str(snr)] = hashlib.sha256(payload["state_hat"].tobytes()).hexdigest()
        paired[str(snr)] = {"position_p50": report["by_snr"][str(snr)]["position_error_m"]["p50"],
                            "position_p90": report["by_snr"][str(snr)]["position_error_m"]["p90"],
                            "velocity_p50": report["by_snr"][str(snr)]["velocity_error_mps"]["p50"]}
    medians = [paired[str(snr)]["position_p50"] for snr in SNR_LIST]
    report["paired_snr_check"] = {
        "identical_cache_files": len(set(file_hashes.values())) != len(SNR_LIST),
        "position_p50_by_snr": paired,
        "low_snr_worse_than_high_snr": medians[0] > medians[-1],
        "monotone_position_p50": all(a >= b - 0.005 for a, b in zip(medians, medians[1:])),
        "rule": "same fixed episode/origin windows and paired seed contract across SNR; only the "
                "physical SNR realization changes",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"overall": report["overall"], "paired": report["paired_snr_check"],
                      "violations": report["sticky_slot_continuity_violations"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()