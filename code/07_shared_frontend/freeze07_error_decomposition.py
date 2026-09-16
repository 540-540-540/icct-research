"""SENS-FREEZE-07 Stage F decomposition: where the train position-error tail comes from."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import freeze07_generate_f01e as generator  # noqa: E402

DATA = ROOT / "data/f01e"
SNR_LIST = [-5, 0, 5, 10, 15, 20]


def percentile(values, fraction):
    return float(np.percentile(values, fraction * 100)) if len(values) else None


def main() -> None:
    from frontend.echo_source import SourceEpisodes

    source = SourceEpisodes()
    metadata = json.loads((DATA / "metadata" / "train.json").read_text())
    by_episode = defaultdict(list)
    for sample in metadata["samples"]:
        by_episode[sample["episode_index"]].append(sample)
    buckets = defaultdict(list)
    costs = defaultdict(list)
    window_p90 = defaultdict(list)
    window_p90_detected = defaultdict(list)
    for snr in SNR_LIST:
        sequences = {index: generator.load_sequence(generator.sequence_path(DATA, snr, index))
                     for index in by_episode}
        for index, samples in by_episode.items():
            episode = source.episodes[index]
            sequence = sequences[index]
            density = "low" if len(episode["source_keys"]) <= 3 else (
                "mid" if len(episode["source_keys"]) <= 6 else "high")
            for sample in samples:
                origin = int(sample["origin_ms"])
                indices = generator.history_indices(sequence, origin)
                exists = sequence["track_exists"][indices]
                detected = sequence["detected"][indices]
                states = sequence["state_hat"][indices]
                alignment = generator.align_window(sequence, episode, origin, source)
                window_errors = []
                window_errors_detected = []
                for entry in alignment:
                    slot = entry["slot"]
                    key = entry["source_key"]
                    costs[snr].append(entry["cost_m"])
                    times = origin + np.arange(-19, 1, dtype=np.int64) * 100
                    xy, exact = generator.gt_track_xy(source.tracks[int(key)], times)
                    for frame in range(20):
                        if not exists[frame, slot] or not exact[frame]:
                            continue
                        error = float(np.linalg.norm(states[frame, slot, :2] - xy[frame]))
                        bucket = "coast" if not detected[frame, slot] else "detected"
                        rho = float(np.hypot(xy[frame, 0] - source.stations[0][0],
                                             xy[frame, 1] - source.stations[0][1]))
                        if rho < 90:
                            range_bucket = "range_lt90"
                        elif rho < 225:
                            range_bucket = "range_90_225"
                        else:
                            range_bucket = "range_gt225"
                        buckets[(snr, bucket, density, range_bucket)].append(error)
                        window_errors.append(error)
                        if detected[frame, slot]:
                            window_errors_detected.append(error)
                window_p90[snr].append(float(np.percentile(window_errors, 90)) if window_errors else 0.0)
                window_p90_detected[snr].append(float(np.percentile(window_errors_detected, 90))
                                                if window_errors_detected else 0.0)
    summary = {}
    for key, values in sorted(buckets.items()):
        summary["|".join(str(part) for part in key)] = {
            "samples": len(values), "p50": percentile(values, 0.50), "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99)}
    artifact = {"stage": "SENS-FREEZE-07 Stage F decomposition",
                "buckets": summary, "alignment_cost": {str(snr): {
                    "median": percentile(costs[snr], 0.50), "p90": percentile(costs[snr], 0.90),
                    "fraction_gt1m": float(np.mean(np.asarray(costs[snr]) > 1.0))} for snr in SNR_LIST},
                "window_p90_gt1_fraction": {str(snr): float((np.asarray(window_p90[snr]) > 1.0).mean())
                                            for snr in SNR_LIST},
                "window_p90_detected_only_gt1_fraction": {
                    str(snr): float((np.asarray(window_p90_detected[snr]) > 1.0).mean())
                    for snr in SNR_LIST},
                "note": "SNR-independent structural tail; coast frames are a minor contributor"}
    out = Path(ROOT) / "reports/f01e/freeze_07/train_error_decomposition.json"
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1) + "\n")
    for key, values in sorted(buckets.items()):
        if key[2] == "high" and (key[-1] == "range_gt225" or key[1] == "coast"):
            print(json.dumps({"bucket": "|".join(str(part) for part in key), "samples": len(values),
                              "p50": percentile(values, 0.50), "p90": percentile(values, 0.90)}))
    for snr in SNR_LIST:
        p90 = np.asarray(window_p90[snr])
        p90d = np.asarray(window_p90_detected[snr])
        print(json.dumps({"snr": snr, "cost_median": percentile(costs[snr], 0.50),
                          "cost_p90": percentile(costs[snr], 0.90),
                          "window_p90_gt1_fraction": float((p90 > 1.0).mean()),
                          "window_p90_detected_gt1_fraction": float((p90d > 1.0).mean()),
                          "coast_error_samples": len(buckets.get((snr, "coast", "low", "range_lt90"), []))}))


if __name__ == "__main__":
    main()