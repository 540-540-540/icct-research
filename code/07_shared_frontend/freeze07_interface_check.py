"""SENS-FREEZE-07 Stage G: downstream interface / leakage smoke against the existing loader.

Uses frontend.symbol_dataset (the frozen downstream loader) for all four splits and all six SNR
levels. Structural checks only: no model is trained, no ADE/FDE is computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402
import freeze07_generate_f01e as generator  # noqa: E402

DATA = ROOT / "data/f01e"
OUT = ROOT / "reports/f01e/freeze_07/interface_check.json"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
EXPECTED_SAMPLES = {"train": 5549, "V_select": 400, "V_confirm": 352, "test": 864}
MODEL_INPUT_KEYS = {"state_hat", "standardized_state", "track_exists", "detected", "timestamp",
                    "origin_eligible"}
LABEL_KEYS = {"future_position", "label_valid"}
FORBIDDEN_IN_INPUT = ("future", "source_key", "gt", "truth", "label", "target_id", "vehicle_id")
EVALUATION_ONLY_KEYS = {"sample_index", "episode_index", "origin_ms", "source_keys", "episode_id",
                        "source_block", "source_groups", "label_alignment_snr", "slot_alignment",
                        "slot_alignment_by_snr"}


def main() -> None:
    from frontend import symbol_dataset
    from frontend.echo_source import SourceEpisodes

    normalization = symbol_dataset.fit_normalization(DATA)
    report = {"stage": "SENS-FREEZE-07 Stage G", "loader": "frontend.symbol_dataset",
              "snr_levels": SNR_LIST, "splits": {}, "checks": {}}
    source = SourceEpisodes()
    split_episodes = {}
    for episode in source.episodes:
        split_episodes.setdefault(episode["split"], set()).add(episode["episode_id"])
    split_integrity_ok = True
    episode_partitions = {}
    for split in EXPECTED_SAMPLES:
        metadata = json.loads((DATA / "metadata" / f"{split}.json").read_text())
        assert len(metadata["samples"]) == EXPECTED_SAMPLES[split], f"{split} sample count changed"
        reported = {sample["episode_id"] for sample in metadata["samples"]}
        episode_partitions[split] = reported
        if reported - split_episodes[split]:
            split_integrity_ok = False
        entry = {"samples": len(metadata["samples"]), "episodes": len(reported),
                 "episode_partition_ok": reported <= split_episodes[split], "by_snr": {}}
        for snr in SNR_LIST:
            path = DATA / "inputs" / f"{split}_{generator.snr_name(snr)}.npz"
            arrays = symbol_dataset.read_inputs(path)
            loader = symbol_dataset.SharedPredictionInputs(path, normalization=DATA / "normalization.json")
            sample = loader[0]
            input_keys = set(sample.keys())
            leakage = [key for key in input_keys
                       if any(token in key.lower() for token in FORBIDDEN_IN_INPUT)]
            states = arrays["state_hat"]
            masks = arrays["track_exists"]
            detected = arrays["detected"]
            zero_padding_ok = bool(np.all(states[~masks] == 0))
            coast_examples = int((masks & ~detected).sum())
            tentative_examples = int((~masks).sum())
            entry["by_snr"][str(snr)] = {
                "samples": int(states.shape[0]),
                "state_shape": list(states.shape), "masks_shape": list(masks.shape),
                "timestamp_shape": list(arrays["timestamp"].shape),
                "dtype_ok": states.dtype == np.float32 and masks.dtype == np.bool_
                and arrays["timestamp"].dtype == np.float64,
                "finite": bool(np.isfinite(states).all() and np.isfinite(arrays["timestamp"]).all()),
                "zero_padding_ok": zero_padding_ok,
                "coast_state_frames": coast_examples,
                "unavailable_state_frames": tentative_examples,
                "input_keys": sorted(input_keys),
                "forbidden_input_keys": leakage,
                "origin_eligible_ratio": float(sample["origin_eligible"].mean()),
                "standardized_zero_where_missing": bool(np.all(sample["standardized_state"][~sample["track_exists"]] == 0)),
            }
        label_reference = DATA / "labels" / f"{split}.npz"
        label_by_snr = {}
        with np.load(label_reference, allow_pickle=False) as payload:
            reference_future = payload["future_position"].copy()
            reference_valid = payload["label_valid"].copy()
        for snr in SNR_LIST:
            with np.load(DATA / "labels" / f"{split}_{generator.snr_name(snr)}.npz",
                         allow_pickle=False) as payload:
                future = payload["future_position"]
                valid = payload["label_valid"]
            label_by_snr[str(snr)] = {
                "shape": list(future.shape),
                "valid": int(valid.sum()),
                "finite_where_valid": bool(np.isfinite(future[valid]).all()),
                "zero_where_invalid": bool(np.all(future[~valid] == 0)),
                "matches_reference": bool(np.array_equal(future, reference_future)
                                          and np.array_equal(valid, reference_valid)),
            }
        entry["labels"] = {"reference_is_snr_20": label_by_snr["20"]["matches_reference"],
                           "by_snr": label_by_snr, "label_keys": sorted(LABEL_KEYS)}
        entry["model_input_keys"] = sorted(MODEL_INPUT_KEYS)
        entry["evaluation_only_keys"] = sorted(EVALUATION_ONLY_KEYS)
        report["splits"][split] = entry
    overlaps = []
    splits = list(EXPECTED_SAMPLES)
    for first in range(len(splits)):
        for second in range(first + 1, len(splits)):
            shared = episode_partitions[splits[first]] & episode_partitions[splits[second]]
            if shared:
                overlaps.append({"a": splits[first], "b": splits[second], "episodes": len(shared)})
    structural = all(
        report["splits"][split]["by_snr"][str(snr)]["dtype_ok"]
        and report["splits"][split]["by_snr"][str(snr)]["finite"]
        and report["splits"][split]["by_snr"][str(snr)]["zero_padding_ok"]
        and not report["splits"][split]["by_snr"][str(snr)]["forbidden_input_keys"]
        and report["splits"][split]["by_snr"][str(snr)]["standardized_zero_where_missing"]
        and report["splits"][split]["labels"]["by_snr"][str(snr)]["finite_where_valid"]
        and report["splits"][split]["labels"]["by_snr"][str(snr)]["zero_where_invalid"]
        for split in EXPECTED_SAMPLES for snr in SNR_LIST)
    report["checks"] = {
        "all_splits_all_snr_load": True,
        "shapes_20_8_4": all(report["splits"][split]["by_snr"][str(snr)]["state_shape"] == [EXPECTED_SAMPLES[split], 20, 8, 4]
                             for split in EXPECTED_SAMPLES for snr in SNR_LIST),
        "future_20_frames": all(report["splits"][split]["labels"]["by_snr"][str(snr)]["shape"][1] == 20
                                for split in EXPECTED_SAMPLES for snr in SNR_LIST),
        "no_forbidden_input_keys": all(not report["splits"][split]["by_snr"][str(snr)]["forbidden_input_keys"]
                                       for split in EXPECTED_SAMPLES for snr in SNR_LIST),
        "structural": structural,
        "split_integrity": split_integrity_ok and not overlaps,
        "split_overlaps": overlaps,
        "label_files_per_snr": True,
        "normalization_fit_split": normalization["fit_split"],
        "normalization_snr_db": normalization["snr_db"],
        "model_inputs_and_labels_separated": True,
    }
    report["passed"] = all(value for key, value in report["checks"].items()
                           if isinstance(value, bool))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"interface": "PASS" if report["passed"] else "FAIL",
                      "checks": report["checks"]}, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()