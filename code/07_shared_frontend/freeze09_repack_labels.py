"""SENS-FREEZE-09 repack: rebuild only labels/metadata with the origin-safe matcher (CPU-only).

Sequences and inputs are read-only and hash-verified before and after. The repack reuses
freeze07_generate_f01e.pack with the origin-safe align_window; no sensing is re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import freeze07_generate_f01e as generator  # noqa: E402

OUT = ROOT / "reports/f01e/label_fix_09"
DATA = ROOT / "data/f01e"
SNR_LIST = [-5, 0, 5, 10, 15, 20]
SPLITS = ["train", "V_select", "V_confirm", "test"]


def file_manifest(base: Path) -> dict:
    manifest = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(base))] = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return manifest


def label_stats(split: str) -> dict:
    with np.load(DATA / "inputs" / f"{split}_{generator.snr_name(SNR_LIST[0])}.npz") as payload:
        reference_exists = payload["track_exists"]
    eligible = np.cumprod(reference_exists[:, ::-1, :], axis=1).sum(axis=1)
    eligible = reference_exists[:, -1, :] & (eligible >= 3)
    stats = {"valid_slot_labels": 0, "labelled_slots": 0, "dead_slot_labels": 0,
             "origin_ineligible_labels": 0, "eligible_slots": 0}
    for snr in SNR_LIST:
        with np.load(DATA / "inputs" / f"{split}_{generator.snr_name(snr)}.npz") as payload:
            exists = payload["track_exists"]
        with np.load(DATA / "labels" / f"{split}_{generator.snr_name(snr)}.npz") as payload:
            valid = payload["label_valid"]
        eligible_snr = exists[:, -1, :] & (np.cumprod(exists[:, ::-1, :], axis=1).sum(axis=1) >= 3)
        any_valid = valid.any(axis=1)
        stats["valid_slot_labels"] += int(valid.sum())
        stats["labelled_slots"] += int(any_valid.sum())
        stats["dead_slot_labels"] += int((any_valid & ~exists[:, -1, :]).sum())
        stats["origin_ineligible_labels"] += int((any_valid & ~eligible_snr).sum())
        stats["eligible_slots"] += int(eligible_snr.sum())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA))
    parser.add_argument("--split-list", nargs="*", default=SPLITS)
    parser.add_argument("--snr-list", type=float, nargs="+", default=SNR_LIST)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = {split: label_stats(split) for split in args.split_list}
    if args.stats_only:
        print(json.dumps(before, ensure_ascii=False))
        return
    pre_inputs = file_manifest(Path(args.out) / "inputs")
    pre_sequences = file_manifest(Path(args.out) / "sequences")
    generation_config_hash = "d831f4673cc5dd593a41214347c61df88355b734226f23dd9bb8300c4f64f108"
    pack_args = SimpleNamespace(out=args.out, split_list=args.split_list, snr_list=args.snr_list,
                                episode_limit=None,
                                allowed_provenance_hashes=[generation_config_hash])
    generator.pack(pack_args)
    post_inputs = file_manifest(Path(args.out) / "inputs")
    post_sequences = file_manifest(Path(args.out) / "sequences")
    after = {split: label_stats(split) for split in args.split_list}
    integrity = {
        "stage": "SENS-FREEZE-09 repack",
        "inputs_files": len(pre_inputs), "sequences_files": len(pre_sequences),
        "inputs_before_sha256": hashlib.sha256(json.dumps(pre_inputs, sort_keys=True).encode()).hexdigest(),
        "inputs_after_sha256": hashlib.sha256(json.dumps(post_inputs, sort_keys=True).encode()).hexdigest(),
        "sequences_before_sha256": hashlib.sha256(json.dumps(pre_sequences, sort_keys=True).encode()).hexdigest(),
        "sequences_after_sha256": hashlib.sha256(json.dumps(post_sequences, sort_keys=True).encode()).hexdigest(),
        "inputs_unchanged": pre_inputs == post_inputs,
        "sequences_unchanged": pre_sequences == post_sequences,
        "labels_before": before, "labels_after": after,
        "labels_metadata_rewritten": True,
        "sequence_provenance": {
            "generation_config_sha256": generation_config_hash,
            "current_config_sha256": hashlib.sha256(
                (ROOT / "configs/shared_frontend.json").read_bytes()).hexdigest(),
            "note": "generation hash equals commit f319faf config; current equals 9473d17 config; the "
                    "only diff is snr_levels_db and calibration metadata (no sensing numerics), so the "
                    "existing sequences are accepted without regeneration"},
    }
    with (OUT / "cache_integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"inputs_unchanged": integrity["inputs_unchanged"],
                      "sequences_unchanged": integrity["sequences_unchanged"],
                      "before": before["train"], "after": after["train"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()