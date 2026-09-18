#!/usr/bin/env python3
"""Validation suite for AutomatumPredictionDataset.

Verifies:
1. Scene-cache equality: Dataset history_state == ISAC sensing cache state_hat (float32 exact equality)
2. Future equality: Dataset future_state == samples.npz future
3. Padding: inactive vehicle slots are strictly zeroed and masked False
4. Cross-SNR alignment: same sample across 5 SNRs has identical metadata and future, differing only in history
5. Missing lookups: 0
6. Non-finite values: 0
7. GT-history leakage test: perturbing samples.npz history does NOT affect Dataset history_state
8. Generic DataLoader smoke: PyTorch DataLoader batching, shuffling, shapes, dtypes, and finite checks

Usage:
  python tools/data_preprocessing/validate_prediction_dataset.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from frontend.automatum_prediction_dataset import (
    CANONICAL_DT,
    AutomatumPredictionDataset,
)

SNR_LEVELS = [-10.0, -5.0, 0.0, 5.0, 10.0]
SPLITS = ["train", "val", "test"]


def test_split_contracts(split: str) -> dict:
    print(f"\n--- Testing Dataset Contract: {split.upper()} ---")

    # Check 5 SNR datasets instantiation
    datasets = {
        lvl: AutomatumPredictionDataset(split=split, snr_db=lvl, root_dir=ROOT, return_tensors=False)
        for lvl in SNR_LEVELS
    }
    N = len(datasets[0.0])
    print(f"[{split}] Total samples: {N:,}")

    # Load raw cache and samples for ground-truth cross-checks
    samples_path = ROOT / f"data/automatum_t_crossing/splits/{split}/samples.npz"
    cache_path = ROOT / f"data/automatum_t_crossing/isac/{split}/sensing_cache.npz"
    raw_samples = np.load(samples_path)
    raw_cache = np.load(cache_path)

    cache_scene = raw_cache["scene_id"]
    cache_frame = raw_cache["frame"]
    cache_veh = raw_cache["vehicle_id"]
    cache_state_hat = raw_cache["state_hat"]
    cache_snr_levels = raw_cache["snr_levels_db"]

    cache_lookup = {
        (int(cache_scene[i]), int(cache_frame[i]), int(cache_veh[i])): i
        for i in range(cache_scene.shape[0])
    }

    # 1 & 2 & 3. Sample checks across 100 deterministic indices
    sample_indices = np.linspace(0, N - 1, min(100, N), dtype=int)
    max_scene_cache_diff = 0.0
    max_future_diff = 0.0
    padding_violation_count = 0
    non_finite_count = 0

    for idx in sample_indices:
        sc = int(raw_samples["scene_id"][idx])
        st = int(raw_samples["start_frame"][idx])
        raw_mask = raw_samples["vehicle_mask"][idx]
        raw_ids = raw_samples["vehicle_ids"][idx]
        raw_fut = raw_samples["future"][idx]

        # Check across all 5 SNRs
        ref_sample = None
        for lvl_idx, lvl in enumerate(SNR_LEVELS):
            item = datasets[lvl][idx]

            hist = item["history_state"]
            fut = item["future_state"]
            mask = item["vehicle_mask"]
            ts = item["history_timestamp"]

            # Finiteness
            if not np.isfinite(hist).all() or not np.isfinite(fut).all() or not np.isfinite(ts).all():
                non_finite_count += 1

            # Future equality
            fut_diff = np.max(np.abs(fut - raw_fut))
            if fut_diff > max_future_diff:
                max_future_diff = fut_diff

            # Padding check
            for slot in range(8):
                if not mask[slot]:
                    if not (hist[:, slot, :] == 0.0).all():
                        padding_violation_count += 1
                    if not (fut[:, slot, :] == 0.0).all():
                        padding_violation_count += 1
                    if item["vehicle_ids"][slot] != -1:
                        padding_violation_count += 1

            # Scene-cache equality
            cache_snr_idx = np.where(np.isclose(cache_snr_levels, lvl))[0][0]
            for slot in range(8):
                if mask[slot]:
                    vid = int(raw_ids[slot])
                    for t in range(20):
                        fr = st + t
                        row_idx = cache_lookup[(sc, fr, vid)]
                        expected_hist = cache_state_hat[cache_snr_idx, row_idx]
                        diff = np.max(np.abs(hist[t, slot] - expected_hist))
                        if diff > max_scene_cache_diff:
                            max_scene_cache_diff = diff

            # 4. Cross-SNR alignment: metadata, future, mask, and ts must be 100% identical
            if ref_sample is None:
                ref_sample = item
            else:
                assert item["scene_id"] == ref_sample["scene_id"]
                assert item["start_frame"] == ref_sample["start_frame"]
                assert (item["vehicle_ids"] == ref_sample["vehicle_ids"]).all()
                assert (item["vehicle_mask"] == ref_sample["vehicle_mask"]).all()
                assert (item["future_state"] == ref_sample["future_state"]).all()
                assert (item["history_timestamp"] == ref_sample["history_timestamp"]).all()

    assert max_scene_cache_diff == 0.0, f"[{split}] scene-cache diff: {max_scene_cache_diff}"
    assert max_future_diff == 0.0, f"[{split}] future diff: {max_future_diff}"
    assert padding_violation_count == 0, f"[{split}] padding violations: {padding_violation_count}"
    assert non_finite_count == 0, f"[{split}] non-finite count: {non_finite_count}"
    print(f"[{split}] Scene-cache equality: EXACT (max diff = 0.00e+00)")
    print(f"[{split}] Future GT equality:    EXACT (max diff = 0.00e+00)")
    print(f"[{split}] Padding rules:         PASS (all padding slots strictly 0.0 / masked False)")
    print(f"[{split}] Cross-SNR alignment:   PASS (metadata/future identical across 5 SNRs)")
    print(f"[{split}] Finiteness:            PASS (0 non-finite values)")

    return {
        "split": split,
        "sample_count": N,
        "scene_cache_diff": float(max_scene_cache_diff),
        "future_diff": float(max_future_diff),
        "padding_violations": padding_violation_count,
        "non_finite_count": non_finite_count,
        "status": "PASS",
    }


def test_gt_history_leakage() -> bool:
    print("\n--- Testing GT-History Leakage Prevention ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        # Mirror data structure for train
        split = "train"
        tmp_splits = tmp_root / f"data/automatum_t_crossing/splits/{split}"
        tmp_isac = tmp_root / f"data/automatum_t_crossing/isac/{split}"
        tmp_splits.mkdir(parents=True, exist_ok=True)
        tmp_isac.mkdir(parents=True, exist_ok=True)

        # Copy original isac sensing cache
        src_cache = ROOT / f"data/automatum_t_crossing/isac/{split}/sensing_cache.npz"
        shutil.copy2(src_cache, tmp_isac / "sensing_cache.npz")

        # Copy and PERTURB samples.npz history
        src_samples = ROOT / f"data/automatum_t_crossing/splits/{split}/samples.npz"
        orig_samples = dict(np.load(src_samples))

        # Severely corrupt the history array in samples.npz with huge random numbers
        corrupted_history = orig_samples["history"] + 999999.0
        orig_samples["history"] = corrupted_history
        np.savez_compressed(tmp_splits / "samples.npz", **orig_samples)

        # Instantiate uncorrupted dataset from real repo
        dataset_clean = AutomatumPredictionDataset(split=split, snr_db=0.0, root_dir=ROOT, return_tensors=False)
        # Instantiate dataset pointing to corrupted samples.npz
        dataset_corrupted = AutomatumPredictionDataset(split=split, snr_db=0.0, root_dir=tmp_root, return_tensors=False)

        # Verify that history_state in dataset_corrupted is 100% IDENTICAL to dataset_clean!
        max_diff = 0.0
        for i in range(100):
            clean_hist = dataset_clean[i]["history_state"]
            corrupted_hist = dataset_corrupted[i]["history_state"]
            diff = np.max(np.abs(clean_hist - corrupted_hist))
            if diff > max_diff:
                max_diff = diff

        assert max_diff == 0.0, f"Leakage detected! max_diff={max_diff}"
        print(f"Corrupted samples.npz.history by +999999.0.")
        print(f"Dataset history_state max difference from uncorrupted baseline: {max_diff:.2e}")
        print(f"GT-history leakage prevention: PASS (ISAC cache is the sole source of history state)")
        return True


def test_generic_dataloader() -> bool:
    print("\n--- Testing Generic PyTorch DataLoader Smoke ---")
    # 1. train @ 0 dB
    train_ds = AutomatumPredictionDataset(split="train", snr_db=0.0, root_dir=ROOT, return_tensors=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)

    # 2. val @ -5 dB
    val_ds = AutomatumPredictionDataset(split="val", snr_db=-5.0, root_dir=ROOT, return_tensors=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # 3. test @ +10 dB
    test_ds = AutomatumPredictionDataset(split="test", snr_db=10.0, root_dir=ROOT, return_tensors=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    for name, loader in [("train@0dB", train_loader), ("val@-5dB", val_loader), ("test@+10dB", test_loader)]:
        print(f"Iterating 10 batches of {name}...")
        for b_idx, batch in enumerate(loader):
            if b_idx >= 10:
                break
            hist = batch["history_state"]
            fut = batch["future_state"]
            mask = batch["vehicle_mask"]
            ts = batch["history_timestamp"]

            assert hist.shape == (4, 20, 8, 4), f"bad hist shape: {hist.shape}"
            assert fut.shape == (4, 20, 8, 4), f"bad fut shape: {fut.shape}"
            assert mask.shape == (4, 8), f"bad mask shape: {mask.shape}"
            assert ts.shape == (4, 20), f"bad ts shape: {ts.shape}"

            assert hist.dtype == torch.float32
            assert fut.dtype == torch.float32
            assert mask.dtype == torch.bool
            assert ts.dtype == torch.float64

            assert torch.isfinite(hist).all()
            assert torch.isfinite(fut).all()
            assert torch.isfinite(ts).all()

            # Verify timestamps step is exactly 3 / 29.97
            dts = torch.diff(ts, dim=1)
            assert torch.allclose(dts, torch.tensor(CANONICAL_DT, dtype=torch.float64), atol=1e-12)

    print("PyTorch DataLoader smoke: PASS (all batches shaped, typed, masked, shuffled successfully)")
    return True


def main() -> None:
    print("==========================================")
    print("AUTOMATUM PREDICTION DATASET VALIDATION")
    print("==========================================")

    results = {}
    for split in SPLITS:
        results[split] = test_split_contracts(split)

    leakage_pass = test_gt_history_leakage()
    dataloader_pass = test_generic_dataloader()

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_revision": "AUTOMATUM-PREDICTION-DATA-FROZEN",
        "splits": results,
        "gt_history_leakage_pass": leakage_pass,
        "generic_dataloader_pass": dataloader_pass,
        "overall_status": "PASS",
    }

    out_json = ROOT / "reports/isac_production_cache/dataset_adapter_validation.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWritten adapter validation summary: {out_json}")
    print("\nALL DATASET ADAPTER TESTS PASSED SUCCESSFULLY")


if __name__ == "__main__":
    main()
