#!/usr/bin/env python3
"""Small executable contract checks, plus full train/val target/cache auditing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.data_preprocessing.build_sind_target_views import (
    target_neighbors, eligible_windows, verify_sensing_parity,
    setup_from_config, calibration_from_config, atomic_json,
)
from tools.data_preprocessing.build_sind_high_interaction import pair_features, active_relation
from frontend.sind_target_dataset import SinDTargetPredictionDataset


def synthetic():
    ids = np.arange(1, 5)
    states = np.array([[0., 0, 1, 0], [1, 0, 0, 0], [2, 0, -1, 0], [3, 0, -2, 0]])
    for target in range(4):
        pick = target_neighbors(ids, states, target)
        assert pick[0] == target and set(ids[pick]) == set(ids)
    ids = np.arange(1, 13)
    states = np.column_stack((np.arange(12) * 10., np.zeros((12, 3))))
    left, right = target_neighbors(ids, states, 0), target_neighbors(ids, states, 11)
    assert set(left) != set(right) and left[0] == 0 and right[0] == 11
    first, last = np.array([0, 0, 0, 0]), np.array([39, 19, 39, 39])
    history, targets = eligible_windows(first, last, 0)
    assert history.tolist() == [0, 1, 2, 3] and targets.tolist() == [0, 2, 3]
    assert 1 in target_neighbors(np.arange(4), states[:4], 0)
    rng = np.random.default_rng(2026)
    for n in (4, 8, 12, 25):
        states = rng.normal(size=(n, 4)) * [20, 20, 5, 5]
        ids = rng.permutation(np.arange(n)) + 100
        for target in range(n):
            ranked = []
            for j in range(n):
                if j == target:
                    continue
                dist, closing, tcpa, dcpa = pair_features(states[target], states[j])
                ranked.append((0 if active_relation((dist, closing, tcpa, dcpa)) else 1,
                               dcpa if np.isfinite(dcpa) else 1e9, -closing, dist, int(ids[j]), j))
            expected = [target] + [row[-1] for row in sorted(ranked)[:7]]
            assert target_neighbors(ids, states, target).tolist() == expected
    try:
        SinDTargetPredictionDataset("test")
    except ValueError:
        pass
    else:
        raise AssertionError("test must be rejected before any file read")
    print("synthetic: N=4 all targets, N>8 own neighborhoods, history-only context, original selector parity, test refusal PASS", flush=True)


def real(split):
    dataset = SinDTargetPredictionDataset(split, return_tensors=False)
    folder = ROOT / "data/sind/target_views_v1" / split
    with np.load(folder / "state_pool.npz", allow_pickle=False) as pool:
        sc, fr, vi = pool["scene_id"], pool["frame"], pool["vehicle_id"]
        timestamps, sensing_gt = pool["timestamp"], pool["sensing_gt_rows"]
    d = dataset
    assert np.all(np.isfinite(d.state_hat)) and np.all(np.isfinite(d.state_gt))
    assert np.all((d.history_indices >= 0) & (d.history_indices < len(d.state_hat)))
    assert np.all((d.future_indices >= 0) & (d.future_indices < len(d.state_gt)))
    assert np.all((d.history_rows >= -1) & (d.history_rows < len(d.history_indices)))
    assert np.array_equal(d.history_rows >= 0, d.vehicle_mask)
    keys = np.column_stack((d.scene_id, d.start_frame, d.vehicle_ids[:, 0]))
    assert len(np.unique(keys, axis=0)) == len(d)
    assert np.array_equal(np.bincount(d.origin_index, minlength=d.num_origins), d.origin_target_count)
    # Reconstruct eligible targets directly from guarded split trajectory spans.
    groups = np.flatnonzero(np.r_[True, (np.diff(sc) != 0) | (np.diff(vi) != 0)])
    ends = np.r_[groups[1:] - 1, len(sc) - 1]
    for origin, (scene, start) in enumerate(zip(d.origin_scene, d.origin_start)):
        same_scene = sc[groups] == scene
        eligible = same_scene & (fr[groups] <= start) & (fr[ends] >= start + 39)
        expected_ids = np.sort(vi[groups[eligible]])
        lo = int(np.searchsorted(d.origin_index, origin, side="left"))
        hi = int(np.searchsorted(d.origin_index, origin, side="right"))
        assert np.array_equal(np.sort(d.vehicle_ids[lo:hi, 0]), expected_ids)
    # All target labels and all context histories must have exact frame/ID keys.
    for low in range(0, len(d), 2048):
        high = min(low + 2048, len(d))
        origin = d.origin_index[low:high]
        future = d.future_indices[low:high]
        assert np.all(sc[future] == d.origin_scene[origin, None])
        assert np.all(fr[future] == d.origin_start[origin, None] + 20 + np.arange(20))
        assert np.all(vi[future] == d.vehicle_ids[low:high, :1])
        selected = d.history_rows[low:high]
        valid = selected >= 0
        sensed = d.history_indices[np.maximum(selected, 0)]
        raw = sensing_gt[sensed]
        assert np.all((sc[raw] == d.origin_scene[origin, None, None])[valid])
        assert np.all((vi[raw] == d.vehicle_ids[low:high, :, None])[valid])
        assert np.all((fr[raw] == d.origin_start[origin, None, None] + np.arange(20))[valid])
        assert np.allclose(timestamps[raw][valid], np.broadcast_to(d.history_timestamp[origin, None, :], raw.shape)[valid], rtol=0, atol=1e-9)
    # Original cache entries must be reused exactly; no sensing changes on overlap.
    with np.load(ROOT / "data/sind/isac" / split / "sensing_cache.npz", allow_pickle=False) as old:
        zero = int(np.flatnonzero(np.isclose(old["snr_levels_db"], 0.))[0])
        states = old["state_hat"][zero]
        lookup = {tuple(map(int, key)): i for i, key in enumerate(zip(old["scene_id"], old["frame"], old["vehicle_id"]))}
    checked = 0
    for i, raw in enumerate(sensing_gt):
        old_index = lookup.get((int(sc[raw]), int(fr[raw]), int(vi[raw])))
        if old_index is not None:
            assert np.array_equal(d.state_hat[i], states[old_index])
            checked += 1
    assert checked == d.manifest["cache_reused_states"]
    csv = pd.read_csv(ROOT / "data/sind/splits" / split / "trajectories.csv")
    csv = csv.sort_values(["scene_id", "vehicle_id", "frame_id"]).reset_index(drop=True)
    source_states = csv[["x", "y", "vx", "vy"]].to_numpy(float)[sensing_gt]
    sensing_keys = list(zip(sc[sensing_gt].tolist(), fr[sensing_gt].tolist(), vi[sensing_gt].tolist()))
    config = json.loads((ROOT / "configs/sind_controlled_isac.json").read_text())
    parity = verify_sensing_parity(sensing_keys, source_states, lookup, states,
                                  setup_from_config(config), calibration_from_config(config))
    for index in np.unique(np.linspace(0, len(d) - 1, 64, dtype=int)):
        sample = d[int(index)]
        assert sample["history_state"].shape == (20, 8, 4)
        assert sample["target_mask"].tolist() == [True] + [False] * 7
        assert np.all(sample["future_state"][:, 1:] == 0)
        assert np.all(sample["history_state"][:, ~sample["vehicle_mask"]] == 0)
        assert sample["target_vehicle_id"] == sample["vehicle_ids"][0]
    weights = len(d) / (d.num_origins * d.origin_target_count[d.origin_index])
    assert np.isclose(weights.mean(), 1.0)
    result = {"split": split, "targets": len(d), "origins": d.num_origins,
              "unique_history_states": len(d.state_hat), "cache_reuse_exact": checked,
              "frozen_current_sensing_parity": parity,
              "all_target_keys_complete_unique": True, "history_and_future_keys_valid": True,
              "test_used": False}
    atomic_json(folder / "validation.json", result)
    print(json.dumps(result), flush=True)
    return set(zip(sc.tolist(), fr.tolist())), set(zip(sc.tolist(), vi.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    args = parser.parse_args()
    synthetic()
    if not args.synthetic_only:
        splits = ("train", "val") if args.split == "all" else (args.split,)
        coverage = [real(split) for split in splits]
        if len(coverage) == 2:
            assert not coverage[0][0].intersection(coverage[1][0]), "train/val frame overlap"
            assert not coverage[0][1].intersection(coverage[1][1]), "train/val physical vehicle overlap"
            print("train/val disjoint frame and physical-vehicle checks PASS; test_used=false", flush=True)


if __name__ == "__main__":
    main()
