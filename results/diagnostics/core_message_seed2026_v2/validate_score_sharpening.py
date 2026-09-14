"""Paired holdout bootstrap for the score-x2 causal probe."""
import json
from pathlib import Path

import numpy as np

import experiment_utils as r
import run as retained
from probe_quantum_bottlenecks import BASE, RESULTS, SEED, build_variant, make_bank


def metric(array, rows):
    value = array[rows]
    count = value[:, 2].sum()
    return np.array([value[:, 0].sum() / (20 * count), value[:, 1].sum() / count])


def main():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        dev_indices = split["circuit_dev_indices"].copy()
    selected = set(r.select_profile_indices(dev_indices, 256).tolist())
    holdout = np.asarray([index for index in dev_indices.tolist() if index not in selected], dtype=np.int64)
    with np.load("/home/js_cn/sensing/data/multitarget_lankershim_v1.npz", allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
        blocks = (data["train_start_index"][holdout] // 200).astype(int)
    bank = make_bank(states, masks, holdout)
    retained.NOISE = r.load_snr_noise_map(
        "/home/js_cn/sensing/results/multitarget_snr/snr_calibration.json"
    )
    arrays = {}
    for name in ("baseline", "score_scale_2"):
        model, _ = build_variant(name)
        _, arrays[name] = retained.evaluate(model, bank, collect=True)
    rows = np.arange(len(holdout))
    baseline = np.mean([metric(arrays["baseline"][str(snr)], rows) for snr in (5, 10, 15, 20)], axis=0)
    candidate = np.mean([metric(arrays["score_scale_2"][str(snr)], rows) for snr in (5, 10, 15, 20)], axis=0)
    point = 100 * (baseline - candidate) / baseline
    unique = np.unique(blocks)
    rng = np.random.default_rng(20260907)
    samples = []
    for _ in range(10000):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled_rows = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
        b = np.mean(
            [metric(arrays["baseline"][str(snr)], sampled_rows) for snr in (5, 10, 15, 20)], axis=0
        )
        c = np.mean(
            [metric(arrays["score_scale_2"][str(snr)], sampled_rows) for snr in (5, 10, 15, 20)], axis=0
        )
        samples.append(100 * (b - c) / b)
    samples = np.asarray(samples)
    report = {
        "seed": SEED,
        "holdout_scenes": int(len(holdout)),
        "unique_time_blocks": int(len(unique)),
        "bootstrap_replicates": 10000,
        "score_scale": 2.0,
        "reduction_percent_ADE_FDE": point.tolist(),
        "block_bootstrap_95_percentile_ci_ADE_FDE": np.percentile(
            samples, [2.5, 97.5], axis=0
        ).T.tolist(),
        "no_retraining": True,
        "no_test": True,
    }
    output = RESULTS / "score_scale_2_holdout_bootstrap.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
