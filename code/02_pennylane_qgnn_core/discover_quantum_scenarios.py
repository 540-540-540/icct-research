"""Outcome-blind dynamic-interaction taxonomy with discovery/confirmation blocks.

All scene definitions and thresholds are written before model prediction arrays
are loaded. The script reuses retained single-core dev predictions only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "converged_protocol_seed2026_v1"
OUTPUT = BASE / "quantum_scenario_discovery_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SPLIT = BASE / "retained_inputs/circuit_split_indices.npz"
SNRS = (5, 10, 15, 20)
RADIUS_SMALL = 12.0
RADIUS_MID = 20.0
HORIZON_S = 2.0
CPA_DISTANCE_M = 4.0
HIGH_QUANTILE = 2.0 / 3.0
REPETITIONS = 10000
SEED = 2026090719


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def midrank01(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / max(1, len(values) - 1)


def scene_features(history, masks):
    names = (
        "small_radius_edge_turnover",
        "small_radius_topology_entropy",
        "two_hop_pair_fraction",
        "predicted_conflict_pairs",
        "maximum_competing_neighbors",
        "heading_diversity_on_mid_edges",
        "closing_mid_edges",
    )
    rows = []
    for states, mask in zip(history, masks):
        valid = np.flatnonzero(mask)
        states = states[:, valid]
        count = len(valid)
        if count < 2:
            rows.append([0.0] * len(names))
            continue
        position = states[..., :2].astype(np.float64)
        velocity = states[..., 2:4].astype(np.float64)
        delta_p = position[:, None, :, :] - position[:, :, None, :]
        distance = np.linalg.norm(delta_p, axis=-1)
        upper = np.triu(np.ones((count, count), dtype=bool), 1)
        pair_count = upper.sum()

        small = distance <= RADIUS_SMALL
        small[:, np.arange(count), np.arange(count)] = False
        turnover = np.logical_xor(small[1:], small[:-1])[:, upper].mean()
        probability = small[:, upper].mean(0)
        entropy = -(probability * np.log2(np.maximum(probability, 1e-12))
                    + (1.0 - probability) * np.log2(np.maximum(1.0 - probability, 1e-12)))
        topology_entropy = entropy.mean()

        final_mid = distance[-1] <= RADIUS_MID
        final_mid[np.arange(count), np.arange(count)] = False
        two_step = (final_mid.astype(np.int64) @ final_mid.astype(np.int64)) > 0
        exact_two_hop = two_step & ~final_mid & ~np.eye(count, dtype=bool)
        two_hop_fraction = exact_two_hop[upper].sum() / pair_count

        last_p = position[-1]
        last_v = velocity[-1]
        pair_p = last_p[None, :, :] - last_p[:, None, :]
        pair_v = last_v[None, :, :] - last_v[:, None, :]
        relative_speed_sq = (pair_v * pair_v).sum(-1)
        closest_t = -(pair_p * pair_v).sum(-1) / np.maximum(relative_speed_sq, 1e-9)
        closest_t = np.clip(closest_t, 0.0, HORIZON_S)
        closest_distance = np.linalg.norm(pair_p + closest_t[..., None] * pair_v, axis=-1)
        conflict = (closest_distance <= CPA_DISTANCE_M) & (distance[-1] <= 45.0)
        conflict &= ~np.eye(count, dtype=bool)
        conflict_pairs = conflict[upper].sum()
        maximum_competing = conflict.sum(1).max(initial=0)

        speed = np.linalg.norm(last_v, axis=-1)
        cross = np.abs(last_v[:, None, 0] * last_v[None, :, 1]
                       - last_v[:, None, 1] * last_v[None, :, 0])
        heading_sine = cross / np.maximum(speed[:, None] * speed[None, :], 1e-9)
        moving_mid = final_mid & (speed[:, None] > 1.0) & (speed[None, :] > 1.0) & upper
        heading_diversity = heading_sine[moving_mid].mean() if moving_mid.any() else 0.0

        radial_rate = (pair_p * pair_v).sum(-1) / np.maximum(distance[-1], 1e-9)
        closing_mid = ((radial_rate < -0.25) & final_mid & upper).sum()
        rows.append([
            turnover,
            topology_entropy,
            two_hop_fraction,
            conflict_pairs,
            maximum_competing,
            heading_diversity,
            closing_mid,
        ])
    matrix = np.asarray(rows, dtype=np.float64)
    return {name: matrix[:, index] for index, name in enumerate(names)}


def summarize(values):
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def load_arm(arm, indices):
    with np.load(RESULTS / f"{arm}_graph_full_dev.npz", allow_pickle=False) as data:
        assert np.array_equal(data["indices"], indices)
        return {snr: data[str(snr)].copy() for snr in SNRS}


def metrics(arrays, rows):
    by_snr = {}
    for snr in SNRS:
        values = arrays[snr][rows]
        weight = values[:, 2].sum()
        by_snr[str(snr)] = {
            "ade_m": float(values[:, 0].sum() / (20.0 * weight)),
            "fde_m": float(values[:, 1].sum() / weight),
        }
    return {
        key: float(np.mean([value[key] for value in by_snr.values()]))
        for key in ("ade_m", "fde_m")
    }


def reduction(reference, candidate):
    return {key: 100.0 * (reference[key] - candidate[key]) / reference[key]
            for key in ("ade_m", "fde_m")}


def bootstrap(reference, candidate, rows, blocks, seed):
    unique = np.unique(blocks[rows])
    rng = np.random.default_rng(seed)
    draws = np.empty((REPETITIONS, 2), dtype=np.float64)
    for iteration in range(REPETITIONS):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled = np.concatenate([rows[blocks[rows] == value] for value in chosen])
        change = reduction(metrics(reference, sampled), metrics(candidate, sampled))
        draws[iteration] = change["ade_m"], change["fde_m"]
    return {
        "unique_blocks": int(len(unique)),
        "ade_m": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
        "fde_m": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
    }


def evaluate_group(predictions, rows, blocks, seed):
    arm_metrics = {arm: metrics(values, rows) for arm, values in predictions.items()}
    q_vs_c = reduction(arm_metrics["classical"], arm_metrics["quantum"])
    q_vs_p = reduction(arm_metrics["plain"], arm_metrics["quantum"])
    return {
        "scenes": int(len(rows)),
        "metrics": arm_metrics,
        "quantum_vs_classical_reduction_percent": q_vs_c,
        "quantum_vs_plain_reduction_percent": q_vs_p,
        "quantum_vs_classical_block_bootstrap_95_ci": bootstrap(
            predictions["classical"], predictions["quantum"], rows, blocks, seed
        ),
    }


def main():
    OUTPUT.mkdir(exist_ok=False)
    with np.load(SPLIT, allow_pickle=False) as split:
        indices = split["circuit_dev_indices"].copy()
    with np.load(DATA, allow_pickle=False) as data:
        history = data["train_states"][indices, :20].copy()
        masks = data["train_mask"][indices].copy()
        starts = data["train_start_index"][indices].copy()
    blocks = (starts // 200).astype(np.int64)
    unique_blocks = np.unique(blocks)
    # Alternating ordered time blocks fixes the split without using outcomes.
    discovery_blocks = unique_blocks[::2]
    confirmation_blocks = unique_blocks[1::2]
    discovery_rows = np.flatnonzero(np.isin(blocks, discovery_blocks))
    confirmation_rows = np.flatnonzero(np.isin(blocks, confirmation_blocks))
    features = scene_features(history, masks)

    # Build a prespecified equal-weight composite using discovery-only empirical ranks.
    component_names = tuple(features)
    discovery_rank = np.column_stack([midrank01(features[name][discovery_rows]) for name in component_names])
    discovery_composite = discovery_rank.mean(1)
    composite = np.empty(len(indices), dtype=np.float64)
    composite[discovery_rows] = discovery_composite
    for row in confirmation_rows:
        percentiles = []
        for name in component_names:
            reference = np.sort(features[name][discovery_rows])
            percentiles.append(np.searchsorted(reference, features[name][row], side="right") / len(reference))
        composite[row] = np.mean(percentiles)
    features["equal_weight_dynamic_interaction_composite"] = composite

    thresholds = {
        name: float(np.quantile(values[discovery_rows], HIGH_QUANTILE, method="higher"))
        for name, values in features.items()
    }
    feature_protocol = {
        "status": "frozen_before_loading_model_outcomes",
        "seed": SEED,
        "data_sha256": sha256(DATA),
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "history_only": True,
        "future_states_used": False,
        "model_outcomes_used": False,
        "split": "ordered unique 200-frame blocks alternating discovery/confirmation",
        "discovery_blocks": discovery_blocks.tolist(),
        "confirmation_blocks": confirmation_blocks.tolist(),
        "discovery_rows": discovery_rows.tolist(),
        "confirmation_rows": confirmation_rows.tolist(),
        "high_group_rule": "feature >= discovery-set 2/3 quantile using method=higher",
        "feature_parameters": {
            "small_radius_m": RADIUS_SMALL,
            "mid_radius_m": RADIUS_MID,
            "constant_velocity_horizon_s": HORIZON_S,
            "closest_approach_distance_m": CPA_DISTANCE_M,
            "composite": "equal mean of seven discovery-empirical percentile ranks",
        },
        "thresholds": thresholds,
        "feature_summaries": {
            name: {
                "discovery": summarize(values[discovery_rows]),
                "confirmation": summarize(values[confirmation_rows]),
            }
            for name, values in features.items()
        },
        "no_retraining": True,
        "no_llm": True,
        "no_test": True,
        "source_sha256": sha256(Path(__file__)),
    }
    protocol_path = OUTPUT / "scenario_protocol_frozen.json"
    protocol_path.write_text(json.dumps(feature_protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Load outcomes only after the complete taxonomy and thresholds are durable.
    predictions = {arm: load_arm(arm, indices) for arm in ("plain", "classical", "quantum")}
    report = {
        "status": "completed",
        "scenario_protocol_sha256": sha256(protocol_path),
        "outcomes_loaded_after_protocol_was_written": True,
        "warning": (
            "This is a single-training-seed screening study. Multiple prespecified features are all reported; "
            "a useful scenario requires the direction to replicate in confirmation blocks."
        ),
        "all_scenes": {},
        "high_interaction_groups": {},
        "confirmation_rule": (
            "candidate only if quantum-vs-classical ADE and FDE are both positive in discovery and confirmation; "
            "strong candidate additionally requires both confirmation confidence intervals above zero"
        ),
        "candidates": [],
        "strong_candidates": [],
        "no_retraining": True,
        "no_llm": True,
        "no_test": True,
    }
    for split_name, rows in (("discovery", discovery_rows), ("confirmation", confirmation_rows)):
        report["all_scenes"][split_name] = evaluate_group(predictions, rows, blocks, SEED + len(report["all_scenes"]))
    for offset, (name, values) in enumerate(features.items()):
        threshold = thresholds[name]
        entry = {"threshold": threshold}
        for split_offset, (split_name, base_rows) in enumerate(
            (("discovery", discovery_rows), ("confirmation", confirmation_rows))
        ):
            rows = base_rows[values[base_rows] >= threshold]
            entry[split_name] = evaluate_group(
                predictions, rows, blocks, SEED + 100 * (offset + 1) + split_offset
            )
            entry[split_name]["feature_summary"] = summarize(values[rows])
        report["high_interaction_groups"][name] = entry
        discovery_change = entry["discovery"]["quantum_vs_classical_reduction_percent"]
        confirmation_change = entry["confirmation"]["quantum_vs_classical_reduction_percent"]
        if min(discovery_change.values()) > 0 and min(confirmation_change.values()) > 0:
            report["candidates"].append(name)
            interval = entry["confirmation"]["quantum_vs_classical_block_bootstrap_95_ci"]
            if interval["ade_m"][0] > 0 and interval["fde_m"][0] > 0:
                report["strong_candidates"].append(name)
    output_path = OUTPUT / "scenario_screening_results.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "completed.json").write_text(json.dumps({
        "status": "completed",
        "protocol_sha256": sha256(protocol_path),
        "results_sha256": sha256(output_path),
        "candidates": report["candidates"],
        "strong_candidates": report["strong_candidates"],
        "no_test": True,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
