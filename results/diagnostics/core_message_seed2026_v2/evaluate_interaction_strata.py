"""Outcome-blind interaction-complexity stratification of retained dev predictions.

The script computes and saves strata before loading any model error arrays.  No
training or test data are used for evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "converged_protocol_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SPLIT = BASE / "retained_inputs/circuit_split_indices.npz"
SNRS = (5, 10, 15, 20)
ARMS = ("plain", "classical", "quantum")
EDGE_RADIUS_M = 45.0
TTC_SCALE_S = 5.0
SEED = 20260907

# Fixed before model outcomes are loaded.  Components are empirical mid-ranks
# on the fixed dev set, then combined by these weights.
WEIGHTS = {
    "active_targets": 0.20,
    "mean_neighbor_degree": 0.30,
    "nearest_pair_proximity": 0.20,
    "closing_ttc_risk": 0.30,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def midrank01(values):
    """Empirical mid-ranks in [0, 1], preserving ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked / max(1, len(values) - 1)


def scene_features(last_state, masks):
    records = []
    for state, mask in zip(last_state, masks):
        valid = np.flatnonzero(mask)
        count = len(valid)
        if count < 2:
            records.append((count, 0.0, 0.0, 0.0, np.inf, 0))
            continue
        position = state[valid, :2].astype(np.float64)
        velocity = state[valid, 2:4].astype(np.float64)
        delta_p = position[None, :, :] - position[:, None, :]
        delta_v = velocity[None, :, :] - velocity[:, None, :]
        distance = np.linalg.norm(delta_p, axis=-1)
        upper = np.triu(np.ones((count, count), dtype=bool), 1)
        close = upper & (distance <= EDGE_RADIUS_M)
        edge_count = int(close.sum())
        mean_degree = 2.0 * edge_count / count
        if edge_count == 0:
            records.append((count, mean_degree, 0.0, 0.0, np.inf, edge_count))
            continue
        pair_distance = distance[close]
        proximity = 1.0 - min(float(pair_distance.min()), EDGE_RADIUS_M) / EDGE_RADIUS_M
        radial_rate = (delta_p * delta_v).sum(-1) / np.maximum(distance, 1e-9)
        closing_speed = np.maximum(-radial_rate[close], 0.0)
        valid_closing = closing_speed > 1e-6
        if valid_closing.any():
            ttc = pair_distance[valid_closing] / closing_speed[valid_closing]
            min_ttc = float(ttc.min())
            ttc_risk = 1.0 / (1.0 + min_ttc / TTC_SCALE_S)
        else:
            min_ttc = np.inf
            ttc_risk = 0.0
        records.append((count, mean_degree, proximity, ttc_risk, min_ttc, edge_count))
    names = (
        "active_targets",
        "mean_neighbor_degree",
        "nearest_pair_proximity",
        "closing_ttc_risk",
        "minimum_positive_ttc_s",
        "undirected_edges_within_45m",
    )
    matrix = np.asarray(records, dtype=np.float64)
    return {name: matrix[:, index] for index, name in enumerate(names)}


def summarize_values(values):
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "finite_fraction": float(np.isfinite(values).mean()),
    }


def build_strata():
    with np.load(SPLIT, allow_pickle=False) as split:
        dev_indices = split["circuit_dev_indices"].copy()
    with np.load(DATA, allow_pickle=False) as data:
        last_state = data["train_states"][dev_indices, 19].copy()
        masks = data["train_mask"][dev_indices].copy()
        starts = data["train_start_index"][dev_indices].copy()
    features = scene_features(last_state, masks)
    ranks = {name: midrank01(features[name]) for name in WEIGHTS}
    score = sum(WEIGHTS[name] * ranks[name] for name in WEIGHTS)

    # Exact near-equal thirds. Stable dataset index breaks score ties without
    # using any model prediction or error.
    order = np.lexsort((dev_indices, score))
    labels = np.empty(len(score), dtype=np.int64)
    cuts = np.linspace(0, len(score), 4, dtype=np.int64)
    for level in range(3):
        labels[order[cuts[level] : cuts[level + 1]]] = level
    names = ("low", "medium", "high")
    protocol = {
        "status": "strata_frozen_before_loading_model_outcomes",
        "source": "last observed state at history index 19 only",
        "edge_radius_m": EDGE_RADIUS_M,
        "ttc_risk": "1/(1+minimum_positive_TTC/5s), zero without a closing pair within 45m",
        "normalization": "empirical mid-rank on fixed 991-scene dev set",
        "weights": WEIGHTS,
        "assignment": "stable score sort split into exact near-equal thirds; dataset index breaks score ties",
        "no_outcome_used_for_stratification": True,
        "no_test": True,
        "dev_indices_sha256": hashlib.sha256(dev_indices.tobytes()).hexdigest(),
        "data_sha256": sha256(DATA),
        "source_sha256": sha256(Path(__file__)),
        "strata": {},
    }
    for level, name in enumerate(names):
        rows = np.flatnonzero(labels == level)
        protocol["strata"][name] = {
            "scenes": int(len(rows)),
            "score": summarize_values(score[rows]),
            "features": {key: summarize_values(value[rows]) for key, value in features.items()},
            "dev_rows": rows.tolist(),
            "dataset_indices": dev_indices[rows].tolist(),
            "time_blocks": sorted(np.unique(starts[rows] // 200).astype(int).tolist()),
        }
    output = RESULTS / "interaction_stratification_protocol.json"
    output.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dev_indices, starts, score, labels, features, protocol, output


def load_arm(arm, dev_indices):
    with np.load(RESULTS / f"{arm}_graph_full_dev.npz", allow_pickle=False) as data:
        assert np.array_equal(data["indices"], dev_indices)
        return {snr: data[str(snr)].copy() for snr in SNRS}


def metrics(arrays, rows):
    by_snr = {}
    for snr in SNRS:
        values = arrays[snr][rows]
        valid_targets = values[:, 2].sum()
        by_snr[str(snr)] = {
            "ade_m": float(values[:, 0].sum() / (20 * valid_targets)),
            "fde_m": float(values[:, 1].sum() / valid_targets),
            "valid_target_weight": int(valid_targets),
        }
    aggregate = {
        metric: float(np.mean([value[metric] for value in by_snr.values()]))
        for metric in ("ade_m", "fde_m")
    }
    return {"aggregate": aggregate, "by_snr": by_snr}


def reduction(reference, candidate):
    return {
        metric: 100.0 * (reference[metric] - candidate[metric]) / reference[metric]
        for metric in ("ade_m", "fde_m")
    }


def bootstrap(reference, candidate, rows, blocks, repetitions=10000):
    unique = np.unique(blocks[rows])
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled = np.concatenate([rows[blocks[rows] == value] for value in chosen])
        ref = metrics(reference, sampled)["aggregate"]
        cand = metrics(candidate, sampled)["aggregate"]
        draws.append([reduction(ref, cand)["ade_m"], reduction(ref, cand)["fde_m"]])
    draws = np.asarray(draws)
    return {
        "unique_time_blocks": int(len(unique)),
        "repetitions": repetitions,
        "ade_m": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
        "fde_m": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
    }


def main():
    dev_indices, starts, score, labels, features, protocol, protocol_path = build_strata()
    # Model outcomes are deliberately loaded only after the protocol is written.
    predictions = {arm: load_arm(arm, dev_indices) for arm in ARMS}
    blocks = (starts // 200).astype(np.int64)
    report = {
        "seed": 2026,
        "stratification_protocol": str(protocol_path),
        "stratification_protocol_sha256": sha256(protocol_path),
        "model_outcomes_loaded_after_protocol_was_written": True,
        "warning": "Bootstrap intervals condition on one training seed; strata are descriptive subgroup analyses.",
        "no_retraining": True,
        "no_llm": True,
        "no_test": True,
        "strata": {},
    }
    comparisons = (
        ("quantum_vs_classical", "classical", "quantum"),
        ("quantum_vs_plain", "plain", "quantum"),
        ("classical_vs_plain", "plain", "classical"),
    )
    for level, name in enumerate(("low", "medium", "high")):
        rows = np.flatnonzero(labels == level)
        arm_metrics = {arm: metrics(predictions[arm], rows) for arm in ARMS}
        result = {
            "scenes": int(len(rows)),
            "complexity_score": summarize_values(score[rows]),
            "features": {key: summarize_values(value[rows]) for key, value in features.items()},
            "metrics": arm_metrics,
            "comparisons": {},
        }
        for comparison, reference_arm, candidate_arm in comparisons:
            point = reduction(
                arm_metrics[reference_arm]["aggregate"], arm_metrics[candidate_arm]["aggregate"]
            )
            result["comparisons"][comparison] = {
                "positive_means_candidate_better": True,
                "reduction_percent": point,
                "paired_time_block_bootstrap_95_ci": bootstrap(
                    predictions[reference_arm], predictions[candidate_arm], rows, blocks
                ),
            }
        report["strata"][name] = result
    output = RESULTS / "interaction_stratified_evaluation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
