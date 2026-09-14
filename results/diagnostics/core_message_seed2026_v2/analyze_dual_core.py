"""Paired full-dev and frozen-stratum analysis for the two-core experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
DUAL = BASE / "dual_core_seed2026_v1"
SINGLE = BASE / "converged_protocol_seed2026_v1"
DATA = ROOT / "data/multitarget_lankershim_v1.npz"
SNRS = (5, 10, 15, 20)
BOOTSTRAP_SEED = 2026090717
REPETITIONS = 10000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prediction(path, expected_indices=None):
    with np.load(path, allow_pickle=False) as data:
        indices = data["indices"].copy()
        arrays = {snr: data[str(snr)].copy() for snr in SNRS}
    if expected_indices is not None:
        assert np.array_equal(indices, expected_indices)
    return indices, arrays


def metrics(arrays, rows):
    by_snr = {}
    for snr in SNRS:
        values = arrays[snr][rows]
        targets = values[:, 2].sum()
        by_snr[str(snr)] = {
            "ade_m": float(values[:, 0].sum() / (20.0 * targets)),
            "fde_m": float(values[:, 1].sum() / targets),
            "valid_target_weight": int(targets),
        }
    aggregate = {
        key: float(np.mean([value[key] for value in by_snr.values()]))
        for key in ("ade_m", "fde_m")
    }
    return {"aggregate": aggregate, "by_snr": by_snr}


def reduction(reference, candidate):
    return {
        key: 100.0 * (reference[key] - candidate[key]) / reference[key]
        for key in ("ade_m", "fde_m")
    }


def paired_bootstrap(reference, candidate, rows, blocks, seed):
    unique = np.unique(blocks[rows])
    rng = np.random.default_rng(seed)
    draws = np.empty((REPETITIONS, 2), dtype=np.float64)
    for iteration in range(REPETITIONS):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled = np.concatenate([rows[blocks[rows] == block] for block in chosen])
        ref = metrics(reference, sampled)["aggregate"]
        cand = metrics(candidate, sampled)["aggregate"]
        change = reduction(ref, cand)
        draws[iteration] = (change["ade_m"], change["fde_m"])
    return {
        "unique_time_blocks": int(len(unique)),
        "repetitions": REPETITIONS,
        "ade_m": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
        "fde_m": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
    }


def comparison(reference_name, candidate_name, predictions, rows, blocks, seed):
    ref_metrics = metrics(predictions[reference_name], rows)
    cand_metrics = metrics(predictions[candidate_name], rows)
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "positive_reduction_means_candidate_better": True,
        "reduction_percent": reduction(ref_metrics["aggregate"], cand_metrics["aggregate"]),
        "paired_time_block_bootstrap_95_ci": paired_bootstrap(
            predictions[reference_name], predictions[candidate_name], rows, blocks, seed
        ),
    }


def main():
    completed = json.loads((DUAL / "completed.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    indices, quantum_dual = load_prediction(DUAL / "quantum_dual_graph_full_dev.npz")
    _, classical_dual = load_prediction(DUAL / "classical_dual_graph_full_dev.npz", indices)
    _, quantum_single = load_prediction(SINGLE / "quantum_graph_full_dev.npz", indices)
    _, classical_single = load_prediction(SINGLE / "classical_graph_full_dev.npz", indices)
    _, plain = load_prediction(SINGLE / "plain_graph_full_dev.npz", indices)
    predictions = {
        "plain": plain,
        "classical_single": classical_single,
        "quantum_single": quantum_single,
        "classical_dual": classical_dual,
        "quantum_dual": quantum_dual,
    }
    with np.load(DATA, allow_pickle=False) as data:
        starts = data["train_start_index"][indices].copy()
    blocks = (starts // 200).astype(np.int64)
    strata_path = SINGLE / "interaction_stratification_protocol.json"
    strata = json.loads(strata_path.read_text(encoding="utf-8"))["strata"]
    all_rows = np.arange(len(indices), dtype=np.int64)
    comparison_specs = (
        ("quantum_dual_vs_classical_dual", "classical_dual", "quantum_dual"),
        ("quantum_dual_vs_quantum_single", "quantum_single", "quantum_dual"),
        ("classical_dual_vs_classical_single", "classical_single", "classical_dual"),
        ("quantum_dual_vs_plain", "plain", "quantum_dual"),
        ("classical_dual_vs_plain", "plain", "classical_dual"),
    )
    report = {
        "training_seed": 2026,
        "status": "completed_posthoc_analysis",
        "dual_completed_sha256": sha256(DUAL / "completed.json"),
        "frozen_stratification_protocol": str(strata_path),
        "frozen_stratification_protocol_sha256": sha256(strata_path),
        "strata_were_not_recomputed": True,
        "no_retraining": True,
        "no_llm": True,
        "no_test": True,
        "warning": (
            "Intervals condition on one training seed and quantify paired scene/time-block "
            "variation only; they do not quantify training-seed uncertainty."
        ),
        "metrics": {name: metrics(values, all_rows) for name, values in predictions.items()},
        "full_dev_comparisons": {},
        "frozen_interaction_strata": {},
    }
    for offset, (name, reference, candidate) in enumerate(comparison_specs):
        report["full_dev_comparisons"][name] = comparison(
            reference, candidate, predictions, all_rows, blocks, BOOTSTRAP_SEED + offset
        )
    for level, stratum_name in enumerate(("low", "medium", "high")):
        rows = np.asarray(strata[stratum_name]["dev_rows"], dtype=np.int64)
        entry = {
            "scenes": int(len(rows)),
            "metrics": {name: metrics(values, rows) for name, values in predictions.items()},
            "comparisons": {},
        }
        # The primary matched comparison and the depth effect within each family.
        for offset, (name, reference, candidate) in enumerate(comparison_specs[:3]):
            entry["comparisons"][name] = comparison(
                reference,
                candidate,
                predictions,
                rows,
                blocks,
                BOOTSTRAP_SEED + 100 * (level + 1) + offset,
            )
        report["frozen_interaction_strata"][stratum_name] = entry
    output = DUAL / "paired_and_stratified_analysis.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
