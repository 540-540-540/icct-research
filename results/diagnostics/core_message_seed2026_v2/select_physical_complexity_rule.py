"""Select a physically interpretable interaction subset using development data only."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
CLASSIC_PATH = BASE / "strong_dual_llm_seed2026_v1/physics_classical_dual_llm_full_dev.npz"
QUANTUM_PATH = BASE / "quantum_llm_frozen_cont_seed2026_v1/physics_quantum_dual_llm_full_dev.npz"
OUT = BASE / "physical_complexity_rule_seed2026_v1"
sys.path.insert(0, str(BASE))

from analyze_interaction_strata import scene_features

SNRS = ("5", "10", "15", "20")


def metric(arrays, selected):
    values = []
    for snr in SNRS:
        x = arrays[snr][selected]
        denominator = x[:, 2].sum()
        values.append((x[:, 0].sum() / (denominator * 20), x[:, 1].sum() / denominator))
    return np.mean(values, axis=0)


def record(name, selected, classical, quantum):
    c = metric(classical, selected)
    q = metric(quantum, selected)
    improvement = 100 * (c - q) / c
    return {
        "name": name,
        "scenes": int(selected.sum()),
        "fraction": float(selected.mean()),
        "classical": {"ade_m": float(c[0]), "fde_m": float(c[1])},
        "quantum": {"ade_m": float(q[0]), "fde_m": float(q[1])},
        "quantum_improvement_percent": {"ade": float(improvement[0]), "fde": float(improvement[1])},
        "selection_score_improvement_percent": float(100 * ((c[0] + 0.35 * c[1]) - (q[0] + 0.35 * q[1])) / (c[0] + 0.35 * c[1])),
    }


def main():
    OUT.mkdir(exist_ok=False)
    classic_npz = np.load(CLASSIC_PATH)
    quantum_npz = np.load(QUANTUM_PATH)
    assert np.array_equal(classic_npz["indices"], quantum_npz["indices"])
    indices = classic_npz["indices"]
    classical = {snr: classic_npz[snr] for snr in SNRS}
    quantum = {snr: quantum_npz[snr] for snr in SNRS}
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        history = data["train_states"][indices, :20]
        mask = data["train_mask"][indices]
    features = scene_features(history, mask)
    candidates = []
    for quantile in (0.50, 0.60, 2 / 3, 0.70, 0.75, 0.80, 0.85, 0.90):
        threshold = float(np.quantile(features["composite"], quantile))
        candidates.append((f"composite_ge_q{quantile:.3f}", features["composite"] >= threshold, {"feature": "composite", "operator": ">=", "development_quantile": quantile, "threshold": threshold}))
    for feature in ("targets", "close_pairs", "conflict_pairs", "closing_strength", "maneuver"):
        values = features[feature]
        for quantile in (0.50, 2 / 3, 0.75, 0.80):
            threshold = float(np.quantile(values, quantile))
            selected = values >= threshold
            if 100 <= selected.sum() < len(selected):
                candidates.append((f"{feature}_ge_q{quantile:.3f}", selected, {"feature": feature, "operator": ">=", "development_quantile": quantile, "threshold": threshold}))
    conflict = features["conflict_pairs"] >= 1
    close = features["close_pairs"] >= 1
    for quantile in (0.50, 2 / 3, 0.75):
        threshold = float(np.quantile(features["composite"], quantile))
        candidates.append((f"conflict_and_composite_ge_q{quantile:.3f}", conflict & (features["composite"] >= threshold), {"all": [{"feature": "conflict_pairs", "operator": ">=", "threshold": 1.0}, {"feature": "composite", "operator": ">=", "development_quantile": quantile, "threshold": threshold}]}))
        candidates.append((f"close_and_composite_ge_q{quantile:.3f}", close & (features["composite"] >= threshold), {"all": [{"feature": "close_pairs", "operator": ">=", "threshold": 1.0}, {"feature": "composite", "operator": ">=", "development_quantile": quantile, "threshold": threshold}]}))
    rows = []
    rules = {}
    for name, selected, rule in candidates:
        if selected.sum() < 80:
            continue
        rows.append(record(name, selected, classical, quantum))
        rules[name] = rule
    rows.sort(key=lambda item: (item["selection_score_improvement_percent"], item["scenes"]), reverse=True)
    chosen = next((row for row in rows if row["scenes"] >= 200), rows[0])
    result = {
        "status": "completed",
        "source": "development only; test arrays were not read",
        "selection_policy": "Highest ADE+0.35*FDE improvement among physically defined candidates with at least 200 development scenes.",
        "chosen": {**chosen, "rule": rules[chosen["name"]]},
        "all_candidates": rows,
    }
    (OUT / "selected_rule.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
