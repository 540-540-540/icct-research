from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/task_redesign"
METRICS = ("ade_m", "fde_m", "J_m")

CANDIDATES = {
    "q_full_ema_fw050": "sind_gate_b_plus_robust_ema999_seed{seed}",
    "q_full_ema_fw075": "sind_gate_b_plus_robust_ema999_fw075_seed{seed}",
    "q_full_ema_fw100": "sind_gate_b_plus_robust_ema999_fw100_seed{seed}",
    "q_latent_ema_fw075": "sind_gate_b_plus_robust_quantum_latent_attention_fw075_seed{seed}",
    "q_wide_ema_fw075_fixed_order": "sind_gate_b_plus_robust_wide_quantum_fw075_seed{seed}",
    "q_full_distill025_fw075_fixed_order": "sind_gate_b_plus_robust_distill025_fw075_seed{seed}",
    "q_latent_distill025_fw075_fixed_order": "sind_gate_b_plus_robust_latent_distill025_fw075_seed{seed}",
    "strong_graph_full_ema": "sind_gate_b_plus_robust_graph_seed{seed}",
    "all_graph_full_ema": "sind_gate_b_plus_robust_all_graph_seed{seed}",
}


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "range": max(values) - min(values),
    }


def main() -> None:
    rows: dict[str, dict] = {}
    for name, pattern in CANDIDATES.items():
        seeds = {}
        for seed in range(2026, 2031):
            path = REPORT_ROOT / pattern.format(seed=seed) / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            seeds[str(seed)] = {
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "train_samples": summary["train_samples"],
                "ema_decay": summary.get("ema_decay"),
                "order_seed": summary.get("order_seed", summary.get("seed")),
                **{metric: summary["validation"]["ic4"][metric] for metric in METRICS},
            }
        if not seeds:
            continue
        rows[name] = {
            "seeds": seeds,
            "aggregate": {
                metric: mean_std([row[metric] for row in seeds.values()]) for metric in METRICS
            },
        }

    comparisons = {}
    baseline = rows.get("all_graph_full_ema", {}).get("seeds", {})
    for name, result in rows.items():
        if not name.startswith("q_"):
            continue
        common = sorted(set(result["seeds"]) & set(baseline))
        if not common:
            continue
        comparisons[name] = {
            "common_seeds": [int(seed) for seed in common],
            "all_graph_minus_q": {
                metric: mean_std([baseline[seed][metric] - result["seeds"][seed][metric]
                                  for seed in common])
                for metric in METRICS
            },
            "directly_matched_order_seed": all(
                baseline[seed]["order_seed"] == result["seeds"][seed]["order_seed"] for seed in common
            ),
        }

    payload = {
        "status": "DEVELOPMENT_ONLY_NOT_FORMAL_EVIDENCE",
        "test_accessed": False,
        "seed_policy": {
            "selection_seeds": [2026, 2027],
            "frozen_confirmation_seeds": [2026, 2027, 2028, 2029, 2030],
            "fixed_batch_order_seed": 20260921,
            "rule": "freeze one recipe, run every predeclared seed, report all outcomes",
        },
        "candidates": rows,
        "paired_development_comparisons": comparisons,
        "warning": "A comparison with directly_matched_order_seed=false is descriptive only.",
    }
    out = REPORT_ROOT / "SIND_GATE_B_PLUS_ROBUSTNESS_SUMMARY_20260921.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
