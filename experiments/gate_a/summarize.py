from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def reduction(reference, candidate, metric):
    return 100.0 * (reference[metric] - candidate[metric]) / reference[metric]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_A_FINAL_SUMMARY_20260921.json")
    parser.add_argument("--bootstrap", default="reports/task_redesign/SIND_GATE_A_PAIRED_BOOTSTRAP_20260921.json")
    args = parser.parse_args()
    bootstrap = json.loads((ROOT / args.bootstrap).read_text())
    seeds, required, diagnostics = {}, {}, {}
    cohorts = ("full2", "ic2", "full4", "ic4")
    for seed in (2026, 2027):
        run = json.loads((ROOT / f"reports/task_redesign/sind_gate_a_seed{seed}/summary.json").read_text())
        models = run["models"]
        seed_result = {"metrics": {name: result["validation"] for name, result in models.items()},
                       "cv_baseline": run["cv_baseline"], "comparisons": {},
                       "training": {name: {key: result[key] for key in
                                             ("parameters", "total_train_samples", "actual_train_samples",
                                              "train_fraction", "batch_size", "optimizer_updates",
                                              "epochs_completed", "checkpoint_epoch", "runtime_s",
                                              "gpu_peak_memory_mb", "undertraining_warning")}
                                    for name, result in models.items()}}
        self_metrics = models["self"]["validation"]
        for name in ("own", "pool", "graph", "all_graph"):
            candidate = models[name]["validation"]
            seed_result["comparisons"][f"{name}_vs_self"] = {
                cohort: {metric + "_reduction_percent": reduction(self_metrics[cohort], candidate[cohort], metric)
                         for metric in ("ade_m", "fde_m", "J_m")} for cohort in cohorts}
        for reference, candidate in (("own", "graph"), ("pool", "graph"),
                                     ("pool", "all_graph"), ("graph", "all_graph")):
            ref, cand = models[reference]["validation"], models[candidate]["validation"]
            seed_result["comparisons"][f"{candidate}_vs_{reference}"] = {
                cohort: {metric + "_reduction_percent": reduction(ref[cohort], cand[cohort], metric)
                         for metric in ("ade_m", "fde_m", "J_m")} for cohort in cohorts}
        seeds[str(seed)] = seed_result
        m = seed_result["metrics"]
        required[f"{seed}_graph_beats_self_ic4"] = all(m["graph"]["ic4"][q] < m["self"]["ic4"][q]
                                                          for q in ("ade_m", "fde_m"))
        required[f"{seed}_graph_beats_own_ic4"] = all(m["graph"]["ic4"][q] < m["own"]["ic4"][q]
                                                         for q in ("ade_m", "fde_m"))
        required[f"{seed}_all_graph_beats_self_ic4"] = all(m["all_graph"]["ic4"][q] < m["self"]["ic4"][q]
                                                              for q in ("ade_m", "fde_m"))
        paired = bootstrap["seeds"][str(seed)]["ic4"]
        required[f"{seed}_graph_vs_self_ci_positive"] = all(
            paired["self_minus_graph"][q]["ci95_m"][0] > 0 for q in ("ade", "fde"))
        required[f"{seed}_all_graph_vs_self_ci_positive"] = all(
            paired["self_minus_all_graph"][q]["ci95_m"][0] > 0 for q in ("ade", "fde"))
        graph_gain = seed_result["comparisons"]["graph_vs_self"]
        diagnostics[f"{seed}_ic4_gain_not_weaker_than_full4"] = all(
            graph_gain["ic4"][q + "_reduction_percent"] >= graph_gain["full4"][q + "_reduction_percent"]
            for q in ("ade_m", "fde_m"))
        diagnostics[f"{seed}_graph_beats_pool_ic4"] = all(m["graph"]["ic4"][q] < m["pool"]["ic4"][q]
                                                          for q in ("ade_m", "fde_m"))
        diagnostics[f"{seed}_all_graph_beats_graph_ic4"] = all(m["all_graph"]["ic4"][q] < m["graph"]["ic4"][q]
                                                              for q in ("ade_m", "fde_m"))
        diagnostics[f"{seed}_both_cities_graph_direction_positive"] = all(
            m["graph"][f"city{city}_ic4"][q] < m["self"][f"city{city}_ic4"][q]
            for city in (0, 1) for q in ("ade_m", "fde_m"))
    if all(required.values()):
        decision = "PASS"
    elif all(value for key, value in required.items() if "ci_positive" not in key):
        decision = "CONDITIONAL PASS"
    else:
        decision = "FAIL"
    output = {
        "gate": "SinD Gate A: Task / Interaction Necessity", "decision": decision,
        "qualification": "two-seed sufficiently-trained validation pilot; sealed test remains unopened",
        "question": "Does history-side neighbor information provide reproducible predictive value beyond a strong target-only model and matched capacity controls on SinD-IC4?",
        "required_conditions": required, "diagnostics": diagnostics,
        "paired_bootstrap": args.bootstrap, "seeds": seeds,
    }
    path = ROOT / args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"decision": decision, "required_conditions": required, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
