from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def reduction(reference, candidate, metric):
    return 100.0 * (reference[metric] - candidate[metric]) / reference[metric]


def main():
    seeds = {}
    for seed in (2026, 2027):
        path = ROOT / f"reports/task_redesign/gate_a_seed{seed}/summary.json"
        run = json.loads(path.read_text())
        models = run["models"]
        seed_result = {"metrics": {}, "comparisons": {}}
        for name, result in models.items():
            seed_result["metrics"][name] = {k: result["validation"][k]
                                             for k in ("full2", "ic2", "full4", "ic4",
                                                       "k04", "k14", "k2plus4", "closing14", "cpa54")}
        self_metrics = models["self"]["validation"]
        for name in ("own", "pool", "graph", "all_graph"):
            candidate = models[name]["validation"]
            seed_result["comparisons"][name] = {
                cohort: {metric + "_reduction_percent": reduction(self_metrics[cohort], candidate[cohort], metric)
                         for metric in ("ade_m", "fde_m", "J_m")}
                for cohort in ("full2", "ic2", "full4", "ic4")}
        graph = models["graph"]["validation"]
        for control in ("own", "pool"):
            baseline = models[control]["validation"]
            seed_result["comparisons"][f"graph_vs_{control}"] = {
                cohort: {metric + "_reduction_percent": reduction(baseline[cohort], graph[cohort], metric)
                         for metric in ("ade_m", "fde_m", "J_m")}
                for cohort in ("full2", "ic2", "full4", "ic4")}
        all_graph = models["all_graph"]["validation"]
        seed_result["comparisons"]["all_graph_vs_graph"] = {
            cohort: {metric + "_reduction_percent": reduction(graph[cohort], all_graph[cohort], metric)
                     for metric in ("ade_m", "fde_m", "J_m")}
            for cohort in ("full2", "ic2", "full4", "ic4")}
        seeds[str(seed)] = seed_result

    conditions = {}
    bootstrap_path = ROOT / "reports/task_redesign/GATE_A_PAIRED_BOOTSTRAP_20260921.json"
    bootstrap = json.loads(bootstrap_path.read_text()) if bootstrap_path.exists() else None
    for seed, result in seeds.items():
        m = result["metrics"]
        conditions[f"{seed}_graph_beats_self_ic4"] = all(m["graph"]["ic4"][q] < m["self"]["ic4"][q] for q in ("ade_m", "fde_m"))
        conditions[f"{seed}_graph_beats_own_ic4"] = all(m["graph"]["ic4"][q] < m["own"]["ic4"][q] for q in ("ade_m", "fde_m"))
        conditions[f"{seed}_graph_beats_pool_ic4"] = all(m["graph"]["ic4"][q] < m["pool"]["ic4"][q] for q in ("ade_m", "fde_m"))
        conditions[f"{seed}_ic4_gain_exceeds_full4"] = all(
            result["comparisons"]["graph"]["ic4"][q + "_reduction_percent"]
            > result["comparisons"]["graph"]["full4"][q + "_reduction_percent"]
            for q in ("ade_m", "fde_m"))
        conditions[f"{seed}_all_graph_beats_limited_graph_ic4"] = all(
            m["all_graph"]["ic4"][q] < m["graph"]["ic4"][q] for q in ("ade_m", "fde_m"))
        if bootstrap:
            paired = bootstrap["seeds"][seed]["ic4"]
            conditions[f"{seed}_all_graph_vs_self_ci_positive"] = all(
                paired["self_minus_all_graph"][q]["ci95_m"][0] > 0 for q in ("ade", "fde"))
            conditions[f"{seed}_all_graph_vs_pool_ci_positive"] = all(
                paired["pool_minus_all_graph"][q]["ci95_m"][0] > 0 for q in ("ade", "fde"))
    decision = "PASS" if all(conditions.values()) else "FAIL"
    output = {"gate": "A: Task / Interaction Necessity", "decision": decision,
              "qualification": "two-seed bounded pilot on validation; sealed test remains unopened",
              "question": "Does history-only neighbor information add predictive value beyond strong target-only and capacity controls?",
              "conditions": conditions, "paired_bootstrap": str(bootstrap_path.relative_to(ROOT)) if bootstrap else None,
              "seeds": seeds}
    path = ROOT / "reports/task_redesign/GATE_A_FINAL_SUMMARY_20260921.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"decision": decision, "conditions": conditions}, indent=2))


if __name__ == "__main__": main()
