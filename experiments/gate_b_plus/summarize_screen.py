from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/task_redesign"


def main() -> None:
    candidates: dict[str, dict] = {}
    pattern = "sind_gate_b_plus_screen_*_seed*/summary.json"
    for path in sorted(REPORT_ROOT.glob(pattern)):
        data = json.loads(path.read_text())
        seed = str(data["seed"])
        weight = data.get("fde_weight", 0.5)
        label = data["mode"] if weight == 0.5 else f"{data['mode']}_fde_weight_{weight:g}"
        candidates.setdefault(label, {"seeds": {}})["seeds"][seed] = {
            "status": data["status"], "epochs": data["epochs_completed"],
            "checkpoint_epoch": data["checkpoint_epoch"], "parameters": data["parameters"],
            "ic4": data["validation"]["ic4"], "path": str(path.relative_to(ROOT)),
        }

    gate_a = json.loads((REPORT_ROOT / "SIND_GATE_A_FINAL_SUMMARY_20260921.json").read_text())
    ranking = []
    for label, candidate in candidates.items():
        if set(candidate["seeds"]) != {"2026", "2027"}:
            continue
        mean = {metric: sum(candidate["seeds"][seed]["ic4"][metric] for seed in ("2026", "2027")) / 2
                for metric in ("ade_m", "fde_m", "J_m")}
        candidate["two_seed_mean_ic4"] = mean
        candidate["strong_graph_relative_improvement_pct"] = {
            seed: {
                metric: 100.0 * (
                    gate_a["seeds"][seed]["metrics"]["graph"]["ic4"][metric]
                    - candidate["seeds"][seed]["ic4"][metric]
                ) / gate_a["seeds"][seed]["metrics"]["graph"]["ic4"][metric]
                for metric in ("ade_m", "fde_m")
            }
            for seed in ("2026", "2027")
        }
        candidate["worst_seed_primary_margin_pct"] = min(
            candidate["strong_graph_relative_improvement_pct"][seed][metric]
            for seed in ("2026", "2027") for metric in ("ade_m", "fde_m")
        )
        candidate["beats_strong_graph_both_primary_metrics_both_seeds"] = all(
            candidate["seeds"][seed]["ic4"][metric]
            < gate_a["seeds"][seed]["metrics"]["graph"]["ic4"][metric]
            for seed in ("2026", "2027") for metric in ("ade_m", "fde_m"))
        ranking.append((candidate["worst_seed_primary_margin_pct"], label))
    ranking.sort(key=lambda row: (
        -row[0],
        candidates[row[1]]["two_seed_mean_ic4"]["ade_m"],
        candidates[row[1]]["two_seed_mean_ic4"]["fde_m"],
        row[1],
    ))
    winner = ranking[0][1]
    output = {"scope": "development screen; not formal evidence", "test_accessed": False,
              "selection_metric": ("maximize the worst per-seed relative margin over Strong Graph "
                                   "across co-primary IC4 ADE and FDE; mean metrics are tie-breakers"),
              "ranking": [{"candidate": label, "worst_seed_primary_margin_pct": margin}
                          for margin, label in ranking],
              "selected_for_next_iteration": winner, "candidates": candidates}
    (REPORT_ROOT / "SIND_GATE_B_PLUS_SCREEN_SUMMARY_20260921.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = ["# SinD Gate B+ quantum-primary development screen", "", "该表仅用于开发筛选，不是正式论文证据；test 未访问。", "",
             "| rank | candidate | mean ADE | mean FDE4 | mean J (aux) | worst-seed margin vs Strong Graph | wins ADE+FDE in both seeds |",
             "|---:|---|---:|---:|---:|---:|---|" ]
    for rank, (margin, label) in enumerate(ranking, 1):
        c = candidates[label]; mean = c["two_seed_mean_ic4"]
        lines.append(f"| {rank} | {label} | {mean['ade_m']:.4f} | {mean['fde_m']:.4f} | {mean['J_m']:.4f} | {margin:+.3f}% | {c['beats_strong_graph_both_primary_metrics_both_seeds']} |")
    lines += ["", f"Selected for next iteration: `{winner}`.", "",
              "完整保留所有候选；ADE/FDE 为共同主指标。选择依据为候选相对 Strong Graph 的最差 seed、最差主指标改善率；均值仅用于同分排序，J 只作辅助描述。",
              "进入正式确认前，必须补齐 matched classical comparator，并在未参与本次筛选的新 seed 上冻结验证。", ""]
    (ROOT / "docs/task_redesign/CODEX_SIND_GATE_B_PLUS_SCREEN_REPORT_20260921.md").write_text("\n".join(lines))
    print(json.dumps({"winner": winner, "ranking": output["ranking"]}, indent=2))


if __name__ == "__main__":
    main()
