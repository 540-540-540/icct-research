from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = "raj_weighted_multij_quantum"
C = "raj_multij_johnson"
METRICS = ("ade_m", "fde_m", "J_m")


def reduction(reference: dict, candidate: dict, metric: str) -> float:
    return 100 * (reference[metric] - candidate[metric]) / reference[metric]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_B_FINAL_SUMMARY_20260921.json")
    parser.add_argument("--report", default="docs/task_redesign/CODEX_SIND_GATE_B_FINAL_REPORT_20260921.md")
    parser.add_argument("--bootstrap", default="reports/task_redesign/SIND_GATE_B_PAIRED_BOOTSTRAP_20260921.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/gate_b_raj_migration.json").read_text())
    gate_a = json.loads((ROOT / "reports/task_redesign/SIND_GATE_A_FINAL_SUMMARY_20260921.json").read_text())
    bootstrap = json.loads((ROOT / args.bootstrap).read_text())
    seeds, conditions = {}, {}
    mean_delta = {metric: [] for metric in METRICS}
    for seed in (2026, 2027):
        root = ROOT / f"reports/task_redesign/sind_gate_b_seed{seed}"
        run = json.loads((root / "summary.json").read_text()); models = run["models"]
        q, c = models[Q]["validation"], models[C]["validation"]
        comparison = {view: {metric + "_absolute_gain_m": c[view][metric] - q[view][metric]
                             for metric in METRICS} | {metric + "_relative_gain_percent": reduction(c[view], q[view], metric)
                                                      for metric in METRICS}
                      for view in ("full2", "ic2", "full4", "ic4")}
        for metric in METRICS: mean_delta[metric].append(c["ic4"][metric] - q["ic4"][metric])
        seed_positive = all(q["ic4"][metric] < c["ic4"][metric] for metric in METRICS)
        ci_positive = all(bootstrap["seeds"][str(seed)]["ic4"][name]["ci95_m"][0] > 0
                          for name in ("ade", "fde", "J"))
        converged = all(not models[kind]["undertraining_warning"] for kind in (Q, C))
        conditions[f"{seed}_raj_improves_all_primary_metrics"] = seed_positive
        conditions[f"{seed}_all_primary_bootstrap_ci_positive"] = ci_positive
        conditions[f"{seed}_converged"] = converged
        checkpoints = {kind: {"path": str((root / f"{kind}_best.pt").relative_to(ROOT)),
                              "sha256": file_sha(root / f"{kind}_best.pt")} for kind in (Q, C)}
        external = gate_a["seeds"][str(seed)]["metrics"]
        seeds[str(seed)] = {
            "metrics": {"self": external["self"], "pool": external["pool"], "strong_graph": external["graph"],
                        "all_graph": external["all_graph"], "matched_johnson": c, "raj_qgnn": q},
            "raj_vs_johnson": comparison, "bootstrap": bootstrap["seeds"][str(seed)],
            "training": {kind: {key: models[kind][key] for key in ("parameters", "epochs_completed",
                         "checkpoint_epoch", "optimizer_updates", "runtime_s", "gpu_peak_memory_mb", "undertraining_warning")}
                         for kind in (Q, C)},
            "train_indices_sha256": run["train_indices_sha256"], "source_sha256": run["source_sha256"],
            "config_sha256": run["config_sha256"], "checkpoints": checkpoints,
        }
    mean_gain = {metric: sum(values) / len(values) for metric, values in mean_delta.items()}
    mean_positive = all(value > 0 for value in mean_gain.values())
    any_seed_all_positive = any(conditions[f"{seed}_raj_improves_all_primary_metrics"] for seed in (2026, 2027))
    if all(conditions.values()): decision = "PASS"
    elif mean_positive and any_seed_all_positive: decision = "CONDITIONAL PASS"
    else: decision = "FAIL"
    output = {
        "gate": "SinD Gate B: frozen Raj QGNN formal validation", "decision": decision,
        "question": "Does the frozen Raj Weighted Multi-J QGNN outperform its matched Multi-J Johnson classical core on SinD-IC4?",
        "preregistered_decision_rule": cfg["decision_rule"], "conditions": conditions,
        "two_seed_mean_absolute_gain_m": mean_gain,
        "test_accessed": False, "legacy_2s_to_2s_gain_percent": {"ADE": 3.006, "FDE": 3.307, "J": 3.161},
        "seeds": seeds, "gate_c_ready": decision == "PASS",
    }
    output_path = ROOT / args.output; output_path.write_text(json.dumps(output, indent=2) + "\n")

    def row(seed: int, name: str) -> str:
        m = seeds[str(seed)]["metrics"][name]["ic4"]
        return f"{m['ade_m']:.4f} / {m['fde_m']:.4f} / {m['J_m']:.4f}"
    report = [
        "# SinD Gate B 最终报告：冻结 Raj QGNN 正式验证", "", f"判决：`{decision}`", "",
        "## Primary IC4 table", "", "| seed | Self | Pool | Strong Graph | AllGraph | Matched Johnson | Raj QGNN |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed in (2026, 2027):
        report.append(f"| {seed} | {row(seed,'self')} | {row(seed,'pool')} | {row(seed,'strong_graph')} | {row(seed,'all_graph')} | {row(seed,'matched_johnson')} | {row(seed,'raj_qgnn')} |")
    report += ["", "表中依次为 ADE / FDE4 / J；J 仅用于 checkpoint selection。", "", "## Raj vs matched Johnson", ""]
    for seed in (2026, 2027):
        comp = seeds[str(seed)]["raj_vs_johnson"]["ic4"]; boot = seeds[str(seed)]["bootstrap"]["ic4"]
        report.append(f"- seed {seed}: ADE {comp['ade_m_relative_gain_percent']:+.3f}%、FDE4 {comp['fde_m_relative_gain_percent']:+.3f}%、J {comp['J_m_relative_gain_percent']:+.3f}%。")
        report.append(f"  Paired 95% CI: ADE {boot['ade']['ci95_m']}，FDE4 {boot['fde']['ci95_m']}，J {boot['J']['ci95_m']}。")
    report += ["", "Two-seed mean matched gain (Johnson - Raj): "
               f"ADE {mean_gain['ade_m']:+.6f} m，FDE4 {mean_gain['fde_m']:+.6f} m，J {mean_gain['J_m']:+.6f} m。",
               "FDE4 均值仍为负，因此不满足 CONDITIONAL PASS 的三指标均正条件。", "",
               "## Absolute-reference finding", "",
               "- Raj 未在两个 seed 上同时超过 Strong Graph 和 AllGraph。",
               "- seed 2026：Raj 的 ADE/FDE4/J 均劣于 AllGraph，FDE4/J 也劣于 Strong Graph。",
               "- seed 2027：Raj ADE 优于两个外部图基线，但 FDE4/J 仍劣于两者。",
               "- 因此本轮不能声称 QGNN top1、量子优势或不可替代性。", "",
               "## Statistical validity scan", "",
               "- Selection: 两个预登记 seed 全部报告，无 seed/cherry-pick。",
               "- Checkpoint: 两模型共用 validation IC4 J 选择规则，未访问 test。",
               "- Multiplicity: Gate 按预登记的 ADE/FDE4/J 联合条件判定，未从子组中挑显著结果。",
               "- Dependence: 95% CI 在 prediction-origin 层面聚类 bootstrap，不把同场景样本当独立。",
               "- Causality: 只报告预测误差差异，不声称交通因果。",
               "- Comparator: matched Johnson 是主对照，Strong Graph/AllGraph 只是绝对参照。", "",
               "## Eight required answers", "",
               f"1. Raj 是否稳定优于 Johnson：{decision}；以两 seed 三指标与 paired CI 的预登记规则为准。",
               "2. ADE 提升：见上方 seed-wise relative gain。", "3. FDE4 提升：见上方 seed-wise relative gain。",
               "4. 两 seed 一致性：见 machine summary conditions。", "5. Paired uncertainty：5,000 次 prediction-origin cluster bootstrap。",
               "6. Strong Graph / AllGraph 位置：主表将其作为 absolute reference，不冒充 matched comparison。",
               f"7. 参数：Raj {seeds['2026']['training'][Q]['parameters']['total']:,}，Johnson {seeds['2026']['training'][C]['parameters']['total']:,}；Johnson 更多。",
               "8. Legacy 2s->2s 约 3% gain 与 IC4 的变化：只作历史比较，不并入新判决。", "",
               "## Boundaries", "", "- test 保持 sealed。", "- Raj core 未改架构、未改 j、未改数据 population 或 checkpoint rule。",
               "- 只有当 Raj 同时胜过 matched Johnson 和外部 Strong Graph/AllGraph，才可写成超过全部已测经典参考。",
               "- Gate C 仅在 Gate B PASS 时 ready；本报告不启动 GPT-2。", ""]
    report_path = ROOT / args.report; report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report))
    print(json.dumps({"decision": decision, "conditions": conditions, "gate_c_ready": output["gate_c_ready"]}, indent=2))


if __name__ == "__main__": main()
