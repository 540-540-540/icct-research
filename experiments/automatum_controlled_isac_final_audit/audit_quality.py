"""Final Audit L/M and §20: audit of the provisional state-quality percentage and candidate
normalization scales for phase 2. Read-only: the quality formula in frontend is not modified."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final_audit import common  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_ge2_errors() -> dict:
    """Pooled n_bs>=2 RMSE per SNR from the BS-count table (avoids the 1-BS geometry tail)."""
    rows = load_rows(common.OUT_DIR / "bs_count_error_scale.csv")
    output = {}
    for split in ("train", "val"):
        output[split] = {}
        for level in common.LEVELS:
            numerator_position = numerator_velocity = count = 0.0
            for entry in rows:
                if entry["split"] != split or float(entry["snr_db"]) != level:
                    continue
                if int(entry["n_bs"]) not in (2, 3) or not entry.get("position_rmse_m"):
                    continue
                n = float(entry["n"])
                numerator_position += n * float(entry["position_rmse_m"]) ** 2
                numerator_velocity += n * float(entry["velocity_rmse_mps"]) ** 2
                count += n
            output[split][str(level)] = {
                "n": int(count),
                "position_rmse_m": math.sqrt(numerator_position / count) if count else None,
                "velocity_rmse_mps": math.sqrt(numerator_velocity / count) if count else None}
    return output


def quality(position_error: float, velocity_error: float, p_ref: float, v_ref: float) -> float:
    return 100.0 * math.exp(-math.sqrt((position_error / p_ref) ** 2
                                       + (velocity_error / v_ref) ** 2))


def main() -> int:
    scale = json.loads((common.OUT_DIR / "prediction_cohort_scale.json").read_text())
    pooled = pooled_ge2_errors()
    common.write_json(common.OUT_DIR / "quality_pooled_ge2.json", pooled)
    train_scale = scale["splits"]["train"]
    median_speed = train_scale["speed"]["all"]["p50"]
    median_frame = train_scale["single_frame_displacement_m"]["p50"]
    median_2s = train_scale["history_span_20frame_m"]["p50"]
    median_nn = train_scale["nearest_neighbor_m"]["all"]["p50"]
    ten_frame_seconds = scale["ten_frame_seconds"]

    reference_grid = (0.25, 0.5, 1.0, 2.0)
    lines = []
    lines.append("# Route-B state-quality 公式审计与第二阶段归一化候选（AUDIT ONLY）\n")
    lines.append("本文件只做审计，**未修改** `frontend/controlled_isac/state_quality.py`、"
                 "`configs/automatum_controlled_isac.json` 或任何 sensing 参数。\n")
    lines.append("## 1. 当前公式\n")
    lines.append("```text\n")
    lines.append("s = sqrt((Δp / 1.0 m)² + (Δv / 1.0 m/s)²);  quality = 100 · exp(-s)\n")
    lines.append("```\n")
    lines.append("## 2. 它不是“准确率”\n")
    lines.append("- 它是**自声明的评分函数**，不度量 sensing 与任何外部真值来源的“正确率”；"
                 "所有误差都来自同一套受控噪声模型。\n")
    lines.append("- `quality` 不区分 position 与 velocity 的物理量纲，1 m 与 1 m/s 是人为选择的"
                 "两把尺子；换尺子会整体平移百分数（见 §4）。\n")
    lines.append("- 它逐状态计算后平均，低误差状态数量多、高速度状态数量少，平均值会被状态构成影响。\n")
    lines.append("## 3. 为什么 1 m / 1 m/s 是人为尺度\n")
    lines.append(f"- 数据集尺度：median speed **{median_speed:.2f} m/s**，median 单帧位移 "
                 f"**{median_frame:.3f} m**，median 1 s 位移 **{train_scale['ten_frame_displacement_m']['p50']:.2f} m**"
                 f"（真实 {ten_frame_seconds:.3f} s），median 2 s history 跨度 **{median_2s:.2f} m**，"
                 f"median 最近邻 **{median_nn:.2f} m**。\n")
    lines.append("- 1 m 位置尺子约为中位单帧位移的 0.76 倍、2 s 跨度的 5%；1 m/s 速度尺子约为中位速度的 7.6%。"
                 "“91%/45%” 主要反映这两个尺子的选择，而不是车辆或下游任务的物理容忍度。\n")
    lines.append("## 4. 换 reference 后百分数如何变化（当前 Candidate B，train，n_bs≥2 汇总 RMSE）\n")
    lines.append("| SNR | pos RMSE (m) | vel RMSE (m/s) | Q(0.25/0.25) | Q(0.5/0.5) | Q(1/1) | Q(2/2) |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for level in common.LEVELS:
        entry = pooled["train"][str(level)]
        row = [f"| {level:+.0f} | {entry['position_rmse_m']:.4f} | {entry['velocity_rmse_mps']:.4f} "]
        for reference in reference_grid:
            value = quality(entry["position_rmse_m"], entry["velocity_rmse_mps"], reference, reference)
            row.append(f"| {value:.1f}")
        lines.append("".join(row) + " |\n")
    lines.append("\n同一批物理误差在更严格的尺子下从“91%/45%”变成低得多的分数，说明百分比本身"
                 "不能作为科学结论，必须连同物理 RMSE 一起报告。\n")
    lines.append("## 5. 第二阶段归一化候选（NOT FROZEN）\n")
    lines.append("### Candidate normalization A：固定任务容忍度（现公式的延续）\n")
    lines.append("- `s = sqrt((Δp/p_ref)² + (Δv/v_ref)²)`，p_ref/v_ref 由下游预测任务敏感度事先声明。\n")
    lines.append("- 优点：简单、与 SNR 无关、无坐标原点奇异性；缺点：尺子人为，容易“调百分数”。\n")
    lines.append("### Candidate normalization B：数据集运动尺度归一化\n")
    lines.append("- 位置除以 median 单帧位移（或 2 s 跨度），速度除以 median speed。\n")
    lines.append("- 优点：与数据尺度绑定；缺点：低速/静止状态（train 约 10.9%，speed<1 m/s）"
                 "会让速度比值爆炸或失去意义，需要在低速区间改用绝对误差。\n")
    lines.append("### Candidate normalization C（推荐给第二阶段讨论）：预测时域影响归一化\n")
    lines.append("- 定义 `s = sqrt((Δp/H)² + (Δv·T_h/H)²)`，其中 `T_h = 20·dt = 2.002 s` 是预测时域，"
                 "`H = median 2 s history 跨度`（train ≈ 20.0 m）。\n")
    lines.append("- 语义：把当前状态误差换算成**在预测时域内可能造成的轨迹位移偏差**，位置与速度同量纲，"
                 "低速状态自动退化为绝对误差贡献，无原点奇异性；缺点：依赖数据集与固定时域。\n")
    lines.append("### 推荐\n")
    lines.append("- 主候选：**C**（可解释为“预测时域诱导位移的相对量级”）；备选：**A**（简单、可声明）。\n")
    lines.append("- 不建议直接采用 B：低速车辆占比较高，百分比在低速端不稳定。\n")
    lines.append("- 以上全部 **NOT FROZEN**，由第二阶段决定。\n")
    (common.OUT_DIR / "quality_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pooled ge2 errors: {json.dumps(pooled['train'], indent=1)}")
    print(f"wrote {common.OUT_DIR / 'quality_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())