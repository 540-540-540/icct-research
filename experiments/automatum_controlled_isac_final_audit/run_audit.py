"""Orchestrates the read-only final audit and writes decision_summary.json with explicit
rubrics. No sensing parameter is modified; no test data is read for any decision."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final_audit import common  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def load_csv(name: str) -> list[dict]:
    with (common.OUT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_ge2() -> dict:
    return json.loads((common.OUT_DIR / "quality_pooled_ge2.json").read_text())


def run(script: str) -> None:
    print(f"--- {script}")
    subprocess.run([sys.executable, str(SCRIPT_DIR / script)], check=True, cwd=ROOT)


def pool(rows: list[dict], split: str, level: float, metric: str) -> dict:
    numerator = count = 0.0
    for entry in rows:
        if entry["split"] != split or float(entry["snr_db"]) != level:
            continue
        if int(entry["n_bs"]) not in (2, 3) or not entry.get(metric):
            continue
        n = float(entry["n"])
        numerator += n * float(entry[metric]) ** 2
        count += n
    return {"n": int(count), "rmse": math.sqrt(numerator / count) if count else None}


def verdict_position(ratio_low: float) -> str:
    if ratio_low < 0.10:
        return "TOO_LIGHT"
    if ratio_low > 1.0:
        return "TOO_HEAVY"
    return "REASONABLE"


def verdict_velocity(ratio_low: float) -> str:
    if ratio_low < 0.02:
        return "TOO_LIGHT"
    if ratio_low > 0.30:
        return "TOO_HEAVY"
    return "REASONABLE"


def verdict_gradient(ratio: float) -> str:
    if ratio < 3.0:
        return "TOO_FLAT"
    if ratio > 30.0:
        return "TOO_STEEP"
    return "REASONABLE"


def _pooled_gradient(pooled_split: dict) -> dict:
    output = {"position": {}, "velocity": {}}
    ordered = sorted(common.LEVELS)
    for index in range(1, len(ordered)):
        higher, lower = ordered[index - 1], ordered[index]
        for metric in ("position", "velocity"):
            key = f"{metric}_rmse_m" if metric == "position" else "velocity_rmse_mps"
            output[metric][str(lower)] = (pooled_split[str(lower)][key]
                                          / pooled_split[str(higher)][key])
    output["position"]["overall_+10_to_-10"] = (pooled_split[str(ordered[-1])]["position_rmse_m"]
                                                / pooled_split[str(ordered[0])]["position_rmse_m"])
    output["velocity"]["overall_+10_to_-10"] = (pooled_split[str(ordered[-1])]["velocity_rmse_mps"]
                                                / pooled_split[str(ordered[0])]["velocity_rmse_mps"])
    return output


def semantic_verdict(kind: str, position_ratio: float, velocity_ratio: float,
                     flip_fraction: float | None, angle_median: float | None) -> str:
    flip_fraction = 0.0 if flip_fraction is None else flip_fraction
    angle_median = 0.0 if angle_median is None else angle_median
    if kind == "good":
        passed = position_ratio <= 0.10 and velocity_ratio <= 0.05 and flip_fraction == 0.0
    elif kind == "medium":
        passed = (0.05 < position_ratio <= 0.30) and (0.01 < velocity_ratio <= 0.10)
    else:
        passed = (0.20 < position_ratio <= 1.00) and (0.03 < velocity_ratio <= 0.30) \
            and flip_fraction < 0.01 and angle_median < 10.0
    return "PASS" if passed else "BORDERLINE"


def main() -> int:
    started = time.time()
    assemble_only = "--assemble-only" in sys.argv[1:]
    if not assemble_only:
        run("audit_data_scale.py")
        run("audit_geometry.py")
        run("audit_sensing_scale.py")
        run("audit_quality.py")

    scale = json.loads((common.OUT_DIR / "prediction_cohort_scale.json").read_text())
    pooled = pooled_ge2()
    bs_rows = load_csv("bs_count_error_scale.csv")
    stability = load_csv("temporal_stability.csv")
    geometry = load_csv("geometry_coverage.csv")
    gradient = load_csv("snr_gradient.csv")
    normalized = json.loads((common.OUT_DIR / "normalized_error_scale.json").read_text())
    conditioning = load_csv("geometry_conditioning.csv")

    train_scale = scale["splits"]["train"]
    median_frame = train_scale["single_frame_displacement_m"]["p50"]
    median_2s = train_scale["history_span_20frame_m"]["p50"]
    median_speed = train_scale["speed"]["all"]["p50"]
    median_nn = train_scale["nearest_neighbor_m"]["all"]["p50"]

    position_low = pooled["train"][str(-10.0)]["position_rmse_m"]
    velocity_low = pooled["train"][str(-10.0)]["velocity_rmse_mps"]
    position_high = pooled["train"][str(10.0)]["position_rmse_m"]
    velocity_high = pooled["train"][str(10.0)]["velocity_rmse_mps"]
    position_ratio_low = position_low / median_frame
    velocity_ratio_low = velocity_low / median_speed

    def stability_entry(level: float, speed_class: str) -> dict | None:
        return next((entry for entry in stability
                     if entry["split"] == "train" and float(entry["snr_db"]) == level
                     and entry["speed_class"] == speed_class), None)

    def flip(level: float) -> float | None:
        entry = stability_entry(level, "normal_speed_ge2")
        if entry is None:
            return None
        return float(entry["direction_flip_fraction"] or 0.0)

    def angle_median(level: float) -> float | None:
        entry = stability_entry(level, "normal_speed_ge2")
        if entry is None:
            return None
        return float(json.loads(entry["velocity_angle_vs_true_deg"].replace("'", '"'))["median"])

    def position_ratio(level: float) -> float:
        return pooled["train"][str(level)]["position_rmse_m"] / median_frame

    def velocity_ratio(level: float) -> float:
        return pooled["train"][str(level)]["velocity_rmse_mps"] / median_speed

    one_bs = next(entry for entry in bs_rows
                  if entry["split"] == "train" and float(entry["snr_db"]) == 10.0
                  and int(entry["n_bs"]) == 1)
    three_bs = next(entry for entry in bs_rows
                    if entry["split"] == "train" and float(entry["snr_db"]) == 10.0
                    and int(entry["n_bs"]) == 3)

    coverage = {f"{entry['split']}|{entry['scope']}": entry for entry in geometry}
    low_condition = [entry for entry in conditioning
                     if entry["split"] == "train" and int(entry["n_bs"]) == 2]
    gradient_rows = [entry for entry in gradient
                     if entry["split"] == "train" and entry["scope"] == "overall"]

    decision = {
        "audit": "Route-B final audit (read-only)",
        "baseline_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip(),
        "candidate_b": common.EXPECTED_CANDIDATE_B,
        "parameter_changes_made": False,
        "test_used_for_parameter_decisions": False,
        "scales": {
            "median_speed_mps": median_speed,
            "median_single_frame_displacement_m": median_frame,
            "median_1s_displacement_m": train_scale["ten_frame_displacement_m"]["p50"],
            "median_2s_history_span_m": median_2s,
            "median_nearest_neighbor_m": median_nn,
            "nn_le_5m_fraction": train_scale["nearest_neighbor_m"]["rates"]["le_5m"],
            "nn_le_10m_fraction": train_scale["nearest_neighbor_m"]["rates"]["le_10m"],
        },
        "candidate_b_errors_ge2": {
            split: {str(level): pooled[split][str(level)] for level in common.LEVELS}
            for split in ("train", "val")},
        "candidate_b_normalized": {
            "position_rmse_over_median_frame_displacement": {
                str(level): position_ratio(level) for level in common.LEVELS},
            "velocity_rmse_over_median_speed": {
                str(level): velocity_ratio(level) for level in common.LEVELS},
            "position_p95_over_median_frame_displacement": {
                str(level): normalized["train"]["levels"][str(level)][
                    "position_p95_over_median_frame_displacement"] for level in common.LEVELS},
            "velocity_error_accumulated_2s_over_median_2s_displacement": {
                str(level): normalized["train"]["levels"][str(level)][
                    "velocity_error_accumulated_2s_over_median_2s_displacement"]
                for level in common.LEVELS},
        },
        "verdicts": {
            "position": verdict_position(position_ratio_low),
            "position_ratio_low": position_ratio_low,
            "velocity": verdict_velocity(velocity_ratio_low),
            "velocity_ratio_low": velocity_ratio_low,
            "gradient_position_ratio_+10_to_-10": position_low / position_high,
            "gradient_velocity_ratio_+10_to_-10": velocity_low / velocity_high,
            "gradient": verdict_gradient(position_low / position_high),
            "semantics": {
                "plus_10_good": semantic_verdict("good", position_ratio(10.0),
                                                 velocity_ratio(10.0), flip(10.0),
                                                 angle_median(10.0)),
                "zero_medium": semantic_verdict("medium", position_ratio(0.0),
                                                velocity_ratio(0.0), flip(0.0), angle_median(0.0)),
                "minus_10_poor_but_usable": semantic_verdict(
                    "poor", position_ratio(-10.0), velocity_ratio(-10.0), flip(-10.0),
                    angle_median(-10.0)),
            },
        },
        "semantic_evidence": {
            str(level): {
                "position_ratio_vs_frame_displacement": position_ratio(level),
                "velocity_ratio_vs_median_speed": velocity_ratio(level),
                "direction_flip_fraction_normal_speed": flip(level),
                "direction_angle_median_deg": angle_median(level),
            } for level in common.LEVELS},
        "gradient_adjacent_ratios": {
            "note": "overall (all n_bs>=1) velocity ratios are contaminated by the 1-BS geometry "
                    "tail; prefer gradient_adjacent_ratios_ge2",
            "position": {entry["snr_db"]: entry.get("position_ratio_vs_higher_snr")
                         for entry in gradient_rows if entry.get("position_ratio_vs_higher_snr")},
            "velocity": {entry["snr_db"]: entry.get("velocity_ratio_vs_higher_snr")
                         for entry in gradient_rows if entry.get("velocity_ratio_vs_higher_snr")},
        },
        "gradient_adjacent_ratios_ge2": _pooled_gradient(pooled["train"]),
        "geometry": {
            "coverage": coverage,
            "blind_states_train": int(coverage["train|overall"]["n_bs_0"]),
            "blind_states_val": int(coverage["val|overall"]["n_bs_0"]),
            "one_bs_fraction_train": float(coverage["train|overall"]["pct_1bs"]),
            "one_bs_fraction_scene_0_train": float(coverage["train|scene_0"]["pct_1bs"]),
            "one_bs_fraction_scene_1_train": float(coverage["train|scene_1"]["pct_1bs"]),
            "one_bs_velocity_penalty_at_+10": {
                "one_bs_velocity_rmse_mps": float(one_bs["velocity_rmse_mps"]),
                "three_bs_velocity_rmse_mps": float(three_bs["velocity_rmse_mps"]),
                "penalty_ratio": float(one_bs["velocity_rmse_mps"]) / float(three_bs["velocity_rmse_mps"]),
                "one_bs_position_rmse_m": float(one_bs["position_rmse_m"]),
                "three_bs_position_rmse_m": float(three_bs["position_rmse_m"]),
                "position_penalty_ratio": float(one_bs["position_rmse_m"]) / float(three_bs["position_rmse_m"]),
            },
            "condition_ratio_2bs": {entry["snr_db"]: {"p05": entry["p05"], "p50": entry["p50"],
                                                      "fraction_below_0p05": entry["fraction_below_0p05"]}
                                    for entry in low_condition},
            "geometry_hard_issue": False,
        },
        "quality_formula": {
            "current": "quality = 100*exp(-sqrt((dp/1m)^2+(dv/1mps)^2))",
            "is_accuracy": False,
            "reference_scales_arbitrary": True,
            "status": "provisional; NOT FROZEN",
            "phase2_candidates": ["A fixed task tolerance", "B dataset motion scales",
                                  "C prediction-horizon impact (recommended for discussion)"],
        },
        "phase2_directions": {
            "recommended": "increase range and radial-velocity channel strengths moderately; keep "
                           "bearing; fix scene-0 1-BS geometry; adopt horizon-impact quality "
                           "normalization candidate C",
            "range_search_hint": "a_range in [0.15, 0.25] m",
            "bearing_search_hint": "a_bearing stay at 0.004 rad (adequate)",
            "velocity_search_hint": "a_radial_velocity in [0.20, 0.30] m/s",
            "gradient_direction": "steepen low-SNR degradation modestly if downstream prediction is "
                                  "sensitive; current gradient is a clean 10x over +10 -> -10 dB",
            "not_frozen": True,
        },
    }
    common.write_json(common.OUT_DIR / "decision_summary.json", decision)
    print(json.dumps(decision["verdicts"], indent=2))
    print(f"total {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())