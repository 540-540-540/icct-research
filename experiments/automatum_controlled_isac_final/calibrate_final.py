"""Phase B: recalibrate (a_range, a_bearing, a_radial_velocity) after the geometry fix.

Objective = target-band loss on the three key SNR levels + strict monotonicity penalty, with
the Final-Audit soft references as tie-break only. Train prediction-history unique states;
val and test are not read. Uses the fast_eval replica for the search (verified against the
real frontend in selfcheck).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final import common, fast_eval  # noqa: E402

GRID = {
    "range": (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    "bearing": (0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012),
    "radial_velocity": (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
}
REFINE_FACTORS = (0.85, 0.925, 1.0, 1.075, 1.15)
CANDIDATE_B = {"range": 0.10, "bearing": 0.004, "radial_velocity": 0.16}


def load_geometry() -> dict:
    payload = json.loads((common.OUT_DIR / "geometry_search.json").read_text())
    selected = payload["selected_scene0_geometry"]
    if selected["status"] != "FROZEN":
        raise SystemExit("geometry search did not reach the target; refusing to calibrate")
    return selected


def calibration_for(config: dict, a: dict) -> dict:
    base = config["measurement"]
    return {"a": {channel: float(a[channel]) for channel in ("range", "bearing", "radial_velocity")},
            "floors": {channel: float(base["floors"][channel])
                       for channel in ("range", "bearing", "radial_velocity")},
            "covariance_floor_m2": float(base.get("covariance_floor_m2", 1e-8)),
            "velocity_prior_sigma_mps": float(config["fusion"]["velocity_prior_sigma_mps"])}


def evaluate_point(prepared, config, a) -> dict:
    output = fast_eval.evaluate(prepared, calibration_for(config, a))
    subset = prepared["visible"].sum(axis=1) >= 2
    result = fast_eval.objective(output, subset)
    result["a"] = {channel: float(a[channel]) for channel in GRID}
    return result


def rank_key(result):
    return (round(result["score"], 9), round(result["soft_reference_loss"], 9),
            result["a"]["range"], result["a"]["bearing"], result["a"]["radial_velocity"])


def candidate_row(name: str, result: dict) -> dict:
    row = {"candidate": name}
    row.update({f"a_{key}": value for key, value in result["a"].items()})
    row["score"] = result["score"]
    row["band_loss"] = result["band_loss"]
    row["soft_reference_loss"] = result["soft_reference_loss"]
    row["monotonicity_violations"] = result["monotonicity_violations"]
    for level, values in result["rmse_by_level"].items():
        row[f"position_rmse_{level}"] = values["position"]
        row[f"velocity_rmse_{level}"] = values["velocity"]
    for key, band in result["bands"].items():
        row[f"in_band_{key}"] = 1 if band["loss"] == 0 else 0
    return row


def main() -> int:
    started = time.time()
    config = common.load_config()
    geometry = load_geometry()
    prepared = fast_eval.prepare(config, geometry, "train")
    print(f"prepared {len(prepared['keys'])} train states with fixed geometry "
          f"({time.time() - started:.1f} s)")

    evaluations = []
    for a_range, a_bearing, a_vr in itertools.product(*GRID.values()):
        evaluations.append(evaluate_point(prepared, config,
                                          {"range": a_range, "bearing": a_bearing,
                                           "radial_velocity": a_vr}))
    best = min(evaluations, key=rank_key)
    history = [{"stage": "coarse", "a": best["a"], "score": best["score"]}]
    print(f"coarse best {best['a']} score {best['score']:.4f}")

    for _ in range(2):
        for channel in GRID:
            local_best = best
            for factor in REFINE_FACTORS:
                candidate = dict(best["a"])
                candidate[channel] = round(best["a"][channel] * factor, 6)
                result = evaluate_point(prepared, config, candidate)
                evaluations.append(result)
                if rank_key(result) < rank_key(local_best):
                    local_best = result
            best = local_best
            history.append({"stage": f"refine_{channel}", "a": best["a"], "score": best["score"]})
    print(f"refined {best['a']} score {best['score']:.4f}")

    ranked = sorted(evaluations, key=rank_key)
    top = ranked[:20]
    reference = evaluate_point(prepared, config, CANDIDATE_B)
    rows = [candidate_row(f"rank_{index + 1:02d}", result) for index, result in enumerate(top)]
    rows.append(candidate_row("reference_candidate_b", reference))
    common.write_csv(common.OUT_DIR / "calibration_candidates.csv", rows)

    payload = {
        "baseline_head": "eaadbe6d826f9f6e83baf3caf3c01bf75ac093d1",
        "geometry": geometry,
        "grid": {key: list(values) for key, values in GRID.items()},
        "n_evaluations": len(evaluations),
        "bands": {f"{level:+g}": band for level, band in fast_eval.BANDS.items()},
        "soft_references": {f"{level:+g}": value
                            for level, value in fast_eval.SOFT_REFERENCES.items()},
        "history": history,
        "top20": top,
        "reference_candidate_b": reference,
        "selected": {
            "a": best["a"],
            "score": best["score"], "band_loss": best["band_loss"],
            "soft_reference_loss": best["soft_reference_loss"],
            "monotonicity_violations": best["monotonicity_violations"],
            "rmse_by_level": best["rmse_by_level"], "bands": best["bands"],
            "status": "FROZEN" if best["band_loss"] == 0 and best["monotonicity_violations"] == 0
            else "CANDIDATE",
        },
    }
    common.write_json(common.OUT_DIR / "calibration_search.json", payload)
    print(json.dumps(payload["selected"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())