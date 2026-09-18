"""Phase A: minimal scene-0 geometry fix.

Search order (work order §9): boresight -> FOV -> small BS position moves. Scene 1 geometry is
never modified. Search uses train prediction-history unique states only. Targets: 0-BS = 0
(hard), 1-BS as low as possible, ideal >=2-BS >= 99.5%.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final import common  # noqa: E402

BORESIGHT_OFFSETS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
FOV_STEPS = (70.0, 75.0, 80.0)
TARGET_ONE_BS_FRACTION = 0.005
POSITION_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
POSITION_STEPS_M = (5.0, 10.0)


def condition_median(positions: np.ndarray, stations: np.ndarray, visible: np.ndarray) -> float | None:
    delta = positions[:, None, :] - stations[None, :, :]
    rho = np.linalg.norm(delta, axis=2)
    unit = delta / np.maximum(rho[:, :, None], 1e-9)
    mask = visible.astype(np.float64)[:, :, None, None]
    outer = unit[:, :, :, None] * unit[:, :, None, :]
    matrix = (mask * outer).sum(axis=1)
    trace = matrix[:, 0, 0] + matrix[:, 1, 1]
    det = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] ** 2
    disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0.0))
    ratio = (trace - disc) / np.maximum(trace + disc, 1e-12)
    keep = visible.sum(axis=1) >= 2
    if not keep.any():
        return None
    return float(np.median(ratio[keep]))


def evaluate_candidate(positions, base_stations, base_boresights, offsets, fov, moves):
    stations = base_stations + np.asarray(moves)
    boresights = base_boresights + np.asarray(offsets)
    visible = common.visibility(positions, stations, boresights, fov)
    counts = visible.sum(axis=1)
    result = {
        "boresight_offsets_deg": [float(v) for v in offsets],
        "boresight_abs_sum": float(np.abs(offsets).sum()),
        "boresight_abs_max": float(np.abs(offsets).max()),
        "fov_half_angle_deg": float(fov),
        "position_moves_m": [[float(v) for v in move] for move in moves],
        "position_move_abs_sum": float(np.abs(np.asarray(moves)).sum()),
        "n_bs_0": int((counts == 0).sum()),
        "n_bs_1": int((counts == 1).sum()),
        "n_bs_2": int((counts == 2).sum()),
        "n_bs_3": int((counts == 3).sum()),
        "n": int(counts.size),
    }
    result["one_bs_fraction"] = result["n_bs_1"] / result["n"]
    result["pct_ge_2bs"] = (result["n_bs_2"] + result["n_bs_3"]) / result["n"]
    result["condition_ratio_median"] = condition_median(positions, stations, visible)
    result["modification_level"] = 0 if (fov == 70.0 and result["position_move_abs_sum"] == 0.0) else (
        1 if fov == 75.0 else (2 if fov == 80.0 else 3))
    return result


def rank_key(result):
    meets = result["one_bs_fraction"] <= TARGET_ONE_BS_FRACTION
    return (0 if meets else 1, 0 if result["n_bs_0"] == 0 else 1, result["modification_level"],
            result["fov_half_angle_deg"], result["position_move_abs_sum"],
            result["boresight_abs_sum"], result["boresight_abs_max"],
            -result["n_bs_3"], -(result["condition_ratio_median"] or 0.0))


def coverage_report(split: str, scene_geometry: dict) -> list[dict]:
    config = common.load_config()
    history = common.unique_history(split)
    lookup = common.state_lookup(split)
    rows = []
    for scene_id in common.SCENES:
        keys = [key for key in history["keys"] if key[0] == scene_id]
        positions = np.asarray([lookup[key][:2] for key in keys])
        geometry = common.scene_geometry(config)[scene_id]
        stations = scene_geometry["stations_xy_m"] if scene_id == 0 \
            else geometry["stations"]
        boresights = scene_geometry["boresights_deg"] if scene_id == 0 \
            else geometry["boresights_deg"]
        fov = scene_geometry["fov_half_angle_deg"] if scene_id == 0 else 70.0
        visible = common.visibility(positions, np.asarray(stations), np.asarray(boresights), fov)
        counts = common.coverage_counts(visible)
        counts.update({"split": split, "scope": f"scene_{scene_id}",
                       "pct_ge_2bs": (counts["n_bs_2"] + counts["n_bs_3"]) / counts["n"],
                       "pct_3bs": counts["n_bs_3"] / counts["n"],
                       "condition_ratio_median": condition_median(
                           positions, np.asarray(stations), visible)})
        rows.append(counts)
    total = {key: sum(row[key] for row in rows) for key in
             ("n", "n_bs_0", "n_bs_1", "n_bs_2", "n_bs_3")}
    total.update({"split": split, "scope": "overall",
                  "pct_ge_2bs": (total["n_bs_2"] + total["n_bs_3"]) / total["n"],
                  "pct_3bs": total["n_bs_3"] / total["n"],
                  "condition_ratio_median": rows[0]["condition_ratio_median"]})
    rows.append(total)
    return rows


def main() -> int:
    started = time.time()
    config = common.load_config()
    history = common.unique_history("train")
    lookup = common.state_lookup("train")
    keys0 = [key for key in history["keys"] if key[0] == 0]
    positions = np.asarray([lookup[key][:2] for key in keys0])
    geometry = common.scene_geometry(config)[0]
    base_stations = geometry["stations"]
    base_boresights = geometry["boresights_deg"]

    evaluations = []
    for offsets in itertools.product(BORESIGHT_OFFSETS, repeat=3):
        result = evaluate_candidate(positions, base_stations, base_boresights, offsets, 70.0,
                                    ((0, 0), (0, 0), (0, 0)))
        evaluations.append(result)
    boresight_best = min(evaluations, key=rank_key)
    print(f"level1 boresight-only best: 1-BS {boresight_best['n_bs_1']} "
          f"({boresight_best['one_bs_fraction']:.4f}), offsets "
          f"{boresight_best['boresight_offsets_deg']}")

    level2_used = False
    if boresight_best["one_bs_fraction"] > TARGET_ONE_BS_FRACTION:
        level2_used = True
        for fov in (75.0, 80.0):
            for offsets in itertools.product(BORESIGHT_OFFSETS, repeat=3):
                result = evaluate_candidate(positions, base_stations, base_boresights, offsets,
                                            fov, ((0, 0), (0, 0), (0, 0)))
                evaluations.append(result)
        best = min(evaluations, key=rank_key)
        print(f"level2 best: fov {best['fov_half_angle_deg']} 1-BS {best['n_bs_1']} "
              f"({best['one_bs_fraction']:.4f}) offsets {best['boresight_offsets_deg']}")

    best = min(evaluations, key=rank_key)
    level3_used = False
    if best["one_bs_fraction"] > TARGET_ONE_BS_FRACTION:
        level3_used = True
        current = best
        for _ in range(2):
            for bs_id in range(3):
                improved = current
                for dx, dy in POSITION_DIRECTIONS:
                    for step in POSITION_STEPS_M:
                        moves = [list(move) for move in current["position_moves_m"]]
                        moves[bs_id] = [moves[bs_id][0] + dx * step, moves[bs_id][1] + dy * step]
                        if math.hypot(*moves[bs_id]) > 12.0:
                            continue
                        result = evaluate_candidate(
                            positions, base_stations, base_boresights,
                            current["boresight_offsets_deg"], current["fov_half_angle_deg"],
                            moves)
                        evaluations.append(result)
                        if rank_key(result) < rank_key(improved):
                            improved = result
                current = improved
            if rank_key(current) == rank_key(best):
                break
            best = current
        print(f"level3 best: moves {best['position_moves_m']} 1-BS {best['n_bs_1']} "
              f"({best['one_bs_fraction']:.4f})")

    ranked = sorted(evaluations, key=rank_key)
    top5 = ranked[:5]
    selected = {
        "scene_id": 0,
        "stations_xy_m": [[float(base_stations[b][0] + selected_move[0]),
                           float(base_stations[b][1] + selected_move[1])]
                          for b, selected_move in enumerate(best["position_moves_m"])],
        "boresights_deg": [float(base_boresights[b] + best["boresight_offsets_deg"][b])
                           for b in range(3)],
        "fov_half_angle_deg": float(best["fov_half_angle_deg"]),
        "status": "FROZEN" if best["one_bs_fraction"] <= TARGET_ONE_BS_FRACTION else "CANDIDATE",
    }
    validation = {split: coverage_report(split, selected) for split in ("train", "val")}
    payload = {
        "baseline_head": "eaadbe6d826f9f6e83baf3caf3c01bf75ac093d1",
        "search": {"levels_used": {"boresight_only": True, "fov": level2_used,
                                   "position_moves": level3_used},
                   "n_evaluations": len(evaluations),
                   "target_one_bs_fraction": TARGET_ONE_BS_FRACTION,
                   "train_scene0_states": len(keys0)},
        "old_scene0_geometry": {"stations_xy_m": base_stations.tolist(),
                                "boresights_deg": base_boresights.tolist(),
                                "fov_half_angle_deg": 70.0},
        "selected_scene0_geometry": selected,
        "top5": top5,
        "coverage_after_fix": validation,
    }
    common.write_json(common.OUT_DIR / "geometry_search.json", payload)
    common.write_csv(common.OUT_DIR / "geometry_final_audit.csv",
                     [row for split in ("train", "val") for row in validation[split]])
    print(json.dumps({"selected": selected, "train": [
        row for row in validation["train"] if row["scope"] == "overall"],
        "val": [row for row in validation["val"] if row["scope"] == "overall"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())