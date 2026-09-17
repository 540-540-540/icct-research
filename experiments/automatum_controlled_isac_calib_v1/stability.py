"""Calibration V1 stability probe on the recommended candidate's per-state output.

Answers the qualitative PASS items of the work order with numbers:
  * no large velocity direction flips at low SNR,
  * no absurd frame-to-frame position jumps,
  * estimates remain trajectory-like (second-difference sanity).

Reads only ``per_state_recommended.csv``; writes ``stability.json`` next to it.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/isac_calibration/automatum_controlled_isac_v1"


def main() -> int:
    rows = list(csv.DictReader((OUT / "per_state_recommended.csv").open(encoding="utf-8")))
    levels = sorted({float(row["snr_db"]) for row in rows})
    payload = {"velocity_direction_flips": {}, "frame_jump": {}, "second_difference": {}}
    for level in levels:
        selected = [row for row in rows if float(row["snr_db"]) == level]
        flips = 0
        for row in selected:
            vx, vy = float(row["vx_hat"]), float(row["vy_hat"])
            tx, ty = float(row["vx_true"]), float(row["vy_true"])
            if vx * tx + vy * ty < 0.0:
                flips += 1
        payload["velocity_direction_flips"][str(level)] = {
            "n": len(selected), "flips": flips, "fraction": flips / len(selected)}

        by_vehicle = defaultdict(dict)
        for row in selected:
            by_vehicle[(row["split"], row["scene_id"], row["vehicle_id"])][int(row["frame"])] = row
        jumps, second_differences = [], []
        for key, frames in by_vehicle.items():
            for frame in sorted(frames):
                if frame + 1 not in frames:
                    continue
                first, second = frames[frame], frames[frame + 1]
                jumps.append(math.hypot(
                    (float(second["x_hat"]) - float(first["x_hat"]))
                    - (float(second["x_true"]) - float(first["x_true"])),
                    (float(second["y_hat"]) - float(first["y_hat"]))
                    - (float(second["y_true"]) - float(first["y_true"]))))
                if frame + 2 in frames:
                    third = frames[frame + 2]
                    positions = [(float(row["x_true"]), float(row["y_true"]), float(row["x_hat"]),
                                  float(row["y_hat"])) for row in (first, second, third)]
                    gt_curvature = math.hypot(
                        positions[2][0] - 2 * positions[1][0] + positions[0][0],
                        positions[2][1] - 2 * positions[1][1] + positions[0][1])
                    hat_curvature = math.hypot(
                        positions[2][2] - 2 * positions[1][2] + positions[0][2],
                        positions[2][3] - 2 * positions[1][3] + positions[0][3])
                    second_differences.append(abs(hat_curvature - gt_curvature))
        payload["frame_jump"][str(level)] = {
            "pairs": len(jumps), "median_m": float(np.median(jumps)) if jumps else None,
            "p95_m": float(np.percentile(jumps, 95)) if jumps else None,
            "max_m": float(np.max(jumps)) if jumps else None}
        payload["second_difference"][str(level)] = {
            "triples": len(second_differences),
            "p95_m": float(np.percentile(second_differences, 95)) if second_differences else None,
            "max_m": float(np.max(second_differences)) if second_differences else None}
    (OUT / "stability.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())