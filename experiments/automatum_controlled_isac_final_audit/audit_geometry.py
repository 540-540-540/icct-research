"""Final Audit J/K: BS geometric coverage of the de-duplicated prediction-history states,
0-BS blind-state forensic audit and the 1-BS situation (proportion only; error penalty is in
audit_sensing_scale). Geometry is read-only and never modified."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_final_audit import common  # noqa: E402
from frontend.controlled_isac.automatum_measurement import setup_from_config  # noqa: E402
from frontend.controlled_isac.measurement import truth_measurement, visible  # noqa: E402


def visible_bs_count(scene_id: int, position, velocity, setup: dict) -> int:
    count = 0
    for bs_id in range(3):
        scene = setup["scenes"][scene_id]
        truth = truth_measurement(position, velocity, scene["stations"][bs_id],
                                  float(scene["boresights"][bs_id]), setup["height"])
        if visible(truth, setup):
            count += 1
    return count


def coverage_rows(split: str) -> tuple[list[dict], list[dict], dict]:
    config = common.load_config()
    setup = setup_from_config(config)
    history = common.unique_history(split)
    lookup, _ = common.state_lookup(split)
    buckets = {}
    blind = []
    for scene_id, frame_id, vehicle in history["keys"]:
        state = lookup[(scene_id, frame_id, vehicle)]
        count = visible_bs_count(scene_id, state[:2], state[2:], setup)
        for scope in ("overall", f"scene_{scene_id}"):
            record = buckets.setdefault((split, scope), {"split": split, "scope": scope, "n": 0,
                                                         "n_bs_0": 0, "n_bs_1": 0, "n_bs_2": 0,
                                                         "n_bs_3": 0})
            record["n"] += 1
            record[f"n_bs_{count}"] += 1
        if count == 0:
            blind.append({"split": split, "scene_id": scene_id, "frame": frame_id,
                          "vehicle_id": vehicle, "x": float(state[0]), "y": float(state[1]),
                          "speed_mps": float(np.hypot(state[2], state[3]))})
    rows = []
    for record in buckets.values():
        n = record["n"]
        record.update({"pct_0bs": record["n_bs_0"] / n, "pct_1bs": record["n_bs_1"] / n,
                       "pct_2bs": record["n_bs_2"] / n, "pct_3bs": record["n_bs_3"] / n,
                       "pct_ge_2bs": (record["n_bs_2"] + record["n_bs_3"]) / n})
        rows.append(record)
    rows.sort(key=lambda row: (row["split"], row["scope"]))
    return rows, blind, {"unique_states": history["unique_states"]}


def blind_forensics(blind: list[dict]) -> dict:
    if not blind:
        return {"blind_states": 0, "verdict": "no 0-BS states"}
    keys = {(row["split"], row["scene_id"], row["frame"], row["vehicle_id"]) for row in blind}
    runs = {}
    for row in blind:
        key = (row["split"], row["scene_id"], row["vehicle_id"])
        runs.setdefault(key, []).append(row["frame"])
    max_run = 0
    run_examples = []
    for key, frames in runs.items():
        ordered = sorted(frames)
        current = 1
        for index in range(1, len(ordered)):
            if ordered[index] == ordered[index - 1] + 1:
                current += 1
            else:
                current = 1
            if current > max_run:
                max_run = current
                run_examples = [f"{key} frames {ordered[index - current + 1]}..{ordered[index]}"]
    clusters = {}
    for row in blind:
        cell = (row["scene_id"], round(row["x"] / 20.0), round(row["y"] / 20.0))
        clusters[cell] = clusters.get(cell, 0) + 1
    worst_cluster = max(clusters.values())
    verdict = "GEOMETRY HARD ISSUE" if (max_run >= 3 or worst_cluster >= 5) else \
        "isolated boundary cases only"
    return {"blind_states": len(blind),
            "vehicles_affected": len(runs),
            "max_consecutive_blind_frames": max_run,
            "worst_20m_grid_cluster": worst_cluster,
            "examples": run_examples[:5],
            "verdict": verdict}


def plot_coverage(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = common.OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "bs_coverage_by_scene.png").unlink(missing_ok=True)
    labels, fractions = [], {0: [], 1: [], 2: [], 3: []}
    for row in rows:
        labels.append(f"{row['split']}\n{row['scope']}")
        for count in range(4):
            fractions[count].append(row[f"n_bs_{count}"] / row["n"])
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    bottom = np.zeros(len(labels))
    colors = ("tab:red", "tab:orange", "tab:blue", "tab:green")
    for count in range(4):
        axis.bar(labels, fractions[count], bottom=bottom, label=f"{count} BS", color=colors[count])
        bottom += np.array(fractions[count])
    axis.set_ylabel("fraction of unique prediction-history states")
    axis.set_title("Automatum Route-B: geometric BS coverage (read-only geometry)")
    axis.legend()
    axis.grid(True, alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(plots / "bs_coverage_by_scene.png", dpi=140)
    plt.close(figure)


def main() -> int:
    started = time.time()
    all_rows, all_blind, summary = [], [], {}
    for split in ("train", "val"):
        rows, blind, meta = coverage_rows(split)
        all_rows.extend(rows)
        all_blind.extend(blind)
        summary[split] = {"unique_states": meta["unique_states"],
                          "coverage": rows, "blind_forensics": blind_forensics(blind)}
        print(f"{split}: " + ", ".join(
            f"{row['scope']}: 0BS {row['n_bs_0']}, 1BS {row['n_bs_1']}, 2BS {row['n_bs_2']}, "
            f"3BS {row['n_bs_3']}" for row in rows if row["scope"] == "overall"))
    common.write_csv(common.OUT_DIR / "geometry_coverage.csv", all_rows)
    common.write_csv(common.OUT_DIR / "geometry_blind_states.csv", all_blind)
    common.write_json(common.OUT_DIR / "geometry_audit.json", summary)
    plot_coverage(all_rows)
    print(f"blind states: {len(all_blind)}; written in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())