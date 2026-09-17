"""AUTOMATUM-CONTROLLED-ISAC Calibration V1 runner.

Runs the deterministic calibration search on train, evaluates every candidate on train and val,
and writes metrics / CSVs / plots under
``reports/isac_calibration/automatum_controlled_isac_v1``. Test data is never touched.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.automatum_controlled_isac_calib_v1 import calibrate  # noqa: E402
from experiments.automatum_controlled_isac_calib_v1.evaluate import (OUT_DIR, aggregate,  # noqa: E402
                                                                     flatten, geometry_audit,
                                                                     load_config, run_split,
                                                                     scene_frames, write_csv,
                                                                     write_json)
from frontend.controlled_isac.automatum_measurement import setup_from_config  # noqa: E402
from frontend.controlled_isac.state_quality import mapping_table  # noqa: E402

CANDIDATE_ORDER = ("A_light", "B_recommended", "C_heavy", "D_very_heavy")
METRIC_KEYS = ("position", "velocity", "quality")


def trajectory_windows(config: dict) -> dict[int, tuple[int, list[int]]]:
    """Deterministic 20-frame train windows with a stable prediction cohort (per scene)."""
    import numpy as np

    samples = np.load(ROOT / "data/automatum_t_crossing/splits/train/samples.npz")
    windows = {}
    for scene_id in (0, 1):
        for index in range(samples["scene_id"].size):
            if int(samples["scene_id"][index]) != scene_id:
                continue
            count = int(samples["num_vehicles"][index])
            if count >= 4:
                ids = samples["vehicle_ids"][index][samples["vehicle_mask"][index]].tolist()
                windows[scene_id] = (int(samples["start_frame"][index]), [int(v) for v in ids])
                break
    return windows


def plot_trajectories(config: dict, calibration: dict) -> None:
    from frontend.automatum_scene_source import AutomatumSceneSource
    from frontend.controlled_isac.automatum_frontend import sense_frame

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup = setup_from_config(config)
    source = AutomatumSceneSource(ROOT / config["dataset"]["train_trajectories"])
    plots = OUT_DIR / "plots"
    for scene_id, (start_frame, vehicle_ids) in trajectory_windows(config).items():
        frames = [source.at_frame(scene_id, start_frame + offset) for offset in range(20)]
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
        for axis, snr_db in zip(axes, (10.0, 0.0, -10.0)):
            for vehicle_id in vehicle_ids:
                truth_x, truth_y, hat_x, hat_y = [], [], [], []
                for scene_frame in frames:
                    index = scene_frame.vehicle_ids.tolist().index(vehicle_id)
                    states = sense_frame(scene_frame, scene_id, snr_db, calibration, setup)
                    state = next(item for item in states if item["vehicle_id"] == vehicle_id)
                    truth_x.append(float(scene_frame.states[index, 0]))
                    truth_y.append(float(scene_frame.states[index, 1]))
                    hat_x.append(state["x_hat"])
                    hat_y.append(state["y_hat"])
                axis.plot(truth_x, truth_y, "-", color="black", linewidth=2.2,
                          label="GT" if vehicle_id == vehicle_ids[0] else None)
                axis.plot(hat_x, hat_y, "--", color="tab:red", linewidth=1.3,
                          label="sensing" if vehicle_id == vehicle_ids[0] else None)
            axis.set_title(f"{snr_db:+.0f} dB ({config['snr']['semantics'][str(snr_db)]})")
            axis.set_xlabel("x (m)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel("y (m)")
        axes[0].legend(fontsize=8)
        figure.suptitle(f"Automatum Route B recommended candidate: scene {scene_id} frames "
                        f"{start_frame}-{start_frame + 19}, same vehicles / same window")
        figure.tight_layout()
        figure.savefig(plots / f"trajectories_scene{scene_id}_good_medium_poor.png", dpi=140)
        plt.close(figure)


def plot_outputs(candidate_rows: list[dict], per_state: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for stale in plots.glob("*.png"):
        stale.unlink()
    levels = list(calibrate.LEVELS)

    for metric, label, unit in (("position", "fused position RMSE", "m"),
                                ("velocity", "fused velocity RMSE", "m/s"),
                                ("quality", "provisional state quality", "%")):
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        for candidate in CANDIDATE_ORDER:
            for split, style, alpha in (("train", "-", 1.0), ("val", "--", 0.5)):
                values = []
                for snr_db in levels:
                    row = next(row for row in candidate_rows
                               if row["candidate"] == candidate and row["split"] == split
                               and row["scope"] == "overall" and row["subset"] == "ge2"
                               and row["snr_db"] == snr_db)
                    values.append(row[f"{metric}_mean"] if metric == "quality"
                                  else row[f"{metric}_rmse"])
                axis.plot(levels, values, style, alpha=alpha, marker="o",
                          label=f"{candidate} ({split})")
        axis.set_xlabel("frame-level SNR (dB)")
        axis.set_ylabel(f"{label} ({unit})")
        axis.set_title(f"Automatum Route B Calibration V1: SNR vs {label}")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(plots / f"snr_vs_{metric}.png", dpi=140)
        plt.close(figure)

    


def main() -> int:
    started = time.time()
    config = load_config()
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    audit = geometry_audit(config)
    write_csv(out / "geometry_coverage.csv", audit)

    search_result = calibrate.search(config)
    write_json(out / "calibration.json", search_result)

    frames_train = scene_frames(config, "train")
    frames_val = scene_frames(config, "val")
    candidate_rows = []
    per_state_rows = []
    for name in CANDIDATE_ORDER:
        parameters = search_result["candidates"][name]["a"]
        calibration = calibrate.calibration_with(config, parameters)
        for split, cache in (("train", frames_train), ("val", frames_val)):
            rows = run_split(config, calibration, split, calibrate.LEVELS, frames_cache=cache)
            if name == "B_recommended":
                per_state_rows.extend(rows)
            for summary in aggregate(rows, calibrate.LEVELS,
                                     {"overall": [0, 1], "scene_0": [0], "scene_1": [1]}):
                row = {"candidate": name, "split": split, "scope": summary["scope"],
                       "subset": summary["subset"], "snr_db": float(summary["snr_db"]),
                       "a_range": parameters["range"], "a_bearing": parameters["bearing"],
                       "a_radial_velocity": parameters["radial_velocity"]}
                row.update(flatten(summary))
                candidate_rows.append(row)
    write_csv(out / "candidate_metrics.csv", candidate_rows)
    write_csv(out / "per_state_recommended.csv", per_state_rows)
    write_csv(out / "state_quality_mapping.csv", mapping_table(
        float(config["state_quality"]["p_ref_m"]), float(config["state_quality"]["v_ref_mps"])))

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    metrics = {
        "revision": config["revision"],
        "status": "Calibration V1 candidates; NOT FROZEN",
        "baseline_head": head,
        "config_sha256": hashlib.sha256(
            (ROOT / "configs/automatum_controlled_isac.json").read_bytes()).hexdigest(),
        "snr_levels_db": list(calibrate.LEVELS),
        "geometry_audit": audit,
        "recommended_candidate": search_result["recommended"],
        "candidates": {name: {"factor_vs_recommended": search_result["candidates"][name][
            "factor_vs_recommended"], "a": search_result["candidates"][name]["a"]}
            for name in CANDIDATE_ORDER},
        "search_history": search_result["history"],
        "candidate_metrics": candidate_rows,
        "state_quality": config["state_quality"],
    }
    write_json(out / "metrics.json", metrics)
    plot_outputs(candidate_rows, per_state_rows)
    recommended_calibration = calibrate.calibration_with(
        config, search_result["candidates"]["B_recommended"]["a"])
    plot_trajectories(config, recommended_calibration)
    print(f"done in {time.time() - started:.1f} s; recommended a = "
          f"{search_result['recommended']['a']}")
    for name in CANDIDATE_ORDER:
        row = next(row for row in candidate_rows
                   if row["candidate"] == name and row["split"] == "train"
                   and row["scope"] == "overall" and row["subset"] == "ge2"
                   and row["snr_db"] == 10.0)
        low = next(row for row in candidate_rows
                   if row["candidate"] == name and row["split"] == "train"
                   and row["scope"] == "overall" and row["subset"] == "ge2"
                   and row["snr_db"] == -10.0)
        print(f"{name}: Q(+10)={row['quality_mean']:.1f} rmse_pos={row['position_rmse']:.3f} "
              f"rmse_vel={row['velocity_rmse']:.3f} | Q(-10)={low['quality_mean']:.1f} "
              f"rmse_pos={low['position_rmse']:.3f} rmse_vel={low['velocity_rmse']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())