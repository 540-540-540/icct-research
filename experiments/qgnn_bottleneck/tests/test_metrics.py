"""Small executable acceptance checks for BDX-01 metric rules."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qgnn_bottleneck.metrics import eligible_mask, evaluate_metrics, scene_rows, worst_overlap


def run():
    exists = np.ones((1, 20, 8), bool)
    exists[:, :, 2:] = False
    eligible = eligible_mask(exists)
    future = np.zeros((1, 20, 8, 2), np.float64)
    pred = np.zeros_like(future)
    pred[:, :, 0, 0] = 1
    pred[:, :5, 1, 0] = 3
    valid = np.zeros((1, 20, 8), bool)
    valid[:, :, 0] = True
    valid[:, :5, 1] = True
    m20 = evaluate_metrics(pred, future, valid, eligible, (20,))[0]
    assert abs(m20["ADE_point_trainrule_eval"] - 1.4) < 1e-12
    assert abs(m20["ADE"] - 2.0) < 1e-12
    assert m20["fde_targets"] == 1 and abs(m20["FDE"] - 1.0) < 1e-12
    valid[:, :, 1] = True
    pred[:, :, 1, 0] = 3
    m20 = evaluate_metrics(pred, future, valid, eligible, (20,))[0]
    assert abs(m20["ADE_point_trainrule_eval"] - m20["ADE"]) < 1e-12

    # Ineligible, partial future, missing step 20, and wholly empty scenes.
    exists2 = np.zeros((4, 20, 8), bool)
    exists2[0, -2:, 0] = True
    exists2[1, -3:, 0] = True
    exists2[2, -3:, 0] = True
    valid2 = np.zeros((4, 20, 8), bool)
    valid2[1, :5, 0] = True
    valid2[2, :19, 0] = True
    e2 = eligible_mask(exists2)
    p2 = np.ones((4, 20, 8, 2)) / np.sqrt(2)
    y2 = np.zeros_like(p2)
    rows = scene_rows(p2, y2, valid2, e2)
    assert rows[0]["ADE"] is None and abs(rows[1]["ADE"] - 1.0) < 1e-12
    assert rows[2]["FDE"] is None and rows[3]["ADE"] is None
    horizons = evaluate_metrics(p2, y2, valid2, e2)
    assert [r["horizon_steps"] for r in horizons] == [5, 10, 20]
    assert horizons[-1]["fde_scenes"] == 0
    overlap = worst_overlap([(0, 0), (1, 0), (2, 0), (3, 0)], [4, 3, 2, 1], [4, 1, 3, 2], .5)
    assert overlap["k"] == 2 and overlap["intersection"] == 1
    episode_values = {"a":[1.0, 1.0, 1.0], "b":[3.0]}
    weighted = np.mean([v for values in episode_values.values() for v in values])
    equal_episode = np.mean([np.mean(values) for values in episode_values.values()])
    assert weighted == 1.5 and equal_episode == 2.0
    return dict(status="passed", point_weighted_ADE=1.4, target_macro_ADE=2.0,
                FDE_does_not_use_last_valid=True, empty_scene_retained=True,
                horizons_seconds=[.5, 1.0, 2.0], explicit_paired_key_test=True,
                episode_weighted_recovers_target_overall=weighted, equal_episode_mean=equal_episode)


if __name__ == "__main__":
    print(run())
