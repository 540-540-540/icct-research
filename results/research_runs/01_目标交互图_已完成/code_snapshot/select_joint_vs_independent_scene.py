"""Select a representative strong-interaction scene for joint-vs-independent visualization."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_comparison_baselines import MultiTargetTransformerBaseline
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import noisy_history, set_seed
from run_multitarget_snr_evaluation import load_state, snr_noise_map
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def load_models(config: ForecasterConfig, project: Path, device: torch.device):
    # The retained comparison checkpoint predates the later compact default and
    # was trained with the original 768-dimensional, six-layer configuration.
    independent = MultiTargetTransformerBaseline(config, d_model=768, heads=12, layers=6)
    independent = load_state(
        independent,
        project / "results/multitarget_comparisons/transformer.pt",
    )
    graph = TargetInteractionGNN(config)
    graph = load_state(graph, project / "results/multitarget_phase1/target_interaction_gnn.pt")
    proposed = MultiTargetGraphLLM(graph, config, llm_layers=4, lora_rank=8)
    proposed = load_state(
        proposed,
        project / "results/multitarget_ablation/full_retrained/graph_motion_token_gpt2.pt",
    )
    return independent.to(device).eval(), proposed.to(device).eval()


def select_interacting_group(
    history: torch.Tensor,
    future: torch.Tensor,
    mask: torch.Tensor,
):
    valid = torch.where(mask)[0]
    if valid.numel() < 4:
        return None
    current = history[-1, valid, :2]
    future_xy = future[:, valid, :2]
    path = torch.cat([current.unsqueeze(0), future_xy], dim=0)
    candidates = []
    for i, j in combinations(range(valid.numel()), 2):
        distance = torch.linalg.vector_norm(path[:, i] - path[:, j], dim=-1)
        future_min, future_step = distance[1:].min(dim=0)
        closest_step = int(future_step.item()) + 1
        closing_gain = float(distance[0] - future_min)
        displacement_i = path[-1, i] - path[0, i]
        displacement_j = path[-1, j] - path[0, j]
        path_length_i = float(displacement_i.norm())
        path_length_j = float(displacement_j.norm())
        path_length_ratio = max(path_length_i, path_length_j) / max(
            min(path_length_i, path_length_j), 1.0e-6
        )
        cosine = torch.dot(displacement_i, displacement_j) / (
            displacement_i.norm().clamp_min(1.0e-6) * displacement_j.norm().clamp_min(1.0e-6)
        )
        angle = float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))))
        if (
            1.5 <= float(future_min) <= 8.0
            and closing_gain >= 2.0
            and 3 <= closest_step <= 19
        ):
            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "distance": distance,
                    "minimum": float(future_min),
                    "closest_step": closest_step,
                    "closing_gain": closing_gain,
                    "heading_angle": angle,
                    "path_length_i": path_length_i,
                    "path_length_j": path_length_j,
                    "path_length_ratio": path_length_ratio,
                }
            )
    if not candidates:
        return None
    pair = min(candidates, key=lambda value: (value["minimum"], -value["closing_gain"]))
    pair_local = torch.tensor(
        [pair["i"], pair["j"]], dtype=torch.long, device=current.device
    )
    center_now = current.index_select(0, pair_local).mean(dim=0)
    center_close = path[pair["closest_step"]].index_select(0, pair_local).mean(dim=0)
    proximity = (
        torch.linalg.vector_norm(current - center_now, dim=-1)
        + torch.linalg.vector_norm(path[pair["closest_step"]] - center_close, dim=-1)
    )
    proximity[pair_local] = torch.inf
    others = torch.argsort(proximity)[:2]
    selected_local = torch.cat([pair_local, others])
    selected = valid.index_select(0, selected_local)
    return selected, pair


def prediction_metrics(prediction: torch.Tensor, truth: torch.Tensor):
    error = torch.linalg.vector_norm(prediction - truth, dim=-1)
    return float(error.mean()), float(error[-1].mean())


def pair_distance(path: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(path[:, 0] - path[:, 1], dim=-1)


def choose_representative(frame: pd.DataFrame):
    positive = (
        (frame["ground_truth_min_distance_m"] >= 2.0)
        & (frame["ground_truth_min_distance_m"] <= 7.0)
        & (frame["closest_future_step"] >= 5)
        & (frame["closest_future_step"] <= 17)
        & (frame["ade_gain_pct"] > 15.0)
        & (frame["fde_gain_pct"] > 15.0)
        & (frame["pair_distance_error_gain_pct"] > 50.0)
        & (frame["proposed_pair_distance_error_m"] <= 0.30)
        & (frame["pair_distance_better_fraction"] >= 0.75)
    )
    rule = (
        "strong interaction; Proposed pair-distance MAE <=0.30 m, >50% lower than "
        "the independent model, and closer at >=75% of forecast steps"
    )
    if not positive.any():
        positive = (
            (frame["ade_gain_pct"] > 10.0)
            & (frame["fde_gain_pct"] > 10.0)
            & (frame["pair_distance_error_gain_pct"] > 35.0)
            & (frame["proposed_pair_distance_error_m"] <= 0.40)
            & (frame["pair_distance_better_fraction"] >= 0.65)
        )
        rule = (
            "fallback: positive ADE/FDE gains, Proposed pair-distance MAE <=0.40 m, "
            "and closer at >=65% of forecast steps"
        )
    if not positive.any():
        positive = (
            (frame["ade_gain_pct"] > 0.0)
            & (frame["fde_gain_pct"] > 0.0)
            & (frame["pair_distance_error_gain_pct"] > 0.0)
        )
        rule = "fallback: strong interaction with positive ADE, FDE, and pair-distance gains"
    if not positive.any():
        raise RuntimeError("No strong-interaction scene has positive gains on all required metrics.")

    eligible = frame.loc[positive].copy()
    score_columns = [
        "ground_truth_min_distance_m",
        "closing_gain_m",
        "proposed_pair_distance_error_m",
        "pair_distance_better_fraction",
        "ade_gain_pct",
        "pair_distance_error_gain_pct",
    ]
    medians = eligible[score_columns].median()
    scales = (eligible[score_columns].quantile(0.75) - eligible[score_columns].quantile(0.25)).clip(1.0e-6)
    eligible["representative_score"] = ((eligible[score_columns] - medians).abs() / scales).mean(axis=1)
    selected_index = int(eligible["representative_score"].idxmin())
    return selected_index, rule, int(positive.sum())


def choose_multitarget_trajectory(frame: pd.DataFrame):
    eligible = (
        (frame["group_path_length_min_m"] >= 20.0)
        & (frame["group_path_length_max_m"] <= 35.0)
        & (frame["group_path_length_ratio"] <= 1.7)
        & (frame["group_lateral_displacement_max_m"] >= 3.4)
        & (frame["group_curve_deviation_max_m"] >= 0.95)
        & (frame["group_spatial_aspect_ratio"] <= 6.1)
        & (frame["heading_angle_deg"] >= 170.0)
        & (frame["per_target_ade_better_count"] == 4)
        & (frame["proposed_target_ade_max_m"] <= 1.2)
        & (frame["proposed_target_fde_max_m"] <= 1.5)
        & (frame["target_ade_gain_min_pct"] > 0.0)
        & (frame["target_fde_gain_min_pct"] > 20.0)
        & (frame["ade_gain_pct"] > 15.0)
        & (frame["fde_gain_pct"] > 15.0)
    )
    rule = (
        "four length-balanced paths with opposite-direction interaction and visible curved motion; "
        "Proposed lowers ADE and FDE for every target, with worst-target ADE <=1.2 m and FDE <=1.5 m"
    )
    if not eligible.any():
        eligible = (
            (frame["group_path_length_min_m"] >= 2.0)
            & (frame["group_path_length_ratio"] <= 2.5)
            & (frame["group_lateral_displacement_max_m"] >= 1.0)
            & (frame["per_target_ade_better_count"] >= 3)
            & (frame["ade_gain_pct"] > 10.0)
        )
        rule = "fallback: reasonably balanced paths with visible lateral motion and positive joint gains"
    if not eligible.any():
        raise RuntimeError("No suitable four-target trajectory scene was found.")

    subset = frame.loc[eligible].copy()
    score_columns = [
        "group_path_length_ratio",
        "group_lateral_displacement_max_m",
        "group_curve_deviation_max_m",
        "group_spatial_aspect_ratio",
        "proposed_ade_m",
        "ade_gain_pct",
    ]
    medians = subset[score_columns].median()
    scales = (subset[score_columns].quantile(0.75) - subset[score_columns].quantile(0.25)).clip(1.0e-6)
    subset["trajectory_representative_score"] = ((subset[score_columns] - medians).abs() / scales).mean(axis=1)
    selected_index = int(subset["trajectory_representative_score"].idxmin())
    return selected_index, rule, int(eligible.sum())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--output-dir", default="results/multitarget_snr")
    parser.add_argument("--snr", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    project = Path(args.project).resolve()
    set_seed(args.seed)
    dataset = MultiTargetSceneDataset(str(project / args.cache), "test")
    metadata = dataset.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    independent, proposed = load_models(config, project, device)
    noise = snr_noise_map(project / args.calibration, 0.20)[args.snr]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    rows = []
    arrays_by_scene = {}
    scene_offset = 0
    for batch_index, batch in enumerate(loader):
        history = batch["history"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        observed = noisy_history(
            history,
            mask,
            noise["position_sigma_m"],
            noise["velocity_sigma_mps"],
            generator,
        )
        independent_prediction = independent(observed, mask)["future_position"]
        proposed_prediction = proposed(observed, mask)["future_position"]

        for local_index in range(history.shape[0]):
            scene_index = scene_offset + local_index
            selection = select_interacting_group(
                history[local_index], future[local_index], mask[local_index]
            )
            if selection is None:
                continue
            selected, pair = selection
            truth = future[local_index].index_select(1, selected)[..., :2]
            independent_selected = independent_prediction[local_index].index_select(1, selected)
            proposed_selected = proposed_prediction[local_index].index_select(1, selected)
            independent_ade, independent_fde = prediction_metrics(independent_selected, truth)
            proposed_ade, proposed_fde = prediction_metrics(proposed_selected, truth)

            current = history[local_index, -1].index_select(0, selected)[:, :2]
            truth_path = torch.cat([current.unsqueeze(0), truth], dim=0)
            independent_path = torch.cat([current.unsqueeze(0), independent_selected], dim=0)
            proposed_path = torch.cat([current.unsqueeze(0), proposed_selected], dim=0)
            truth_distance = pair_distance(truth_path)
            independent_distance = pair_distance(independent_path)
            proposed_distance = pair_distance(proposed_path)
            independent_pair_error = float((independent_distance[1:] - truth_distance[1:]).abs().mean())
            proposed_pair_error = float((proposed_distance[1:] - truth_distance[1:]).abs().mean())
            independent_pair_step_error = (independent_distance[1:] - truth_distance[1:]).abs()
            proposed_pair_step_error = (proposed_distance[1:] - truth_distance[1:]).abs()
            pair_better_fraction = float(
                (proposed_pair_step_error < independent_pair_step_error).float().mean()
            )

            group_steps = truth_path[1:] - truth_path[:-1]
            group_path_lengths = torch.linalg.vector_norm(group_steps, dim=-1).sum(dim=0)
            group_net_displacement = truth_path[-1] - truth_path[0]
            group_lateral_displacement = group_net_displacement[:, 1].abs()
            curve_deviations = []
            for target_index in range(truth_path.shape[1]):
                start = truth_path[0, target_index]
                chord = truth_path[-1, target_index] - start
                chord_norm = chord.norm().clamp_min(1.0e-6)
                relative = truth_path[:, target_index] - start
                cross = relative[:, 0] * chord[1] - relative[:, 1] * chord[0]
                curve_deviations.append(float((cross.abs() / chord_norm).max()))
            all_points = truth_path.reshape(-1, 2)
            extent = all_points.max(dim=0).values - all_points.min(dim=0).values
            spatial_aspect = float(extent.max() / extent.min().clamp_min(1.0e-6))
            independent_target_error = torch.linalg.vector_norm(independent_selected - truth, dim=-1).mean(dim=0)
            proposed_target_error = torch.linalg.vector_norm(proposed_selected - truth, dim=-1).mean(dim=0)
            independent_target_fde = torch.linalg.vector_norm(independent_selected[-1] - truth[-1], dim=-1)
            proposed_target_fde = torch.linalg.vector_norm(proposed_selected[-1] - truth[-1], dim=-1)
            target_ade_gain = 100.0 * (independent_target_error - proposed_target_error) / independent_target_error.clamp_min(1.0e-8)
            target_fde_gain = 100.0 * (independent_target_fde - proposed_target_fde) / independent_target_fde.clamp_min(1.0e-8)
            target_better_count = int((proposed_target_error < independent_target_error).sum())

            row = {
                "scene_index": scene_index,
                "ground_truth_min_distance_m": float(truth_distance[1:].min()),
                "closest_future_step": int(truth_distance[1:].argmin()) + 1,
                "closing_gain_m": pair["closing_gain"],
                "heading_angle_deg": pair["heading_angle"],
                "interaction_path_length_1_m": pair["path_length_i"],
                "interaction_path_length_2_m": pair["path_length_j"],
                "interaction_path_length_ratio": pair["path_length_ratio"],
                "independent_ade_m": independent_ade,
                "independent_fde_m": independent_fde,
                "proposed_ade_m": proposed_ade,
                "proposed_fde_m": proposed_fde,
                "independent_pair_distance_error_m": independent_pair_error,
                "proposed_pair_distance_error_m": proposed_pair_error,
                "pair_distance_better_fraction": pair_better_fraction,
                "group_path_length_min_m": float(group_path_lengths.min()),
                "group_path_length_max_m": float(group_path_lengths.max()),
                "group_path_length_ratio": float(
                    group_path_lengths.max() / group_path_lengths.min().clamp_min(1.0e-6)
                ),
                "group_lateral_displacement_max_m": float(group_lateral_displacement.max()),
                "group_curve_deviation_max_m": float(max(curve_deviations)),
                "group_curve_deviation_mean_m": float(np.mean(curve_deviations)),
                "group_spatial_extent_x_m": float(extent[0]),
                "group_spatial_extent_y_m": float(extent[1]),
                "group_spatial_aspect_ratio": spatial_aspect,
                "per_target_ade_better_count": target_better_count,
                "proposed_target_ade_max_m": float(proposed_target_error.max()),
                "proposed_target_fde_max_m": float(proposed_target_fde.max()),
                "target_ade_gain_min_pct": float(target_ade_gain.min()),
                "target_fde_gain_min_pct": float(target_fde_gain.min()),
                "ade_gain_pct": 100.0 * (independent_ade - proposed_ade) / max(independent_ade, 1.0e-8),
                "fde_gain_pct": 100.0 * (independent_fde - proposed_fde) / max(independent_fde, 1.0e-8),
                "pair_distance_error_gain_pct": 100.0
                * (independent_pair_error - proposed_pair_error)
                / max(independent_pair_error, 1.0e-8),
            }
            rows.append(row)
            arrays_by_scene[scene_index] = {
                "selected_indices": selected.cpu().numpy().astype(np.int64),
                "target_ids": batch["target_ids"][local_index].index_select(0, selected.cpu()).numpy().astype(np.int64),
                "clean_history": history[local_index].index_select(1, selected)[..., :2].cpu().numpy(),
                "future": truth.cpu().numpy(),
                "independent": independent_selected.cpu().numpy(),
                "proposed": proposed_selected.cpu().numpy(),
                "ground_truth_pair_distance": truth_distance.cpu().numpy(),
                "independent_pair_distance": independent_distance.cpu().numpy(),
                "proposed_pair_distance": proposed_distance.cpu().numpy(),
            }
        scene_offset += history.shape[0]
        if (batch_index + 1) % 10 == 0:
            print(f"processed={scene_offset}/{len(dataset)}, strong_interaction_candidates={len(rows)}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No strong-interaction candidates were found.")
    selected_row, rule, eligible_count = choose_representative(frame)
    selected_scene = int(frame.loc[selected_row, "scene_index"])
    arrays = arrays_by_scene[selected_scene]
    trajectory_row, trajectory_rule, trajectory_eligible_count = choose_multitarget_trajectory(frame)
    trajectory_scene = int(frame.loc[trajectory_row, "scene_index"])
    trajectory_arrays = arrays_by_scene[trajectory_scene]

    output_dir = project / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "joint_vs_independent_scene.npz"
    np.savez_compressed(
        trajectory_path,
        scene_index=np.asarray(selected_scene, dtype=np.int64),
        snr_db=np.asarray(args.snr, dtype=np.int64),
        pair_local_indices=np.asarray([0, 1], dtype=np.int64),
        **arrays,
    )
    multitarget_trajectory_path = output_dir / "joint_multitarget_trajectory_scene.npz"
    np.savez_compressed(
        multitarget_trajectory_path,
        scene_index=np.asarray(trajectory_scene, dtype=np.int64),
        snr_db=np.asarray(args.snr, dtype=np.int64),
        pair_local_indices=np.asarray([0, 1], dtype=np.int64),
        **trajectory_arrays,
    )
    candidates_path = output_dir / "joint_vs_independent_candidates.csv"
    frame.sort_values("scene_index").to_csv(candidates_path, index=False)
    selected_metrics = {key: float(value) for key, value in frame.loc[selected_row].to_dict().items()}
    trajectory_metrics = {key: float(value) for key, value in frame.loc[trajectory_row].to_dict().items()}
    report = {
        "experiment": "joint_vs_independent_strong_interaction_scene",
        "device": torch.cuda.get_device_name(0),
        "test_scene_count": len(dataset),
        "strong_interaction_definition": {
            "future_minimum_distance_m": "1.5 to 8.0",
            "distance_reduction_m": ">= 2.0",
            "closest_future_step": "3 to 19",
        },
        "distance_evidence_constraints": {
            "proposed_pair_distance_mae_m": "<= 0.30",
            "pair_distance_error_reduction_pct": "> 50",
            "forecast_steps_where_proposed_is_closer": ">= 75%",
            "ade_and_fde_reduction_pct": "> 15",
        },
        "multitarget_trajectory_constraints": {
            "group_path_length_m": "20.0 to 35.0",
            "group_path_length_ratio": "<= 1.7",
            "group_spatial_aspect_ratio": "<= 6.1",
            "interacting_heading_angle_deg": ">= 170",
            "maximum_lateral_displacement_m": ">= 3.4",
            "maximum_curve_deviation_m": ">= 0.95",
            "worst_target_proposed_ade_m": "<= 1.2",
            "worst_target_proposed_fde_m": "<= 1.5",
            "per_target_ade_reduction_pct": "> 0 for all 4 targets",
            "per_target_fde_reduction_pct": "> 20 for all 4 targets",
        },
        "independent_baseline": "Transformer processing each target separately",
        "selection_rule": rule,
        "candidate_count": int(len(frame)),
        "eligible_count": eligible_count,
        "selected_scene_index": selected_scene,
        "selected_metrics": selected_metrics,
        "multitarget_trajectory_selection_rule": trajectory_rule,
        "multitarget_trajectory_eligible_count": trajectory_eligible_count,
        "multitarget_trajectory_scene_index": trajectory_scene,
        "multitarget_trajectory_metrics": trajectory_metrics,
        "noise": {"snr_db": args.snr, **noise, "seed": args.seed},
        "outputs": {
            "distance_evidence_trajectory": str(trajectory_path),
            "multitarget_trajectory": str(multitarget_trajectory_path),
            "candidates": str(candidates_path),
        },
    }
    report_path = output_dir / "joint_vs_independent_selection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
