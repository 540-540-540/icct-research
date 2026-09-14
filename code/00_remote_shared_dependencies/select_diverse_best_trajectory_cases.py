"""Select four diverse test trajectories where Proposed wins both ADE and FDE.

The full test split is evaluated at one fixed SNR.  Selection is performed per
target, not per scene.  A candidate is eligible only when the Proposed method
has the lowest ADE and the lowest FDE among all compared methods.  Four cases
are then selected from distinct scenes using motion-shape diversity features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_comparison_baselines import MultiTargetLSTMBaseline, MultiTargetTransformerBaseline
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import noisy_history, set_seed
from run_multitarget_snr_evaluation import load_state, snr_noise_map
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


MODEL_KEYS = ("lstm", "transformer", "target_interaction_gnn", "proposed")


def load_models(config: ForecasterConfig, project: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    comparison = project / "results/multitarget_comparisons_final"
    graph_checkpoint = project / "results/multitarget_phase1/target_interaction_gnn.pt"
    proposed_graph = load_state(TargetInteractionGNN(config), graph_checkpoint)
    models = {
        "lstm": load_state(
            MultiTargetLSTMBaseline(config, hidden_dim=256, layers=2), comparison / "lstm.pt"
        ),
        "transformer": load_state(
            MultiTargetTransformerBaseline(config, d_model=256, heads=8, layers=4),
            comparison / "transformer.pt",
        ),
        "target_interaction_gnn": load_state(TargetInteractionGNN(config), graph_checkpoint),
        "proposed": load_state(
            MultiTargetGraphLLM(proposed_graph, config, llm_layers=4, lora_rank=8),
            project / "results/multitarget_ablation/full_retrained/graph_motion_token_gpt2.pt",
        ),
    }
    return {name: model.to(device).eval() for name, model in models.items()}


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def localize_trajectory(history: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Translate to the prediction origin and align recent motion with +x."""
    origin = history[-1]
    velocity = history[-1] - history[max(0, len(history) - 4)]
    if float(np.linalg.norm(velocity)) < 1.0e-6:
        velocity = future[min(2, len(future) - 1)] - origin
    heading = float(np.arctan2(velocity[1], velocity[0]))
    cosine, sine = np.cos(heading), np.sin(heading)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return (history - origin) @ rotation.T, (future - origin) @ rotation.T, heading


def motion_features(history: np.ndarray, future: np.ndarray) -> dict[str, float]:
    local_history, local_future, heading = localize_trajectory(history, future)
    path = np.vstack([np.zeros((1, 2), dtype=np.float64), local_future])
    steps = np.diff(path, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = lengths > 1.0e-6
    headings = np.unwrap(np.arctan2(steps[valid, 1], steps[valid, 0]))
    heading_change = float(wrap_angle(headings[-1] - headings[0])) if len(headings) >= 2 else 0.0
    path_length = float(lengths.sum())
    direct = float(np.linalg.norm(local_future[-1]))
    cross_track = float(np.max(np.abs(local_future[:, 1])))
    net_lateral = float(local_future[-1, 1])
    speed_first = float(lengths[: max(1, len(lengths) // 3)].mean())
    speed_last = float(lengths[-max(1, len(lengths) // 3) :].mean())
    return {
        "initial_heading_rad": heading,
        "net_longitudinal_m": float(local_future[-1, 0]),
        "net_lateral_m": net_lateral,
        "path_length_m": path_length,
        "path_efficiency": direct / max(path_length, 1.0e-8),
        "heading_change_deg": float(np.degrees(heading_change)),
        "max_lateral_excursion_m": cross_track,
        "speed_ratio": speed_last / max(speed_first, 1.0e-8),
    }


def motion_label(row: pd.Series) -> str:
    lateral = float(row["net_lateral_m"])
    turn = float(row["heading_change_deg"])
    excursion = float(row["max_lateral_excursion_m"])
    if abs(turn) <= 8.0 and abs(lateral) <= 1.0 and excursion <= 1.25:
        return "straight"
    if abs(turn) <= 10.0 and abs(lateral) >= 1.0:
        return "lateral_shift"
    if turn >= 12.0 or lateral >= 2.5:
        return "left_curve"
    if turn <= -12.0 or lateral <= -2.5:
        return "right_curve"
    return "mild_curve"


def robust_standardize(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    standardized = (values - median) / scale
    return np.clip(standardized, -4.0, 4.0)


def choose_diverse(frame: pd.DataFrame, count: int = 4) -> tuple[pd.DataFrame, dict]:
    eligible = frame.loc[frame["eligible"]].copy()
    if len(eligible) < count:
        raise RuntimeError(f"Only {len(eligible)} candidates satisfy simultaneous ADE/FDE optimality.")

    # Remove nearly stationary cases and gross error outliers before visual case selection.
    error_cap_ade = float(eligible["proposed_ade_m"].quantile(0.80))
    error_cap_fde = float(eligible["proposed_fde_m"].quantile(0.80))
    pool = eligible.loc[
        (eligible["path_length_m"] >= 3.0)
        & (eligible["proposed_ade_m"] <= error_cap_ade)
        & (eligible["proposed_fde_m"] <= error_cap_fde)
    ].copy()
    if len(pool) < count:
        pool = eligible.loc[eligible["path_length_m"] >= 3.0].copy()
    if len(pool) < count:
        pool = eligible.copy()

    pool["motion_type"] = pool.apply(motion_label, axis=1)
    pool = pool.loc[pool["motion_type"] != "straight"].copy()
    pool["heading_sin"] = np.sin(pool["initial_heading_rad"].to_numpy(dtype=np.float64))
    pool["heading_cos"] = np.cos(pool["initial_heading_rad"].to_numpy(dtype=np.float64))
    pool["lateral_ratio"] = (
        pool["net_lateral_m"].abs() / pool["path_length_m"].clip(lower=1.0e-8)
    )
    pool = pool.loc[
        pool["heading_change_deg"].abs().ge(35.0)
        & pool["lateral_ratio"].ge(0.15)
    ].copy()
    if len(pool) < count:
        raise RuntimeError(
            "Fewer than four candidates satisfy the strong-curvature rule: "
            "abs(heading change) >= 35 deg and lateral/path ratio >= 0.15."
        )
    feature_columns = [
        "net_lateral_m",
        "heading_change_deg",
        "max_lateral_excursion_m",
        "path_efficiency",
        "speed_ratio",
        "heading_sin",
        "heading_cos",
    ]
    features = robust_standardize(pool, feature_columns)
    ade_rank = pool["proposed_ade_m"].rank(pct=True).to_numpy()
    fde_rank = pool["proposed_fde_m"].rank(pct=True).to_numpy()
    gain_rank = (
        pool["ade_margin_m"].rank(pct=True, ascending=False).to_numpy()
        + pool["fde_margin_m"].rank(pct=True, ascending=False).to_numpy()
    ) / 2.0
    quality_penalty = 0.40 * ade_rank + 0.40 * fde_rank + 0.20 * gain_rank

    selected_positions: list[int] = []
    selected_scenes: set[int] = set()
    selected_labels: set[str] = set()
    category_counts = pool["motion_type"].value_counts().to_dict()
    requirements = ("vertical_strong_curve", "right_curve", "left_curve", "diverse_strong_curve")
    vertical_mask = pool["heading_cos"].abs().le(0.35)
    if not vertical_mask.any():
        raise RuntimeError("No strong-curvature candidate has a near-vertical global direction.")

    for required_type in requirements:
        distances = np.linalg.norm(features[:, None, :] - features[selected_positions][None, :, :], axis=2)
        min_distance = distances.min(axis=1) if selected_positions else np.zeros(len(pool))
        score = min_distance - 0.45 * quality_penalty
        for position in range(len(pool)):
            is_required_type = (
                bool(vertical_mask.iloc[position])
                if required_type == "vertical_strong_curve"
                else (
                    True
                    if required_type == "diverse_strong_curve"
                    else str(pool.iloc[position]["motion_type"]) == required_type
                )
            )
            if (
                not is_required_type
                or position in selected_positions
                or int(pool.iloc[position]["scene_index"]) in selected_scenes
            ):
                score[position] = -np.inf
        choice = int(np.argmax(score))
        if not np.isfinite(score[choice]):
            raise RuntimeError(f"Could not select a distinct-scene case for {required_type}.")
        selected_positions.append(choice)
        selected_scenes.add(int(pool.iloc[choice]["scene_index"]))
        selected_labels.add(str(pool.iloc[choice]["motion_type"]))

    selected = pool.iloc[selected_positions].copy().reset_index(drop=True)
    selected["panel_index"] = np.arange(1, len(selected) + 1)
    report = {
        "full_target_count": int(len(frame)),
        "simultaneous_ade_fde_winner_count": int(len(eligible)),
        "selection_pool_count": int(len(pool)),
        "error_filter": {
            "proposed_ade_80th_percentile_m": error_cap_ade,
            "proposed_fde_80th_percentile_m": error_cap_fde,
            "minimum_path_length_m": 3.0,
        },
        "diversity_features": feature_columns,
        "required_motion_types": list(requirements),
        "strong_curvature_rule": "abs(heading change) >= 35 deg and abs(local lateral displacement)/path length >= 0.15",
        "vertical_case_rule": "abs(cos(initial heading)) <= 0.35",
        "motion_type_counts_in_pool": {str(key): int(value) for key, value in category_counts.items()},
        "distinct_scene_constraint": True,
    }
    return selected, report


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
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
    device = torch.device("cuda")
    models = load_models(config, project, device)
    noise = snr_noise_map(project / args.calibration, 0.20)
    number_of_scenes = len(dataset) if args.max_scenes is None else min(len(dataset), args.max_scenes)

    rows: list[dict] = []
    arrays: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for scene_index in range(number_of_scenes):
        item = dataset[scene_index]
        history = item["history"].unsqueeze(0).to(device)
        future = item["future"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        generator = torch.Generator(device=device).manual_seed(
            args.seed + args.snr * 100000 + scene_index
        )
        observed = noisy_history(
            history,
            mask,
            noise[args.snr]["position_sigma_m"],
            noise[args.snr]["velocity_sigma_mps"],
            generator,
        )
        predictions = {
            key: model(observed, mask)["future_position"][0, ..., :2].cpu()
            for key, model in models.items()
        }
        valid_indices = torch.where(item["mask"])[0].tolist()
        for target_index in valid_indices:
            target_future = item["future"][:, target_index, :2].cpu()
            metrics: dict[str, float] = {}
            for key in MODEL_KEYS:
                distance = torch.linalg.vector_norm(predictions[key][:, target_index] - target_future, dim=-1)
                metrics[f"{key}_ade_m"] = float(distance.mean())
                metrics[f"{key}_fde_m"] = float(distance[-1])
            baseline_ade = min(metrics[f"{key}_ade_m"] for key in MODEL_KEYS[:-1])
            baseline_fde = min(metrics[f"{key}_fde_m"] for key in MODEL_KEYS[:-1])
            eligible = (
                metrics["proposed_ade_m"] <= baseline_ade + args.tolerance
                and metrics["proposed_fde_m"] <= baseline_fde + args.tolerance
            )
            clean_history = item["history"][:, target_index, :2].cpu().numpy()
            clean_future = target_future.numpy()
            row = {
                "scene_index": scene_index,
                "target_index": int(target_index),
                "target_id": int(item["target_ids"][target_index]),
                "eligible": bool(eligible),
                "ade_margin_m": baseline_ade - metrics["proposed_ade_m"],
                "fde_margin_m": baseline_fde - metrics["proposed_fde_m"],
                **metrics,
                **motion_features(clean_history, clean_future),
            }
            rows.append(row)
            if eligible:
                origin = clean_history[-1]
                translated_history = clean_history - origin
                translated_future = clean_future - origin
                translated_predictions = {}
                for key in MODEL_KEYS:
                    translated_predictions[key] = (
                        predictions[key][:, target_index].numpy() - origin
                    ).astype(np.float32)
                arrays[(scene_index, int(target_index))] = {
                    "clean_history": translated_history.astype(np.float32),
                    "future": translated_future.astype(np.float32),
                    **translated_predictions,
                }
        if (scene_index + 1) % 100 == 0 or scene_index + 1 == number_of_scenes:
            winners = sum(int(row["eligible"]) for row in rows)
            print(
                f"processed_scenes={scene_index + 1}/{number_of_scenes} "
                f"targets={len(rows)} simultaneous_winners={winners}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    selected, selection_report = choose_diverse(frame, count=4)
    selected_arrays = [arrays[(int(row.scene_index), int(row.target_index))] for row in selected.itertuples()]
    output_dir = project / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "diverse_best_trajectory_cases"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        snr_db=np.asarray(args.snr, dtype=np.int64),
        scene_indices=selected["scene_index"].to_numpy(dtype=np.int64),
        target_indices=selected["target_index"].to_numpy(dtype=np.int64),
        target_ids=selected["target_id"].to_numpy(dtype=np.int64),
        motion_types=selected["motion_type"].to_numpy(dtype="U32"),
        clean_history=np.stack([case["clean_history"] for case in selected_arrays]),
        future=np.stack([case["future"] for case in selected_arrays]),
        **{
            key: np.stack([case[key] for case in selected_arrays])
            for key in MODEL_KEYS
        },
    )
    frame["motion_type"] = frame.apply(motion_label, axis=1)
    frame.to_csv(output_dir / f"{stem}_all_targets.csv", index=False)
    selected.to_csv(output_dir / f"{stem}_selected.csv", index=False)
    report = {
        "experiment": "full_test_diverse_per_target_qualitative_selection",
        "device": torch.cuda.get_device_name(0),
        "test_scene_count": int(number_of_scenes),
        "dataset_test_scene_count": int(len(dataset)),
        "snr_db": int(args.snr),
        "base_seed": int(args.seed),
        "noise_seed_protocol": "base seed + snr * 100000 + scene index",
        "eligibility_rule": "Proposed ADE <= every baseline ADE AND Proposed FDE <= every baseline FDE",
        "coordinate_frame": "global x-y axes translated to the last observed position; no rotation",
        "selection": selection_report,
        "selected_cases": selected.to_dict(orient="records"),
        "outputs": {
            "npz": str(output_dir / f"{stem}.npz"),
            "all_targets_csv": str(output_dir / f"{stem}_all_targets.csv"),
            "selected_csv": str(output_dir / f"{stem}_selected.csv"),
        },
    }
    (output_dir / f"{stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--output-dir", default="results/multitarget_snr")
    parser.add_argument("--snr", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6026)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
