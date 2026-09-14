"""Select and export a transparent representative multi-target trajectory scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import noisy_history, set_seed
from run_multitarget_snr_evaluation import load_state, snr_noise_map
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def select_four_interacting_targets(item: dict) -> Optional[np.ndarray]:
    valid = torch.where(item["mask"])[0]
    if len(valid) < 4:
        return None
    positions = item["history"][-1, valid, :2]
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(1.0e6)
    minimum = float(distances.min())
    if not 4.0 <= minimum <= 15.0:
        return None
    pair = torch.nonzero(distances == distances.min(), as_tuple=False)[0]
    center = positions[pair].mean(dim=0)
    nearest = torch.argsort(torch.linalg.vector_norm(positions - center, dim=-1))[:4]
    return valid[nearest].numpy()


def metrics(prediction: torch.Tensor, future: torch.Tensor) -> tuple[float, float, np.ndarray, np.ndarray]:
    distance = torch.linalg.vector_norm(prediction - future, dim=-1)
    target_ade = distance.mean(dim=0).cpu().numpy()
    target_fde = distance[-1].cpu().numpy()
    return float(distance.mean()), float(distance[-1].mean()), target_ade, target_fde


def load_models(config: ForecasterConfig, project: Path, device: torch.device) -> Dict[str, torch.nn.Module]:
    graph_checkpoint = project / "results/multitarget_phase1/target_interaction_gnn.pt"
    gnn = load_state(TargetInteractionGNN(config), graph_checkpoint)
    proposed_graph = load_state(TargetInteractionGNN(config), graph_checkpoint)
    proposed = MultiTargetGraphLLM(proposed_graph, config, llm_layers=4, lora_rank=8)
    proposed = load_state(
        proposed,
        project / "results/multitarget_ablation/full_retrained/graph_motion_token_gpt2.pt",
    )
    return {"target_interaction_gnn": gnn.to(device).eval(), "proposed": proposed.to(device).eval()}


@torch.inference_mode()
def evaluate_candidate(
    item: dict,
    selected: np.ndarray,
    scene_index: int,
    models: Dict[str, torch.nn.Module],
    noise_map: dict,
    device: torch.device,
    base_seed: int,
) -> tuple[dict, dict]:
    history = item["history"].unsqueeze(0).to(device)
    future = item["future"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
    selected_future = future[0].index_select(1, selected_tensor)[..., :2]
    row = {"scene_index": scene_index}
    arrays = {
        "clean_history": history[0].index_select(1, selected_tensor)[..., :2].cpu().numpy(),
        "future": selected_future.cpu().numpy(),
        "selected_indices": selected.astype(np.int64),
        "target_ids": item["target_ids"][selected].numpy().astype(np.int64),
    }
    for snr in (5, 20):
        generator = torch.Generator(device=device).manual_seed(base_seed + snr * 100000 + scene_index)
        observed = noisy_history(
            history,
            mask,
            noise_map[snr]["position_sigma_m"],
            noise_map[snr]["velocity_sigma_mps"],
            generator,
        )
        arrays[f"observed_snr{snr}"] = observed[0].index_select(1, selected_tensor)[..., :2].cpu().numpy()
        for model_key, model in models.items():
            prediction = model(observed, mask)["future_position"][0].index_select(1, selected_tensor)
            ade, fde, target_ade, target_fde = metrics(prediction, selected_future)
            row[f"{model_key}_ade_snr{snr}"] = ade
            row[f"{model_key}_fde_snr{snr}"] = fde
            arrays[f"{model_key}_snr{snr}"] = prediction.cpu().numpy()
            arrays[f"{model_key}_target_ade_snr{snr}"] = target_ade
            arrays[f"{model_key}_target_fde_snr{snr}"] = target_fde
        row[f"ade_gain_snr{snr}"] = (
            row[f"target_interaction_gnn_ade_snr{snr}"] - row[f"proposed_ade_snr{snr}"]
        ) / row[f"target_interaction_gnn_ade_snr{snr}"]
        row[f"fde_gain_snr{snr}"] = (
            row[f"target_interaction_gnn_fde_snr{snr}"] - row[f"proposed_fde_snr{snr}"]
        ) / row[f"target_interaction_gnn_fde_snr{snr}"]
        row[f"target_error_cv_snr{snr}"] = float(
            np.std(arrays[f"proposed_target_ade_snr{snr}"]) /
            max(np.mean(arrays[f"proposed_target_ade_snr{snr}"]), 1.0e-8)
        )
    return row, arrays


def choose_representative(frame: pd.DataFrame) -> tuple[int, pd.DataFrame, dict]:
    metric_cols = [
        "proposed_ade_snr5", "proposed_fde_snr5",
        "proposed_ade_snr20", "proposed_fde_snr20",
    ]
    improvement_cols = ["ade_gain_snr5", "fde_gain_snr5", "ade_gain_snr20", "fde_gain_snr20"]
    quantiles = frame[metric_cols].quantile([0.25, 0.5, 0.75])
    positive = (frame[improvement_cols] > 0).all(axis=1)
    middle = ((frame[metric_cols] >= quantiles.loc[0.25]) & (frame[metric_cols] <= quantiles.loc[0.75])).all(axis=1)
    eligible = positive & middle
    rule = "positive ADE/FDE gains at 5 and 20 dB; all four proposed metrics within candidate IQR"
    if not eligible.any():
        quantiles = frame[metric_cols].quantile([0.10, 0.5, 0.90])
        middle = ((frame[metric_cols] >= quantiles.loc[0.10]) & (frame[metric_cols] <= quantiles.loc[0.90])).all(axis=1)
        eligible = positive & middle
        rule = "fallback: positive ADE/FDE gains at 5 and 20 dB; all four proposed metrics within 10th-90th percentiles"
    if not eligible.any():
        eligible = positive
        rule = "fallback: positive ADE/FDE gains at 5 and 20 dB"
    if not eligible.any():
        raise RuntimeError("No candidate scene improves both ADE and FDE at both SNR levels.")

    medians = frame[metric_cols].median()
    iqr = (frame[metric_cols].quantile(0.75) - frame[metric_cols].quantile(0.25)).clip(lower=1.0e-6)
    frame = frame.copy()
    frame["median_distance_score"] = ((frame[metric_cols] - medians).abs() / iqr).mean(axis=1)
    frame["imbalance_score"] = frame[["target_error_cv_snr5", "target_error_cv_snr20"]].mean(axis=1)
    frame["selection_score"] = frame["median_distance_score"] + 0.25 * frame["imbalance_score"]
    frame["eligible"] = eligible
    selected_index = int(frame.loc[eligible, "selection_score"].idxmin())
    metadata = {
        "rule": rule,
        "candidate_count": int(len(frame)),
        "positive_gain_count": int(positive.sum()),
        "eligible_count": int(eligible.sum()),
        "metric_medians": {key: float(value) for key, value in medians.items()},
        "selected_row_index": selected_index,
    }
    return selected_index, frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--output-dir", default="results/multitarget_snr")
    parser.add_argument("--seed", type=int, default=6026)
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
    device = torch.device("cuda")
    models = load_models(config, project, device)
    noise = snr_noise_map(project / args.calibration, 0.20)

    rows = []
    arrays_by_scene = {}
    for scene_index in range(len(dataset)):
        item = dataset[scene_index]
        selected = select_four_interacting_targets(item)
        if selected is None:
            continue
        row, arrays = evaluate_candidate(
            item, selected, scene_index, models, noise, device, args.seed
        )
        rows.append(row)
        arrays_by_scene[scene_index] = arrays
    frame = pd.DataFrame(rows)
    selected_row, frame, selection = choose_representative(frame)
    selected_scene = int(frame.loc[selected_row, "scene_index"])
    arrays = arrays_by_scene[selected_scene]

    output_dir = project / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "trajectory_examples_revised.npz",
        scene_index=np.asarray(selected_scene, dtype=np.int64),
        **arrays,
    )
    frame.sort_values("selection_score").to_csv(
        output_dir / "trajectory_selection_candidates.csv", index=False
    )
    selected_metrics = frame.loc[selected_row].to_dict()
    report = {
        "experiment": "representative_interaction_scene_selection",
        "device": torch.cuda.get_device_name(0),
        "seed_protocol": "scene seed = base seed + snr * 100000 + scene index",
        "base_seed": args.seed,
        "selection": selection,
        "selected_scene_index": selected_scene,
        "selected_indices": arrays["selected_indices"].tolist(),
        "target_ids": arrays["target_ids"].tolist(),
        "selected_metrics": {
            key: (bool(value) if isinstance(value, (np.bool_, bool)) else float(value))
            for key, value in selected_metrics.items()
            if key != "scene_index"
        },
        "outputs": {
            "trajectory": str(output_dir / "trajectory_examples_revised.npz"),
            "candidates": str(output_dir / "trajectory_selection_candidates.csv"),
        },
    }
    (output_dir / "trajectory_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
