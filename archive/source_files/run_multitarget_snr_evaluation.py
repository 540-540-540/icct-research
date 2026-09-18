"""Evaluate trained multi-target predictors at empirically calibrated SNR levels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_comparison_baselines import (
    MultiTargetLSTMBaseline,
    MultiTargetTCNBaseline,
    MultiTargetTransformerBaseline,
)
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, evaluate_constant_velocity, noisy_history, set_seed
from target_interaction_graph import ForecasterConfig, IndependentGRUForecaster, TargetInteractionGNN


def load_state(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return model


def snr_noise_map(calibration_path: Path, velocity_at_20db: float) -> Dict[int, Dict[str, float]]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    position = {
        int(key): float(value["position_per_axis_rmse_m"])
        for key, value in payload["calibration"].items()
    }
    reference = position[20]
    return {
        snr: {
            "position_sigma_m": sigma,
            "velocity_sigma_mps": velocity_at_20db * sigma / reference,
        }
        for snr, sigma in sorted(position.items())
    }


def make_models(
    config: ForecasterConfig,
    project: Path,
    proposed_checkpoint: Path,
) -> Dict[str, torch.nn.Module]:
    phase1 = project / "results/multitarget_phase1"
    comparison = project / "results/multitarget_comparisons_final"
    graph_checkpoint = phase1 / "target_interaction_gnn.pt"

    models: Dict[str, torch.nn.Module] = {
        "independent_gru": load_state(IndependentGRUForecaster(config), phase1 / "independent_gru.pt"),
        "lstm": load_state(MultiTargetLSTMBaseline(config, hidden_dim=256, layers=2), comparison / "lstm.pt"),
        "tcn": load_state(MultiTargetTCNBaseline(config, channels=(64, 128, 256)), comparison / "tcn.pt"),
        "transformer": load_state(
            MultiTargetTransformerBaseline(config, d_model=256, heads=8, layers=4),
            comparison / "transformer.pt",
        ),
        "target_interaction_gnn": load_state(TargetInteractionGNN(config), graph_checkpoint),
    }
    proposed_graph = load_state(TargetInteractionGNN(config), graph_checkpoint)
    proposed = MultiTargetGraphLLM(proposed_graph, config, llm_layers=4, lora_rank=8)
    models["graph_motion_token_gpt2"] = load_state(proposed, proposed_checkpoint)
    return models


def select_interaction_scene(dataset: MultiTargetSceneDataset, min_targets: int = 4) -> tuple[int, np.ndarray]:
    best = None
    for index in range(len(dataset)):
        item = dataset[index]
        valid = torch.where(item["mask"])[0]
        if len(valid) < min_targets:
            continue
        positions = item["history"][-1, valid, :2]
        distances = torch.cdist(positions, positions)
        distances.fill_diagonal_(1.0e6)
        minimum = float(distances.min())
        if 4.0 <= minimum <= 15.0:
            pair = torch.nonzero(distances == distances.min(), as_tuple=False)[0]
            center = positions[pair].mean(dim=0)
            nearest = torch.argsort(torch.linalg.vector_norm(positions - center, dim=-1))[:min_targets]
            selected = valid[nearest].numpy()
            score = abs(minimum - 8.0)
            if best is None or score < best[0]:
                best = (score, index, selected)
    if best is None:
        raise RuntimeError("No suitable four-target interaction scene was found.")
    return int(best[1]), best[2]


@torch.inference_mode()
def export_trajectory_example(
    dataset: MultiTargetSceneDataset,
    models: Dict[str, torch.nn.Module],
    noise_map: Dict[int, Dict[str, float]],
    device: torch.device,
    output_path: Path,
    seed: int,
) -> dict:
    scene_index, selected = select_interaction_scene(dataset)
    item = dataset[scene_index]
    history = item["history"].unsqueeze(0).to(device)
    future = item["future"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
    exported = {
        "scene_index": np.asarray(scene_index, dtype=np.int64),
        "selected_indices": selected.astype(np.int64),
        "target_ids": item["target_ids"][selected].numpy(),
        "future": future[0, :, :, :2].index_select(1, selected_tensor).cpu().numpy(),
    }
    for snr in (5, 20):
        generator = torch.Generator(device=device).manual_seed(seed)
        observed = noisy_history(
            history,
            mask,
            noise_map[snr]["position_sigma_m"],
            noise_map[snr]["velocity_sigma_mps"],
            generator,
        )
        exported[f"history_snr{snr}"] = observed[0, :, :, :2].index_select(1, selected_tensor).cpu().numpy()
        for model_name in ("target_interaction_gnn", "graph_motion_token_gpt2"):
            prediction = models[model_name](observed, mask)["future_position"]
            exported[f"{model_name}_snr{snr}"] = prediction[0].index_select(1, selected_tensor).cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **exported)
    return {
        "scene_index": scene_index,
        "selected_indices": selected.tolist(),
        "target_ids": item["target_ids"][selected].tolist(),
        "path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--output-dir", default="results/multitarget_snr")
    parser.add_argument(
        "--proposed-checkpoint",
        default="results/multitarget_ablation/full_retrained/graph_motion_token_gpt2.pt",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--velocity-noise-20db", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    project = Path(args.project).resolve()
    set_seed(args.seed)
    test_dataset = MultiTargetSceneDataset(str(project / args.cache), "test")
    metadata = test_dataset.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    noise_map = snr_noise_map(project / args.calibration, args.velocity_noise_20db)
    device = torch.device("cuda")
    proposed_checkpoint = project / args.proposed_checkpoint
    models = make_models(config, project, proposed_checkpoint)
    for model in models.values():
        model.to(device).eval()

    results: Dict[str, Dict[str, dict]] = {}
    for snr, noise in noise_map.items():
        snr_results = {
            "constant_velocity": evaluate_constant_velocity(
                loader,
                config,
                device,
                noise["position_sigma_m"],
                noise["velocity_sigma_mps"],
                args.seed + 2000,
            )
        }
        for name, model in models.items():
            snr_results[name] = evaluate(
                model,
                loader,
                device,
                noise["position_sigma_m"],
                noise["velocity_sigma_mps"],
                args.seed + 2000,
            )
            print(json.dumps({"snr_db": snr, "model": name, **snr_results[name]}), flush=True)
        results[str(snr)] = snr_results

    proposed_best = {
        snr: all(
            values["graph_motion_token_gpt2"]["ade_m"] < metrics["ade_m"]
            and values["graph_motion_token_gpt2"]["fde_m"] < metrics["fde_m"]
            for name, metrics in values.items()
            if name != "graph_motion_token_gpt2"
        )
        for snr, values in results.items()
    }
    output_dir = project / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = export_trajectory_example(
        test_dataset,
        models,
        noise_map,
        device,
        output_dir / "trajectory_examples.npz",
        args.seed + 3000,
    )
    payload = {
        "experiment": "multi_target_empirically_calibrated_snr_evaluation",
        "device": torch.cuda.get_device_name(0),
        "snr_db": list(noise_map),
        "noise_protocol": {
            "position": "Per-axis RMSE measured from first-point RF localization cache at each SNR.",
            "velocity": "Scaled from 0.20 m/s at 20 dB in proportion to empirical position RMSE.",
            "same_standard_normal_draws_across_models_and_snr": True,
            "test_noise_seed": args.seed + 2000,
            "training_protocol": "All checkpoints were trained once near the empirically calibrated 20 dB condition.",
        },
        "proposed_checkpoint": str(proposed_checkpoint),
        "noise_map": {str(key): value for key, value in noise_map.items()},
        "test": results,
        "proposed_best_on_ade_and_fde": proposed_best,
        "trajectory_example": trajectory,
    }
    (output_dir / "comparison_snr_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
