"""Export representative-scene predictions for selected comparison models."""
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


def load_models(config: ForecasterConfig, project: Path, device: torch.device) -> dict:
    phase1 = project / "results/multitarget_phase1"
    comparison = project / "results/multitarget_comparisons_final"
    graph_checkpoint = phase1 / "target_interaction_gnn.pt"
    proposed_graph = load_state(TargetInteractionGNN(config), graph_checkpoint)
    models = {
        "lstm": load_state(MultiTargetLSTMBaseline(config, hidden_dim=256, layers=2), comparison / "lstm.pt"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--selection", default="results/multitarget_snr/trajectory_selection_report.json")
    parser.add_argument("--output-dir", default="results/multitarget_snr")
    parser.add_argument("--seed", type=int, default=6026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    project = Path(args.project).resolve()
    set_seed(args.seed)
    selection = json.loads((project / args.selection).read_text(encoding="utf-8"))
    scene_index = int(selection["selected_scene_index"])
    selected = np.asarray(selection["selected_indices"], dtype=np.int64)
    dataset = MultiTargetSceneDataset(str(project / args.cache), "test")
    item = dataset[scene_index]
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
    history = item["history"].unsqueeze(0).to(device)
    future = item["future"].unsqueeze(0).to(device)
    mask = item["mask"].unsqueeze(0).to(device)
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
    selected_future = future[0].index_select(1, selected_tensor)[..., :2]
    arrays = {
        "scene_index": np.asarray(scene_index, dtype=np.int64),
        "selected_indices": selected,
        "target_ids": item["target_ids"][selected].numpy().astype(np.int64),
        "clean_history": history[0].index_select(1, selected_tensor)[..., :2].cpu().numpy(),
        "future": selected_future.cpu().numpy(),
    }
    rows = []
    with torch.inference_mode():
        for snr in (5, 20):
            generator = torch.Generator(device=device).manual_seed(args.seed + snr * 100000 + scene_index)
            observed = noisy_history(
                history,
                mask,
                noise[snr]["position_sigma_m"],
                noise[snr]["velocity_sigma_mps"],
                generator,
            )
            for model_key, model in models.items():
                prediction = model(observed, mask)["future_position"][0].index_select(1, selected_tensor)
                distance = torch.linalg.vector_norm(prediction - selected_future, dim=-1)
                arrays[f"{model_key}_snr{snr}"] = prediction.cpu().numpy()
                rows.append(
                    {
                        "snr_db": snr,
                        "model_key": model_key,
                        "ade_m": float(distance.mean()),
                        "fde_m": float(distance[-1].mean()),
                    }
                )

    output_dir = project / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "trajectory_examples_comparison.npz", **arrays)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "trajectory_example_model_metrics.csv", index=False)
    report = {
        "scene_index": scene_index,
        "target_ids": arrays["target_ids"].tolist(),
        "models": list(models),
        "metrics": rows,
        "noise_seed_protocol": "scene seed = base seed + snr * 100000 + scene index",
        "output": str(output_dir / "trajectory_examples_comparison.npz"),
    }
    (output_dir / "trajectory_example_model_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
