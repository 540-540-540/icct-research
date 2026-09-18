"""Evaluate the full Graph+LLM and trained ablations across calibrated SNR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_ablation import VARIANTS
from run_multitarget_experiment import evaluate, set_seed
from run_multitarget_snr_evaluation import snr_noise_map
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def build_variant(
    config: ForecasterConfig,
    graph_state: dict,
    options: dict,
    checkpoint_path: Path,
) -> MultiTargetGraphLLM:
    graph = TargetInteractionGNN(config)
    graph.load_state_dict(graph_state)
    kwargs = {"graph_backbone": graph, "config": config, "llm_layers": 4, "lora_rank": 8}
    kwargs.update(options)
    model = MultiTargetGraphLLM(**kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--phase1-dir", default="results/multitarget_phase1")
    parser.add_argument("--full-checkpoint", default="results/multitarget_graph_llm/graph_motion_token_gpt2.pt")
    parser.add_argument("--ablation-dir", default="results/multitarget_ablation")
    parser.add_argument("--output", default="results/multitarget_snr/ablation_snr_results.json")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--velocity-noise-20db", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    set_seed(args.seed)
    dataset = MultiTargetSceneDataset(args.cache, "test")
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
    noise_map = snr_noise_map(Path(args.calibration), args.velocity_noise_20db)
    graph_state = torch.load(Path(args.phase1_dir) / "target_interaction_gnn.pt", map_location="cpu")["model_state"]
    variants = {
        "full_model": (
            VARIANTS["full_retrained"]["model"],
            Path(args.ablation_dir) / "full_retrained" / "graph_motion_token_gpt2.pt",
        ),
        **{
            name: (definition["model"], Path(args.ablation_dir) / name / "graph_motion_token_gpt2.pt")
            for name, definition in VARIANTS.items()
            if name != "full_retrained"
        },
    }
    device = torch.device("cuda")
    test = {str(snr): {} for snr in noise_map}
    parameter_summary = {}
    for name, (options, checkpoint_path) in variants.items():
        set_seed(args.seed)
        model = build_variant(config, graph_state, options, checkpoint_path).to(device).eval()
        parameter_summary[name] = model.trainable_parameter_summary()
        for snr, noise in noise_map.items():
            metrics = evaluate(
                model,
                loader,
                device,
                noise["position_sigma_m"],
                noise["velocity_sigma_mps"],
                args.seed + 2000,
            )
            test[str(snr)][name] = metrics
            print(json.dumps({"snr_db": snr, "variant": name, **metrics}), flush=True)
        del model
        torch.cuda.empty_cache()

    degradation = {}
    for snr, values in test.items():
        full = values["full_model"]
        degradation[snr] = {
            name: {
                "ade_increase_percent": (metrics["ade_m"] - full["ade_m"]) / full["ade_m"] * 100.0,
                "fde_increase_percent": (metrics["fde_m"] - full["fde_m"]) / full["fde_m"] * 100.0,
            }
            for name, metrics in values.items()
            if name != "full_model"
        }
    payload = {
        "experiment": "multi_target_graph_llm_ablation_snr",
        "device": torch.cuda.get_device_name(0),
        "snr_db": list(noise_map),
        "noise_map": {str(key): value for key, value in noise_map.items()},
        "same_test_noise_seed": args.seed + 2000,
        "parameter_summary": parameter_summary,
        "test": test,
        "degradation_vs_full_percent": degradation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
