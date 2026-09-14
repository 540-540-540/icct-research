"""Train strict component ablations for the graph-conditioned motion-token LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, set_seed
from run_multitarget_graph_llm import train_graph_llm
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


VARIANTS = {
    "full_retrained": {
        "model": {},
        "token_weight": 0.035,
        "description": "Full model retrained under the identical empirically calibrated 20 dB noise.",
    },
    "without_uncertainty": {
        "model": {"use_uncertainty": False},
        "token_weight": 0.035,
        "description": "Replace the learned uncertainty temperature with a fixed 0.22 temperature.",
    },
    "without_soft_token": {
        "model": {"use_soft_tokens": False},
        "token_weight": 0.035,
        "description": "Replace each soft motion-token distribution with its hard argmax token.",
    },
    "without_graph_context": {
        "model": {"use_graph_context": False},
        "token_weight": 0.035,
        "description": "Remove graph node context from the LLM token sequence and correction head.",
    },
    "without_token_loss": {
        "model": {},
        "token_weight": 0.0,
        "description": "Remove the future motion-token auxiliary cross-entropy loss.",
    },
    "without_lora": {
        "model": {"lora_rank": 0},
        "token_weight": 0.035,
        "description": "Keep GPT-2 frozen and remove all LoRA adapters.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--phase1-dir", default="results/multitarget_phase1")
    parser.add_argument("--output-dir", default="results/multitarget_ablation")
    parser.add_argument("--calibration", default="results/multitarget_snr/snr_calibration.json")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = set(selected) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    position_noise = float(calibration["calibration"]["20"]["position_per_axis_rmse_m"])
    train_dataset = MultiTargetSceneDataset(args.cache, "train")
    validation_dataset = MultiTargetSceneDataset(args.cache, "val")
    test_dataset = MultiTargetSceneDataset(args.cache, "test")
    metadata = train_dataset.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    graph_path = Path(args.phase1_dir) / "target_interaction_gnn.pt"
    graph_state = torch.load(graph_path, map_location="cpu")["model_state"]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    combined = {}

    for variant_name in selected:
        variant_dir = output_root / variant_name
        result_path = variant_dir / "result.json"
        checkpoint_path = variant_dir / "graph_motion_token_gpt2.pt"
        if args.skip_existing and result_path.exists():
            combined[variant_name] = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"Skipping completed variant: {variant_name}", flush=True)
            continue

        set_seed(args.seed)
        graph = TargetInteractionGNN(config)
        graph.load_state_dict(graph_state)
        model_options = {
            "graph_backbone": graph,
            "config": config,
            "llm_layers": 4,
            "lora_rank": 8,
        }
        model_options.update(VARIANTS[variant_name]["model"])
        model = MultiTargetGraphLLM(**model_options)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            **loader_options,
        )
        if args.skip_existing and checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(saved["model_state"])
            model.to(device)
            validation = saved["validation_metrics"]
            print(f"Recovered completed checkpoint: {variant_name}", flush=True)
        else:
            validation = train_graph_llm(
                model,
                train_loader,
                validation_loader,
                device,
                args.epochs,
                args.learning_rate,
                position_noise,
                args.velocity_noise,
                checkpoint_path,
                variant_dir / "training.jsonl",
                args.seed,
                token_weight=float(VARIANTS[variant_name]["token_weight"]),
                model_name=variant_name,
            )
        test = evaluate(
            model,
            test_loader,
            device,
            position_noise,
            args.velocity_noise,
            args.seed + 2000,
        )
        serializable_options = dict(model_options)
        serializable_options.update({"graph_backbone": "TargetInteractionGNN", "config": "ForecasterConfig"})
        result = {
            "variant": variant_name,
            "description": VARIANTS[variant_name]["description"],
            "model_options": serializable_options,
            "token_loss_weight": float(VARIANTS[variant_name]["token_weight"]),
            "training_noise": {
                "snr_db": 20,
                "position_sigma_m": position_noise,
                "velocity_sigma_mps": args.velocity_noise,
            },
            "parameter_summary": model.trainable_parameter_summary(),
            "validation": validation,
            "test_20db": test,
        }
        variant_dir.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        combined[variant_name] = result
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    payload = {
        "experiment": "multi_target_graph_llm_component_ablation",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "variants": combined,
    }
    (output_root / "ablation_20db_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
