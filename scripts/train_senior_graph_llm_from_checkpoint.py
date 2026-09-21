"""Validation-only Motion-Token GPT-2 training from a frozen graph checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "code" / "00_remote_shared_dependencies"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCHIVE))

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, set_seed
from run_multitarget_graph_llm import train_graph_llm
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN
from prediction.senior_raj_backbone import SeniorRajQGNN


def loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("classical", "quantum"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train_data = MultiTargetSceneDataset(args.cache, "train")
    val_data = MultiTargetSceneDataset(args.cache, "val")
    metadata = train_data.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    if args.arm == "classical":
        backbone = TargetInteractionGNN(config)
    else:
        backbone = SeniorRajQGNN(
            config,
            quantum_scale=0.05,
            projection_hidden=128,
            projection_depth=1,
            classical_layers=2,
            quantum_first=True,
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone.load_state_dict(checkpoint["model_state"])

    set_seed(args.seed)
    original_cwd = Path.cwd()
    try:
        os.chdir(ROOT / "models")
        model = MultiTargetGraphLLM(backbone, config, llm_layers=4, lora_rank=8)
    finally:
        os.chdir(original_cwd)
    train_loader = loader(train_data, args.batch_size, args.workers, True, args.seed)
    val_loader = loader(val_data, args.batch_size, args.workers, False, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    validation = train_graph_llm(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.learning_rate,
        args.position_noise,
        args.velocity_noise,
        output_dir / "graph_motion_token_gpt2.pt",
        output_dir / "training.jsonl",
        args.seed,
        model_name=f"{args.arm}_graph_motion_token_gpt2",
    )
    quantum_ablated_validation = None
    if args.arm == "quantum":
        model.graph_backbone.set_quantum_scale(0.0)
        quantum_ablated_validation = evaluate(
            model,
            val_loader,
            device,
            args.position_noise,
            args.velocity_noise,
            args.seed + 1000,
        )
    result = {
        "arm": args.arm,
        "seed": args.seed,
        "selection_split": "validation_only",
        "graph_checkpoint": args.checkpoint,
        "graph_backbone_frozen": True,
        "gpt2_base_frozen": True,
        "lora_and_heads_trainable": True,
        "validation": validation,
        "quantum_ablated_validation": quantum_ablated_validation,
        "parameter_summary": model.trainable_parameter_summary(),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
