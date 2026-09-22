"""Validation-only screen for non-zero Raj quantum coupling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "code" / "00_remote_shared_dependencies"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCHIVE))

from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, set_seed, train_model
from target_interaction_graph import ForecasterConfig
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
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument(
        "--independent-checkpoint",
        default="reports/senior_r0_reproduction/phase1/independent_gru.pt",
    )
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--frozen-epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--quantum-scale", type=float, required=True)
    parser.add_argument("--projection-hidden", type=int, default=128)
    parser.add_argument("--projection-depth", type=int, default=1)
    parser.add_argument("--classical-layers", type=int, default=0)
    parser.add_argument("--quantum-first", action="store_true")
    parser.add_argument("--quantum-only", action="store_true")
    parser.add_argument("--metric-aligned-loss", action="store_true")
    args = parser.parse_args()
    if args.frozen_epochs < 0 or args.frozen_epochs >= args.epochs:
        raise ValueError("frozen_epochs must be in [0, epochs)")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = MultiTargetSceneDataset(args.cache, "train")
    val_data = MultiTargetSceneDataset(args.cache, "val")
    metadata = train_data.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )
    train_loader = loader(train_data, args.batch_size, args.workers, True, args.seed)
    val_loader = loader(val_data, args.batch_size, args.workers, False, args.seed)

    set_seed(args.seed + 17)
    model = SeniorRajQGNN(
        config,
        quantum_scale=args.quantum_scale,
        projection_hidden=args.projection_hidden,
        projection_depth=args.projection_depth,
        classical_layers=args.classical_layers,
        quantum_first=args.quantum_first,
    )
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    else:
        checkpoint = torch.load(args.independent_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        allowed = ("core.", "raj_projection.", "quantum_scale", "graph_layers.")
        if unexpected or any(not key.startswith(allowed) for key in missing):
            raise RuntimeError(f"Unexpected warm-start mismatch: missing={missing} unexpected={unexpected}")
    if args.quantum_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("core.", "raj_projection.")))
    initial_core = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(("core.", "raj_projection."))
    }
    frozen_validation = None
    if args.frozen_epochs:
        for module in (model.history_encoder, model.node_projection, model.time_embedding, model.decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model, frozen_validation = train_model(
            f"senior_raj_qgnn_quantum_only_{args.quantum_scale:g}",
            model,
            train_loader,
            val_loader,
            device,
            args.frozen_epochs,
            args.learning_rate,
            2e-4,
            args.position_noise,
            args.velocity_noise,
            output_dir / "raj_qgnn_frozen.pt",
            output_dir / "frozen_training.jsonl",
            args.seed + 17,
            graph_weighting=True,
            use_amp=False,
            metric_aligned_loss=args.metric_aligned_loss,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    model, active_validation = train_model(
        f"senior_raj_qgnn_joint_{args.quantum_scale:g}",
        model,
        train_loader,
        val_loader,
        device,
        args.epochs - args.frozen_epochs,
        args.learning_rate,
        2e-4,
        args.position_noise,
        args.velocity_noise,
        output_dir / "raj_qgnn.pt",
        output_dir / "training.jsonl",
        args.seed + 17,
        graph_weighting=True,
        use_amp=False,
        metric_aligned_loss=args.metric_aligned_loss,
    )
    final_state = model.state_dict()
    quantum_parameter_delta = sum(
        (final_state[key].detach().cpu() - initial).square().sum().item()
        for key, initial in initial_core.items()
    ) ** 0.5
    model.set_quantum_scale(0.0)
    ablated_validation = evaluate(
        model,
        val_loader,
        device,
        args.position_noise,
        args.velocity_noise,
        args.seed + 1017,
    )
    result = {
        "seed": args.seed,
        "resume_checkpoint": args.resume_checkpoint,
        "selection_split": "validation_only",
        "quantum_scale": args.quantum_scale,
        "projection_hidden": args.projection_hidden,
        "projection_depth": args.projection_depth,
        "classical_layers": args.classical_layers,
        "quantum_first": args.quantum_first,
        "quantum_only": args.quantum_only,
        "metric_aligned_loss": args.metric_aligned_loss,
        "frozen_epochs": args.frozen_epochs,
        "joint_epochs": args.epochs - args.frozen_epochs,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "quantum_parameter_delta_l2": quantum_parameter_delta,
        "frozen_validation": frozen_validation,
        "active_validation": active_validation,
        "quantum_ablated_validation": ablated_validation,
        "active_gain_over_ablation_percent": {
            metric: 100.0 * (ablated_validation[metric] - active_validation[metric]) / ablated_validation[metric]
            for metric in ("ade_m", "fde_m")
        },
    }
    if quantum_parameter_delta <= 1e-8:
        raise RuntimeError("Quantum parameters did not update")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
