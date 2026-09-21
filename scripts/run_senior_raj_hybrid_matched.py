"""Matched classical-GNN versus Raj-hybrid-QGNN experiment."""
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
from run_multitarget_experiment import evaluate, parameter_count, set_seed, train_model
from target_interaction_graph import ForecasterConfig, IndependentGRUForecaster, TargetInteractionGNN
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


def train_arm(
    name,
    model,
    independent_state,
    train_data,
    val_data,
    args,
    output_dir,
    device,
):
    missing, unexpected = model.load_state_dict(independent_state, strict=False)
    if name == "target_interaction_gnn":
        allowed = ("graph_layers.",)
    else:
        allowed = ("graph_layers.", "core.", "raj_projection.", "quantum_scale")
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise RuntimeError(f"Unexpected {name} warm-start mismatch: missing={missing} unexpected={unexpected}")
    set_seed(args.seed + 17)
    train_loader = loader(train_data, args.batch_size, args.workers, True, args.seed + 17)
    val_loader = loader(val_data, args.batch_size, args.workers, False, args.seed + 17)
    return train_model(
        name,
        model,
        train_loader,
        val_loader,
        device,
        args.graph_epochs,
        args.learning_rate,
        2e-4,
        args.position_noise,
        args.velocity_noise,
        output_dir / f"{name}.pt",
        output_dir / f"{name}_training.jsonl",
        args.seed + 17,
        graph_weighting=True,
        use_amp=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--independent-epochs", type=int, default=35)
    parser.add_argument("--graph-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--quantum-scale", type=float, default=0.15)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = MultiTargetSceneDataset(args.cache, "train")
    val_data = MultiTargetSceneDataset(args.cache, "val")
    test_data = MultiTargetSceneDataset(args.cache, "test")
    metadata = train_data.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=128,
    )

    independent = IndependentGRUForecaster(config)
    independent_train = loader(train_data, args.batch_size, args.workers, True, args.seed)
    independent_val = loader(val_data, args.batch_size, args.workers, False, args.seed)
    independent, independent_validation = train_model(
        "independent_gru",
        independent,
        independent_train,
        independent_val,
        device,
        args.independent_epochs,
        1e-3,
        1e-4,
        args.position_noise,
        args.velocity_noise,
        output_dir / "independent_gru.pt",
        output_dir / "independent_training.jsonl",
        args.seed,
        graph_weighting=False,
    )
    independent_state = {key: value.detach().cpu() for key, value in independent.state_dict().items()}

    classical, classical_validation = train_arm(
        "target_interaction_gnn",
        TargetInteractionGNN(config),
        independent_state,
        train_data,
        val_data,
        args,
        output_dir,
        device,
    )
    quantum, quantum_validation = train_arm(
        "raj_hybrid_qgnn",
        SeniorRajQGNN(
            config,
            quantum_scale=args.quantum_scale,
            projection_hidden=128,
            projection_depth=1,
            classical_layers=1,
            quantum_first=True,
        ),
        independent_state,
        train_data,
        val_data,
        args,
        output_dir,
        device,
    )

    test_loader = loader(test_data, args.batch_size, args.workers, False, args.seed)
    classical_test = evaluate(
        classical, test_loader, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )
    quantum_test = evaluate(
        quantum, test_loader, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )
    val_loader = loader(val_data, args.batch_size, args.workers, False, args.seed)
    quantum.set_quantum_scale(0.0)
    quantum_ablated_validation = evaluate(
        quantum, val_loader, device, args.position_noise, args.velocity_noise, args.seed + 1017
    )

    result = {
        "experiment": "senior_r0_matched_gnn_vs_raj_hybrid_qgnn",
        "seed": args.seed,
        "protocol": {
            "cache": args.cache,
            "epochs": args.graph_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "quantum_scale": args.quantum_scale,
            "quantum_first": True,
            "classical_layers_in_qgnn": 1,
            "raj_core": "RajWeightedMultiJQGNNCore(j=2+j=3, rounds=3)",
            "test_noise_seed": args.seed + 2000,
        },
        "parameters": {
            "target_interaction_gnn": parameter_count(classical),
            "raj_hybrid_qgnn": parameter_count(quantum),
        },
        "independent_validation": independent_validation,
        "validation": {
            "target_interaction_gnn": classical_validation,
            "raj_hybrid_qgnn": quantum_validation,
            "raj_quantum_ablated": quantum_ablated_validation,
        },
        "test": {
            "target_interaction_gnn": classical_test,
            "raj_hybrid_qgnn": quantum_test,
        },
        "qgnn_gain_over_gnn_percent": {
            metric: 100.0 * (classical_test[metric] - quantum_test[metric]) / classical_test[metric]
            for metric in ("ade_m", "fde_m")
        },
        "quantum_validation_gain_over_ablation_percent": {
            metric: 100.0 * (quantum_ablated_validation[metric] - quantum_validation[metric])
            / quantum_ablated_validation[metric]
            for metric in ("ade_m", "fde_m")
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
