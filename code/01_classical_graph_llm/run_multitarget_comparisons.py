"""Train conventional baselines from the first paper on the multi-target split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from multitarget_comparison_baselines import (
    MultiTargetLSTMBaseline,
    MultiTargetTCNBaseline,
    MultiTargetTransformerBaseline,
)
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, parameter_count, set_seed, train_model
from target_interaction_graph import ForecasterConfig


def make_train_loader(dataset, batch_size: int, workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def make_eval_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LSTM/TCN/Transformer comparison on multi-target scenes")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--phase1-results", default="results/multitarget_phase1/phase1_results.json")
    parser.add_argument("--phase2-results", default="results/multitarget_graph_llm/phase2_results.json")
    parser.add_argument("--output-dir", default="results/multitarget_comparisons")
    parser.add_argument("--models", default="lstm,tcn,transformer")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lstm-epochs", type=int, default=35)
    parser.add_argument("--tcn-epochs", type=int, default=35)
    parser.add_argument("--transformer-epochs", type=int, default=25)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-proposed-best", action="store_true")
    args = parser.parse_args(argv)

    selected = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    unknown = set(selected) - {"lstm", "tcn", "transformer"}
    if unknown:
        raise ValueError("Unsupported comparison models: %s" % sorted(unknown))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this experiment on the configured remote GPU server.")
    device = torch.device("cuda")
    set_seed(args.seed)

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
    validation_loader = make_eval_loader(validation_dataset, args.batch_size, args.workers)
    test_loader = make_eval_loader(test_dataset, args.batch_size, args.workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"

    factories = {
        "lstm": lambda: MultiTargetLSTMBaseline(config, hidden_dim=256, layers=2),
        "tcn": lambda: MultiTargetTCNBaseline(config, channels=(64, 128, 256)),
        # A compact encoder is better matched to the 20-step history and the
        # 6,000-scene training split than the earlier 43 M-parameter diagnostic.
        "transformer": lambda: MultiTargetTransformerBaseline(config, d_model=256, heads=8, layers=4),
    }
    epochs = {
        "lstm": args.lstm_epochs,
        "tcn": args.tcn_epochs,
        "transformer": args.transformer_epochs,
    }
    learning_rates = {"lstm": 1.0e-3, "tcn": 8.0e-4, "transformer": 8.0e-4}
    validation_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics: Dict[str, Dict[str, float]] = {}
    parameters: Dict[str, int] = {}

    for model_index, name in enumerate(selected):
        model_seed = args.seed + 100 * (model_index + 1)
        set_seed(model_seed)
        model = factories[name]()
        parameters[name] = parameter_count(model)
        train_loader = make_train_loader(train_dataset, args.batch_size, args.workers, model_seed)
        model, best_validation = train_model(
            name=name,
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            epochs=epochs[name],
            learning_rate=learning_rates[name],
            weight_decay=2.0e-4,
            position_sigma=args.position_noise,
            velocity_sigma=args.velocity_noise,
            checkpoint_path=output_dir / (name + ".pt"),
            log_path=log_path,
            seed=model_seed,
            graph_weighting=False,
            use_amp=name != "transformer",
        )
        validation_metrics[name] = best_validation
        test_metrics[name] = evaluate(
            model,
            test_loader,
            device,
            args.position_noise,
            args.velocity_noise,
            args.seed + 2000,
        )
        del model
        torch.cuda.empty_cache()

    phase1 = json.loads(Path(args.phase1_results).read_text(encoding="utf-8"))
    phase2 = json.loads(Path(args.phase2_results).read_text(encoding="utf-8"))
    all_test = {
        "constant_velocity": phase1["test"]["constant_velocity"],
        "independent_gru": phase1["test"]["independent_gru"],
        **test_metrics,
        "target_interaction_gnn": phase1["test"]["target_interaction_gnn"],
        "graph_motion_token_gpt2": phase2["test"]["graph_motion_token_gpt2"],
    }
    proposed = all_test["graph_motion_token_gpt2"]
    comparisons = {}
    for name, metrics in all_test.items():
        if name == "graph_motion_token_gpt2":
            continue
        comparisons[name] = {
            "ade_reduction_percent": (metrics["ade_m"] - proposed["ade_m"]) / metrics["ade_m"] * 100.0,
            "fde_reduction_percent": (metrics["fde_m"] - proposed["fde_m"]) / metrics["fde_m"] * 100.0,
            "interaction_ade_reduction_percent": (
                (metrics["interaction_ade_m"] - proposed["interaction_ade_m"])
                / metrics["interaction_ade_m"]
                * 100.0
            ),
        }
    proposed_best = all(
        proposed["ade_m"] < metrics["ade_m"] and proposed["fde_m"] < metrics["fde_m"]
        for name, metrics in all_test.items()
        if name != "graph_motion_token_gpt2"
    )
    result = {
        "experiment": "multi_target_conventional_model_comparison",
        "device": torch.cuda.get_device_name(0),
        "metadata": metadata,
        "fairness": {
            "same_scene_cache": args.cache,
            "same_temporal_split": True,
            "same_position_noise_sigma_m": args.position_noise,
            "same_velocity_noise_sigma_mps": args.velocity_noise,
            "same_test_noise_seed": args.seed + 2000,
            "oracle_track_ids": True,
            "independent_baselines_process_each_target_separately": True,
        },
        "parameters": parameters,
        "validation": validation_metrics,
        "test": all_test,
        "proposed_reduction_percent": comparisons,
        "proposed_best_on_ade_and_fde": proposed_best,
    }
    result_path = output_dir / "comparison_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.require_proposed_best and not proposed_best:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
