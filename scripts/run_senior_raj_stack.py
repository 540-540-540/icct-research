"""Train Raj QGNN and Raj + Motion-Token GPT-2 on the senior R0 protocol."""
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
from run_multitarget_experiment import evaluate, set_seed, train_model
from run_multitarget_graph_llm import train_graph_llm
from target_interaction_graph import ForecasterConfig, IndependentGRUForecaster
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase1-batch-size", type=int, default=128)
    parser.add_argument("--llm-batch-size", type=int, default=24)
    parser.add_argument("--independent-epochs", type=int, default=35)
    parser.add_argument("--raj-epochs", type=int, default=30)
    parser.add_argument("--llm-epochs", type=int, default=16)
    parser.add_argument("--raj-learning-rate", type=float, default=4e-4)
    parser.add_argument("--llm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--quantum-scale", type=float, default=0.05)
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
    train_raj = loader(train_data, args.phase1_batch_size, args.workers, True, args.seed)
    val_raj = loader(val_data, args.phase1_batch_size, args.workers, False, args.seed)
    test_raj = loader(test_data, args.phase1_batch_size, args.workers, False, args.seed)

    independent = IndependentGRUForecaster(config)
    independent, independent_validation = train_model(
        "independent_gru",
        independent,
        train_raj,
        val_raj,
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
    independent_test = evaluate(
        independent, test_raj, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )
    raj = SeniorRajQGNN(config, quantum_scale=args.quantum_scale)
    missing, unexpected = raj.load_state_dict(independent.state_dict(), strict=False)
    allowed = ("core.", "raj_projection.", "quantum_scale")
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise RuntimeError(f"Unexpected Raj warm-start mismatch: missing={missing} unexpected={unexpected}")
    raj, raj_validation = train_model(
        "senior_raj_qgnn",
        raj,
        train_raj,
        val_raj,
        device,
        args.raj_epochs,
        args.raj_learning_rate,
        2e-4,
        args.position_noise,
        args.velocity_noise,
        output_dir / "raj_qgnn.pt",
        output_dir / "raj_training.jsonl",
        args.seed + 17,
        graph_weighting=True,
        use_amp=False,
    )
    raj_test = evaluate(
        raj, test_raj, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )
    phase1 = {
        "seed": args.seed,
        "validation": raj_validation,
        "test": raj_test,
        "parameters": sum(p.numel() for p in raj.parameters()),
    }
    (output_dir / "raj_phase1_results.json").write_text(
        json.dumps(phase1, indent=2), encoding="utf-8"
    )

    # Recreate loaders so the LLM stage gets a fresh, seed-controlled shuffle.
    train_llm = loader(train_data, args.llm_batch_size, args.workers, True, args.seed)
    val_llm = loader(val_data, args.llm_batch_size, args.workers, False, args.seed)
    test_llm = loader(test_data, args.llm_batch_size, args.workers, False, args.seed)
    original_cwd = Path.cwd()
    try:
        os.chdir(ROOT / "models")
        model = MultiTargetGraphLLM(raj, config, llm_layers=4, lora_rank=8)
    finally:
        os.chdir(original_cwd)
    llm_validation = train_graph_llm(
        model,
        train_llm,
        val_llm,
        device,
        args.llm_epochs,
        args.llm_learning_rate,
        args.position_noise,
        args.velocity_noise,
        output_dir / "raj_graph_motion_token_gpt2.pt",
        output_dir / "raj_llm_training.jsonl",
        args.seed,
        model_name="raj_graph_motion_token_gpt2",
    )
    llm_test = evaluate(
        model, test_llm, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )

    senior = json.loads(
        (
            ROOT
            / "reports"
            / "senior_r0_reproduction"
            / "phase2"
            / "phase2_results.json"
        ).read_text(encoding="utf-8")
    )["test"]
    reference = senior["graph_motion_token_gpt2"]
    result = {
        "experiment": "senior_r0_raj_qgnn_integration",
        "seed": args.seed,
        "protocol": {
            "cache": args.cache,
            "same_independent_gru_warm_start_as_senior_gnn": True,
            "same_motion_token_gpt2": True,
            "raj_core": "RajWeightedMultiJQGNNCore(j=2+j=3, rounds=3)",
            "test_noise_seed": args.seed + 2000,
            "quantum_scale": args.quantum_scale,
        },
        "independent_gru": {"validation": independent_validation, "test": independent_test},
        "raj_qgnn": {"validation": raj_validation, "test": raj_test},
        "raj_graph_motion_token_gpt2": {"validation": llm_validation, "test": llm_test},
        "senior_seed2026_reference": reference,
        "gain_vs_senior_reference_percent": {
            "ade": 100 * (reference["ade_m"] - llm_test["ade_m"]) / reference["ade_m"],
            "fde": 100 * (reference["fde_m"] - llm_test["fde_m"]) / reference["fde_m"],
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
