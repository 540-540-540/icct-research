"""One-shot final-test evaluation for the frozen Raj QGNN + GPT-2 models."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
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
from target_interaction_graph import ForecasterConfig
from prediction.senior_raj_backbone import SeniorRajQGNN


def build_model(config: ForecasterConfig, checkpoint_path: Path, seed: int) -> MultiTargetGraphLLM:
    backbone = SeniorRajQGNN(
        config,
        quantum_scale=0.05,
        projection_hidden=128,
        projection_depth=1,
        classical_layers=2,
        quantum_first=True,
    )
    set_seed(seed)
    original_cwd = Path.cwd()
    try:
        os.chdir(ROOT / "models")
        model = MultiTargetGraphLLM(
            backbone,
            config,
            llm_layers=4,
            lora_rank=8,
            freeze_graph_backbone=True,
            quantum_coordinate_dim=64,
        )
    finally:
        os.chdir(original_cwd)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def summarize(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-dir", default="results/senior_raj_frozen_single_model"
    )
    parser.add_argument(
        "--output", default="results/senior_raj_frozen_single_model/final_test_summary.json"
    )
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
    frozen_dir = Path(args.frozen_dir)
    enabled_rows = []
    ablated_rows = []
    for seed in (2026, 2027, 2028):
        checkpoint = frozen_dir / f"seed{seed}" / "graph_motion_token_gpt2.pt"
        model = build_model(config, checkpoint, seed).cuda()
        enabled = evaluate(model, loader, torch.device("cuda"), 0.35, 0.20, seed + 2000)
        model.graph_backbone.set_quantum_scale(0.0)
        ablated = evaluate(model, loader, torch.device("cuda"), 0.35, 0.20, seed + 2000)
        enabled_rows.append({"seed": seed, **enabled})
        ablated_rows.append({"seed": seed, **ablated})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    enabled_ade = summarize(enabled_rows, "ade_m")
    enabled_fde = summarize(enabled_rows, "fde_m")
    ablated_ade = summarize(ablated_rows, "ade_m")
    ablated_fde = summarize(ablated_rows, "fde_m")
    senior_ade = 0.5911531277186552
    senior_fde = 1.195073048045906
    result = {
        "evaluation_split": "test",
        "test_opened_once_after_freeze": True,
        "selection_changed_after_test": False,
        "enabled_per_seed": enabled_rows,
        "quantum_ablated_per_seed": ablated_rows,
        "enabled_three_seed": {
            "ade_m": enabled_ade,
            "fde_m": enabled_fde,
        },
        "quantum_ablated_three_seed": {
            "ade_m": ablated_ade,
            "fde_m": ablated_fde,
        },
        "senior_original_graph_gpt2_test": {
            "ade_m": senior_ade,
            "fde_m": senior_fde,
        },
        "gain_over_senior_original_percent": {
            "ade": 100.0 * (senior_ade - enabled_ade["mean"]) / senior_ade,
            "fde": 100.0 * (senior_fde - enabled_fde["mean"]) / senior_fde,
        },
        "enabled_gain_over_quantum_ablation_percent": {
            "ade": 100.0 * (ablated_ade["mean"] - enabled_ade["mean"]) / ablated_ade["mean"],
            "fde": 100.0 * (ablated_fde["mean"] - enabled_fde["mean"]) / ablated_fde["mean"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
