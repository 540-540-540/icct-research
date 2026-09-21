"""Matched Raj QGNN versus multi-j JohnsonGIN on the senior R0 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "code" / "00_remote_shared_dependencies"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCHIVE))

from multitarget_scene_dataset import MultiTargetSceneDataset
from prediction.qgnn_paper_native.model import build_paper_model
from run_multitarget_experiment import evaluate, set_seed
from run_multitarget_graph_llm import train_graph_llm


KINDS = {
    "quantum": "raj_weighted_multij_quantum",
    "classical": "raj_multij_johnson",
}


class SeniorProtocolAdapter(nn.Module):
    """Expose FinalModel through the senior evaluator/trainer contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history, mask):
        # The original successful Raj training is full-fp32; keep its complex
        # operations and downstream GPT-2 out of the senior AMP wrapper.
        with torch.autocast(device_type=history.device.type, enabled=False):
            output = self.model(history.float(), mask)
        output["future_position"] = output["prediction"]
        return output

    def future_token_ids(self, history, future):
        return self.model.llm.future_token_ids(history, future)

    def trainable_parameter_summary(self):
        return self.model.parameter_summary()


def llm_initialization_hash(model: SeniorProtocolAdapter) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.model.llm.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_loader(dataset, batch_size, workers, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--position-noise", type=float, default=.35)
    parser.add_argument("--velocity-noise", type=float, default=.20)
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
    train_loader = make_loader(train_data, args.batch_size, args.workers, True, args.seed)
    val_loader = make_loader(val_data, args.batch_size, args.workers, False, args.seed)
    test_loader = make_loader(test_data, args.batch_size, args.workers, False, args.seed)

    model = SeniorProtocolAdapter(build_paper_model(KINDS[args.kind], seed=args.seed)).to(device)
    initial_hash = llm_initialization_hash(model)
    validation = train_graph_llm(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.learning_rate,
        args.position_noise,
        args.velocity_noise,
        output_dir / "model.pt",
        output_dir / "training.jsonl",
        args.seed,
        model_name=f"raj_matched_{args.kind}",
    )
    test = evaluate(
        model,
        test_loader,
        device,
        args.position_noise,
        args.velocity_noise,
        args.seed + 2000,
    )
    result = {
        "experiment": "senior_r0_matched_raj_quantum_vs_classical",
        "kind": args.kind,
        "implementation": KINDS[args.kind],
        "seed": args.seed,
        "protocol": {
            "cache": args.cache,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "position_noise_sigma_m": args.position_noise,
            "velocity_noise_sigma_mps": args.velocity_noise,
            "test_noise_seed": args.seed + 2000,
            "joint_graph_llm_training": True,
            "same_final_motion_gpt2": True,
        },
        "llm_initialization_sha256": initial_hash,
        "parameters": model.trainable_parameter_summary(),
        "validation": validation,
        "test": test,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
