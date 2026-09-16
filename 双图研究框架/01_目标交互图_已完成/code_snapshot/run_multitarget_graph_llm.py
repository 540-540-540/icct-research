"""Phase-2 experiment: refine the accepted interaction GNN with GPT-2/LoRA."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MultiTargetTimeLLM import MultiTargetGraphLLM
from multitarget_scene_dataset import MultiTargetSceneDataset
from run_multitarget_experiment import evaluate, noisy_history, prediction_loss, set_seed
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def token_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    valid = mask[:, None, :].expand_as(targets).reshape(-1)
    return losses[valid].mean()


def train_graph_llm(
    model: MultiTargetGraphLLM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    position_sigma: float,
    velocity_sigma: float,
    checkpoint_path: Path,
    log_path: Path,
    seed: int,
    token_weight: float = 0.035,
    model_name: str = "graph_motion_token_gpt2",
) -> Dict[str, float]:
    model.to(device)
    lora_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    parameter_groups = [
        {"params": head_parameters, "lr": learning_rate},
        {"params": lora_parameters, "lr": learning_rate * 0.25},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.08)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    initial = evaluate(model, validation_loader, device, position_sigma, velocity_sigma, seed + 1000)
    best_score = initial["ade_m"] + 0.35 * initial["fde_m"]
    best_metrics = initial
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"model": model_name, "epoch": 0, "validation": initial}) + "\n")
        for epoch in range(1, epochs + 1):
            model.train()
            coordinate_total = 0.0
            token_total = 0.0
            examples = 0
            started = time.time()
            for batch in train_loader:
                history = batch["history"].to(device, non_blocking=True)
                future = batch["future"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                observed = noisy_history(history, mask, position_sigma, velocity_sigma)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output = model(observed, mask)
                    coordinate = prediction_loss(output, future, mask, graph_weighting=True)
                    targets = model.future_token_ids(observed, future)
                    classification = token_loss(output["token_logits"], targets, mask)
                    loss = coordinate + token_weight * classification
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], max_norm=3.0
                )
                scaler.step(optimizer)
                scaler.update()
                batch_size = history.shape[0]
                coordinate_total += float(coordinate.item()) * batch_size
                token_total += float(classification.item()) * batch_size
                examples += batch_size
            scheduler.step()
            validation = evaluate(model, validation_loader, device, position_sigma, velocity_sigma, seed + 1000)
            score = validation["ade_m"] + 0.35 * validation["fde_m"]
            record = {
                "model": model_name,
                "epoch": epoch,
                "coordinate_loss": coordinate_total / max(examples, 1),
                "token_ce": token_total / max(examples, 1),
                "validation": validation,
                "head_lr": optimizer.param_groups[0]["lr"],
                "lora_lr": optimizer.param_groups[1]["lr"],
                "seconds": time.time() - started,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            if score < best_score - 1.0e-5:
                best_score = score
                best_metrics = validation
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 6:
                print("Early stopping %s at epoch %d" % (model_name, epoch), flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "validation_metrics": best_metrics,
            "parameter_summary": model.trainable_parameter_summary(),
        },
        checkpoint_path,
    )
    return best_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train graph-conditioned motion-token GPT-2")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--phase1-dir", default="results/multitarget_phase1")
    parser.add_argument("--output-dir", default="results/multitarget_graph_llm")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--llm-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-improvement", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    train_dataset = MultiTargetSceneDataset(args.cache, "train")
    validation_dataset = MultiTargetSceneDataset(args.cache, "val")
    test_dataset = MultiTargetSceneDataset(args.cache, "test")
    metadata = train_dataset.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=args.hidden_dim,
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **options)
    test_loader = DataLoader(test_dataset, shuffle=False, **options)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script on the configured remote GPU server.")
    device = torch.device("cuda")

    phase1_dir = Path(args.phase1_dir)
    phase1_results = json.loads((phase1_dir / "phase1_results.json").read_text(encoding="utf-8"))
    if not phase1_results.get("graph_gate_passed", False):
        raise RuntimeError("Phase-1 GNN gate did not pass; refusing to start the LLM stage.")
    graph = TargetInteractionGNN(config)
    graph_checkpoint = torch.load(phase1_dir / "target_interaction_gnn.pt", map_location="cpu")
    graph.load_state_dict(graph_checkpoint["model_state"])
    model = MultiTargetGraphLLM(
        graph_backbone=graph,
        config=config,
        llm_layers=args.llm_layers,
        lora_rank=args.lora_rank,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics = train_graph_llm(
        model,
        train_loader,
        validation_loader,
        device,
        args.epochs,
        args.learning_rate,
        args.position_noise,
        args.velocity_noise,
        output_dir / "graph_motion_token_gpt2.pt",
        output_dir / "training.jsonl",
        args.seed,
    )
    test_metrics = evaluate(model, test_loader, device, args.position_noise, args.velocity_noise, args.seed + 2000)
    graph_test = phase1_results["test"]["target_interaction_gnn"]
    gain = {
        "ade_vs_gnn": (graph_test["ade_m"] - test_metrics["ade_m"]) / graph_test["ade_m"] * 100.0,
        "fde_vs_gnn": (graph_test["fde_m"] - test_metrics["fde_m"]) / graph_test["fde_m"] * 100.0,
        "interaction_ade_vs_gnn": (
            (graph_test["interaction_ade_m"] - test_metrics["interaction_ade_m"])
            / graph_test["interaction_ade_m"]
            * 100.0
        ),
    }
    passed = test_metrics["ade_m"] < graph_test["ade_m"] and test_metrics["fde_m"] < graph_test["fde_m"]
    result = {
        "phase": "graph_motion_token_gpt2_gate",
        "device": torch.cuda.get_device_name(0),
        "model": {
            "llm": "local GPT-2",
            "llm_layers": args.llm_layers,
            "adaptation": "LoRA rank %d" % args.lora_rank,
            "tokenization": "uncertainty-guided agent-centric motion tokens",
            "parameter_summary": model.trainable_parameter_summary(),
        },
        "validation": validation_metrics,
        "test": {
            "independent_gru": phase1_results["test"]["independent_gru"],
            "target_interaction_gnn": graph_test,
            "graph_motion_token_gpt2": test_metrics,
        },
        "relative_gain_percent": gain,
        "llm_gate_passed": passed,
        "gate_rule": "Graph+LLM test ADE and FDE must both be lower than the accepted GNN.",
    }
    (output_dir / "phase2_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.require_improvement and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
