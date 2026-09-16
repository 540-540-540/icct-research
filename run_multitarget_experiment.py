"""Phase-1 experiment: independent prediction versus target-interaction GNN."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from evaluate_multitarget import finalize_metric_sums, trajectory_metric_sums
from multitarget_scene_dataset import MultiTargetSceneDataset, build_scene_cache
from target_interaction_graph import ForecasterConfig, IndependentGRUForecaster, TargetInteractionGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noisy_history(
    history: torch.Tensor,
    mask: torch.Tensor,
    position_sigma: float,
    velocity_sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    observed = history.clone()
    if position_sigma > 0:
        noise = torch.randn(observed[..., :2].shape, device=observed.device, dtype=observed.dtype, generator=generator)
        observed[..., :2] += noise * position_sigma
    if velocity_sigma > 0:
        noise = torch.randn(observed[..., 2:4].shape, device=observed.device, dtype=observed.dtype, generator=generator)
        observed[..., 2:4] += noise * velocity_sigma
    return observed * mask[:, None, :, None]


def prediction_loss(
    output: Dict[str, torch.Tensor],
    future: torch.Tensor,
    mask: torch.Tensor,
    graph_weighting: bool,
) -> torch.Tensor:
    error = F.smooth_l1_loss(output["future_position"], future[..., :2], beta=1.0, reduction="none").sum(dim=-1)
    time_weight = torch.linspace(0.5, 1.5, error.shape[1], device=error.device)[None, :, None]
    target_weight = mask[:, None, :].to(error.dtype)
    if graph_weighting:
        final_position = future[:, 0, :, :2]
        relative = final_position[:, :, None, :] - final_position[:, None, :, :]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        interacting = ((distance < 15.0) & pair_mask & ~eye).any(dim=-1)
        target_weight = target_weight * (1.0 + 0.35 * interacting[:, None, :].to(error.dtype))
    main_loss = (error * time_weight * target_weight).sum() / (time_weight * target_weight).sum().clamp_min(1.0)
    final_loss = (error[:, -1] * mask).sum() / mask.sum().clamp_min(1)
    return main_loss + 0.15 * final_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    position_sigma: float,
    velocity_sigma: float,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        observed = noisy_history(history, mask, position_sigma, velocity_sigma, generator)
        output = model(observed, mask)
        components = trajectory_metric_sums(
            output["future_position"], future[..., :2], mask, observed[:, -1, :, :2]
        )
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
    return finalize_metric_sums(sums)


@torch.no_grad()
def evaluate_constant_velocity(
    loader: DataLoader,
    config: ForecasterConfig,
    device: torch.device,
    position_sigma: float,
    velocity_sigma: float,
    seed: int,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    times = (torch.arange(config.prediction_length, device=device) + 1) * config.dt
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        observed = noisy_history(history, mask, position_sigma, velocity_sigma, generator)
        prediction = observed[:, -1, None, :, :2] + observed[:, -1, None, :, 2:4] * times[None, :, None, None]
        components = trajectory_metric_sums(prediction, future[..., :2], mask, observed[:, -1, :, :2])
        for key, value in components.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
    return finalize_metric_sums(sums)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    position_sigma: float,
    velocity_sigma: float,
    checkpoint_path: Path,
    log_path: Path,
    seed: int,
    graph_weighting: bool,
    use_amp: bool = True,
) -> Tuple[nn.Module, Dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.08)
    amp_enabled = device.type == "cuda" and use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    initial_metrics = evaluate(model, validation_loader, device, position_sigma, velocity_sigma, seed + 1000)
    best_score = initial_metrics["ade_m"] + 0.35 * initial_metrics["fde_m"]
    best_metrics = initial_metrics
    best_state = copy.deepcopy(model.state_dict())
    patience = 8
    stale_epochs = 0

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"model": name, "epoch": 0, "validation": initial_metrics}) + "\n")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            examples = 0
            started = time.time()
            for batch in train_loader:
                history = batch["history"].to(device, non_blocking=True)
                future = batch["future"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                observed = noisy_history(history, mask, position_sigma, velocity_sigma)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(observed, mask)
                    loss = prediction_loss(output, future, mask, graph_weighting=graph_weighting)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                batch_size = history.shape[0]
                running_loss += float(loss.item()) * batch_size
                examples += batch_size
            scheduler.step()
            validation = evaluate(model, validation_loader, device, position_sigma, velocity_sigma, seed + 1000)
            score = validation["ade_m"] + 0.35 * validation["fde_m"]
            record = {
                "model": name,
                "epoch": epoch,
                "train_loss": running_loss / max(examples, 1),
                "validation": validation,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - started,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            if score < best_score - 1.0e-5:
                best_score = score
                best_metrics = validation
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping %s at epoch %d" % (name, epoch), flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(
        {"model_state": best_state, "validation_metrics": best_metrics, "name": name},
        checkpoint_path,
    )
    return model, best_metrics


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train and gate the multi-target interaction graph")
    parser.add_argument("--csv", default="Lankershim_Vehicle_Trajectories.csv")
    parser.add_argument("--cache", default="data/multitarget_lankershim_v1.npz")
    parser.add_argument("--output-dir", default="results/multitarget_phase1")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--prediction-length", type=int, default=20)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--max-train-scenes", type=int, default=6000)
    parser.add_argument("--max-val-scenes", type=int, default=1200)
    parser.add_argument("--max-test-scenes", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--independent-epochs", type=int, default=35)
    parser.add_argument("--graph-epochs", type=int, default=30)
    parser.add_argument("--independent-lr", type=float, default=1.0e-3)
    parser.add_argument("--graph-lr", type=float, default=4.0e-4)
    parser.add_argument("--position-noise", type=float, default=0.35)
    parser.add_argument("--velocity-noise", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-improvement", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    cache_path = Path(args.cache)
    if args.build_cache or not cache_path.exists():
        counts = build_scene_cache(
            csv_path=args.csv,
            output_path=str(cache_path),
            history_length=args.history_length,
            prediction_length=args.prediction_length,
            max_targets=args.max_targets,
            max_train_scenes=args.max_train_scenes,
            max_val_scenes=args.max_val_scenes,
            max_test_scenes=args.max_test_scenes,
            seed=args.seed,
        )
        print("Scene cache:", counts, flush=True)

    train_dataset = MultiTargetSceneDataset(str(cache_path), "train")
    validation_dataset = MultiTargetSceneDataset(str(cache_path), "val")
    test_dataset = MultiTargetSceneDataset(str(cache_path), "test")
    metadata = train_dataset.metadata
    config = ForecasterConfig(
        history_length=int(metadata["history_length"]),
        prediction_length=int(metadata["prediction_length"]),
        dt=float(metadata["dt"]),
        hidden_dim=args.hidden_dim,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=loader_generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment is configured for the remote CUDA GPU; CUDA is unavailable.")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"

    constant_velocity = evaluate_constant_velocity(
        test_loader, config, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )
    independent = IndependentGRUForecaster(config)
    independent, independent_validation = train_model(
        "independent_gru",
        independent,
        train_loader,
        validation_loader,
        device,
        args.independent_epochs,
        args.independent_lr,
        1.0e-4,
        args.position_noise,
        args.velocity_noise,
        output_dir / "independent_gru.pt",
        log_path,
        args.seed,
        graph_weighting=False,
    )
    independent_test = evaluate(
        independent, test_loader, device, args.position_noise, args.velocity_noise, args.seed + 2000
    )

    graph = TargetInteractionGNN(config)
    missing, unexpected = graph.load_state_dict(independent.state_dict(), strict=False)
    if unexpected or any(not key.startswith("graph_layers") for key in missing):
        raise RuntimeError("Unexpected graph warm-start mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    graph, graph_validation = train_model(
        "target_interaction_gnn",
        graph,
        train_loader,
        validation_loader,
        device,
        args.graph_epochs,
        args.graph_lr,
        2.0e-4,
        args.position_noise,
        args.velocity_noise,
        output_dir / "target_interaction_gnn.pt",
        log_path,
        args.seed + 17,
        graph_weighting=True,
    )
    graph_test = evaluate(graph, test_loader, device, args.position_noise, args.velocity_noise, args.seed + 2000)

    ade_gain = (independent_test["ade_m"] - graph_test["ade_m"]) / independent_test["ade_m"] * 100.0
    fde_gain = (independent_test["fde_m"] - graph_test["fde_m"]) / independent_test["fde_m"] * 100.0
    interaction_gain = (
        (independent_test["interaction_ade_m"] - graph_test["interaction_ade_m"])
        / independent_test["interaction_ade_m"]
        * 100.0
    )
    passed = graph_test["ade_m"] < independent_test["ade_m"] and graph_test["fde_m"] < independent_test["fde_m"]
    result = {
        "phase": "target_interaction_graph_gate",
        "device": torch.cuda.get_device_name(0),
        "metadata": metadata,
        "parameters": {
            "independent_gru": parameter_count(independent),
            "target_interaction_gnn": parameter_count(graph),
        },
        "validation": {"independent_gru": independent_validation, "target_interaction_gnn": graph_validation},
        "test": {
            "constant_velocity": constant_velocity,
            "independent_gru": independent_test,
            "target_interaction_gnn": graph_test,
        },
        "relative_gain_percent": {
            "ade": ade_gain,
            "fde": fde_gain,
            "interaction_ade": interaction_gain,
        },
        "graph_gate_passed": passed,
        "gate_rule": "GNN test ADE and FDE must both be lower than the independent GRU under identical noise.",
    }
    result_path = output_dir / "phase1_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.require_improvement and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
