from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel
from experiments.gate_a.run_pilot import evaluate
from experiments.gate_b_plus.models import GateBPlusQuantumModel


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def screen_loss(prediction: torch.Tensor, batch: dict, fde_weight: float) -> torch.Tensor:
    mask = batch["future_mask"]
    error = torch.linalg.vector_norm(prediction - batch["future_xy"], dim=-1)
    weight = batch["target_weight"]
    ade_valid = mask.any(1)
    ade = (error * mask).sum(1) / mask.sum(1).clamp_min(1)
    loss = (ade[ade_valid] * weight[ade_valid]).sum() / weight[ade_valid].sum().clamp_min(1e-8)
    endpoint = mask[:, -1]
    if endpoint.any():
        loss = loss + fde_weight * (error[endpoint, -1] * weight[endpoint]).sum() / weight[endpoint].sum().clamp_min(1e-8)
    return loss


def distillation_loss(prediction: torch.Tensor, teacher: torch.Tensor, batch: dict,
                      fde_weight: float) -> torch.Tensor:
    proxy = dict(batch)
    proxy["future_xy"] = teacher.detach()
    return screen_loss(prediction, proxy, fde_weight)


def effective_rank_loss(latents: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for name in ("j2", "j3", "q2", "q3"):
        if name not in latents:
            continue
        centered = latents[name] - latents[name].mean(0, keepdim=True)
        energy = torch.linalg.svdvals(centered).square()
        probability = energy / energy.sum().clamp_min(1e-8)
        maximum_rank = max(min(centered.shape[0] - 1, centered.shape[1]), 2)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum()
        losses.append(1.0 - entropy / np.log(maximum_rank))
    if not losses:
        raise ValueError("no quantum latents available for effective-rank regularization")
    return torch.stack(losses).mean()


def attention_balance_loss(latents: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = latents["neighbor_mask"]
    count = mask.sum(1)
    valid = count > 1
    if not valid.any():
        return latents["attention_j2"].new_zeros(())
    entropies = []
    for name in ("attention_j2", "attention_j3"):
        weights = latents[name]
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1) / count.clamp_min(2).float().log()
        entropies.append(entropy)
    return (entropies[0][valid] - entropies[1][valid]).square().mean()


def load_matching_modules(model: GateBPlusQuantumModel, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    for name in ("encoder", "time", "decoder"):
        prefix = name + "."
        module_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        getattr(model, name).load_state_dict(module_state, strict=True)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("neighbor_residual", "dual_quantum_view", "full_multiscale",
                                           "multiscale_quantum_view", "fused_multiscale",
                                           "cross_order_multiscale", "quantum_latent_attention",
                                           "all_neighbor_quantum_latent_attention", "horizon_multiscale",
                                           "horizon_independent", "horizon_trajectory", "horizon_kinematic",
                                           "horizon_adaptive", "horizon_ranknorm",
                                           "multiscale_quantum_attention", "target_conditioned_quantum_attention",
                                           "stochastic_multiscale_quantum_attention",
                                           "dual_readout_quantum_attention",
                                           "residual_multiscale_quantum_attention"),
                        required=True)
    parser.add_argument("--seed", type=int, choices=tuple(range(2026, 2031)), required=True)
    parser.add_argument("--order-seed", type=int, default=None,
                        help="Fixed batch-order seed. Omit to retain the legacy seed-coupled order.")
    parser.add_argument("--dropout-seed", type=int, default=None,
                        help="Optional fixed per-epoch stochastic-layer seed, independent of initialization.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many validation epochs without a new best; 0 disables.")
    parser.add_argument("--checkpoint-average-k", type=int, choices=(1, 5), default=1,
                        help="Average the top-k validation EMA states; use 1 for the legacy best checkpoint.")
    parser.add_argument("--fde-weight", type=float, choices=(0.5, 0.75, 1.0), default=0.5)
    parser.add_argument("--core-kind", choices=("quantum", "quantum_wide", "johnson", "johnson_wide", "johnson_large"),
                        default="quantum")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, choices=(0.0, 0.995, 0.999), default=0.0)
    parser.add_argument("--warmstart-classical", default=None,
                        help="Classical best.pt used only to initialize the shared temporal/decoder modules.")
    parser.add_argument("--warmstart-quantum", default=None,
                        help="Quantum best.pt used to initialize the complete student before stage-two training.")
    parser.add_argument("--distill-teacher", default=None,
                        help="Frozen classical best.pt used for development-only trajectory distillation.")
    parser.add_argument("--distill-weight", type=float, choices=(0.0, 0.1, 0.25, 0.5, 1.0), default=0.0)
    parser.add_argument("--latent-rank-weight", type=float, choices=(0.0, 0.01, 0.05), default=0.0)
    parser.add_argument("--attention-balance-weight", type=float, choices=(0.0, 0.05, 0.2), default=0.0)
    parser.add_argument("--quantum-init", choices=("random", "near_identity"), default="random",
                        help="Initialization of the trainable compound quantum evolutions.")
    parser.add_argument("--quantum-init-scale", type=float, choices=(0.01, 0.05, 0.1), default=0.01,
                        help="Maximum absolute angle for the fixed near-identity pattern.")
    parser.add_argument("--target-loss-weight", type=float, choices=(0.0, 0.25, 0.5), default=0.0,
                        help="Auxiliary supervised weight for a history-only predicted 4 s target.")
    parser.add_argument("--branch-drop-probability", type=float, choices=(0.0, 0.2), default=0.0,
                        help="Probability of dropping exactly one order-specific quantum readout.")
    parser.add_argument("--config", default="configs/gate_b_plus_screen.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.distill_weight and not args.distill_teacher:
        raise ValueError("--distill-weight requires --distill-teacher")
    if args.target_loss_weight and args.mode != "target_conditioned_quantum_attention":
        raise ValueError("--target-loss-weight requires target_conditioned_quantum_attention")
    if args.branch_drop_probability and args.mode != "stochastic_multiscale_quantum_attention":
        raise ValueError("--branch-drop-probability requires stochastic_multiscale_quantum_attention")
    if args.warmstart_classical and args.warmstart_quantum:
        raise ValueError("choose only one warm-start source")
    cfg = json.loads((ROOT / args.config).read_text())
    seed = args.seed; seed_all(seed)
    order_seed = seed if args.order_seed is None else args.order_seed
    target_epochs = args.epochs or cfg["training"]["epochs"]
    train_limit = args.train_limit or cfg["training"]["train_limit"]
    out = ROOT / args.output
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint, state_path = out / "best.pt", out / "last.pt"
    train = GateADataset(ROOT / cfg["benchmark"], "train", train_limit, seed)
    val = GateADataset(ROOT / cfg["benchmark"], "val")
    val_loader = DataLoader(val, cfg["training"]["batch_size"] * 2, shuffle=False, num_workers=2)
    device = torch.device("cuda:0")
    model = GateBPlusQuantumModel(args.mode, dt_s=cfg["dt_s"], rounds=cfg["rounds"],
                                  core_kind=args.core_kind,
                                  branch_drop_probability=args.branch_drop_probability).to(device)
    if args.quantum_init == "near_identity" and not state_path.exists():
        model.initialize_quantum_evolution_near_identity(args.quantum_init_scale)
    warmstart = str(args.warmstart_classical or "")
    quantum_warmstart = str(args.warmstart_quantum or "")
    teacher_path = str(args.distill_teacher or "")
    if warmstart and not state_path.exists():
        load_matching_modules(model, ROOT / warmstart)
    if quantum_warmstart and not state_path.exists():
        source = torch.load(ROOT / quantum_warmstart, map_location="cpu", weights_only=True)["model"]
        if args.mode in {"horizon_trajectory", "horizon_kinematic"}:
            prefix = "trajectory_heads." if args.mode == "horizon_trajectory" else "kinematic_head."
            missing, unexpected = model.load_state_dict(source, strict=False)
            if unexpected or not missing or any(not key.startswith(prefix) for key in missing):
                raise RuntimeError(f"invalid partial quantum warm-start: missing={missing}, unexpected={unexpected}")
        else:
            model.load_state_dict(source)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, min_lr=1e-5)
    records, best, best_epoch, updates, elapsed, ema = [], float("inf"), 0, 0, 0.0, None
    checkpoint_candidates = []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if (state["mode"] != args.mode or state["seed"] != seed
                or state.get("fde_weight", 0.5) != args.fde_weight
                or state.get("core_kind", "quantum") != args.core_kind
                or state.get("train_limit", cfg["training"]["train_limit"]) != train_limit
                or state.get("ema_decay", 0.0) != args.ema_decay
                or state.get("order_seed", state["seed"]) != order_seed
                or state.get("dropout_seed") != args.dropout_seed
                or state.get("warmstart_classical", "") != warmstart
                or state.get("warmstart_quantum", "") != quantum_warmstart
                or state.get("distill_teacher", "") != teacher_path
                or state.get("distill_weight", 0.0) != args.distill_weight
                or state.get("latent_rank_weight", 0.0) != args.latent_rank_weight
                or state.get("attention_balance_weight", 0.0) != args.attention_balance_weight
                or state.get("quantum_init", "random") != args.quantum_init
                or state.get("quantum_init_scale", 0.01) != args.quantum_init_scale
                or state.get("target_loss_weight", 0.0) != args.target_loss_weight
                or state.get("branch_drop_probability", 0.0) != args.branch_drop_probability
                or state.get("checkpoint_average_k", 1) != args.checkpoint_average_k
                or state.get("patience", 0) != args.patience):
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        records, best, best_epoch = state["records"], state["best"], state["best_epoch"]
        updates, elapsed, ema = state["updates"], state["elapsed_s"], state.get("ema")
        checkpoint_candidates = state.get("checkpoint_candidates", [])
        if ema is not None:
            ema = {key: value.to(device) for key, value in ema.items()}
    teacher = None
    if teacher_path:
        teacher_state = torch.load(ROOT / teacher_path, map_location="cpu", weights_only=True)
        teacher_kind = teacher_state["kind"]
        teacher = GateAModel(teacher_kind, train.arrays["history_state"].shape[2],
                             dt_s=cfg["dt_s"]).to(device)
        teacher.load_state_dict(teacher_state["model"])
        teacher.eval()
    started = time.monotonic(); stopped_early = False
    for epoch in range(len(records) + 1, target_epochs + 1):
        if args.dropout_seed is not None:
            torch.manual_seed(args.dropout_seed + 7919 * epoch)
            torch.cuda.manual_seed_all(args.dropout_seed + 7919 * epoch)
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(order_seed + 1009 * epoch)
        ).tolist()
        loader = DataLoader(Subset(train, order), cfg["training"]["batch_size"], shuffle=False, num_workers=2)
        model.train(); losses = []; supervised_losses = []; distill_losses = []; rank_losses = []
        target_losses = []
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if args.latent_rank_weight or args.attention_balance_weight or args.target_loss_weight:
                prediction, latents = model(batch["history_state"], batch["node_mask"], return_aux=True)
                rank_loss = (effective_rank_loss(latents) if args.latent_rank_weight
                             else prediction.new_zeros(()))
                balance_loss = (attention_balance_loss(latents) if args.attention_balance_weight
                                else prediction.new_zeros(()))
            else:
                prediction = model(batch["history_state"], batch["node_mask"])
                rank_loss = prediction.new_zeros(())
                balance_loss = prediction.new_zeros(())
            supervised = screen_loss(prediction, batch, args.fde_weight)
            target_loss = prediction.new_zeros(())
            if args.target_loss_weight:
                endpoint = batch["future_mask"][:, -1]
                if endpoint.any():
                    endpoint_error = torch.linalg.vector_norm(
                        latents["quantum_target"][endpoint] - batch["future_xy"][endpoint, -1], dim=-1
                    )
                    endpoint_weight = batch["target_weight"][endpoint]
                    target_loss = ((endpoint_error * endpoint_weight).sum()
                                   / endpoint_weight.sum().clamp_min(1e-8))
            distilled = prediction.new_zeros(())
            if teacher is not None and args.distill_weight:
                with torch.no_grad():
                    teacher_prediction = teacher(batch["history_state"], batch["node_mask"])
                distilled = distillation_loss(prediction, teacher_prediction, batch, args.fde_weight)
            loss = (supervised + args.distill_weight * distilled + args.latent_rank_weight * rank_loss
                    + args.attention_balance_weight * balance_loss
                    + args.target_loss_weight * target_loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            if args.ema_decay:
                current = model.state_dict()
                if ema is None:
                    ema = {key: value.detach().clone() for key, value in current.items()}
                else:
                    for key, value in current.items():
                        if value.is_floating_point() or value.is_complex():
                            ema[key].mul_(args.ema_decay).add_(value.detach(), alpha=1 - args.ema_decay)
                        else:
                            ema[key].copy_(value)
            losses.append(float(loss.detach())); supervised_losses.append(float(supervised.detach()))
            distill_losses.append(float(distilled.detach())); rank_losses.append(float(rank_loss.detach())); updates += 1
            target_losses.append(float(target_loss.detach()))
        raw_state = None
        if ema is not None:
            raw_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(ema)
        metrics = evaluate(model, val_loader, device, dt_s=cfg["dt_s"])
        score = metrics["ic4"]["J_m"]; scheduler.step(score)
        records.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "supervised_loss": float(np.mean(supervised_losses)),
                        "distillation_loss": float(np.mean(distill_losses)),
                        "latent_rank_loss": float(np.mean(rank_losses)),
                        "target_loss": float(np.mean(target_losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"], "validation": metrics})
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "mode": args.mode, "seed": seed,
                        "core_kind": args.core_kind, "train_limit": train_limit,
                        "ema_decay": args.ema_decay, "epoch": epoch, "score": score}, checkpoint)
        if args.checkpoint_average_k > 1:
            candidate = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint_candidates.append({"score": score, "epoch": epoch, "model": candidate})
            checkpoint_candidates.sort(key=lambda row: row["score"])
            checkpoint_candidates = checkpoint_candidates[:args.checkpoint_average_k]
        if raw_state is not None:
            model.load_state_dict(raw_state)
        elapsed_now = elapsed + time.monotonic() - started
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "mode": args.mode, "seed": seed, "fde_weight": args.fde_weight,
                 "core_kind": args.core_kind, "train_limit": train_limit,
                 "ema_decay": args.ema_decay, "order_seed": order_seed,
                 "dropout_seed": args.dropout_seed, "ema": ema,
                 "warmstart_classical": warmstart, "distill_teacher": teacher_path,
                 "warmstart_quantum": quantum_warmstart,
                 "distill_weight": args.distill_weight, "latent_rank_weight": args.latent_rank_weight,
                 "attention_balance_weight": args.attention_balance_weight,
                 "quantum_init": args.quantum_init,
                 "quantum_init_scale": args.quantum_init_scale,
                 "target_loss_weight": args.target_loss_weight,
                 "branch_drop_probability": args.branch_drop_probability,
                 "checkpoint_average_k": args.checkpoint_average_k,
                 "patience": args.patience,
                 "checkpoint_candidates": checkpoint_candidates,
                 "records": records, "best": best, "best_epoch": best_epoch,
                 "updates": updates, "elapsed_s": elapsed_now}
        tmp = state_path.with_suffix(".tmp"); torch.save(state, tmp); tmp.replace(state_path)
        atomic_json(out / "heartbeat.json", {"mode": args.mode, "epoch": epoch, "best_epoch": best_epoch,
                                               "best_ic4_J": best, "elapsed_s": elapsed_now})
        print(json.dumps({"mode": args.mode, "epoch": epoch, "ic4": metrics["ic4"],
                          "best_epoch": best_epoch}), flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            stopped_early = True
            break
    if args.checkpoint_average_k > 1:
        if len(checkpoint_candidates) != args.checkpoint_average_k:
            raise RuntimeError(f"only {len(checkpoint_candidates)} checkpoint candidates available")
        averaged = {}
        for key in checkpoint_candidates[0]["model"]:
            values = [row["model"][key] for row in checkpoint_candidates]
            averaged[key] = (torch.stack(values).mean(0)
                             if values[0].is_floating_point() or values[0].is_complex() else values[0])
        saved = {"model": averaged, "mode": args.mode, "seed": seed,
                 "core_kind": args.core_kind, "train_limit": train_limit,
                 "ema_decay": args.ema_decay,
                 "epoch": [row["epoch"] for row in checkpoint_candidates],
                 "score": [row["score"] for row in checkpoint_candidates]}
        torch.save(saved, checkpoint)
    else:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    final = evaluate(model, val_loader, device, dt_s=cfg["dt_s"])
    ic4 = final["ic4"]
    references = {
        2026: {"ade_m": 1.20587880037012, "fde_m": 3.1602730388136586, "J_m": 2.7860153197769493},
        2027: {"ade_m": 1.2112257490871445, "fde_m": 3.2048305122071574, "J_m": 2.813641005190723},
    }
    reference = references.get(seed)
    passed = reference is not None and all(ic4[key] <= reference[key] for key in reference)
    status = ("SCREEN_PASS" if passed else "SCREEN_FAIL") if reference is not None else "DEVELOPMENT_COMPLETED"
    summary = {"revision": cfg["revision"], "status": status,
               "not_formal_evidence": True, "test_accessed": False, "mode": args.mode, "seed": seed,
               "fde_weight": args.fde_weight, "core_kind": args.core_kind,
               "model_seed": seed, "order_seed": order_seed, "dropout_seed": args.dropout_seed,
               "warmstart_classical": warmstart, "distill_teacher": teacher_path,
               "warmstart_quantum": quantum_warmstart,
               "distill_weight": args.distill_weight, "latent_rank_weight": args.latent_rank_weight,
               "attention_balance_weight": args.attention_balance_weight,
               "quantum_init": args.quantum_init,
               "quantum_init_scale": args.quantum_init_scale,
               "target_loss_weight": args.target_loss_weight,
               "branch_drop_probability": args.branch_drop_probability,
               "checkpoint_average_k": args.checkpoint_average_k,
               "target_epochs": target_epochs, "early_stopping_patience": args.patience,
               "stopped_early": stopped_early,
               "train_samples": len(train), "train_population": len(train.arrays["history_state"]),
               "ema_decay": args.ema_decay,
               "parameters": model.parameter_audit(), "epochs_completed": len(records),
               "checkpoint_epoch": saved["epoch"], "optimizer_updates": updates,
               "runtime_s": elapsed + time.monotonic() - started, "validation": final,
               "frozen_raj_epoch12_reference": reference, "training": records}
    atomic_json(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "mode": args.mode, "ic4": ic4}, indent=2))


if __name__ == "__main__":
    main()
