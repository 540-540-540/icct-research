"""Single-seed development run for the future-token QGNN+LLM route."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/js_cn/sensing")
BASE = ROOT / "diagnostics/core_message_seed2026_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

import experiment_utils as utils
import run as retained
from future_token_model import FutureTokenCoreGraphLLM, future_compact_state, restore_future_compact
from model import CoreGraphLLM, restore_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss

SEED = 2026
GRAPH_DIR = BASE / "physics_aligned_dual_seed2026_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_banks():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz") as split:
        train_indices = split["circuit_train_indices"].copy()
        dev_indices = split["circuit_dev_indices"].copy()
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    selection_indices = utils.select_profile_indices(dev_indices, 256)

    def bank(indices):
        return {
            "history": torch.from_numpy(states[indices, :20]).float().cuda(),
            "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
            "mask": torch.from_numpy(masks[indices]).bool().cuda(),
        }

    return bank(train_indices), bank(selection_indices), bank(dev_indices), train_indices, selection_indices, dev_indices


def restore_models(arm):
    graph_name = f"physics_{arm}_dual"
    source_dir = (
        BASE / "quantum_llm_frozen_cont_seed2026_v1"
        if arm == "quantum" else BASE / "strong_dual_llm_seed2026_v1"
    )
    graph_checkpoint = torch.load(
        GRAPH_DIR / f"{graph_name}_graph_selected.pt", map_location="cpu", weights_only=True
    )
    source_checkpoint = torch.load(
        source_dir / f"{graph_name}_llm_selected.pt", map_location="cpu", weights_only=True
    )

    old_graph = build_physics_aligned_dual_graph(graph_name)
    old_graph.load_state_dict(graph_checkpoint["state"])
    old_model = CoreGraphLLM(old_graph)
    restore_compact(old_model, source_checkpoint["state"])

    new_graph = build_physics_aligned_dual_graph(graph_name)
    new_graph.load_state_dict(graph_checkpoint["state"])
    new_model = FutureTokenCoreGraphLLM(new_graph)
    missing, extra = new_model.load_state_dict(source_checkpoint["state"], strict=False)
    assert not extra, extra
    allowed_prefixes = (
        "gpt2.", "future_gpt2.", "future_type_embedding", "token_residual_head.",
        "adaptive_gate.", "fusion_strength",
    )
    assert all(name.startswith(allowed_prefixes) for name in missing), missing
    new_model.initialize_future_branch_from_baseline()
    return old_model, new_model, source_checkpoint, source_dir, graph_name


def candidate_ade(output, future, mask):
    distance = (output["token_future_position"] - future[..., :2]).norm(dim=-1)
    return (distance * mask[:, None, :]).sum() / (mask.sum().clamp_min(1) * distance.shape[1])


def token_accuracy(logits, targets, mask):
    correct = (logits.argmax(dim=-1) == targets) & mask[:, None, :]
    return correct.sum().float() / (mask.sum().clamp_min(1) * targets.shape[1])


def smoke_check(old_model, new_model, source_checkpoint, selection, output_dir, source_dir, graph_name):
    old_model = old_model.cuda().eval()
    new_model = new_model.cuda().eval()
    history = selection["history"][:2]
    future = selection["future"][:2]
    mask = selection["mask"][:2]
    with torch.no_grad():
        old_output = old_model(history, mask)
        new_output = new_model(history, mask)
    maximum_difference = float(
        (old_output["future_position"] - new_output["future_position"]).abs().max()
    )
    assert maximum_difference < 2e-6, maximum_difference
    assert float(new_output["fusion_strength"].abs()) == 0.0

    # Gradient checks: CE reaches future queries/LoRA, candidate loss reaches the
    # coordinate residual, and final ADE reaches the fusion strength.
    new_model.train()
    new_model.zero_grad(set_to_none=True)
    output = new_model(history, mask)
    targets = new_model.future_token_ids(history, future)
    loss = (
        prediction_loss(output, future, mask, True)
        + 0.05 * token_loss(output["token_logits"], targets, mask)
        + 0.15 * candidate_ade(output, future, mask)
    )
    loss.backward()
    gradients = {
        "future_queries": float(new_model.future_queries.grad.abs().max()),
        "token_head": float(new_model.token_head[-1].weight.grad.abs().max()),
        "token_residual": float(new_model.token_residual_head[-1].weight.grad.abs().max()),
        "fusion_strength": float(new_model.fusion_strength.grad.abs()),
        "future_lora": max(
            float(parameter.grad.abs().max())
            for name, parameter in new_model.future_gpt2.named_parameters()
            if "lora_" in name and parameter.grad is not None
        ),
    }
    assert all(np.isfinite(value) and value > 0 for value in gradients.values()), gradients
    evidence = {
        "source_epoch": int(source_checkpoint["epoch"]),
        "epoch0_exact_max_abs_m": maximum_difference,
        "fusion_strength": 0.0,
        "gradient_max": gradients,
        "source_checkpoint_sha256": sha256(source_dir / f"{graph_name}_llm_selected.pt"),
        "graph_checkpoint_sha256": sha256(GRAPH_DIR / f"{graph_name}_graph_selected.pt"),
    }
    utils.atomic_json(output_dir / "smoke.json", evidence)
    del old_model
    torch.cuda.empty_cache()
    return new_model


def set_trainable(model, joint):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.future_queries.requires_grad_(True)
    model.future_type_embedding.requires_grad_(True)
    for parameter in model.token_head.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.future_gpt2.named_parameters():
        parameter.requires_grad_("lora_" in name)
    for parameter in model.token_residual_head.parameters():
        parameter.requires_grad_(True)
    if joint:
        model.fusion_strength.requires_grad_(True)
        for parameter in model.adaptive_gate.parameters():
            parameter.requires_grad_(True)


def train(model, train_bank, selection_bank, output_dir, epochs=16, warmup_epochs=4):
    set_trainable(model, joint=False)
    named = dict(model.named_parameters())
    query_head = [
        parameter for name, parameter in named.items()
        if name.startswith(("future_queries", "future_type_embedding", "token_head.", "token_residual_head."))
    ]
    lora = [
        parameter for name, parameter in named.items()
        if name.startswith("future_gpt2.") and "lora_" in name
    ]
    fusion = [model.fusion_strength] + list(model.adaptive_gate.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": query_head, "lr": 2e-4},
            {"params": lora, "lr": 2.5e-5},
            {"params": fusion, "lr": 4e-4},
        ],
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    initial, _ = retained.evaluate(model, selection_bank)
    best_score = initial["aggregate"]["ade_m"] + 0.35 * initial["aggregate"]["fde_m"]
    best_epoch = 0
    checkpoint = output_dir / "future_token_qgnn_llm_selected.pt"
    torch.save({"state": future_compact_state(model), "epoch": 0, "metrics": initial}, checkpoint)

    with (output_dir / "training.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            if epoch == warmup_epochs + 1:
                set_trainable(model, joint=True)
            started = time.time()
            utils.set_seed(SEED + 50000 + epoch)
            model.train()
            noise_generator = torch.Generator(device="cuda").manual_seed(SEED + 50000 + epoch)
            order = torch.randperm(
                len(train_bank["history"]), generator=torch.Generator().manual_seed(SEED + 50000 + epoch)
            ).cuda()
            totals = {"coordinate": 0.0, "token": 0.0, "candidate": 0.0, "accuracy": 0.0}
            count = 0
            for offset in range(0, len(order), 24):
                indices = order[offset:offset + 24]
                snr = (5, 10, 15, 20)[(offset // 24 + epoch) % 4]
                history, future, mask = retained.make_batch(train_bank, indices, snr, noise_generator)
                optimizer.zero_grad(set_to_none=True)
                output = model(history, mask)
                targets = model.future_token_ids(history, future)
                coordinate = prediction_loss(output, future, mask, True)
                tokens = token_loss(output["token_logits"], targets, mask)
                candidate = candidate_ade(output, future, mask)
                if epoch <= warmup_epochs:
                    loss = 0.05 * tokens + 0.25 * candidate
                else:
                    loss = coordinate + 0.05 * tokens + 0.15 * candidate
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 3.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                size = len(indices)
                totals["coordinate"] += float(coordinate.detach()) * size
                totals["token"] += float(tokens.detach()) * size
                totals["candidate"] += float(candidate.detach()) * size
                totals["accuracy"] += float(token_accuracy(output["token_logits"], targets, mask).detach()) * size
                count += size
            scheduler.step()
            metrics, _ = retained.evaluate(model, selection_bank)
            current_score = metrics["aggregate"]["ade_m"] + 0.35 * metrics["aggregate"]["fde_m"]
            if current_score < best_score:
                best_score = current_score
                best_epoch = epoch
                torch.save({"state": future_compact_state(model), "epoch": epoch, "metrics": metrics}, checkpoint)
            record = {
                "epoch": epoch,
                "phase": "token_warmup" if epoch <= warmup_epochs else "joint_fusion",
                "coordinate_loss": totals["coordinate"] / count,
                "token_ce": totals["token"] / count,
                "candidate_ade_m": totals["candidate"] / count,
                "token_accuracy": totals["accuracy"] / count,
                "fusion_strength": float(torch.tanh(model.fusion_strength).detach()),
                "metrics": metrics,
                "score": current_score,
                "best_epoch": best_epoch,
                "seconds": time.time() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            utils.atomic_json(output_dir / "progress.json", {"status": "training", "last": record})
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_future_compact(model, selected["state"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="future_token_qgnn_llm_seed2026_v1")
    parser.add_argument("--arm", choices=("quantum", "classical"), default="quantum")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    utils.set_seed(SEED)
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    retained.NOISE = utils.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    train_bank, selection_bank, dev_bank, train_indices, selection_indices, dev_indices = load_banks()
    old_model, model, source_checkpoint, source_dir, graph_name = restore_models(args.arm)
    model = smoke_check(
        old_model, model, source_checkpoint, selection_bank, output_dir, source_dir, graph_name
    )
    if args.smoke_only:
        utils.atomic_json(output_dir / "completed.json", {"status": "smoke_completed"})
        return

    protocol = {
        "seed": SEED,
        "arm": args.arm,
        "route": "frozen retained graph+LLM plus trainable causal future-token GPT-2 branch",
        "source": str(source_dir / f"{graph_name}_llm_selected.pt"),
        "source_epoch": int(source_checkpoint["epoch"]),
        "epochs": 16,
        "token_warmup_epochs": 4,
        "selection": "four-SNR macro ADE + 0.35 FDE on fixed 256-scene subset",
        "no_test": True,
        "success_constraints": [
            "new QGNN+LLM improves retained QGNN+LLM",
            "LLM increment over QGNN remains positive",
            "matched classical GNN+same LLM control is required before a paper claim",
        ],
        "indices": {
            "train": train_indices.tolist(),
            "selection": selection_indices.tolist(),
            "dev": dev_indices.tolist(),
        },
    }
    utils.atomic_json(output_dir / "protocol.json", protocol)
    selected = train(model, train_bank, selection_bank, output_dir)
    dev_metrics, arrays = retained.evaluate(model, dev_bank, collect=True)
    np.savez_compressed(output_dir / "future_token_qgnn_llm_full_dev.npz", **arrays, indices=dev_indices)
    summary = {
        "selected_epoch": int(selected["epoch"]),
        "selection": selected["metrics"],
        "full_dev": dev_metrics,
        "fusion_strength": float(torch.tanh(model.fusion_strength).detach()),
        "source_epoch": int(source_checkpoint["epoch"]),
        "no_test": True,
    }
    utils.atomic_json(output_dir / "summary.json", summary)
    done = {"status": "completed", "summary": summary}
    utils.atomic_json(output_dir / "completed.json", done)
    utils.atomic_json(output_dir / "progress.json", done)
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
