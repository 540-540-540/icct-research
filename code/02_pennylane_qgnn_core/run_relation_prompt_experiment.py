"""Train the direct graph-relation-token to LLM prompt interface."""
from __future__ import annotations

import argparse
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
from quantum_relation_prompt_model import RelationPromptCoreGraphLLM
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import candidate_ade, load_banks, token_accuracy
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss

SEED = 2026
GRAPH_DIR = BASE / "physics_aligned_dual_seed2026_v1"


def restore_relation_state(model, state):
    missing, extra = model.load_state_dict(state, strict=False)
    assert not extra, extra
    allowed = []
    for name in missing:
        frozen_gpt = (
            (name.startswith("gpt2.") or name.startswith("future_gpt2."))
            and "lora_" not in name
        )
        relation_new = name.startswith("relation_")
        if frozen_gpt or relation_new:
            allowed.append(name)
    assert len(allowed) == len(missing), missing


def build_pair(arm):
    graph_name = f"physics_{arm}_dual"
    source_dir = BASE / f"future_token_{'qgnn' if arm == 'quantum' else 'classical'}_llm_seed2026_v1"
    source_checkpoint = torch.load(
        source_dir / ("future_token_qgnn_llm_selected.pt" if arm == "quantum" else "future_token_qgnn_llm_selected.pt"),
        map_location="cpu", weights_only=True,
    )
    graph_checkpoint = torch.load(
        GRAPH_DIR / f"{graph_name}_graph_selected.pt", map_location="cpu", weights_only=True
    )

    reference_graph = build_physics_aligned_dual_graph(graph_name)
    reference_graph.load_state_dict(graph_checkpoint["state"])
    reference = FutureTokenCoreGraphLLM(reference_graph)
    restore_future_compact(reference, source_checkpoint["state"])

    relation_graph = build_physics_aligned_dual_graph(graph_name)
    relation_graph.load_state_dict(graph_checkpoint["state"])
    relation = RelationPromptCoreGraphLLM(relation_graph)
    restore_relation_state(relation, source_checkpoint["state"])
    return reference, relation, source_checkpoint, source_dir


def configure_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("relation_"):
            parameter.requires_grad_(True)


def train(model, train_bank, selection_bank, output_dir, epochs):
    configure_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=4e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    initial, _ = retained.evaluate(model, selection_bank)
    best_score = initial["aggregate"]["ade_m"] + 0.35 * initial["aggregate"]["fde_m"]
    best_epoch = 0
    checkpoint = output_dir / "relation_prompt_selected.pt"
    torch.save({"state": future_compact_state(model), "epoch": 0, "metrics": initial}, checkpoint)

    with (output_dir / "training.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            started = time.time()
            utils.set_seed(SEED + 70000 + epoch)
            model.train()
            generator = torch.Generator(device="cuda").manual_seed(SEED + 70000 + epoch)
            order = torch.randperm(
                len(train_bank["history"]), generator=torch.Generator().manual_seed(SEED + 70000 + epoch)
            ).cuda()
            totals = {"coordinate": 0.0, "token": 0.0, "candidate": 0.0, "accuracy": 0.0}
            count = 0
            for offset in range(0, len(order), 24):
                indices = order[offset:offset + 24]
                snr = (5, 10, 15, 20)[(offset // 24 + epoch) % 4]
                history, future, mask = retained.make_batch(train_bank, indices, snr, generator)
                optimizer.zero_grad(set_to_none=True)
                output = model(history, mask)
                targets = model.future_token_ids(history, future)
                coordinate = prediction_loss(output, future, mask, True)
                tokens = token_loss(output["token_logits"], targets, mask)
                candidate = candidate_ade(output, future, mask)
                loss = coordinate + 0.05 * tokens + 0.15 * candidate
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 3.0, error_if_nonfinite=True)
                optimizer.step()
                size = len(indices)
                totals["coordinate"] += float(coordinate.detach()) * size
                totals["token"] += float(tokens.detach()) * size
                totals["candidate"] += float(candidate.detach()) * size
                totals["accuracy"] += float(token_accuracy(output["token_logits"], targets, mask).detach()) * size
                count += size
            scheduler.step()
            metrics, _ = retained.evaluate(model, selection_bank)
            score = metrics["aggregate"]["ade_m"] + 0.35 * metrics["aggregate"]["fde_m"]
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save({"state": future_compact_state(model), "epoch": epoch, "metrics": metrics}, checkpoint)
            record = {
                "epoch": epoch,
                "coordinate_loss": totals["coordinate"] / count,
                "token_ce": totals["token"] / count,
                "candidate_ade_m": totals["candidate"] / count,
                "token_accuracy": totals["accuracy"] / count,
                "relation_prompt_strength": float(torch.tanh(model.relation_prompt_strength).detach()),
                "relation_coordinate_strength": float(torch.tanh(model.relation_coordinate_strength).detach()),
                "relation_attention_entropy": float(model.last_relation_attention_entropy),
                "relation_risk_activation": float(model.last_relation_risk_activation),
                "metrics": metrics,
                "score": score,
                "best_epoch": best_epoch,
                "seconds": time.time() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            utils.atomic_json(output_dir / "progress.json", {"status": "training", "last": record})
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_relation_state(model, selected["state"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("quantum", "classical"), default="quantum")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    utils.set_seed(SEED)
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    retained.NOISE = utils.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    train_bank, selection_bank, dev_bank, train_indices, selection_indices, dev_indices = load_banks()
    reference, model, source_checkpoint, source_dir = build_pair(args.arm)
    reference, model = reference.cuda().eval(), model.cuda().eval()
    with torch.no_grad():
        old = reference(selection_bank["history"][:2], selection_bank["mask"][:2])["future_position"]
        new = model(selection_bank["history"][:2], selection_bank["mask"][:2])["future_position"]
    exact_difference = float((old - new).abs().max())
    assert exact_difference < 2e-6, exact_difference
    configure_trainable(model)
    model.train()
    output = model(selection_bank["history"][:2], selection_bank["mask"][:2])
    loss = prediction_loss(output, selection_bank["future"][:2], selection_bank["mask"][:2], True)
    loss.backward()
    strength_gradient = float(model.relation_prompt_strength.grad.abs())
    coordinate_strength_gradient = float(model.relation_coordinate_strength.grad.abs())
    assert np.isfinite(strength_gradient) and strength_gradient > 0
    assert np.isfinite(coordinate_strength_gradient) and coordinate_strength_gradient > 0
    smoke = {
        "arm": args.arm,
        "source": str(source_dir),
        "source_epoch": int(source_checkpoint["epoch"]),
        "epoch0_exact_max_abs_m": exact_difference,
        "relation_strength_gradient": strength_gradient,
        "relation_coordinate_strength_gradient": coordinate_strength_gradient,
    }
    utils.atomic_json(output_dir / "smoke.json", smoke)
    if args.smoke_only:
        utils.atomic_json(output_dir / "completed.json", {"status": "smoke_completed"})
        return
    protocol = {
        "seed": SEED,
        "arm": args.arm,
        "epochs": args.epochs,
        "source": str(source_dir),
        "interface": "edge-level layer-1/layer-2 circuit observables + Pauli uncertainty + physical edge features -> relation tokens -> future-query cross-attention -> GPT-2",
        "frozen": "entire selected graph/future-token model; only relation-token interface is trained",
        "selection": "fixed four-SNR macro ADE + 0.35 FDE; no test",
        "indices": {"train": train_indices.tolist(), "selection": selection_indices.tolist(), "dev": dev_indices.tolist()},
    }
    utils.atomic_json(output_dir / "protocol.json", protocol)
    selected = train(model, train_bank, selection_bank, output_dir, args.epochs)
    full_on, arrays = retained.evaluate(model, dev_bank, collect=True)
    np.savez_compressed(output_dir / "relation_prompt_full_dev.npz", **arrays, indices=dev_indices)
    saved_strength = model.relation_prompt_strength.detach().clone()
    saved_coordinate_strength = model.relation_coordinate_strength.detach().clone()
    with torch.no_grad():
        model.relation_prompt_strength.zero_()
        model.relation_coordinate_strength.zero_()
    full_off, _ = retained.evaluate(model, dev_bank)
    with torch.no_grad():
        model.relation_prompt_strength.copy_(saved_strength)
        model.relation_coordinate_strength.copy_(saved_coordinate_strength)
    summary = {
        "arm": args.arm,
        "selected_epoch": int(selected["epoch"]),
        "selection": selected["metrics"],
        "full_dev_relation_on": full_on,
        "full_dev_relation_off_ablation": full_off,
        "relation_prompt_strength": float(torch.tanh(model.relation_prompt_strength).detach()),
        "relation_coordinate_strength": float(torch.tanh(model.relation_coordinate_strength).detach()),
        "no_test": True,
    }
    utils.atomic_json(output_dir / "summary.json", summary)
    utils.atomic_json(output_dir / "completed.json", {"status": "completed", "summary": summary})
    utils.atomic_json(output_dir / "progress.json", {"status": "completed", "summary": summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
