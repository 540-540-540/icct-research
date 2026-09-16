"""Matched quantum/classical test of QGNN-attention-routed LLM future states."""
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
from attention_routed_fusion_model import AttentionRoutedFutureLLM
from future_token_model import FutureTokenCoreGraphLLM, future_compact_state, restore_future_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss

SEED = 2026
GRAPH_DIR = BASE / "physics_aligned_dual_seed2026_v1"


def restore_routed(model, state):
    missing, extra = model.load_state_dict(state, strict=False)
    assert not extra, extra
    for name in missing:
        frozen_gpt = (
            (name.startswith("gpt2.") or name.startswith("future_gpt2."))
            and "lora_" not in name
        )
        assert frozen_gpt or name.startswith("routed_"), name


def build_pair(arm):
    graph_name = f"physics_{arm}_dual"
    source_dir = BASE / f"future_token_{'qgnn' if arm == 'quantum' else 'classical'}_llm_seed2026_v1"
    source = torch.load(
        source_dir / "future_token_qgnn_llm_selected.pt", map_location="cpu", weights_only=True
    )
    graph_state = torch.load(
        GRAPH_DIR / f"{graph_name}_graph_selected.pt", map_location="cpu", weights_only=True
    )["state"]
    reference_graph = build_physics_aligned_dual_graph(graph_name)
    reference_graph.load_state_dict(graph_state)
    reference = FutureTokenCoreGraphLLM(reference_graph)
    restore_future_compact(reference, source["state"])
    routed_graph = build_physics_aligned_dual_graph(graph_name)
    routed_graph.load_state_dict(graph_state)
    routed = AttentionRoutedFutureLLM(routed_graph)
    restore_routed(routed, source["state"])
    return reference, routed, source, source_dir


def configure_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("routed_"):
            parameter.requires_grad_(True)


def train(model, train_bank, selection_bank, output_dir, epochs):
    configure_trainable(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=4e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    initial, _ = retained.evaluate(model, selection_bank)
    best_score = initial["aggregate"]["ade_m"] + 0.35 * initial["aggregate"]["fde_m"]
    best_epoch = 0
    checkpoint = output_dir / "attention_routed_selected.pt"
    torch.save({"state": future_compact_state(model), "epoch": 0, "metrics": initial}, checkpoint)
    with (output_dir / "training.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            started = time.time()
            utils.set_seed(SEED + 90000 + epoch)
            model.train()
            generator = torch.Generator(device="cuda").manual_seed(SEED + 90000 + epoch)
            order = torch.randperm(
                len(train_bank["history"]), generator=torch.Generator().manual_seed(SEED + 90000 + epoch)
            ).cuda()
            loss_sum = 0.0
            residual_sum = 0.0
            count = 0
            for offset in range(0, len(order), 24):
                indices = order[offset:offset + 24]
                snr = (5, 10, 15, 20)[(offset // 24 + epoch) % 4]
                history, future, mask = retained.make_batch(train_bank, indices, snr, generator)
                optimizer.zero_grad(set_to_none=True)
                output = model(history, mask)
                loss = prediction_loss(output, future, mask, True)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 3.0, error_if_nonfinite=True)
                optimizer.step()
                size = len(indices)
                loss_sum += float(loss.detach()) * size
                residual_sum += float(model.last_routed_residual_mean) * size
                count += size
            scheduler.step()
            metrics, _ = retained.evaluate(model, selection_bank)
            score = metrics["aggregate"]["ade_m"] + 0.35 * metrics["aggregate"]["fde_m"]
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save({"state": future_compact_state(model), "epoch": epoch, "metrics": metrics}, checkpoint)
            record = {
                "epoch": epoch, "train_loss": loss_sum / count,
                "routed_residual_mean_m": residual_sum / count,
                "metrics": metrics, "score": score, "best_epoch": best_epoch,
                "seconds": time.time() - started,
            }
            log.write(json.dumps(record) + "\n"); log.flush()
            print(json.dumps(record), flush=True)
            utils.atomic_json(output_dir / "progress.json", {"status": "training", "last": record})
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_routed(model, selected["state"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("quantum", "classical"), default="quantum")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == "jscn" and torch.cuda.is_available()
    torch.set_num_threads(4); utils.set_seed(SEED)
    output_dir = BASE / args.output_name; output_dir.mkdir(exist_ok=False)
    retained.NOISE = utils.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    train_bank, selection_bank, dev_bank, train_indices, selection_indices, dev_indices = load_banks()
    reference, model, source, source_dir = build_pair(args.arm)
    reference, model = reference.cuda().eval(), model.cuda().eval()
    history, future, mask = selection_bank["history"][:2], selection_bank["future"][:2], selection_bank["mask"][:2]
    with torch.no_grad():
        before = reference(history, mask)["future_position"]
        after = model(history, mask)["future_position"]
    exact = float((before - after).abs().max()); assert exact < 2e-6
    configure_trainable(model); model.train(); output = model(history, mask)
    loss = prediction_loss(output, future, mask, True); loss.backward()
    final_gradient = float(model.routed_interaction_head[-1].weight.grad.abs().max())
    assert np.isfinite(final_gradient) and final_gradient > 0
    utils.atomic_json(output_dir / "smoke.json", {
        "arm": args.arm, "source": str(source_dir), "source_epoch": int(source["epoch"]),
        "epoch0_exact_max_abs_m": exact, "routed_final_layer_gradient": final_gradient,
    })
    if args.smoke_only:
        utils.atomic_json(output_dir / "completed.json", {"status": "smoke_completed"}); return
    utils.atomic_json(output_dir / "protocol.json", {
        "seed": SEED, "arm": args.arm, "epochs": args.epochs, "source": str(source_dir),
        "interface": "two graph attention maps route per-target GPT-2 future hidden states; coordinate residual has no independent graph-only or LLM-only input",
        "frozen": "complete selected graph+future-token model; train routed projection/head only",
        "selection": "fixed four-SNR macro ADE + 0.35 FDE; no test",
        "indices": {"train": train_indices.tolist(), "selection": selection_indices.tolist(), "dev": dev_indices.tolist()},
    })
    selected = train(model, train_bank, selection_bank, output_dir, args.epochs)
    model.attention_ablation_mode = "learned"
    learned, arrays = retained.evaluate(model, dev_bank, collect=True)
    np.savez_compressed(output_dir / "attention_routed_full_dev.npz", **arrays, indices=dev_indices)
    model.attention_ablation_mode = "uniform"; uniform, _ = retained.evaluate(model, dev_bank)
    model.attention_ablation_mode = "off"; off, _ = retained.evaluate(model, dev_bank)
    model.attention_ablation_mode = "learned"
    summary = {
        "arm": args.arm, "selected_epoch": int(selected["epoch"]), "selection": selected["metrics"],
        "full_dev_learned_attention": learned, "full_dev_uniform_attention": uniform,
        "full_dev_interaction_off": off, "no_test": True,
    }
    utils.atomic_json(output_dir / "summary.json", summary)
    utils.atomic_json(output_dir / "completed.json", {"status": "completed", "summary": summary})
    utils.atomic_json(output_dir / "progress.json", {"status": "completed", "summary": summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
