"""Outcome-blind high-interaction training for temporal graph-routed LLM fusion."""
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE))

import experiment_utils as utils
import run as retained
from analyze_interaction_strata import scene_features
from future_token_model import FutureTokenCoreGraphLLM, future_compact_state, restore_future_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss
from temporal_risk_routed_model import TemporalRiskRoutedFutureLLM

SEED = 2026
GRAPH_DIR = BASE / "physics_aligned_dual_seed2026_v1"


def restore_temporal(model, state):
    missing, extra = model.load_state_dict(state, strict=False)
    assert not extra, extra
    for name in missing:
        frozen_gpt = ((name.startswith("gpt2.") or name.startswith("future_gpt2.")) and "lora_" not in name)
        assert frozen_gpt or name.startswith("temporal_route_"), name


def build_pair(arm):
    graph_name = f"physics_{arm}_dual"
    source_dir = BASE / f"future_token_{'qgnn' if arm == 'quantum' else 'classical'}_llm_seed2026_v1"
    source = torch.load(source_dir / "future_token_qgnn_llm_selected.pt", map_location="cpu", weights_only=True)
    graph_state = torch.load(GRAPH_DIR / f"{graph_name}_graph_selected.pt", map_location="cpu", weights_only=True)["state"]
    graph = build_physics_aligned_dual_graph(graph_name); graph.load_state_dict(graph_state)
    reference = FutureTokenCoreGraphLLM(graph); restore_future_compact(reference, source["state"])
    graph = build_physics_aligned_dual_graph(graph_name); graph.load_state_dict(graph_state)
    routed = TemporalRiskRoutedFutureLLM(graph); restore_temporal(routed, source["state"])
    return reference, routed, source, source_dir


def subset(bank, positions):
    index = torch.as_tensor(positions, device="cuda", dtype=torch.long)
    return {key: value.index_select(0, index) for key, value in bank.items()}


def complexity_groups(indices):
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        features = scene_features(data["train_states"][indices, :20], data["train_mask"][indices])
    cuts = np.quantile(features["composite"], [1 / 3, 2 / 3])
    score = features["composite"]
    return {
        "low": np.flatnonzero(score <= cuts[0]),
        "medium": np.flatnonzero((score > cuts[0]) & (score <= cuts[1])),
        "high": np.flatnonzero(score > cuts[1]),
    }, cuts


def sampled_order(groups, total, epoch):
    generator = torch.Generator().manual_seed(SEED + 110000 + epoch)
    counts = {"high": int(0.60 * total), "medium": int(0.20 * total)}
    counts["low"] = total - counts["high"] - counts["medium"]
    pieces = []
    for name in ("high", "medium", "low"):
        source = torch.as_tensor(groups[name], dtype=torch.long)
        draw = torch.randint(len(source), (counts[name],), generator=generator)
        pieces.append(source[draw])
    order = torch.cat(pieces)
    return order[torch.randperm(len(order), generator=generator)].cuda()


def configure_trainable(model):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("temporal_route_"): parameter.requires_grad_(True)


def train(model, train_bank, train_groups, selection, output_dir, epochs):
    configure_trainable(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    initial, _ = retained.evaluate(model, selection)
    best_score = initial["aggregate"]["ade_m"] + 0.5 * initial["aggregate"]["fde_m"]
    best_epoch = 0; checkpoint = output_dir / "temporal_route_selected.pt"
    torch.save({"state": future_compact_state(model), "epoch": 0, "metrics": initial}, checkpoint)
    with (output_dir / "training.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            started = time.time(); utils.set_seed(SEED + 110000 + epoch); model.train()
            generator = torch.Generator(device="cuda").manual_seed(SEED + 110000 + epoch)
            order = sampled_order(train_groups, len(train_bank["history"]), epoch)
            loss_sum = residual_sum = active_sum = 0.0; count = 0
            for offset in range(0, len(order), 24):
                positions = order[offset:offset + 24]
                snr = (5, 10, 15, 20)[(offset // 24 + epoch) % 4]
                history, future, mask = retained.make_batch(train_bank, positions, snr, generator)
                optimizer.zero_grad(set_to_none=True); output = model(history, mask)
                loss = prediction_loss(output, future, mask, True); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 3.0, error_if_nonfinite=True); optimizer.step()
                size = len(positions); count += size
                loss_sum += float(loss.detach()) * size
                residual_sum += float(model.last_temporal_route_residual_mean) * size
                active_sum += float(model.last_temporal_route_active_fraction) * size
            scheduler.step(); metrics, _ = retained.evaluate(model, selection)
            score = metrics["aggregate"]["ade_m"] + 0.5 * metrics["aggregate"]["fde_m"]
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save({"state": future_compact_state(model), "epoch": epoch, "metrics": metrics}, checkpoint)
            record = {"epoch": epoch, "train_loss": loss_sum / count, "residual_mean_m": residual_sum / count,
                      "active_fraction": active_sum / count, "metrics": metrics, "score": score,
                      "best_epoch": best_epoch, "seconds": time.time() - started}
            log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True)
            utils.atomic_json(output_dir / "progress.json", {"status": "training", "last": record})
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True); restore_temporal(model, selected["state"])
    return selected


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=("quantum", "classical"), required=True)
    parser.add_argument("--output-name", required=True); parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true"); args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == "jscn" and torch.cuda.is_available()
    torch.set_num_threads(4); utils.set_seed(SEED)
    output_dir = BASE / args.output_name; output_dir.mkdir(exist_ok=False)
    retained.NOISE = utils.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    train_bank, _, dev_bank, train_indices, _, dev_indices = load_banks()
    train_groups, train_cuts = complexity_groups(train_indices); dev_groups, dev_cuts = complexity_groups(dev_indices)
    high = dev_groups["high"]; selection_positions, confirmation_positions = high[::2], high[1::2]
    selection, confirmation = subset(dev_bank, selection_positions), subset(dev_bank, confirmation_positions)
    reference, model, source, source_dir = build_pair(args.arm); reference, model = reference.cuda().eval(), model.cuda().eval()
    history, future, mask = selection["history"][:2], selection["future"][:2], selection["mask"][:2]
    with torch.no_grad(): exact = float((reference(history, mask)["future_position"] - model(history, mask)["future_position"]).abs().max())
    assert exact < 2e-6; configure_trainable(model); model.train(); output = model(history, mask)
    prediction_loss(output, future, mask, True).backward()
    gradient = float(model.temporal_route_head[-1].weight.grad.abs().max()); assert np.isfinite(gradient) and gradient > 0
    utils.atomic_json(output_dir / "smoke.json", {"epoch0_exact_max_abs_m": exact, "final_head_gradient": gradient})
    if args.smoke_only: utils.atomic_json(output_dir / "completed.json", {"status": "smoke_completed"}); return
    utils.atomic_json(output_dir / "protocol.json", {"seed": SEED, "arm": args.arm, "epochs": args.epochs,
        "source": str(source_dir), "train_sampling": {"high": 0.6, "medium": 0.2, "low": 0.2},
        "complexity": "outcome-blind target count, close pairs, conflicts, closing strength, maneuver",
        "high_dev_split": {"selection_positions": selection_positions.tolist(), "confirmation_positions": confirmation_positions.tolist()},
        "train_cuts": train_cuts.tolist(), "dev_cuts": dev_cuts.tolist(), "no_test": True})
    source_selection, _ = retained.evaluate(reference, selection); source_confirmation, _ = retained.evaluate(reference, confirmation)
    selected = train(model, train_bank, train_groups, selection, output_dir, args.epochs)
    confirmation_result, arrays = retained.evaluate(model, confirmation, collect=True)
    np.savez_compressed(output_dir / "high_confirmation.npz", **arrays, indices=dev_indices[confirmation_positions])
    full_dev, _ = retained.evaluate(model, dev_bank)
    model.attention_ablation_mode = "uniform"; confirmation_uniform, _ = retained.evaluate(model, confirmation)
    model.attention_ablation_mode = "off"; confirmation_off, _ = retained.evaluate(model, confirmation)
    summary = {"arm": args.arm, "selected_epoch": int(selected["epoch"]), "selection": selected["metrics"],
        "source_selection": source_selection, "source_confirmation": source_confirmation,
        "confirmation_learned": confirmation_result, "confirmation_uniform": confirmation_uniform,
        "confirmation_off": confirmation_off, "full_dev_secondary": full_dev, "no_test": True}
    utils.atomic_json(output_dir / "summary.json", summary); utils.atomic_json(output_dir / "completed.json", {"status": "completed", "summary": summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__": main()
