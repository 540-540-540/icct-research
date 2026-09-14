"""Matched high-interaction LLM refinement for classical and quantum dual graph backbones."""
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

import experiment_utils as r
import run as retained
from analyze_interaction_strata import scene_features
from model import CoreGraphLLM, compact_state, restore_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss

SEED = 2026
ARMS = ("physics_classical_dual", "physics_quantum_dual")
GRAPH_DIR = BASE / "physics_aligned_dual_seed2026_v1"
CLASSIC_SOURCE = BASE / "strong_dual_llm_seed2026_v1"
QUANTUM_SOURCE = BASE / "quantum_llm_frozen_cont_seed2026_v1"
SNRS = (5, 10, 15, 20)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bank(states, masks, indices):
    return {
        "history": torch.from_numpy(states[indices, :20]).float().cuda(),
        "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
        "mask": torch.from_numpy(masks[indices]).bool().cuda(),
    }


def prepare_banks():
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz") as split:
        train_idx = split["circuit_train_indices"].copy()
        dev_idx = split["circuit_dev_indices"].copy()
    with np.load(ROOT / "data/multitarget_lankershim_v1.npz") as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()

    train_score = scene_features(states[train_idx, :20], masks[train_idx])["composite"]
    dev_score = scene_features(states[dev_idx, :20], masks[dev_idx])["composite"]
    train_cut = float(np.quantile(train_score, 2 / 3))
    dev_cut = float(np.quantile(dev_score, 2 / 3))
    high_train_idx = train_idx[train_score > train_cut]
    high_dev_idx = dev_idx[dev_score > dev_cut]
    selection_idx = r.select_profile_indices(high_dev_idx, min(256, len(high_dev_idx)))
    banks = {
        "train_high": make_bank(states, masks, high_train_idx),
        "selection_high": make_bank(states, masks, selection_idx),
        "dev_high": make_bank(states, masks, high_dev_idx),
        "dev_full": make_bank(states, masks, dev_idx),
    }
    indices = {
        "train_high": high_train_idx.tolist(),
        "selection_high": selection_idx.tolist(),
        "dev_high": high_dev_idx.tolist(),
        "dev_full": dev_idx.tolist(),
    }
    return banks, indices, {"train_cut": train_cut, "dev_cut": dev_cut}


def load_model(arm):
    graph = build_physics_aligned_dual_graph(arm)
    graph.load_state_dict(torch.load(GRAPH_DIR / f"{arm}_graph_selected.pt", map_location="cpu", weights_only=True)["state"])
    model = CoreGraphLLM(graph)
    source_dir = CLASSIC_SOURCE if arm == "physics_classical_dual" else QUANTUM_SOURCE
    checkpoint = torch.load(source_dir / f"{arm}_llm_selected.pt", map_location="cpu", weights_only=True)
    restore_compact(model, checkpoint["state"])
    return model, checkpoint, source_dir


def train(model, arm, train_bank, selection_bank, output_dir, epochs):
    for parameter in model.graph_backbone.parameters():
        parameter.requires_grad_(False)
    named = dict(model.named_parameters())
    lora = [p for n, p in named.items() if p.requires_grad and "lora_" in n]
    final_coordinate = [p for n, p in named.items() if p.requires_grad and n.startswith("coordinate_head.4.")]
    reserved = {id(p) for p in lora + final_coordinate}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in reserved]
    optimizer = torch.optim.AdamW(
        [
            {"params": heads, "lr": 1e-4},
            {"params": final_coordinate, "lr": 3e-4},
            {"params": lora, "lr": 2.5e-5},
        ],
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    initial, _ = retained.evaluate(model, selection_bank)
    best_score = initial["aggregate"]["ade_m"] + 0.35 * initial["aggregate"]["fde_m"]
    best_epoch = 0
    checkpoint_path = output_dir / f"{arm}_high_llm_selected.pt"
    torch.save({"state": compact_state(model), "epoch": 0, "metrics": initial}, checkpoint_path)
    with (output_dir / f"{arm}_high_llm.jsonl").open("x") as log:
        for epoch in range(1, epochs + 1):
            started = time.time()
            r.set_seed(SEED + 40000 + epoch)
            model.train()
            cuda_generator = torch.Generator(device="cuda").manual_seed(SEED + 40000 + epoch)
            order = torch.randperm(len(train_bank["history"]), generator=torch.Generator().manual_seed(SEED + 40000 + epoch)).cuda()
            coordinate_sum = token_sum = 0.0
            count = 0
            for start in range(0, len(order), 24):
                batch_indices = order[start : start + 24]
                snr = SNRS[(start // 24 + epoch) % len(SNRS)]
                history, future, mask = retained.make_batch(train_bank, batch_indices, snr, cuda_generator)
                optimizer.zero_grad(set_to_none=True)
                output = model(history, mask)
                coordinate = prediction_loss(output, future, mask, True)
                token = token_loss(output["token_logits"], model.future_token_ids(history, future), mask)
                loss = coordinate + 0.005 * token
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0)
                optimizer.step()
                coordinate_sum += float(coordinate.detach()) * len(batch_indices)
                token_sum += float(token.detach()) * len(batch_indices)
                count += len(batch_indices)
            scheduler.step()
            metrics, _ = retained.evaluate(model, selection_bank)
            score = metrics["aggregate"]["ade_m"] + 0.35 * metrics["aggregate"]["fde_m"]
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save({"state": compact_state(model), "epoch": epoch, "metrics": metrics}, checkpoint_path)
            record = {
                "arm": arm,
                "epoch": epoch,
                "coordinate_loss": coordinate_sum / count,
                "token_ce": token_sum / count,
                "selection_high": metrics,
                "score": score,
                "best_epoch": best_epoch,
                "seconds": time.time() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            r.atomic_json(output_dir / "progress.json", {"status": "training", "last": record})
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restore_compact(model, selected["state"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="high_complexity_llm_refine_seed2026_v1")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == "jscn" and torch.cuda.is_available()
    assert "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    retained.NOISE = r.load_snr_noise_map(ROOT / "results/multitarget_snr/snr_calibration.json")
    banks, indices, cuts = prepare_banks()
    source_files = [Path(__file__), BASE / "physics_aligned_dual_model.py", BASE / "model.py"]
    protocol = {
        "seed": SEED,
        "purpose": "Matched refinement on the outcome-blind highest interaction-complexity training tertile.",
        "arms": list(ARMS),
        "epochs": args.epochs,
        "graph_frozen": True,
        "token_weight": 0.005,
        "head_lr": 1e-4,
        "coordinate_final_lr": 3e-4,
        "lora_lr": 2.5e-5,
        "selection_metric": "high-complexity ADE + 0.35*FDE",
        "no_test": True,
        "cuts": cuts,
        "indices": indices,
        "source_hashes": {str(path): sha(path) for path in source_files},
    }
    r.atomic_json(output_dir / "protocol.json", protocol)
    summaries = {}
    for arm in ARMS:
        r.set_seed(SEED)
        model, source_checkpoint, source_dir = load_model(arm)
        model = model.cuda()
        initial, _ = retained.evaluate(model, banks["selection_high"])
        smoke = {
            "arm": arm,
            "source_dir": str(source_dir),
            "source_epoch": int(source_checkpoint["epoch"]),
            "initial_selection_high": initial,
        }
        r.atomic_json(output_dir / f"{arm}_smoke.json", smoke)
        if args.smoke_only:
            summaries[arm] = smoke
            del model
            torch.cuda.empty_cache()
            continue
        selected = train(model, arm, banks["train_high"], banks["selection_high"], output_dir, args.epochs)
        high_metrics, high_arrays = retained.evaluate(model, banks["dev_high"], collect=True)
        full_metrics, full_arrays = retained.evaluate(model, banks["dev_full"], collect=True)
        np.savez_compressed(output_dir / f"{arm}_high_dev.npz", **high_arrays, indices=np.asarray(indices["dev_high"]))
        np.savez_compressed(output_dir / f"{arm}_full_dev.npz", **full_arrays, indices=np.asarray(indices["dev_full"]))
        summaries[arm] = {
            "selected_epoch": int(selected["epoch"]),
            "selection_high": selected["metrics"],
            "high_dev": high_metrics,
            "full_dev": full_metrics,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        r.atomic_json(output_dir / "summary.json", summaries)
        del model
        torch.cuda.empty_cache()
    status = {"status": "smoke_completed" if args.smoke_only else "completed", "summaries": summaries, "no_test": True}
    r.atomic_json(output_dir / "completed.json", status)
    r.atomic_json(output_dir / "progress.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
