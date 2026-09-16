"""Convergence-controlled comparison with PennyLane as the quantum backend."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import experiment_utils as r
import run as retained
from model import CoreGraphLLM, build_graph, compact_state, restore_compact
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss


ROOT = Path("/home/js_cn/sensing")
BASE = Path(__file__).resolve().parent
SEED = 2026
ARMS = ("plain", "classical", "quantum")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_stage(model, arm, phase, max_epochs, patience, train, dev, output_dir, input_hashes):
    is_llm = phase == "llm"
    named = dict(model.named_parameters())
    if is_llm:
        groups = [
            {"params": [p for n, p in named.items() if p.requires_grad and n.startswith("graph_backbone.")], "lr": 1e-4},
            {"params": [p for n, p in named.items() if p.requires_grad and not n.startswith("graph_backbone.") and "lora_" not in n], "lr": 3e-4},
            {"params": [p for n, p in named.items() if p.requires_grad and "lora_" in n], "lr": 7.5e-5},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": 1e-3}]
    optimizer = torch.optim.AdamW(groups, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    state_fn = compact_state if is_llm else graph_state
    checkpoint_path = output_dir / f"{arm}_{phase}_selected.pt"
    log_path = output_dir / f"{arm}_{phase}.jsonl"

    initial_eval, _ = retained.evaluate(model, dev)
    best_score = retained.score(initial_eval)
    best_epoch = 0
    wait = 0
    torch.save({"state": state_fn(model), "epoch": 0, "metrics": initial_eval}, checkpoint_path)
    start_state = {name: value.detach().cpu().clone() for name, value in named.items() if value.requires_grad}
    gradmax = {name: 0.0 for name, value in named.items() if value.requires_grad}
    records = [{"epoch": 0, "metrics": initial_eval, "score": best_score}]
    phase_offset = 20000 if is_llm else 10000
    stop_reason = "max_epochs"

    with log_path.open("x", encoding="utf-8") as log:
        for epoch in range(1, max_epochs + 1):
            started = time.time()
            r.set_seed(SEED + phase_offset + epoch)
            model.train()
            generator = torch.Generator(device="cuda").manual_seed(SEED + phase_offset + epoch)
            order = torch.randperm(
                len(train["history"]),
                generator=torch.Generator().manual_seed(SEED + phase_offset + epoch),
            ).cuda()
            loss_sum = 0.0
            count = 0
            digest = r.InputDigest()
            for start in range(0, len(order), 24):
                indices = order[start : start + 24]
                snr = (5, 10, 15, 20)[(start // 24 + epoch) % 4]
                history, future, mask = retained.make_batch(train, indices, snr, generator)
                digest.update(history)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(history, mask)
                loss = prediction_loss(prediction, future, mask, graph_weighting=True)
                if is_llm:
                    loss = loss + 0.035 * token_loss(
                        prediction["token_logits"], model.future_token_ids(history, future), mask
                    )
                assert torch.isfinite(loss)
                loss.backward()
                for name, value in named.items():
                    if value.grad is not None:
                        assert torch.isfinite(value.grad).all(), name
                        if name in gradmax:
                            gradmax[name] = max(gradmax[name], float(value.grad.abs().max()))
                torch.nn.utils.clip_grad_norm_(
                    [value for value in named.values() if value.requires_grad], 3.0, error_if_nonfinite=True
                )
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
                count += len(indices)

            input_digest = digest.hexdigest()
            key = f"{phase}_{epoch}"
            if key in input_hashes:
                assert input_hashes[key] == input_digest
            else:
                input_hashes[key] = input_digest
            scheduler.step()
            evaluation, _ = retained.evaluate(model, dev)
            current_score = retained.score(evaluation)
            if current_score < best_score - 1e-4:
                best_score = current_score
                best_epoch = epoch
                wait = 0
                torch.save({"state": state_fn(model), "epoch": epoch, "metrics": evaluation}, checkpoint_path)
            else:
                wait += 1
            record = {
                "arm": arm,
                "phase": phase,
                "epoch": epoch,
                "max_epochs": max_epochs,
                "train_loss": loss_sum / count,
                "metrics": evaluation,
                "score": current_score,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "wait": wait,
                "patience": patience,
                "seconds": time.time() - started,
                "input_sha256": input_digest,
            }
            records.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            r.atomic_json(output_dir / "progress.json", {"status": "training", "last": record})
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if wait >= patience:
                stop_reason = "early_stopping"
                break

    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if is_llm:
        restore_compact(model, selected["state"])
    else:
        model.load_state_dict(selected["state"], strict=True)
    recheck, _ = retained.evaluate(model, dev)
    assert all(
        abs(recheck["aggregate"][metric] - selected["metrics"]["aggregate"][metric]) < 1e-7
        for metric in ("ade_m", "fde_m")
    )
    audit = {
        "selected_epoch": best_epoch,
        "executed_epochs": records[-1]["epoch"],
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": 1e-4,
        "stop_reason": stop_reason,
        "metrics": recheck,
        "gradient_max": gradmax,
        "parameter_max_change": {
            name: float((value.detach().cpu() - start_state[name]).abs().max())
            for name, value in named.items()
            if name in start_state
        },
        "never_received_gradient": [name for name, value in gradmax.items() if value == 0],
        "records": records,
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    r.atomic_json(output_dir / f"{arm}_{phase}_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="converged_protocol_seed2026_v1")
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert platform.node() == "jscn"
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    r.set_seed(SEED)
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)

    cache = ROOT / "data/multitarget_lankershim_v1.npz"
    with np.load(BASE / "retained_inputs/circuit_split_indices.npz", allow_pickle=False) as split:
        train_indices = split["circuit_train_indices"].copy()
        dev_indices = split["circuit_dev_indices"].copy()
    with np.load(cache, allow_pickle=False) as data:
        states = data["train_states"].copy()
        masks = data["train_mask"].copy()
    selection_indices = r.select_profile_indices(dev_indices, 256)

    def bank(indices):
        return {
            "history": torch.from_numpy(states[indices, :20]).float().cuda(),
            "future": torch.from_numpy(states[indices, 20:]).float().cuda(),
            "mask": torch.from_numpy(masks[indices]).bool().cuda(),
        }

    train_bank = bank(train_indices)
    selection_bank = bank(selection_indices)
    full_dev_bank = bank(dev_indices)
    calibration = ROOT / "results/multitarget_snr/snr_calibration.json"
    retained.NOISE = r.load_snr_noise_map(calibration)

    source_paths = [
        BASE / "model.py",
        BASE / "run_converged_training.py",
        BASE / "pennylane_core.py",
        BASE / "quantum_core.py",
        ROOT / "target_interaction_graph.py",
        ROOT / "MultiTargetTimeLLM.py",
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    r.atomic_json(
        output_dir / "protocol.json",
        {
            "seed": SEED,
            "arms": list(ARMS),
            "quantum_backend": "PennyLane default.qubit analytic statevector",
            "graph": {"max_epochs": 40, "patience": 6},
            "llm": {"max_epochs": 20, "patience": 4},
            "min_delta": 1e-4,
            "batch_size": 24,
            "selection": "four-SNR macro ADE + 0.35 FDE",
            "train_indices": train_indices.tolist(),
            "selection_indices": selection_indices.tolist(),
            "full_dev_indices": dev_indices.tolist(),
            "same_input_hashes_across_arms_required": True,
            "no_test": True,
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.executable,
            "source_hashes": source_hashes,
        },
    )

    input_hashes = {}
    summaries = {}
    common_graph_hash = None
    common_llm_hash = None
    for arm in ARMS:
        r.set_seed(SEED)
        backend = "pennylane" if arm == "quantum" else "torch"
        graph = build_graph(arm, quantum_backend=backend).cuda()
        shared_hash = r.tensor_mapping_sha256(
            {name: value.detach().cpu() for name, value in graph.named_parameters() if not name.startswith("graph_layers.1.")}
        )
        if common_graph_hash is None:
            common_graph_hash = shared_hash
        assert shared_hash == common_graph_hash
        graph_audit = train_stage(
            graph, arm, "graph", 40, 6, train_bank, selection_bank, output_dir, input_hashes
        )
        graph_full, arrays = retained.evaluate(graph, full_dev_bank, collect=True)
        np.savez_compressed(output_dir / f"{arm}_graph_full_dev.npz", **arrays, indices=dev_indices)

        r.set_seed(SEED)
        model = CoreGraphLLM(graph).cuda()
        llm_hash = r.tensor_mapping_sha256(
            {name: value.detach().cpu() for name, value in model.named_parameters() if not name.startswith("graph_backbone.")}
        )
        if common_llm_hash is None:
            common_llm_hash = llm_hash
        assert llm_hash == common_llm_hash
        frozen_hash = r.tensor_mapping_sha256(
            {name: value.detach().cpu() for name, value in model.named_parameters() if not value.requires_grad}
        )
        llm_audit = train_stage(
            model, arm, "llm", 20, 4, train_bank, selection_bank, output_dir, input_hashes
        )
        assert frozen_hash == r.tensor_mapping_sha256(
            {name: value.detach().cpu() for name, value in model.named_parameters() if not value.requires_grad}
        )
        llm_full, arrays = retained.evaluate(model, full_dev_bank, collect=True)
        np.savez_compressed(output_dir / f"{arm}_llm_full_dev.npz", **arrays, indices=dev_indices)
        summaries[arm] = {
            "graph": graph_full,
            "llm": llm_full,
            "graph_selected_epoch": graph_audit["selected_epoch"],
            "graph_executed_epochs": graph_audit["executed_epochs"],
            "graph_stop_reason": graph_audit["stop_reason"],
            "llm_selected_epoch": llm_audit["selected_epoch"],
            "llm_executed_epochs": llm_audit["executed_epochs"],
            "llm_stop_reason": llm_audit["stop_reason"],
        }
        r.atomic_json(output_dir / "summary.json", summaries)
        del model, graph
        torch.cuda.empty_cache()

    for path, digest in source_hashes.items():
        assert sha256(path) == digest
    completed = {
        "status": "completed",
        "seed": SEED,
        "summaries": summaries,
        "source_hashes_verified": True,
        "same_input_hashes_across_arms_verified": True,
        "no_test": True,
    }
    r.atomic_json(output_dir / "completed.json", completed)
    r.atomic_json(output_dir / "progress.json", {"status": "completed", "summaries": summaries})
    print(json.dumps(completed, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
