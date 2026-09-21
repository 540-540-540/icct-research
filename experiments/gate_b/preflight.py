from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.models import GateAModel
from experiments.gate_a.run_pilot import batch_loss
from experiments.gate_b.models import GateBRajModel


def digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_b_raj_migration.json")
    parser.add_argument("--output", default="reports/task_redesign/SIND_GATE_B_READINESS_20260921.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    dataset = GateADataset(ROOT / cfg["benchmark"], "train", limit=2, seed=2026)
    batch = {k: torch.stack([dataset[i][k] for i in range(2)]) for k in dataset[0]}
    history = batch["history_state"][:, :, :8].contiguous()
    mask = batch["node_mask"][:, :8].contiguous()
    checks: dict[str, bool] = {
        "test_not_constructed": True,
        "target_slot_zero_active": bool(mask[:, 0].all()),
        "input_is_20_by_8_by_4": tuple(history.shape[1:]) == (20, 8, 4),
        "future_is_label_only": "future" not in GateBRajModel.forward.__code__.co_varnames,
    }
    parameters, shared = {}, {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kind in cfg["models"]:
        torch.manual_seed(2026)
        model = GateBRajModel(kind, dt_s=cfg["dt_s"], rounds=cfg["rounds"]).to(device)
        model.eval()
        h, m = history.to(device), mask.to(device)
        with torch.no_grad():
            prediction = model(h, m)
            altered = h.clone()
            for row in range(len(altered)):
                altered[row, :, ~m[row]] = 1e4
            padded_prediction = model(altered, m)
        checks[f"{kind}_shape"] = tuple(prediction.shape) == (2, 40, 2)
        checks[f"{kind}_finite"] = bool(torch.isfinite(prediction).all())
        checks[f"{kind}_padded_invariant"] = bool(torch.allclose(prediction, padded_prediction, atol=1e-6, rtol=1e-6))
        model.train()
        pred = model(h, m)
        loss_batch = {k: v.to(device) for k, v in batch.items()}
        loss = batch_loss(pred, loss_batch)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        checks[f"{kind}_loss_and_gradients_finite"] = bool(torch.isfinite(loss) and gradients and all(torch.isfinite(g).all() for g in gradients))
        parameters[kind] = model.parameter_audit()
        shared[kind] = {
            "encoder": digest(model.encoder), "interaction_projection": digest(model.interaction),
            "time_embedding": digest(model.time), "decoder": digest(model.decoder),
        }
    quantum, classical = cfg["models"]
    checks["matched_shared_initialization"] = shared[quantum] == shared[classical]
    checks["same_noncore_parameter_count"] = all(
        parameters[quantum][key] == parameters[classical][key]
        for key in ("temporal_encoder", "interaction_projection", "time_embedding", "decoder")
    )
    all_graph = GateAModel("all_graph", dataset.arrays["history_state"].shape[2], dt_s=cfg["dt_s"])
    checks["all_neighbor_reference_accepts_full_context"] = all_graph.max_nodes > 8
    payload = {
        "revision": cfg["revision"], "status": "PASS" if all(checks.values()) else "FAIL",
        "test_accessed": False, "device": str(device), "checks": checks,
        "parameters": parameters, "shared_initialization_sha256": shared,
        "all_neighbor_reference_parameters": sum(p.numel() for p in all_graph.parameters()),
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
