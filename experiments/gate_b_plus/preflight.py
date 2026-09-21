from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gate_a.data import GateADataset
from experiments.gate_a.run_pilot import batch_loss
from experiments.gate_b_plus.models import GateBPlusQuantumModel


def main() -> None:
    cfg = json.loads((ROOT / "configs/gate_b_plus_screen.json").read_text())
    dataset = GateADataset(ROOT / cfg["benchmark"], "train", limit=2, seed=2026)
    batch = {key: torch.stack([dataset[i][key] for i in range(2)]).cuda() for key in dataset[0]}
    checks = {"test_artifact_absent": not (ROOT / cfg["benchmark"] / "test.npz").exists(),
              "future_not_model_input": "future" not in GateBPlusQuantumModel.forward.__code__.co_varnames}
    profiles = {}
    for mode in cfg["modes"]:
        torch.manual_seed(2026)
        model = GateBPlusQuantumModel(mode, dt_s=cfg["dt_s"], rounds=cfg["rounds"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["history_state"], batch["node_mask"])
            loss = batch_loss(prediction, batch); loss.backward(); optimizer.step()
        core_grad = torch.sqrt(sum(parameter.grad.square().sum() for parameter in model.core.parameters()
                                   if parameter.grad is not None))
        checks[f"{mode}_shape"] = tuple(prediction.shape) == (2, 40, 2)
        checks[f"{mode}_finite"] = bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
        checks[f"{mode}_quantum_core_gradient"] = float(core_grad) > 1e-8
        profiles[mode] = {"parameters": model.parameter_audit(), "quantum_core_gradient_l2": float(core_grad)}
    torch.manual_seed(2026)
    johnson = GateBPlusQuantumModel("full_multiscale", dt_s=cfg["dt_s"], rounds=cfg["rounds"],
                                    core_kind="johnson").cuda()
    optimizer = torch.optim.AdamW(johnson.parameters(), lr=cfg["training"]["learning_rate"])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = johnson(batch["history_state"], batch["node_mask"])
        loss = batch_loss(prediction, batch); loss.backward(); optimizer.step()
    core_grad = torch.sqrt(sum(parameter.grad.square().sum() for parameter in johnson.core.parameters()
                               if parameter.grad is not None))
    checks["johnson_full_multiscale_shape"] = tuple(prediction.shape) == (2, 40, 2)
    checks["johnson_full_multiscale_finite"] = bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
    checks["johnson_full_multiscale_core_gradient"] = float(core_grad) > 1e-8
    profiles["johnson_full_multiscale"] = {"parameters": johnson.parameter_audit(),
                                            "core_gradient_l2": float(core_grad)}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "test_accessed": False,
              "checks": checks, "profiles": profiles}
    out = ROOT / "reports/task_redesign/SIND_GATE_B_PLUS_SCREEN_PREFLIGHT_20260921.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
