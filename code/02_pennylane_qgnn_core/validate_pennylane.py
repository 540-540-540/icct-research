"""Validate the PennyLane circuit against the retained PyTorch statevector core."""
import argparse
import json
from pathlib import Path

import torch

from model import build_graph
from pennylane_core import PennyLaneStatevectorPQC
from quantum_core import BatchedStatevectorPQC, QuantumCircuitConfig


BASE = Path(__file__).resolve().parent


def max_abs(left, right):
    return float((left - right).detach().abs().max().cpu())


def circuit_parity(dtype=torch.float64):
    torch.manual_seed(2026)
    config = QuantumCircuitConfig()
    reference = BatchedStatevectorPQC(config)
    candidate = PennyLaneStatevectorPQC(config)
    reference.weights.data = reference.weights.data.to(dtype=dtype)
    candidate.weights.data = candidate.weights.data.to(dtype=dtype)
    with torch.no_grad():
        candidate.weights.copy_(reference.weights)

    angles_ref = torch.randn(7, config.n_qubits, dtype=dtype, requires_grad=True)
    angles_pl = angles_ref.detach().clone().requires_grad_(True)
    probe = torch.randn(7, 2 * config.n_qubits, dtype=dtype)

    output_ref = reference(angles_ref)
    output_pl = candidate(angles_pl)
    loss_ref = (output_ref * probe).sum()
    loss_pl = (output_pl * probe).sum()
    grad_angles_ref, grad_weights_ref = torch.autograd.grad(
        loss_ref, (angles_ref, reference.weights)
    )
    grad_angles_pl, grad_weights_pl = torch.autograd.grad(
        loss_pl, (angles_pl, candidate.weights)
    )
    return {
        "dtype": str(dtype),
        "batch_size": 7,
        "output_max_abs_error": max_abs(output_ref, output_pl),
        "input_gradient_max_abs_error": max_abs(grad_angles_ref, grad_angles_pl),
        "weight_gradient_max_abs_error": max_abs(grad_weights_ref, grad_weights_pl),
    }


def checkpoint_parity():
    checkpoint_path = BASE / "development" / "quantum_graph_selected.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state"]

    reference = build_graph("quantum")
    candidate = build_graph("quantum", quantum_backend="pennylane")
    reference.load_state_dict(state, strict=True)
    candidate.load_state_dict(state, strict=True)
    reference.eval()
    candidate.eval()

    generator = torch.Generator().manual_seed(20260906)
    history = torch.randn(2, reference.config.history_length, 3, 4, generator=generator)
    # Keep positions in a compact neighborhood so the same nontrivial graph edges are active.
    history[..., :2] *= 0.2
    target_mask = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        output_ref = reference(history, target_mask)
        output_pl = candidate(history, target_mask)
    fields = ("future_position", "node_features", "attention", "adjacency")
    errors = {}
    for field in fields:
        if output_ref[field].dtype == torch.bool:
            errors[field] = int((output_ref[field] != output_pl[field]).sum())
        else:
            errors[field] = max_abs(output_ref[field], output_pl[field])
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_shape": list(history.shape),
        "field_errors": errors,
    }


def gpu_forward_backward_smoke():
    if not torch.cuda.is_available():
        return {"available": False, "passed": False}
    checkpoint = torch.load(
        BASE / "development" / "quantum_graph_selected.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = build_graph("quantum", quantum_backend="pennylane")
    model.load_state_dict(checkpoint["state"], strict=True)
    # cuDNN RNNs require training mode for backward; this is only a finite-gradient smoke test.
    model.cuda().train()
    generator = torch.Generator().manual_seed(20260906)
    history = torch.randn(1, model.config.history_length, 2, 4, generator=generator).cuda()
    history[..., :2] *= 0.2
    mask = torch.ones(1, 2, dtype=torch.bool, device="cuda")
    output = model(history, mask)["future_position"]
    loss = output.square().mean()
    loss.backward()
    gradient = model.graph_layers[1].core.weights.grad
    passed = bool(torch.isfinite(output).all() and gradient is not None and torch.isfinite(gradient).all())
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "output_shape": list(output.shape),
        "loss": float(loss.detach().cpu()),
        "core_gradient_max_abs": float(gradient.detach().abs().max().cpu()),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "pennylane_validation" / "parity.json")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    report = {
        "pennylane": __import__("pennylane").__version__,
        "torch": torch.__version__,
        "circuit": circuit_parity(),
        "checkpoint": checkpoint_parity(),
        "gpu_forward_backward": gpu_forward_backward_smoke(),
        "tolerance": args.tolerance,
    }
    numeric_errors = [
        report["circuit"]["output_max_abs_error"],
        report["circuit"]["input_gradient_max_abs_error"],
        report["circuit"]["weight_gradient_max_abs_error"],
        *[
            value
            for name, value in report["checkpoint"]["field_errors"].items()
            if name != "adjacency"
        ],
    ]
    report["passed"] = (
        max(numeric_errors) <= args.tolerance
        and report["checkpoint"]["field_errors"]["adjacency"] == 0
        and report["gpu_forward_backward"]["passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("PennyLane parity validation failed")


if __name__ == "__main__":
    main()
