"""CPU-only check of terminal RZ visibility; never trains or reads a dataset."""
import argparse
import json
import sys
from pathlib import Path

import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgnn_inherited.graph import InheritedQGNNGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'results/qgnn_inherited100/seed2026/qgnn/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/qgnn_inherited100/readout_observability.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20814)
    # Only load a trusted checkpoint produced by this project.
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    parameters = payload['model']
    angles = torch.randn(4, 12, dtype=torch.float64) * 0.7
    graph = InheritedQGNNGraph().double()

    @qml.qnode(qml.device('default.qubit', wires=6, shots=None), interface='torch', diff_method='backprop')
    def circuit_state(data, weights):
        for layer in range(3):
            for q in range(6):
                qml.RY(data[..., q], wires=q)
                qml.RZ(data[..., q + 6], wires=q)
            for q in range(6):
                phi, theta, omega = weights[layer, q]
                qml.RZ(phi, wires=q)
                qml.RY(theta, wires=q)
                qml.RZ(omega, wires=q)
            for q in range(6):
                qml.CNOT(wires=[q, (q + 1) % 6])
        return qml.state()

    indices = torch.arange(64)
    signs = torch.stack([1 - 2 * ((indices >> (5 - q)) & 1) for q in range(6)], -1).double()
    z_signs = torch.cat([signs, signs * signs.roll(-1, dims=1)], -1)

    def readouts(state):
        old = state.abs().square() @ z_signs
        extra = torch.stack([
            (state.conj() * state[:, indices ^ (1 << (5 - q))]).sum(-1).real
            for q in range(6)
        ], -1)
        return torch.cat([old, extra], -1)

    results = []
    for i, layer in enumerate(graph.graph_layers):
        weights = parameters[f'graph.graph_layers.{i}.core.weights'].detach().double()
        with torch.no_grad():
            layer.core.weights.copy_(weights)
        observed = readouts(circuit_state(angles, weights))
        reference = layer.core(angles)
        reconstruction_error = (observed[:, :12] - reference).abs().max().item()

        def terminal_readouts(omega):
            changed = weights.clone()
            changed[-1, :, 2] = omega
            return readouts(circuit_state(angles, changed))

        omega = weights[-1, :, 2].clone().requires_grad_()
        jacobian = torch.autograd.functional.jacobian(terminal_readouts, omega)
        old_derivatives = jacobian[:, :12].abs().amax(dim=(0, 1))
        x_derivatives = jacobian[:, 12:].abs().amax(dim=(0, 1))
        shifted = terminal_readouts(omega.detach() + 0.43)
        old_shift = (shifted[:, :12] - observed[:, :12]).abs().max().item()
        x_shift = (shifted[:, 12:] - observed[:, 12:]).abs().max().item()

        # Preserve old normalization and inject raw, bounded X features only
        # through a new zero-initialized projection (two four-head outputs).
        z_latent = layer.latent_norm(observed[:, :12])
        original_head = torch.cat([layer.score(z_latent), layer.gate(z_latent)], -1)
        extra_head = torch.zeros(8, 6, dtype=torch.float64, requires_grad=True)
        extended_head = original_head + observed[:, 12:] @ extra_head.T
        identity_error = (extended_head - original_head).abs().max().item()
        derivative = torch.autograd.grad((extended_head * torch.randn_like(extended_head)).sum(), extra_head)[0]

        assert reconstruction_error < 1e-12, reconstruction_error
        assert old_derivatives.max().item() < 1e-12
        assert old_shift < 1e-12
        assert torch.all(x_derivatives > 1e-8), x_derivatives
        assert x_shift > 1e-6
        assert identity_error == 0.0
        assert derivative.abs().max().item() > 1e-8
        results.append(dict(
            layer=i, reconstruction_max_error=reconstruction_error,
            old_terminal_omega_derivatives=old_derivatives.tolist(),
            x_terminal_omega_derivatives=x_derivatives.tolist(),
            old_readout_phase_shift_max=old_shift, x_readout_phase_shift_max=x_shift,
            zero_extension_head_error=identity_error,
            zero_extension_head_gradient_max=derivative.abs().max().item(),
        ))

    report = dict(
        status='passed', checkpoint=str(args.checkpoint), diagnostic_seed=20814,
        checkpoint_epoch=payload.get('progress', {}).get('epoch'),
        scope='CPU; four synthetic relation vectors; trained graph weights; float64 gate comparison.',
        optimizer_steps=0, dataset_access=False, prediction_evaluation=False,
        architecture_changes_applied=False, layers=results,
        proposed_extra_parameters=2 * 8 * 6,
        limitation='Observability and zero-extension checks only; no ADE/FDE or quantum-advantage evidence.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
