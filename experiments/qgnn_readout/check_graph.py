"""CPU structural/gradient checks only; no dataset, optimizer, or ADE/FDE claims."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_inherited.graph import InheritedQGNNGraph
from experiments.qgnn_readout.graph import XReadoutQGNNGraph


def _seed():
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)


def _reference_readouts(angles, weights):
    @qml.qnode(qml.device('default.qubit', wires=6, shots=None, seed=0),
               interface='torch', diff_method='backprop')
    def circuit_state(data, circuit_weights):
        for depth in range(3):
            for q in range(6):
                qml.RY(data[..., q], wires=q)
                qml.RZ(data[..., q + 6], wires=q)
            for q in range(6):
                phi, theta, omega = circuit_weights[depth, q]
                qml.RZ(phi, wires=q)
                qml.RY(theta, wires=q)
                qml.RZ(omega, wires=q)
            for q in range(6):
                qml.CNOT(wires=[q, (q + 1) % 6])
        return qml.state()

    state = circuit_state(angles, weights)
    indices = torch.arange(64)
    signs = torch.stack([1 - 2 * ((indices >> (5 - q)) & 1) for q in range(6)], -1).double()
    z = state.abs().square() @ torch.cat([signs, signs * signs.roll(-1, dims=1)], -1)
    x = torch.stack([(state.conj() * state[:, indices ^ (1 << (5 - q))]).sum(-1).real
                     for q in range(6)], -1)
    return torch.cat([z, x], -1)


def run_checks():
    torch.set_num_threads(1)
    _seed()
    original = InheritedQGNNGraph(seed=2026).cpu().eval()
    old_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    _seed()
    candidate = XReadoutQGNNGraph(seed=2026).cpu().eval()
    assert torch.equal(torch.get_rng_state(), old_rng[0])
    current_numpy = np.random.get_state()
    assert current_numpy[0] == old_rng[1][0]
    assert np.array_equal(current_numpy[1], old_rng[1][1])
    assert current_numpy[2:] == old_rng[1][2:]
    assert random.getstate() == old_rng[2]

    old_parameters = dict(original.named_parameters())
    new_parameters = dict(candidate.named_parameters())
    extra_keys = sorted(set(new_parameters) - set(old_parameters))
    assert extra_keys == [f'graph_layers.{i}.{name}' for i in range(2) for name in ('x_gate', 'x_score')]
    assert set(original.state_dict()).issubset(candidate.state_dict())
    for name, parameter in original.state_dict().items():
        assert torch.equal(parameter, candidate.state_dict()[name]), name
    for name in extra_keys:
        assert new_parameters[name].shape == (4, 6)
        assert torch.count_nonzero(new_parameters[name]) == 0
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in original.parameters())
    assert parameter_delta == 96

    state = torch.randn(1, 3, 4, 4)
    state[..., :2] *= 4
    standardized = state / state.new_tensor([20., 20., 15., 15.])
    exists = torch.tensor([[[True, True, True, False], [False, False, False, False],
                            [True, False, False, False]]])
    detected = exists.clone()
    detected[:, 0, 1] = False
    old_state = state.clone().requires_grad_()
    new_state = state.clone().requires_grad_()
    old_z = standardized.clone().requires_grad_()
    new_z = standardized.clone().requires_grad_()
    old_output = original(old_state, old_z, exists, detected)
    new_output = candidate(new_state, new_z, exists, detected)
    assert new_output.shape == (1, 3, 4, 128)
    assert torch.isfinite(new_output).all()
    assert torch.count_nonzero(new_output[~exists]) == 0
    torch.testing.assert_close(new_output, old_output, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(old_output)
    (old_output * probe).sum().backward()
    (new_output * probe).sum().backward()
    gradient_max_error = 0.0
    for name, parameter in old_parameters.items():
        expected, actual = parameter.grad, new_parameters[name].grad
        assert expected is not None and actual is not None, name
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=name)
        gradient_max_error = max(gradient_max_error, (actual - expected).abs().max().item())
    for old_input, new_input in ((old_state, new_state), (old_z, new_z)):
        torch.testing.assert_close(new_input.grad, old_input.grad, atol=3e-6, rtol=3e-5)
    extra_gradients = {name: new_parameters[name].grad.abs().max().item() for name in extra_keys}
    assert all(value > 1e-8 for value in extra_gradients.values()), extra_gradients

    # Training-mode identity includes identical dropout masks and downstream RNG.
    original.train()
    candidate.train()
    torch.manual_seed(91)
    train_original = original(state, standardized, exists, detected)
    train_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    train_candidate = candidate(state, standardized, exists, detected)
    torch.testing.assert_close(train_candidate, train_original, atol=1e-6, rtol=1e-5)
    assert torch.equal(torch.get_rng_state(), train_rng)
    original.eval()
    candidate.eval()

    with torch.no_grad():
        dirty_state, dirty_z = state.clone(), standardized.clone()
        dirty_state[~exists] = float('nan')
        dirty_z[~exists] = float('nan')
        dirty_detected = detected.float()
        dirty_detected[~exists] = float('nan')
        torch.testing.assert_close(candidate(dirty_state, dirty_z, exists, dirty_detected), new_output)
        perm = torch.tensor([3, 1, 0, 2])
        permuted = candidate(state[..., perm, :], standardized[..., perm, :], exists[..., perm], detected[..., perm])
        torch.testing.assert_close(permuted, new_output[..., perm, :], atol=3e-6, rtol=3e-5)
        short = candidate(state[:, :1, :3], standardized[:, :1, :3], exists[:, :1, :3], detected[:, :1, :3])
        torch.testing.assert_close(short, new_output[:, :1, :3], atol=3e-6, rtol=3e-5)
        empty = candidate(state, standardized, torch.zeros_like(exists), detected)
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0

    # Synthetic proxy loss establishes a usable phase path once increments move.
    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        for layer in candidate.graph_layers:
            layer.x_score.copy_(torch.linspace(-0.15, 0.2, 24).reshape(4, 6))
            layer.x_gate.copy_(torch.linspace(0.2, -0.1, 24).reshape(4, 6))
    (candidate(state, standardized, exists, detected) * probe).sum().backward()
    phase_gradients = [layer.core.weights.grad[-1, :, 2].abs().tolist() for layer in candidate.graph_layers]
    assert all(min(values) > 1e-8 for values in phase_gradients), phase_gradients

    reference_errors = []
    for layer in candidate.graph_layers:
        core = layer.core.double()
        angles = (torch.randn(3, 12, dtype=torch.float64) * 0.7).requires_grad_()
        reference_angles = angles.detach().clone().requires_grad_()
        reference_weights = core.weights.detach().clone().requires_grad_()
        actual = core(angles)
        reference = _reference_readouts(reference_angles, reference_weights)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        assert actual.shape == (3, 18)
        assert actual.detach().abs().max() <= 1 + 1e-12
        core_probe = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad((actual * core_probe).sum(), (angles, core.weights))
        expected_gradients = torch.autograd.grad((reference * core_probe).sum(), (reference_angles, reference_weights))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)
        reference_errors.append((actual - reference).abs().max().item())

    return dict(
        passed=True, device='cpu', parameter_delta=parameter_delta, shared_initialization_exact=True,
        initialization_rng_unchanged=True, training_dropout_rng_unchanged=True,
        zero_increment_output_max_error=(new_output - old_output).abs().max().item(),
        zero_increment_old_parameter_gradient_max_error=gradient_max_error,
        zero_increment_new_parameter_gradient_max=extra_gradients,
        nonzero_increment_terminal_omega_abs_gradients=phase_gradients,
        independent_full18_readout_max_errors=reference_errors,
        checks=['shared parameter names and values', 'zero increment output/input/old-parameter gradients',
                'learnable zero-initialized increments', 'terminal phase visibility with nonzero increments',
                'training dropout identity', 'masks/empty frames/permutation/padding',
                'independent 18-readout state simulation and gradients'],
        graph_sources={name: (ROOT / name).read_text(encoding='utf-8') for name in (
            'experiments/qgnn_readout/graph.py', 'experiments/qgnn_inherited/graph.py',
            'code/02_pennylane_qgnn_core/physics_aligned_dual_model.py',
            'code/00_remote_shared_dependencies/target_interaction_graph.py')},
        optimizer_steps=0, dataset_access=False, prediction_evaluation=False,
        limitation='Structural and synthetic-gradient evidence only; no ADE/FDE or quantum-advantage evidence.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(run_checks(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    print(report)
