"""CPU structure and gradient checks; no dataset, optimizer, or ADE/FDE."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_directed_context.graph import DirectedContextQGNNGraph
from experiments.qgnn_inherited.graph import InheritedQGNNGraph


def _seed():
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)


def _rng_state():
    return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()


def _assert_rng_equal(expected):
    actual = _rng_state()
    assert torch.equal(actual[0], expected[0])
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert actual[2] == expected[2]


def run_checks():
    torch.set_num_threads(1)
    _seed()
    original = InheritedQGNNGraph(seed=2026).cpu().eval()
    expected_rng = _rng_state()
    _seed()
    candidate = DirectedContextQGNNGraph(seed=2026).cpu().eval()
    _assert_rng_equal(expected_rng)

    old_parameters = dict(original.named_parameters())
    new_parameters = dict(candidate.named_parameters())
    extra_keys = sorted(set(new_parameters) - set(old_parameters))
    expected_extra = [f'graph_layers.{layer}.{name}' for layer in range(2)
                      for name in ('receiver_context', 'sender_context')]
    assert extra_keys == expected_extra
    assert set(original.state_dict()).issubset(candidate.state_dict())
    for name, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    for name in extra_keys:
        assert new_parameters[name].shape == (5, 128)
        assert torch.count_nonzero(new_parameters[name]) == 0
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in original.parameters())
    assert parameter_delta == 2560

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
    torch.testing.assert_close(new_output, old_output, atol=1e-6, rtol=1e-5)
    assert new_output.shape == (1, 3, 4, 128)
    assert torch.isfinite(new_output).all() and torch.count_nonzero(new_output[~exists]) == 0

    probe = torch.randn_like(old_output)
    (old_output * probe).sum().backward()
    (new_output * probe).sum().backward()
    shared_gradient_error = 0.
    for name, parameter in old_parameters.items():
        expected, actual = parameter.grad, new_parameters[name].grad
        assert expected is not None and actual is not None, name
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=name)
        shared_gradient_error = max(shared_gradient_error, float((actual - expected).abs().max()))
    for old_input, new_input in ((old_state, new_state), (old_z, new_z)):
        torch.testing.assert_close(new_input.grad, old_input.grad, atol=3e-6, rtol=3e-5)
    extra_gradients = {name: dict(norm=float(new_parameters[name].grad.norm()),
                                  rms=float(new_parameters[name].grad.square().mean().sqrt()))
                       for name in extra_keys}
    assert all(np.isfinite(row['norm']) and row['norm'] > 1e-8 for row in extra_gradients.values())

    original.train()
    candidate.train()
    torch.manual_seed(91)
    train_original = original(state, standardized, exists, detected)
    expected_dropout_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    train_candidate = candidate(state, standardized, exists, detected)
    torch.testing.assert_close(train_candidate, train_original, atol=1e-6, rtol=1e-5)
    assert torch.equal(torch.get_rng_state(), expected_dropout_rng)
    original.eval()
    candidate.eval()

    with torch.no_grad():
        layer, old_layer = candidate.graph_layers[0], original.graph_layers[0]
        nodes = torch.randn(2, 4, 128)
        edges = torch.zeros(2, 4, 4, 7)
        neutral = layer.relation_features(nodes, edges)
        torch.testing.assert_close(neutral, old_layer.relation_features(nodes, edges), atol=0, rtol=0)
        layer.receiver_context.copy_(torch.linspace(-.04, .05, 640).reshape(5, 128))
        layer.sender_context.copy_(torch.linspace(.03, -.02, 640).reshape(5, 128))
        directed = layer.relation_features(nodes, edges)
        diagonal = torch.arange(4)
        torch.testing.assert_close(directed[:, diagonal, diagonal], neutral[:, diagonal, diagonal], atol=0, rtol=0)
        asymmetry = float((directed[:, :, :, 7:] - directed.transpose(1, 2)[:, :, :, 7:]).abs().max())
        assert asymmetry > 1e-6

        dirty_state, dirty_z = state.clone(), standardized.clone()
        dirty_state[~exists] = float('nan')
        dirty_z[~exists] = float('nan')
        dirty_detected = detected.float()
        dirty_detected[~exists] = float('nan')
        dirty_output = candidate(dirty_state, dirty_z, exists, dirty_detected)
        assert torch.isfinite(dirty_output).all()
        perm = torch.tensor([3, 1, 0, 2])
        permuted = candidate(state[..., perm, :], standardized[..., perm, :], exists[..., perm], detected[..., perm])
        torch.testing.assert_close(permuted, candidate(state, standardized, exists, detected)[..., perm, :], atol=3e-6, rtol=3e-5)
        empty = candidate(state, standardized, torch.zeros_like(exists), detected)
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0

    sources = ('experiments/qgnn_directed_context/graph.py',
               'experiments/qgnn_directed_context/check_graph.py',
               'experiments/qgnn_inherited/graph.py',
               'code/02_pennylane_qgnn_core/physics_aligned_dual_model.py',
               'code/00_remote_shared_dependencies/target_interaction_graph.py')
    return dict(
        passed=True, device='cpu', parameter_delta=parameter_delta,
        shared_initialization_exact=True, initialization_rng_unchanged=True,
        training_dropout_rng_unchanged=True,
        zero_residual_output_max_error=float((new_output - old_output).detach().abs().max()),
        zero_residual_shared_gradient_max_error=shared_gradient_error,
        zero_residual_new_parameter_gradients=extra_gradients,
        directed_context_breaks_swap_symmetry=asymmetry,
        self_edges_unchanged_with_nonzero_residual=True,
        checks=['shared parameter names and values', 'zero-residual output/input/shared gradients',
                'new-parameter gradient reachability', 'training dropout identity',
                'directed off-diagonal context and unchanged self edges',
                'masks, empty frames, permutation, and dirty padding'],
        graph_sources={name: (ROOT/name).read_text(encoding='utf-8') for name in sources},
        optimizer_steps=0, dataset_access=False, prediction_evaluation=False,
        limitation='Structural and synthetic-gradient evidence only; no ADE/FDE evidence.',
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
