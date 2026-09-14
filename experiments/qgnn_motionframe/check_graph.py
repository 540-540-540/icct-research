"""CPU synthetic checks; no dataset, optimizer, GPU, or prediction evaluation."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_inherited.graph import InheritedQGNNGraph, _ReplicaSafePQC, _base
from experiments.qgnn_motionframe.graph import MotionFrameQGNNGraph, receiver_motion_edges


def _seed():
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)


def _equivalence(original, candidate, state, exists, detected, check_input_gradients):
    original.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    old_x, new_x = [state.clone().requires_grad_() for _ in range(2)]
    z = state / state.new_tensor([20., 20., 15., 15.])
    old_z, new_z = [z.clone().requires_grad_() for _ in range(2)]
    expected = original(old_x, old_z, exists, detected)
    actual = candidate(new_x, new_z, exists, detected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    errors = []
    for (name, old), (new_name, new) in zip(original.named_parameters(), candidate.named_parameters()):
        assert name == new_name
        assert old.grad is not None and new.grad is not None, name
        assert torch.isfinite(new.grad).all(), name
        torch.testing.assert_close(new.grad, old.grad, atol=3e-6, rtol=3e-5, msg=name)
        errors.append((new.grad - old.grad).abs().max().item())
    for x in (new_x, new_z):
        assert x.grad is not None and torch.isfinite(x.grad).all()
    if check_input_gradients:
        for old, new in ((old_x, new_x), (old_z, new_z)):
            torch.testing.assert_close(new.grad, old.grad, atol=3e-6, rtol=3e-5)
    return dict(output_max_error=(actual-expected).abs().max().item(),
                old_parameter_gradient_max_error=max(errors),
                input_gradients_matched=check_input_gradients)


def run_checks():
    torch.set_num_threads(1)
    _seed()
    original = InheritedQGNNGraph(seed=2026).cpu().eval()
    old_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    _seed()
    candidate = MotionFrameQGNNGraph(seed=2026).cpu().eval()
    assert torch.equal(torch.get_rng_state(), old_rng[0])
    rng = np.random.get_state()
    assert rng[0] == old_rng[1][0] and np.array_equal(rng[1], old_rng[1][1])
    assert rng[2:] == old_rng[1][2:] and random.getstate() == old_rng[2]
    assert list(original.state_dict()) == list(candidate.state_dict())
    for name, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    candidate.load_state_dict(original.state_dict(), strict=True)
    count = sum(p.numel() for p in candidate.parameters())
    quantum_count = sum(layer.core.weights.numel() for layer in candidate.graph_layers)
    assert count == 219278 and quantum_count == 108
    for layer in candidate.graph_layers:
        assert type(layer.core) is _ReplicaSafePQC
        assert layer.core.n_qubits == 6 and layer.core.depth == 3
        assert layer.heads == 4 and layer.head_dim == 32
        assert layer.latent_norm.normalized_shape == (12,)
        first, second = [layer.core._replicate_for_data_parallel() for _ in range(2)]
        assert first._circuit is not second._circuit and first._device is not second._device

    state = torch.randn(1, 3, 4, 4)
    state[..., :2] *= 4
    state[..., 2] = 0
    state[..., 3] = 2
    exists = torch.tensor([[[True, True, True, False], [False, False, False, False],
                            [True, True, False, False]]])
    detected = exists.clone()
    detected[:, 0, 1] = False
    forward_y = _equivalence(original, candidate, state, exists, detected, False)
    slow = state.clone()
    slow[..., 2:] = torch.tensor([[0., 0.], [.3, .4], [-.4, .2], [.1, -.2]])
    fallback = _equivalence(original, candidate, slow, exists, detected, True)

    # The unrotated condition preserves training dropout draws as well.
    z = state / state.new_tensor([20., 20., 15., 15.])
    original.train()
    candidate.train()
    torch.manual_seed(91)
    train_expected = original(state, z, exists, detected)
    dropout_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    train_actual = candidate(state, z, exists, detected)
    torch.testing.assert_close(train_actual, train_expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(torch.get_rng_state(), dropout_rng)
    original.eval()
    candidate.eval()

    # Hand-computed directed pair: row i is the receiver, column j the sender.
    pos = torch.tensor([[[0., 0.], [3., 4.]]], dtype=torch.float64)
    velocity = torch.tensor([[[0., 2.], [3., 0.]]], dtype=torch.float64)
    edges, _ = _base.build_edge_features(pos, velocity)
    motion = receiver_motion_edges(edges, velocity)
    torch.testing.assert_close(motion[0, 0, 1, :4], torch.tensor([.1, 4/30, .2, -2/15], dtype=torch.float64))
    torch.testing.assert_close(motion[0, 1, 0, :4], torch.tensor([4/30, -.1, -2/15, -.2], dtype=torch.float64))
    assert torch.equal(motion[..., 4:], edges[..., 4:])
    backwards = velocity.new_tensor([0., -2.]).expand_as(velocity)
    reversed_edges = receiver_motion_edges(edges, backwards)
    torch.testing.assert_close(reversed_edges[..., :4], -edges[..., :4], atol=0, rtol=0)
    assert torch.equal(reversed_edges[..., 4:], edges[..., 4:])
    # Exactly 1 m/s selects heading; immediately below it selects global axes.
    threshold_velocity = velocity.new_tensor([[[1., 0.], [.999, 0.]]])
    threshold_edges = receiver_motion_edges(edges, threshold_velocity)
    torch.testing.assert_close(threshold_edges[0, 0, 1, :2], edges.new_tensor([-4/30, .1]))
    torch.testing.assert_close(threshold_edges[0, 1, 0], edges[0, 1, 0], atol=0, rtol=0)

    # Only the physical quantum encoding is invariant, not the entire model.
    rotation_errors = []
    for angle in (.37, -1.2, math.pi):
        c, s = math.cos(angle), math.sin(angle)
        rotation = edges.new_tensor([[c, -s], [s, c]])
        rotated_pos = pos @ rotation.T + pos.new_tensor([17., -9.])
        rotated_velocity = velocity @ rotation.T
        global_edges, _ = _base.build_edge_features(rotated_pos, rotated_velocity)
        aligned = receiver_motion_edges(global_edges, rotated_velocity)
        torch.testing.assert_close(aligned, motion, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(torch.tanh(aligned), torch.tanh(motion), atol=1e-12, rtol=1e-12)
        rotation_errors.append((aligned-motion).abs().max().item())

    state[..., 2:] = torch.tensor([[2., 0.], [0., -3.], [2., 3.], [.2, .3]])
    z = state / state.new_tensor([20., 20., 15., 15.])
    observed, handles = {}, []
    for i, layer in enumerate(candidate.graph_layers):
        def layer_input(module, args, index=i):
            observed[index, 'layer'] = tuple(x.detach().clone() for x in args)
            observed[index, 'quantum_edge_pointer'] = args[3].data_ptr()
        def value_input(module, args, index=i):
            observed[index, 'value_edges'] = args[0].detach().clone()
        def core_input(module, args, index=i):
            observed[index, 'angles'] = args[0].detach().clone()
        handles.extend([layer.register_forward_pre_hook(layer_input),
                        layer.edge_value.register_forward_pre_hook(value_input),
                        layer.core.register_forward_pre_hook(core_input)])
    with torch.no_grad():
        output = candidate(state, z, exists, detected)
    for handle in handles:
        handle.remove()
    assert output.shape == (1, 3, 4, 128) and torch.isfinite(output).all()
    assert torch.count_nonzero(output[~exists]) == 0
    clean = torch.where(exists[..., None], state, 0).reshape(-1, 4, 4)
    active = exists.reshape(-1, 4).any(-1).nonzero().squeeze(-1)
    physical = clean[active]
    raw, _ = _base.build_edge_features(physical[..., :2], physical[..., 2:])
    expected_quantum = receiver_motion_edges(raw, physical[..., 2:])
    assert observed[0, 'quantum_edge_pointer'] == observed[1, 'quantum_edge_pointer']
    for i, layer in enumerate(candidate.graph_layers):
        nodes, raw_argument, adjacency, quantum_argument = observed[i, 'layer']
        torch.testing.assert_close(raw_argument, raw, atol=0, rtol=0)
        torch.testing.assert_close(observed[i, 'value_edges'], raw, atol=0, rtol=0)
        torch.testing.assert_close(quantum_argument, expected_quantum, atol=0, rtol=0)
        relations = layer.relation_features(nodes, expected_quantum)
        old_relations = original.graph_layers[i].relation_features(nodes, raw)
        torch.testing.assert_close(relations[..., 7:], old_relations[..., 7:], atol=0, rtol=0)
        selected = adjacency.reshape(-1).nonzero().squeeze(-1)
        angles = layer.encode_angles(relations.reshape(-1, 12)[selected])
        torch.testing.assert_close(observed[i, 'angles'], angles, atol=0, rtol=0)

    with torch.no_grad():
        dirty_state, dirty_z = state.clone(), z.clone()
        dirty_state[~exists] = float('nan')
        dirty_z[~exists] = float('nan')
        dirty_detected = detected.float()
        dirty_detected[~exists] = float('nan')
        torch.testing.assert_close(candidate(dirty_state, dirty_z, exists, dirty_detected), output)
        perm = torch.tensor([3, 1, 0, 2])
        permuted = candidate(state[..., perm, :], z[..., perm, :], exists[..., perm], detected[..., perm])
        torch.testing.assert_close(permuted, output[..., perm, :], atol=3e-6, rtol=3e-5)
        short = candidate(state[:, :1, :3], z[:, :1, :3], exists[:, :1, :3], detected[:, :1, :3])
        torch.testing.assert_close(short, output[:, :1, :3], atol=3e-6, rtol=3e-5)
        single_frame = candidate(state[:, 2:3], z[:, 2:3], exists[:, 2:3], detected[:, 2:3])
        torch.testing.assert_close(single_frame, output[:, 2:3], atol=3e-6, rtol=3e-5)
        empty = candidate(state, z, torch.zeros_like(exists), detected)
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0
        for field in ('state', 'standardized', 'detected'):
            bad_x, bad_z, bad_d = state.clone(), z.clone(), detected.float()
            {'state': bad_x, 'standardized': bad_z, 'detected': bad_d}[field][0, 0, 0] = float('nan')
            try:
                candidate(bad_x, bad_z, exists, bad_d)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Live nonfinite {field} was admitted')

    candidate.zero_grad(set_to_none=True)
    x, standardized = state.clone().requires_grad_(), z.clone().requires_grad_()
    (candidate(x, standardized, exists, detected) * torch.randn_like(output)).sum().backward()
    for name, parameter in candidate.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert torch.isfinite(x.grad).all() and torch.isfinite(standardized.grad).all()
    core_gradients = [layer.core.weights.grad.abs().max().item() for layer in candidate.graph_layers]
    assert min(core_gradients) > 1e-8

    sources = ('experiments/qgnn_motionframe/graph.py', 'experiments/qgnn_motionframe/check_graph.py',
               'experiments/qgnn_inherited/graph.py', 'code/02_pennylane_qgnn_core/physics_aligned_dual_model.py',
               'code/00_remote_shared_dependencies/target_interaction_graph.py')
    return dict(passed=True, device='cpu', graph_parameters=count, quantum_parameters=quantum_count,
                parameter_delta=0, state_dict_exact=True, initialization_rng_unchanged=True,
                training_dropout_rng_unchanged=True, positive_y_equivalence=forward_y,
                low_speed_equivalence=fallback, physical_rotation_max_errors=rotation_errors,
                core_gradient_max=core_gradients, classical_edge_value_input_unchanged=True,
                low_speed_threshold_mps=1.0, threshold_selected_without_validation=True,
                checks=['receiver i <- sender j; right then forward', 'negative-Y sign and unchanged last three edges',
                        'same state_dict/parameter order/values and initialization RNG', 'no new parameters/readouts',
                        'positive-Y old output and parameter gradients', 'fallback output/parameter/input gradients',
                        'training dropout identity', 'physical quantum encoding under common rigid rotations',
                        'raw classical edge_value, unchanged context, aligned quantum input in both layers',
                        'finite gradients; masks/padding/empty frames/permutation/frame mapping; live NaN rejection'],
                graph_sources={name: (ROOT / name).read_text(encoding='utf-8') for name in sources},
                optimizer_steps=0, dataset_access=False, prediction_evaluation=False,
                limitation='Synthetic structural evidence only. The 1 m/s fallback is preset and not smooth at its boundary. '
                           'Positive-Y input-velocity gradients may differ by design. Full-model rotation invariance, '
                           'ADE/FDE improvement and quantum advantage are not established.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'graph_sources'}, indent=2))
