"""Numerical contracts for relation D; no fitting and no dataset access."""
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_factorial.graph import FactorialGraph
from experiments.qgat_relation.graph import RelationGraph, build_edge_features
from prediction.classical import _source


def evaluate(graph, physical, z, mask, detected, reference=False):
    raw = graph.features(physical, z, mask, detected, reference=reference)
    output = graph.readout(raw.to(graph.readout[0].weight.dtype)) * mask[..., None]
    return torch.cat((raw, output), -1)


def gradients(output, weight, physical, z, graph, shared_only=False):
    variables = [('physical', physical), ('standardized', z)] + [
        (name, p) for name, p in graph.named_parameters() if not (shared_only and name == 'edge_encoder.weight')]
    values = torch.autograd.grad((output * weight).sum(), [v for _, v in variables], allow_unused=True)
    result = {}
    for (name, variable), value in zip(variables, values):
        value = torch.zeros_like(variable) if value is None else value
        assert torch.isfinite(value).all(), name
        result[name] = value.detach()
    return result


def gradient_errors(left, right):
    assert left.keys() == right.keys()
    return {key: float((left[key] - right[key]).abs().max()) for key in left}


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected ValueError')


def main():
    torch.set_num_threads(2)
    report = {'checks': [], 'created_utc': datetime.now(timezone.utc).isoformat()}
    torch.manual_seed(20260913)
    original = FactorialGraph(2, 45).double()
    following_original = torch.randn(8)
    torch.manual_seed(20260913)
    graph = RelationGraph().double()
    following_relation = torch.randn(8)
    assert torch.equal(following_original, following_relation)
    assert all(torch.equal(value, graph.state_dict()[key]) for key, value in original.state_dict().items())
    report['checks'].append({'case': 'same seed preserves all shared initialization and following RNG', 'passed': True})
    physical = (torch.randn(3, 8, 4, dtype=torch.float64) * 12).requires_grad_()
    z = torch.randn(3, 8, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[1, [0, 3, 7]] = True
    mask[2] = True
    detected = mask & (torch.rand(3, 8) > .3)
    a = evaluate(original, physical, z, mask, detected)
    b = evaluate(graph, physical, z, mask, detected)
    weight = torch.randn_like(a)
    errors = gradient_errors(gradients(a, weight, physical, z, original),
                             gradients(b, weight, physical, z, graph, shared_only=True))
    error = float((a - b).abs().max().detach())
    assert error < 1e-9 and max(errors.values()) < 1e-7
    report['checks'].append({'case': 'zero relation equals original D raw and final output and shared/input gradients',
                             'output_error': error, 'gradient_errors': errors})
    zero_grad = gradients(evaluate(graph, physical, z, mask, detected), weight, physical, z, graph)['edge_encoder.weight']
    assert zero_grad.abs().max() > 1e-8
    report['checks'].append({'case': 'zero initialized relation has finite nonzero training gradient',
                             'max_abs_gradient': float(zero_grad.abs().max()), 'norm': float(zero_grad.norm())})

    with torch.no_grad():
        graph.edge_encoder.weight.uniform_(-.35, .35)
    for n in (1, 3, 8):
        physical = (torch.randn(1, n, 4, dtype=torch.float64) * 9).requires_grad_()
        z = torch.randn(1, n, 4, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(1, n, dtype=torch.bool)
        detected = mask.clone()
        if n == 8:
            mask[0, [1, 4]] = False
            detected = mask & (torch.rand(1, n) > .3)
        local = evaluate(graph, physical, z, mask, detected)
        full = evaluate(graph, physical, z, mask, detected, reference='full')
        weight = torch.randn_like(local)
        errors = gradient_errors(gradients(local, weight, physical, z, graph),
                                 gradients(full, weight, physical, z, graph))
        error = float((local - full).abs().max().detach())
        assert error < 1e-9 and max(errors.values()) < 1e-7
        raw = local[..., :32]
        assert raw[..., -1].min() >= -1e-10 and raw[..., -1].max() <= 1 + 1e-10
        assert raw[..., :15].square().sum(-1).max() <= 3 + 1e-9
        torch.testing.assert_close(raw[..., 16:31].square().sum(-1), 3 * raw[..., 15].square(), atol=1e-9, rtol=1e-8)
        report['checks'].append({'case': 'nonzero relation exact contraction vs independent seven qubit circuit',
                                 'n': n, 'output_error': error, 'gradient_errors': errors})

    assert build_edge_features is _source.build_edge_features
    physical = torch.tensor([[[0., 0., 10., 0.], [10., 0., 5., 0.]]], dtype=torch.float64)
    edges, _ = build_edge_features(physical[..., :2], physical[..., 2:])
    expected = torch.tensor([[1/3, 0, -1/3, 0, 2/9, 1/3, .1],
                             [-1/3, 0, 1/3, 0, 2/9, 1/3, .1]], dtype=torch.float64)
    torch.testing.assert_close(torch.stack((edges[0, 0, 1], edges[0, 1, 0])), expected, atol=1e-12, rtol=1e-12)
    mask = torch.ones(1, 2, dtype=torch.bool)
    z = torch.randn_like(physical)
    approaching = evaluate(graph, physical, z, mask, mask)
    departing_physical = physical.clone()
    departing_physical[0, 1, 2] = 15.
    departing = evaluate(graph, departing_physical, z, mask, mask)
    raw_delta = float((approaching[..., :32] - departing[..., :32]).abs().max().detach())
    output_delta = float((approaching[..., 32:] - departing[..., 32:]).abs().max().detach())
    assert raw_delta > 1e-6 and output_delta > 1e-6
    report['checks'].append({'case': 'shared GNN directed edges and fixed standardized features physical velocity intervention',
                             'edge_i_receives_j': edges[0, 0, 1].tolist(), 'edge_j_receives_i': edges[0, 1, 0].tolist(),
                             'raw_change': raw_delta, 'readout_change': output_delta})

    physical = torch.randn(2, 3, 8, 4, dtype=torch.float64) * 16
    z = torch.randn_like(physical)
    mask = torch.zeros(2, 3, 8, dtype=torch.bool)
    mask[:, 1, [0, 3, 7]] = True
    mask[:, 2] = True
    detected = mask & (torch.rand_like(mask.float()) > .3)
    result = graph(physical, z, mask, detected)
    p = torch.tensor([7, 3, 1, 6, 4, 0, 2, 5])
    permuted = graph(physical[..., p, :], z[..., p, :], mask[..., p], detected[..., p])
    perm_error = float((permuted - result[..., p, :]).abs().max().detach())
    assert perm_error < 1e-9 and not result[~mask].any()
    dirty_physical, dirty_z = physical.clone(), z.clone()
    dirty_physical[~mask] = float('nan')
    dirty_z[~mask] = float('nan')
    assert torch.equal(result, graph(dirty_physical, dirty_z, mask, detected))
    dirty_physical.requires_grad_()
    dirty_z.requires_grad_()
    dirty_output = graph(dirty_physical, dirty_z, mask, detected)
    dirty_grad = torch.autograd.grad((dirty_output * torch.randn_like(dirty_output)).sum(),
                                    (dirty_physical, dirty_z, *graph.parameters()))
    assert all(torch.isfinite(g).all() for g in dirty_grad)
    assert not dirty_grad[0][~mask].any() and not dirty_grad[1][~mask].any()
    invalid_detected = detected.clone()
    invalid_detected[0, 0, 0] = True
    rejects(lambda: graph(physical, z, mask, invalid_detected))
    invalid_physical = physical.clone()
    invalid_physical[0, 1, 0, 2] = float('nan')
    rejects(lambda: graph(invalid_physical, z, mask, detected))
    invalid_z = z.clone()
    invalid_z[0, 1, 0, 0] = float('inf')
    rejects(lambda: graph(physical, invalid_z, mask, detected))
    report['checks'].append({'case': 'permutation empty frames sparse masks invalid NaNs finite gradients and input rejection',
                             'permutation_error': perm_error})

    z = torch.randn(1, 3, 4, dtype=torch.float64)
    physical = torch.zeros_like(z)
    physical[0, :, 0] = torch.tensor([0., 45., 90.00001])
    mask = torch.ones(1, 3, dtype=torch.bool)
    phi, pairs = graph.relation_angles(physical, mask)
    assert pairs[0, 0, 1] and not pairs[0, 0, 2]
    assert not phi[0, torch.arange(3), torch.arange(3)].any() and not phi[~pairs].any()
    inside = graph.features(physical, z, mask, mask)
    direct = graph.features(physical[:, :2], z[:, :2], mask[:, :2], mask[:, :2])
    torch.testing.assert_close(inside[:, 0], direct[:, 0], atol=1e-10, rtol=1e-9)
    physical[0, 1, 0] = 45.00001
    outside = graph.features(physical, z, mask, mask)
    self_only = original.features(physical[:, :1], z[:, :1], mask[:, :1], mask[:, :1])
    torch.testing.assert_close(outside[:, 0], self_only[:, 0], atol=1e-10, rtol=1e-9)
    # Out-of-range source velocity must not affect this receiver through relations.
    physical[0, 1:, 2:] = 123.
    torch.testing.assert_close(outside[:, 0], graph.features(physical, z, mask, mask)[:, 0], atol=1e-10, rtol=1e-9)
    report['checks'].append({'case': '45m inclusive boundary out-of-range isolation and identity self rotations', 'passed': True})

    fallback = RelationGraph().double()
    with torch.no_grad():
        fallback.encoder.weight.zero_()
        fallback.encoder.bias.zero_()
        fallback.theta.zero_()
        fallback.theta[1, 0, 0] = math.pi
    physical = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    z = torch.randn_like(physical, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    for reference in (False, 'full'):
        raw = fallback.features(physical, z, mask, mask, reference=reference)
        assert not raw.any()
        values = gradients(raw, torch.randn_like(raw), physical, z, fallback)
        assert all(not g.any() for g in values.values())
    with torch.no_grad():
        fallback.theta[1, 0, 0] = math.pi - 4e-6
    recovered = fallback.features(physical, z, mask, mask)
    assert recovered[..., -1].min() > 1e-12 and recovered.abs().max() > .9
    report['checks'].append({'case': 'nonempty nearzero postselection has finite zero raw gradients and recovers above threshold',
                             'threshold': 1e-12, 'recovered_min_probability': float(recovered[..., -1].min().detach())})

    costs = []
    for device in (['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']):
        for model in (FactorialGraph(2, 45), RelationGraph()):
            model = model.to(device)
            z = torch.randn(1, 20, 8, 4, device=device)
            physical = z * 12
            mask = torch.ones(1, 20, 8, dtype=torch.bool, device=device)
            seconds = []
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = model(physical, z, mask, mask)
                (output * torch.randn_like(output)).sum().backward()
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            costs.append({'device': device, 'graph': type(model).__name__, 'shape': [1, 20, 8, 4],
                          'warmup_seconds': seconds[0], 'forward_backward_seconds': seconds[1],
                          'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None})
    report.update(passed=True, backend='PennyLane default.qubit exact complex128 backprop shots=None',
                  graph_parameters_original=sum(p.numel() for p in original.parameters()),
                  graph_parameters_relation=sum(p.numel() for p in graph.parameters()),
                  added_parameters=graph.edge_encoder.weight.numel(), costs=costs,
                  limitations=['Numerical equivalence and differentiability only; no accuracy or convergence claim.',
                               'No dataset access, optimizer updates, or formal training in this check.'])
    folder = ROOT / 'reports/qgat_relation'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'graph_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
