"""CPU numerical contracts for relation-conditioned values; no fitting or dataset access."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation.graph import RelationGraph
from experiments.qgat_value.graph import RelationValueGraph


def evaluate(graph, physical, z, mask, detected, reference=False):
    raw = graph.features(physical, z, mask, detected, reference=reference)
    output = graph.readout(raw.to(graph.readout[0].weight.dtype)) * mask[..., None]
    return torch.cat((raw, output), -1)


def gradients(output, weight, physical, z, graph, shared_only=False):
    variables = [('physical', physical), ('standardized', z)] + [
        (name, p) for name, p in graph.named_parameters()
        if not (shared_only and name == 'value_edge_encoder.weight')]
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
    score_only = RelationGraph().double().cpu()
    following_score = torch.randn(8)
    torch.manual_seed(20260913)
    graph = RelationValueGraph().double().cpu()
    following_value = torch.randn(8)
    assert torch.equal(following_score, following_value)
    assert set(graph.state_dict()) - set(score_only.state_dict()) == {'value_edge_encoder.weight'}
    assert all(torch.equal(value, graph.state_dict()[key]) for key, value in score_only.state_dict().items())
    assert graph.value_edge_encoder.weight.shape == (4, 7) and not graph.value_edge_encoder.weight.any()
    assert graph.value_edge_encoder.weight is not graph.edge_encoder.weight
    report['checks'].append({'case': 'same seed preserves shared initialization and following RNG; independent 28 value weights',
                             'passed': True})
    # The zero-value equivalence must include a working, nonzero score relation.
    with torch.no_grad():
        score_only.edge_encoder.weight.uniform_(-.35, .35)
    loaded = graph.load_state_dict(score_only.state_dict(), strict=False)
    assert loaded.missing_keys == ['value_edge_encoder.weight'] and not loaded.unexpected_keys
    physical = (torch.randn(3, 8, 4, dtype=torch.float64) * 12).requires_grad_()
    z = torch.randn(3, 8, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[1, [0, 3, 7]] = True
    mask[2] = True
    detected = mask & (torch.rand(3, 8) > .3)
    a = evaluate(score_only, physical, z, mask, detected)
    b = evaluate(graph, physical, z, mask, detected)
    weight = torch.randn_like(a)
    errors = gradient_errors(gradients(a, weight, physical, z, score_only),
                             gradients(b, weight, physical, z, graph, shared_only=True))
    error = float((a - b).abs().max().detach())
    assert error < 1e-9 and max(errors.values()) < 1e-7
    report['checks'].append({'case': 'zero value rotation equals current RelationGraph with nonzero score weights',
                             'output_error': error, 'all_shared_and_input_gradient_errors': errors})
    value_gradient = gradients(evaluate(graph, physical, z, mask, detected), weight,
                               physical, z, graph)['value_edge_encoder.weight']
    assert value_gradient.abs().max() > 1e-8
    report['checks'].append({'case': 'zero initialized value branch has finite nonzero gradient',
                             'max_abs_gradient': float(value_gradient.abs().max()), 'norm': float(value_gradient.norm())})

    with torch.no_grad():
        graph.value_edge_encoder.weight.uniform_(-.3, .3)
    for n in (1, 3, 8):
        batch = 2 if n == 8 else 1
        physical = (torch.randn(batch, n, 4, dtype=torch.float64) * 9).requires_grad_()
        z = torch.randn(batch, n, 4, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(batch, n, dtype=torch.bool)
        if n == 8:
            mask[0] = False
            mask[1, [1, 4]] = False
        detected = mask & (torch.rand(batch, n) > .3)
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
        report['checks'].append({'case': 'nonzero score and value exact contraction vs independent seven qubit circuit',
                                 'n': n, 'empty_frame_included': n == 8, 'output_error': error,
                                 'all_parameter_and_input_gradient_errors': errors})

    # With Q/K/V and score fixed, each Rv is unitary and cannot change success p.
    physical = torch.tensor([[[0., 0., 10., 0.], [10., 0., 5., 0.],
                              [20., 0., 15., 0.], [30., 0., 8., 1.]]],
                            dtype=torch.float64, requires_grad=True)
    z = torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    baseline = evaluate(score_only, physical, z, mask, mask)
    changed = evaluate(graph, physical, z, mask, mask)
    probability_error = float((baseline[..., 31] - changed[..., 31]).abs().max().detach())
    raw_change = float((baseline[..., :31] - changed[..., :31]).abs().max().detach())
    readout_change = float((baseline[..., 32:] - changed[..., 32:]).abs().max().detach())
    assert probability_error < 1e-12 and raw_change > 1e-6 and readout_change > 1e-6
    pweight = torch.randn_like(baseline[..., 31])
    baseline_pgrad = gradients(baseline[..., 31], pweight, physical, z, score_only)
    changed_pgrad = gradients(changed[..., 31], pweight, physical, z, graph)
    value_pgrad = changed_pgrad.pop('value_edge_encoder.weight')
    errors = gradient_errors(baseline_pgrad, changed_pgrad)
    assert value_pgrad.abs().max() < 1e-10 and max(errors.values()) < 1e-9
    report['checks'].append({'case': 'value-only rotation changes readout but preserves score success probability and its shared gradients',
                             'probability_error': probability_error, 'raw_change': raw_change,
                             'readout_change': readout_change, 'shared_probability_gradient_errors': errors,
                             'value_probability_gradient_max_abs': float(value_pgrad.abs().max())})

    value_only = RelationValueGraph().double().cpu()
    with torch.no_grad():
        value_only.edge_encoder.weight.zero_()
        value_only.value_edge_encoder.weight.zero_()
        value_only.value_edge_encoder.weight[0, 2] = .6
        value_only.value_edge_encoder.weight[1, 2] = -.25
        value_only.value_edge_encoder.weight[2, 5] = .4
        value_only.value_edge_encoder.weight[3, 5] = .2
    disabled = RelationGraph().double().cpu()
    disabled.load_state_dict({k: v for k, v in value_only.state_dict().items() if k != 'value_edge_encoder.weight'})
    physical = torch.tensor([[[0., 0., 10., 0.], [10., 0., 5., 0.]]], dtype=torch.float64)
    z = torch.tensor([[[.2, -.4, .3, .7], [.8, .1, -.2, .4]]], dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    departing = physical.clone()
    departing[0, 1, 2] = 15.
    no_value_a = evaluate(disabled, physical, z, mask, mask)
    no_value_b = evaluate(disabled, departing, z, mask, mask)
    torch.testing.assert_close(no_value_a, no_value_b, atol=1e-12, rtol=1e-12)
    approaching_output = evaluate(value_only, physical, z, mask, mask)
    departing_output = evaluate(value_only, departing, z, mask, mask)
    probability_error = float((approaching_output[..., 31] - departing_output[..., 31]).abs().max().detach())
    raw_change = float((approaching_output[..., :31] - departing_output[..., :31]).abs().max().detach())
    readout_change = float((approaching_output[..., 32:] - departing_output[..., 32:]).abs().max().detach())
    assert probability_error < 1e-12 and raw_change > 1e-6 and readout_change > 1e-6
    report['checks'].append({'case': 'score relation disabled and QKV fixed: physical relative velocity acts through value branch alone',
                             'probability_error': probability_error, 'raw_change': raw_change, 'readout_change': readout_change})

    physical = torch.randn(2, 3, 8, 4, dtype=torch.float64) * 16
    z = torch.randn_like(physical)
    mask = torch.zeros(2, 3, 8, dtype=torch.bool)
    mask[:, 1, [0, 3, 7]] = True
    mask[:, 2] = True
    detected = mask & (torch.rand_like(mask.float()) > .3)
    output = graph(physical, z, mask, detected)
    permutation = torch.tensor([7, 3, 1, 6, 4, 0, 2, 5])
    permuted = graph(physical[..., permutation, :], z[..., permutation, :],
                     mask[..., permutation], detected[..., permutation])
    error = float((permuted - output[..., permutation, :]).abs().max().detach())
    assert error < 1e-9 and not output[~mask].any()
    dirty_physical, dirty_z = physical.clone(), z.clone()
    dirty_physical[~mask] = float('nan')
    dirty_z[~mask] = float('nan')
    assert torch.equal(output, graph(dirty_physical, dirty_z, mask, detected))
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
    report['checks'].append({'case': 'permutation empty frames sparse masks invalid NaN slots and gradients; invalid active inputs rejected',
                             'permutation_error': error})

    z = torch.randn(1, 3, 4, dtype=torch.float64)
    physical = torch.zeros_like(z)
    physical[0, :, 0] = torch.tensor([0., 45., 90.00001])
    mask = torch.ones(1, 3, dtype=torch.bool)
    score_phi, pairs = graph.relation_angles(physical, mask)
    value_phi = graph.value_relation_angles(physical, pairs)
    assert pairs[0, 0, 1] and not pairs[0, 0, 2]
    for phi in (score_phi, value_phi):
        assert not phi[0, torch.arange(3), torch.arange(3)].any() and not phi[~pairs].any()
    inside = graph.features(physical, z, mask, mask)
    direct = graph.features(physical[:, :2], z[:, :2], mask[:, :2], mask[:, :2])
    torch.testing.assert_close(inside[:, 0], direct[:, 0], atol=1e-10, rtol=1e-9)
    physical[0, 1, 0] = 45.00001
    outside = graph.features(physical, z, mask, mask)
    self_only = score_only.features(physical[:, :1], z[:, :1], mask[:, :1], mask[:, :1])
    torch.testing.assert_close(outside[:, 0], self_only[:, 0], atol=1e-10, rtol=1e-9)
    physical[0, 1:, 2:] = 123.
    torch.testing.assert_close(outside[:, 0], graph.features(physical, z, mask, mask)[:, 0], atol=1e-10, rtol=1e-9)
    report['checks'].append({'case': '45m inclusive boundary and out-of-range isolation; score and value self rotations both identity',
                             'passed': True})

    fallback = RelationValueGraph().double().cpu()
    with torch.no_grad():
        fallback.encoder.weight.zero_()
        fallback.encoder.bias.zero_()
        fallback.theta.zero_()
        fallback.theta[1, 0, 0] = math.pi
        fallback.value_edge_encoder.weight.uniform_(-.3, .3)
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
    assert torch.isfinite(recovered).all() and recovered[..., -1].min() > 1e-12 and recovered.abs().max() > 1e-3
    report['checks'].append({'case': 'nonempty nearzero postselection remains finite with active value rotation and recovers above threshold',
                             'threshold': 1e-12, 'recovered_min_probability': float(recovered[..., -1].min().detach())})

    # Exercise normal float32 model parameters with complex128 local quantum states on CPU.
    runtime = RelationValueGraph().cpu()
    z = torch.randn(1, 20, 8, 4)
    mask = torch.ones(1, 20, 8, dtype=torch.bool)
    output = runtime(z * 12, z, mask, mask)
    (output * torch.randn_like(output)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in runtime.parameters())
    assert runtime.value_edge_encoder.weight.grad.abs().max() > 1e-8
    report['checks'].append({'case': 'float32 parameter CPU forward/backward with complex128 quantum path',
                             'shape': [1, 20, 8, 4], 'output_dtype': str(output.dtype), 'passed': True})
    added = graph.value_edge_encoder.weight.numel()
    assert added == 28 and sum(p.numel() for p in graph.parameters()) - sum(p.numel() for p in score_only.parameters()) == added
    report.update(passed=True, device='cpu', backend='PennyLane default.qubit exact complex128 backprop shots=None',
                  graph_parameters_score_only=sum(p.numel() for p in score_only.parameters()),
                  graph_parameters_relation_value=sum(p.numel() for p in graph.parameters()),
                  added_value_parameters=added, optimizer_steps_executed=0, gpu_work_executed=False,
                  graph_sources={path: (ROOT / path).read_text() for path in (
                      'experiments/qgat_value/graph.py', 'experiments/qgat_value/check_graph.py',
                      'experiments/qgat_relation/graph.py', 'experiments/qgat_factorial/graph.py',
                      'experiments/qgat_candidate/graph.py', 'prediction/classical.py',
                      'code/00_remote_shared_dependencies/target_interaction_graph.py')},
                  limitations=['Numerical equivalence and differentiability only; no prediction-performance claim.',
                               'No dataset access, optimizer updates, GPU execution, or formal training.'])
    folder = ROOT / 'reports/qgat_value'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'graph_checks.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'graph_sources'},
                     indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
