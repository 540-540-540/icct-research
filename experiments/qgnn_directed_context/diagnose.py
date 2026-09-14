"""Read-only train-set and paired baseline-gradient diagnostics for candidate A."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_directed_context.run import Predictor, REPORT, configuration, provenance, shared, training


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0)
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                min=float(quantiles[0]), p05=float(quantiles[1]), p25=float(quantiles[2]),
                median=float(quantiles[3]), p75=float(quantiles[4]), p95=float(quantiles[5]),
                max=float(quantiles[6]))


def _distribution(values):
    return {str(int(key)):int(value) for key, value in sorted(Counter(np.asarray(values).tolist()).items())}


def label_diagnostics(train):
    """Describe supervision weights without loading any non-train split."""
    label_valid = np.asarray(train.labels['label_valid'], dtype=bool)
    exists = np.asarray(train.loaders[20].arrays['track_exists'], dtype=bool)
    if label_valid.shape != (train.n, 20, 8) or exists.shape != (train.n, 20, 8):
        raise ValueError('Unexpected train supervision/history shape')
    eligible = exists[:, -1] & (exists.sum(1) >= 3)
    raw_lengths = label_valid.sum(1)
    lengths = raw_lengths * eligible

    def uneven_stats(matrix):
        multi = uneven = 0
        for row in matrix:
            positive = row[row > 0]
            if len(positive) >= 2:
                multi += 1
                uneven += int(positive.min() != positive.max())
        return dict(scenes_with_two_or_more_labeled_targets=multi,
                    unequal_length_scenes=uneven,
                    unequal_length_scene_fraction=uneven / multi if multi else 0.)

    point_denominators = lengths.sum(1)
    target_denominators = (lengths > 0).sum(1)
    supervised_lengths = lengths[lengths > 0]
    tv = []
    max_weight_ratio = []
    for row in lengths:
        positive = row[row > 0].astype(np.float64)
        if positive.size:
            point_weights = positive / positive.sum()
            macro_weights = np.full(positive.size, 1 / positive.size)
            tv.append(.5 * np.abs(point_weights - macro_weights).sum())
            max_weight_ratio.append(float(point_weights.max() / macro_weights[0]))
    result = dict(
        origins=train.n, slots_per_origin=8, future_steps=20,
        label_valid_length_distribution_all_slots=_distribution(raw_lengths.reshape(-1)),
        label_valid_length_distribution_labeled_targets=_distribution(raw_lengths[raw_lengths > 0]),
        effective_length_distribution_eligible_labeled_targets=_distribution(lengths[lengths > 0]),
        eligible_targets=int(eligible.sum()),
        eligible_targets_without_future_labels=int((eligible & (raw_lengths == 0)).sum()),
        eligible_targets_shorter_than_20_including_zero=int((raw_lengths[eligible] < 20).sum()),
        effective_supervised_targets=int(supervised_lengths.size),
        effective_targets_shorter_than_20=int((supervised_lengths < 20).sum()),
        effective_targets_shorter_than_20_fraction=float((supervised_lengths < 20).mean()),
        raw_label_lengths=uneven_stats(raw_lengths), effective_training_lengths=uneven_stats(lengths),
        ineligible_labeled_targets=int(((raw_lengths > 0) & ~eligible).sum()),
        point_weighted_vs_vehicle_macro=dict(
            definition=('Within each origin, training ADE gives each valid point equal weight (denominator sum H_i); '
                        'vehicle-macro ADE gives each supervised vehicle equal weight after its own denominator H_i.'),
            total_valid_points=int(point_denominators.sum()),
            total_supervised_vehicles=int(target_denominators.sum()),
            point_denominator_per_origin=_summary(point_denominators),
            vehicle_denominator_per_origin=_summary(target_denominators),
            origins_with_different_vehicle_weights=int(sum(value > 0 for value in tv)),
            weight_total_variation_per_supervised_origin=_summary(tv),
            largest_vehicle_weight_over_macro_weight=_summary(max_weight_ratio)),
    )
    expected = dict(eligible_targets=34425, eligible_targets_without_future_labels=83,
                    eligible_targets_shorter_than_20_including_zero=1352,
                    effective_supervised_targets=34342, effective_targets_shorter_than_20=1269,
                    unequal_length_scenes=929)
    actual = dict(eligible_targets=result['eligible_targets'],
                  eligible_targets_without_future_labels=result['eligible_targets_without_future_labels'],
                  eligible_targets_shorter_than_20_including_zero=result['eligible_targets_shorter_than_20_including_zero'],
                  effective_supervised_targets=result['effective_supervised_targets'],
                  effective_targets_shorter_than_20=result['effective_targets_shorter_than_20'],
                  unequal_length_scenes=result['effective_training_lengths']['unequal_length_scenes'])
    if actual != expected:
        raise ValueError(f'Frozen train supervision audit changed: expected {expected}, got {actual}')
    return result


def _tensor_stats(tensor):
    tensor = tensor.detach().double()
    finite = torch.isfinite(tensor)
    return dict(elements=tensor.numel(), finite=bool(finite.all()),
                nonzero_elements=int(torch.count_nonzero(tensor)),
                rms=float(tensor.square().mean().sqrt()),
                max_abs=float(tensor.abs().max()))


def _gradient_stats(named_parameters, predicate=lambda name: True):
    tensors = [parameter.grad.detach() for name, parameter in named_parameters
               if predicate(name) and parameter.grad is not None]
    count = sum(tensor.numel() for tensor in tensors)
    if not count:
        return dict(tensors=0, elements=0, finite=True, nonzero_elements=0, rms=0., max_abs=0., l2_norm=0.)
    square_sum = sum(float(tensor.detach().double().square().sum()) for tensor in tensors)
    return dict(tensors=len(tensors), elements=count,
                finite=all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
                nonzero_elements=sum(int(torch.count_nonzero(tensor)) for tensor in tensors),
                rms=float((square_sum / count) ** .5),
                max_abs=max(float(tensor.abs().max()) for tensor in tensors),
                l2_norm=float(square_sum ** .5))


def _module_gradients(model):
    named = list(model.named_parameters())
    groups = {
        'graph_input': lambda name: name.startswith('graph.input.'),
        'graph_layer_0': lambda name: name.startswith('graph.graph_layers.0.'),
        'graph_layer_1': lambda name: name.startswith('graph.graph_layers.1.'),
        'quantum_core': lambda name: '.core.weights' in name,
        'angle_calibration': lambda name: '.angle_scale_raw' in name or '.angle_bias' in name,
        'score_gate': lambda name: '.score.' in name or '.gate.' in name,
        'message_value': lambda name: any(part in name for part in ('.value_norm.', '.value.', '.edge_value.')),
        'directed_context': lambda name: name.endswith(('receiver_context', 'sender_context')),
        'temporal_adapter': lambda name: name.startswith('temporal.adapter.'),
        'temporal_lora': lambda name: name.startswith('temporal.llm.') and 'lora_' in name,
        'temporal_head': lambda name: name.startswith('temporal.head.'),
    }
    return {name:_gradient_stats(named, predicate) for name, predicate in groups.items()}


def _checkpoint_drift(left, right):
    groups = {
        'pqc': lambda name: '.core.weights' in name,
        'angle_calibration': lambda name: '.angle_scale_raw' in name or '.angle_bias' in name,
        'score_gate': lambda name: '.score.' in name or '.gate.' in name,
        'message_value': lambda name: any(part in name for part in ('.value_norm.', '.value.', '.edge_value.')),
        'graph_projection': lambda name: name.startswith('temporal.adapter.graph_projection.'),
        'lora': lambda name: name.startswith('temporal.llm.') and 'lora_' in name,
    }
    result = {}
    for group, predicate in groups.items():
        names = [name for name in left if predicate(name)]
        if not names or any(name not in right for name in names):
            raise ValueError(f'Checkpoint drift group is unavailable: {group}')
        delta_square = sum(float((right[name].double() - left[name].double()).square().sum()) for name in names)
        reference_square = sum(float(left[name].double().square().sum()) for name in names)
        result[group] = dict(tensors=len(names), elements=sum(left[name].numel() for name in names),
                             delta_l2=delta_square ** .5,
                             relative_l2=(delta_square / max(reference_square, 1e-24)) ** .5)
    return result


def _state_equal(before, model):
    after = training.mutable_state(model)
    return before.keys() == after.keys() and all(torch.equal(value, after[name]) for name, value in before.items())


def _run_backward(model, inputs, truth, rng, device):
    model.to(device).train()
    model.zero_grad(set_to_none=True)
    training.restore_rng(rng, device)
    output = training.forward(model, inputs)
    metric = training.masked_trajectory_loss(**output, **truth)
    if metric['skip_optimizer'] or not torch.isfinite(metric['loss']):
        raise ValueError('Fixed train batch has no finite trajectory supervision')
    metric['loss'].backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise FloatingPointError('Nonfinite fixed-batch gradient')
    return output, metric


def paired_gradient_diagnostics(train, cfg, device):
    ids = shared.origins(train.n, cfg['batch_size'])
    snrs = np.resize(np.asarray(cfg['snr_db']), len(ids))
    inputs, truth = train.batch(ids, snrs, device)

    baseline = ROOT/'results/qgnn_readout/seed2026/original'
    summary = json.loads((baseline/'summary.json').read_text())
    if summary['status'] != 'complete' or summary['best_metrics']['epoch'] != 75:
        raise ValueError('Current original-Q best checkpoint is not the accepted epoch-75 baseline')
    checkpoint = torch.load(baseline/'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['progress']['best_J'] != summary['best_metrics']['J']:
        raise ValueError('Original-Q best checkpoint and summary disagree')

    original = Predictor('original', cfg)
    deterministic_initial = training.mutable_state(original)
    candidate_probe = Predictor('directed_context', cfg)
    training.load_weights(original, checkpoint['model'])
    expected_missing = {f'graph.graph_layers.{layer}.{role}' for layer in range(2)
                        for role in ('receiver_context', 'sender_context')}
    candidate_mutable = training.mutable_state(candidate_probe)
    if set(candidate_mutable) - set(checkpoint['model']) != expected_missing:
        raise ValueError('Original-Q checkpoint is incompatible with the directed candidate')
    candidate_checkpoint = dict(checkpoint['model'])
    candidate_checkpoint.update({name:candidate_mutable[name] for name in expected_missing})
    training.load_weights(candidate_probe, candidate_checkpoint)
    original_state, candidate_state = original.state_dict(), candidate_probe.state_dict()
    common_state_exact = (set(original_state).issubset(candidate_state)
                          and all(torch.equal(value, candidate_state[name])
                                  for name, value in original_state.items()))
    if not common_state_exact:
        raise AssertionError('Fresh shared initialization differs')
    del original_state, candidate_state

    latest = torch.load(baseline/'latest.pt', map_location='cpu', weights_only=False)
    checkpoint_drift = dict(
        deterministic_initial_to_best=_checkpoint_drift(deterministic_initial, checkpoint['model']),
        best_to_latest=_checkpoint_drift(checkpoint['model'], latest['model']),
        latest_epoch=int(latest['progress']['epoch']))

    training.seed_all(cfg['seed'] + 900001)
    rng = training.rng_state(device)
    original_before = training.mutable_state(original)
    original_output, original_metric = _run_backward(original, inputs, truth, rng, device)
    original_unchanged = _state_equal(original_before, original)
    saved_output = {name:value.detach().cpu() for name, value in original_output.items()}
    saved_loss = original_metric['loss'].detach().cpu()
    original_grads = {name:parameter.grad.detach().cpu().clone()
                      for name, parameter in original.named_parameters() if parameter.grad is not None}
    del original, original_output, original_metric
    torch.cuda.empty_cache()

    candidate = candidate_probe
    candidate_before = training.mutable_state(candidate)
    candidate_output, candidate_metric = _run_backward(candidate, inputs, truth, rng, device)
    candidate_unchanged = _state_equal(candidate_before, candidate)
    if not original_unchanged or not candidate_unchanged:
        raise AssertionError('Backward changed model state without an optimizer step')
    output_errors = {name:float((value.detach().cpu() - saved_output[name]).abs().max())
                     for name, value in candidate_output.items() if value.dtype != torch.bool}
    output_equal = all(torch.equal(value.detach().cpu(), saved_output[name])
                       for name, value in candidate_output.items())
    loss_error = float((candidate_metric['loss'].detach().cpu() - saved_loss).abs())
    torch.testing.assert_close(candidate_metric['loss'].detach().cpu(), saved_loss, atol=2e-6, rtol=2e-6)
    for name, value in candidate_output.items():
        torch.testing.assert_close(value.detach().cpu(), saved_output[name], atol=2e-6, rtol=2e-6, msg=name)

    named = list(candidate.named_parameters())
    candidate_common = {name:parameter.grad.detach().cpu() for name, parameter in named
                        if name in original_grads and parameter.grad is not None}
    if set(candidate_common) != set(original_grads):
        raise AssertionError('Fresh shared gradient coverage differs')
    gradient_error = 0.
    for name, actual in candidate_common.items():
        torch.testing.assert_close(actual, original_grads[name], atol=3e-5, rtol=3e-4, msg=name)
        gradient_error = max(gradient_error, float((actual - original_grads[name]).abs().max()))

    contexts = {}
    for layer in range(2):
        for role in ('receiver_context', 'sender_context'):
            name = f'graph.graph_layers.{layer}.{role}'
            gradient = dict(named)[name].grad
            if gradient is None:
                raise AssertionError(f'Missing real-task gradient: {name}')
            contexts[name] = _tensor_stats(gradient)
    if not all(row['finite'] and row['nonzero_elements'] for row in contexts.values()):
        raise AssertionError('Directed-context parameters lack finite nonzero real-task gradients')

    pqc = {f'layer_{layer}':_tensor_stats(candidate.graph.graph_layers[layer].core.weights.grad)
           for layer in range(2)}
    if not all(row['finite'] and row['nonzero_elements'] for row in pqc.values()):
        raise AssertionError('A PQC layer lacks a finite nonzero real-task gradient')
    global_stats = _gradient_stats(named)
    graph_stats = _gradient_stats(named, lambda name: name.startswith('graph.'))
    clip = dict(threshold=1., global_preclip_l2=global_stats['l2_norm'],
                graph_preclip_l2=graph_stats['l2_norm'],
                hypothetical_global_clip_coefficient=min(1., 1. / max(global_stats['l2_norm'], 1e-12)),
                would_clip=global_stats['l2_norm'] > 1.)
    mu_pi = _mu_pi_diagnostics(candidate, inputs)
    return dict(
        baseline=dict(path=str((baseline/'best.pt').relative_to(ROOT)), best_epoch=75,
                      metrics=summary['best_metrics'], sha256=shared.digest(baseline/'best.pt'),
                      latest_sha256=shared.digest(baseline/'latest.pt'),
                      historical_trained_checkpoint_loaded=True, checkpoint_drift=checkpoint_drift),
        fixed_batch=dict(origin_ids=ids.tolist(), snr_db=snrs.tolist(), size=len(ids)),
        same_rng_state_replayed=True, shared_baseline_state_exact=common_state_exact,
        mutable_state_unchanged_by_backward=dict(original=original_unchanged, directed_context=candidate_unchanged),
        output_exact=output_equal, output_close=True, output_max_abs_errors=output_errors,
        loss_close=True, loss_max_abs_error=loss_error,
        shared_gradients_close=True, shared_gradient_tensors=len(original_grads),
        shared_gradient_max_abs_error=gradient_error,
        directed_context_gradients=contexts, pqc_gradients=pqc,
        major_module_gradients=_module_gradients(candidate),
        preclip_norms=clip, evaluation_mu_pi=mu_pi,
    )


def _mu_pi_diagnostics(model, inputs):
    captures = [dict() for _ in range(2)]
    handles = []
    for index, layer in enumerate(model.graph.graph_layers):
        handles.append(layer.register_forward_pre_hook(
            lambda module, args, i=index: captures[i].update(inputs=tuple(value.detach() for value in args))))
        handles.append(layer.score.register_forward_hook(
            lambda module, args, output, i=index: captures[i].update(score=output.detach())))
        handles.append(layer.gate.register_forward_hook(
            lambda module, args, output, i=index: captures[i].update(gate=output.detach())))
    model.eval()
    try:
        with torch.no_grad():
            model.graph(*(inputs[name] for name in ('state_hat', 'standardized_state', 'track_exists', 'detected')))
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    with torch.no_grad():
        for index, (layer, captured) in enumerate(zip(model.graph.graph_layers, captures)):
            nodes, edge_features, adjacency = captured['inputs']
            b, n, h = nodes.shape
            selected = adjacency.reshape(-1).nonzero().squeeze(-1)
            score = nodes.new_zeros(b*n*n, layer.heads).index_copy(0, selected, captured['score']).view(b, n, n, layer.heads)
            gate_logits = nodes.new_zeros(b*n*n, layer.heads).index_copy(0, selected, captured['gate']).view(b, n, n, layer.heads)
            attention = torch.softmax(score.masked_fill(~adjacency[..., None], -1e4), dim=2)
            coefficients = attention * (2 * torch.sigmoid(gate_logits))
            mu = coefficients.sum(2)
            pi = coefficients / mu[:, :, None, :].clamp_min(1e-12)
            live_receiver = adjacency.any(2)[..., None].expand_as(mu)
            live_edge = adjacency[..., None].expand_as(coefficients)
            values = layer.value(layer.value_norm(nodes)).view(b, n, layer.heads, layer.head_dim)
            edge_values = layer.edge_value(edge_features).view(b, n, n, layer.heads, layer.head_dim)
            messages = values[:, None] + edge_values
            direct = (coefficients[..., None] * messages).sum(2)
            rebuilt = mu[..., None] * (pi[..., None] * messages).sum(2)
            reconstruction_error = float((direct[live_receiver] - rebuilt[live_receiver]).abs().max())
            result[f'layer_{index}'] = dict(
                mode='eval; attention dropout disabled',
                mu_sum_attention_times_gate=_summary(mu[live_receiver].cpu().numpy()),
                pi_valid_edge_distribution=_summary(pi[live_edge].cpu().numpy()),
                pi_row_sum_max_abs_error=float((pi.sum(2)[live_receiver] - 1).abs().max()),
                aggregate_reconstruction_max_abs_error=reconstruction_error,
                active_receiver_heads=int(live_receiver.sum()), valid_edge_heads=int(live_edge.sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('D0 requires one CUDA device for the real full-model gradient check')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = configuration()
    train = shared.dataset('train')
    if train.split != 'train' or train.n != cfg['train_origins']:
        raise ValueError('D0 accepts the frozen full train split only')
    source = provenance(cfg)
    report = dict(passed=False, source=source,
                  meaning_of_passed='D0 diagnostics completed; no ADE/FDE improvement is implied.',
                  dataset_access=['train'], V_select_opened=False, confirmation_opened=False, test_opened=False,
                  optimizer_created=False, optimizer_steps=0, configuration_revision=cfg['revision'])
    report['label_valid_diagnostics'] = label_diagnostics(train)
    report['paired_full_model'] = paired_gradient_diagnostics(train, cfg, device)
    report['passed'] = True
    training.write_json(REPORT/'d0.json', report)
    print(json.dumps({key:value for key, value in report.items()
                      if key not in ('label_valid_diagnostics', 'paired_full_model')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
