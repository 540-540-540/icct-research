"""Bounded F04 small-head models, candidate selection and confirmation gate."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .classical import GNNGraph
from .quantum import QuantumGraph

CANDIDATES = [dict(id='Q_A', model='qgnn', depth=3, graph_lr=3e-4),
              dict(id='Q_B', model='qgnn', depth=2, graph_lr=3e-4),
              dict(id='Q_C', model='qgnn', depth=3, graph_lr=1e-4),
              dict(id='G_A', model='gnn', depth=2, graph_lr=3e-4),
              dict(id='G_B', model='gnn', depth=2, graph_lr=1e-4)]
SMALL_HEAD = dict(layers=2, hidden=128, heads=4, ffn=256, dropout=0,
                  position='learned_absolute_0_to_19', activation='gelu',
                  prediction_head='128_SiLU_128_40')


class SmallTemporal(nn.Module):
    def __init__(self, graph_dim):
        super().__init__()
        self.state_projection = nn.Linear(4, 128)
        self.marker_projection = nn.Linear(2, 128, bias=False)
        self.position = nn.Embedding(20, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0,
                                           activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 40))
        self.graph_projection = nn.Linear(graph_dim, 128, bias=False)
        if graph_dim == 24:
            with torch.no_grad():
                self.graph_projection.weight[:, 21:].zero_()

    def forward(self, state_hat, standardized_state, track_exists, detected, graph_features):
        b, t, n, _ = state_hat.shape
        if t != 20 or not 1 <= n <= 8 or standardized_state.shape != state_hat.shape:
            raise ValueError('Expected matching twenty-frame state histories')
        m = track_exists.bool()
        eligible = m[:, -1] & (m.sum(1) >= 3)
        z = torch.where(m[..., None], standardized_state, 0).float()
        q = torch.where(m[..., None], graph_features, 0).float()
        markers = torch.stack((m, detected.bool()), -1).float()
        tokens = self.state_projection(z) + self.marker_projection(markers) + self.graph_projection(q)
        tokens = (tokens * m[..., None]).permute(0, 2, 1, 3).reshape(b*n, t, 128)
        masks = m.permute(0, 2, 1).reshape(b*n, t)
        active = eligible.reshape(-1).nonzero(as_tuple=True)[0]
        correction = tokens.new_zeros((b*n, 20, 2)) + tokens.sum() * 0
        if active.numel():
            valid = masks[active]
            steps = torch.arange(t, device=tokens.device)
            causal = steps[:, None] >= steps[None, :]
            allowed = causal[None] & valid[:, None, :]
            # An unavailable query attends only itself, avoiding all-masked NaNs;
            # unavailable keys remain excluded from every valid query.
            allowed |= (~valid[:, :, None]) & torch.eye(t, dtype=torch.bool, device=tokens.device)
            attention_mask = (~allowed).repeat_interleave(4, dim=0)
            hidden = self.encoder(tokens[active] + self.position(steps)[None], mask=attention_mask)
            correction = correction.index_copy(0, active, self.head(hidden[:, -1]).reshape(-1, 20, 2))
        correction = correction.reshape(b, n, 20, 2).permute(0, 2, 1, 3)
        origin = state_hat[:, -1].to(correction.dtype)
        horizon = torch.arange(1, 21, device=origin.device, dtype=origin.dtype)[None, :, None, None] * .1
        prediction = origin[:, None, :, :2] + horizon * origin[:, None, :, 2:] + correction
        return dict(prediction=torch.where(eligible[:, None, :, None], prediction, 0), origin_eligible=eligible)


class SmallPredictor(nn.Module):
    def __init__(self, name, normalization, depth=3, seed=2025):
        super().__init__()
        self.display_name = name.upper()
        self.name = name
        torch.manual_seed(seed)
        self.graph = QuantumGraph(normalization['quantum_scale'], depth) if name == 'qgnn' else GNNGraph()
        # Same temporal initialization seed; graph width is the declared difference.
        torch.manual_seed(seed)
        self.temporal = SmallTemporal(24 if name == 'qgnn' else 128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        if self.name == 'qgnn':
            features = self.graph.forward_history(state_hat, track_exists)
        else:
            features = self.graph(state_hat, standardized_state, track_exists, detected)
        return self.temporal(state_hat, standardized_state, track_exists, detected, features)


def balanced_indices(metadata, limit, seed=2025):
    """Round-robin shuffled source blocks; never consult future labels."""
    groups = {}
    for sample in metadata['samples']:
        groups.setdefault(str(sample['source_block']), []).append(sample['sample_index'])
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(groups[k])) for k in sorted(groups)]
    rng.shuffle(queues)
    result = []
    while queues and len(result) < limit:
        next_queues = []
        for queue in queues:
            if len(result) == limit:
                break
            result.append(int(queue.pop()))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return result


def choose_candidate(results):
    best = min(float(r['best_metrics']['J']) for r in results)
    if not math.isfinite(best) or best < 0:
        raise ValueError('Invalid selection metric')
    tied = [r for r in results if float(r['best_metrics']['J']) <= best + .005 * max(best, 1e-12)]
    for r in tied:
        if not math.isfinite(r['measured_step_seconds']) or r['measured_step_seconds'] <= 0:
            raise ValueError('Candidate tie-breaking requires measured step cost')
    return min(tied, key=lambda r: (r['measured_step_seconds'], r['parameter_count'],
                                    next(i for i, c in enumerate(CANDIDATES) if c['id'] == r['candidate']['id'])))


def confirmation_decision(pairs, gates):
    """Combine common absolute denominators before forming relative changes."""
    if set(pairs) != {'2023', '2024'}:
        return dict(status='INCOMPLETE', reason='Both fixed confirmation seeds required')
    if any(gates.get(k) != 'passed' for k in ('data', 'numerical', 'cost')):
        return dict(status='STOP', reason='Data, numerical or cost gate failed', gates=gates)
    for pair in pairs.values():
        for model in ('qgnn', 'gnn'):
            if any(not math.isfinite(pair[model][k]) or pair[model][k] < 0 for k in ('ADE', 'FDE', 'J')):
                return dict(status='STOP', reason='Non-finite or invalid confirmation metric')
    q = {k: sum(p['qgnn'][k] for p in pairs.values()) / 2 for k in ('ADE', 'FDE', 'J')}
    g = {k: sum(p['gnn'][k] for p in pairs.values()) / 2 for k in ('ADE', 'FDE', 'J')}
    if min(g.values()) <= 1e-12:
        return dict(status='REVIEW', reason='Degenerate near-zero comparator denominator', qgnn=q, gnn=g)
    improvement = (g['J'] - q['J']) / g['J']
    deltas = {s: p['gnn']['J'] - p['qgnn']['J'] for s, p in pairs.items()}
    protection = all(q[k] <= g[k] * 1.005 for k in ('ADE', 'FDE'))
    status = 'REVIEW'
    if improvement >= .02 and min(deltas.values()) > 0 and protection:
        status = 'PASS'
    elif max(deltas.values()) < 0 and improvement <= -.02:
        status = 'STOP'
    return dict(status=status, qgnn=q, gnn=g, relative_J_improvement=improvement,
                seed_delta_J=deltas, single_metric_guard_passed=protection, gates=gates,
                aggregation='equal fixed seeds; same per-SNR scene denominators; not mean of percentages')


def budget_gate(budget):
    required = ('available_gpu_hours', 'total_planned_gpu_hours', 'f04_reserved_gpu_hours', 'review_reserved_gpu_hours')
    if not budget.get('sealed'):
        raise ValueError('Training budget has not been sealed')
    if any(not isinstance(budget.get(k), (float, int)) or not math.isfinite(budget[k]) or budget[k] <= 0 for k in required):
        raise ValueError('Budget requires positive measured and reserved GPU-hour amounts')
    if budget['total_planned_gpu_hours'] > .7 * budget['available_gpu_hours'] + 1e-9:
        raise ValueError('Total plan exceeds 70% of available capacity')
    if budget['f04_reserved_gpu_hours'] > .2 * budget['total_planned_gpu_hours'] + 1e-9:
        raise ValueError('F04 reservation exceeds 20% of planned capacity')
    if budget['review_reserved_gpu_hours'] > budget['f04_reserved_gpu_hours']:
        raise ValueError('Gray review reserve must be included in F04 budget')
    if budget.get('batch_size') != 16 or not 1 <= budget.get('micro_batch', 0) <= 16:
        raise ValueError('Effective batch 16 and a measured micro batch are required')
    if any(budget.get('gates', {}).get(k) != 'passed' for k in ('data', 'numerical', 'cost')):
        raise ValueError('F00-F03 data/numerical/cost gates have not passed')
    return budget


def self_check():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    m = torch.ones(2, 20, 3, dtype=torch.bool)
    m[0, :7, 0] = False
    m[1, :, 2] = False
    x, q = torch.randn(2, 20, 3, 4), torch.randn(2, 20, 3, 24, requires_grad=True)
    model = SmallTemporal(24)
    out = model(x, x, m, m, q)
    assert torch.isfinite(out['prediction']).all()
    out['prediction'].square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    changed = x.clone(); changed[~m] = float('nan')
    changed_q = q.detach().clone(); changed_q[~m] = float('nan')
    assert torch.equal(out['prediction'], model(changed, changed, m, m, changed_q)['prediction'])
    permutation = torch.tensor([2,0,1])
    permuted = model(x[:,:,permutation], x[:,:,permutation], m[:,:,permutation], m[:,:,permutation], q[:,:,permutation])
    assert torch.allclose(out['prediction'][:,:,permutation], permuted['prediction'], atol=2e-6)
    no = torch.zeros_like(m)
    assert torch.equal(model(x,x,no,no,q)['prediction'], torch.zeros_like(out['prediction']))
    metrics = lambda value: dict(ADE=value, FDE=value, J=1.5*value)
    pairs = {str(s): dict(qgnn=metrics(.97), gnn=metrics(1.)) for s in (2023,2024)}
    gates = dict(data='passed', numerical='passed', cost='passed')
    assert confirmation_decision(pairs, gates)['status'] == 'PASS'
    pairs['2024']['qgnn'] = metrics(1.04)
    assert confirmation_decision(pairs, gates)['status'] == 'REVIEW'
    pairs['2023']['qgnn'] = metrics(1.04)
    assert confirmation_decision(pairs, gates)['status'] == 'STOP'
    assert confirmation_decision({}, gates)['status'] == 'INCOMPLETE'
    trials = [dict(candidate=CANDIDATES[i], best_metrics=dict(J=1+i*.001),
                   measured_step_seconds=3-i, parameter_count=100) for i in range(3)]
    assert choose_candidate(trials)['candidate']['id'] == 'Q_C'
    metadata = dict(samples=[dict(source_block=i//7, sample_index=i) for i in range(21)])
    ids = balanced_indices(metadata, 12)
    assert len(set(ids)) == 12 and [sum(i//7 == b for i in ids) for b in range(3)] == [4,4,4]
    return dict(status='passed', checks=['finite leading masks and gradients', 'padding NaN isolation',
                  'vehicle permutation', 'empty scene', 'PASS REVIEW STOP INCOMPLETE',
                  'cost tie break', 'source-block balanced manifest'])
