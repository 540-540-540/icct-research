"""Single-seed dev experiment for an edge-level QGNN--LLM future decoder."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/js_cn/sensing')
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
HERE = ROOT / '双图研究框架/未来词QGNN_LLM实验_2026-09-09'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

import experiment_utils as utils
import run as retained
from analyze_interaction_strata import scene_features
from future_token_model import FutureTokenCoreGraphLLM, future_compact_state, restore_future_compact
from physics_aligned_dual_model_effective import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss
from temporal_pair_refinement_model import TemporalPairRefinementFutureLLM

SEED = 2026
GRAPH_DIR = BASE / 'physics_aligned_dual_seed2026_v1'


def restore(model, state):
    missing, extra = model.load_state_dict(state, strict=False)
    assert not extra, extra
    allowed = [
        name for name in missing
        if ((name.startswith('gpt2.') or name.startswith('future_gpt2.')) and 'lora_' not in name)
        or name.startswith('pair_')
    ]
    assert len(allowed) == len(missing), missing


def build(arm):
    graph_name = f'physics_{arm}_dual'
    source_path = BASE / f"future_token_{'qgnn' if arm == 'quantum' else 'classical'}_llm_seed2026_v1" / 'future_token_qgnn_llm_selected.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    graph_state = torch.load(
        GRAPH_DIR / f'{graph_name}_graph_selected.pt', map_location='cpu', weights_only=True
    )['state']

    reference_graph = build_physics_aligned_dual_graph(graph_name)
    reference_graph.load_state_dict(graph_state)
    reference = FutureTokenCoreGraphLLM(reference_graph)
    restore_future_compact(reference, source['state'])

    graph = build_physics_aligned_dual_graph(graph_name)
    graph.load_state_dict(graph_state)
    model = TemporalPairRefinementFutureLLM(graph)
    restore(model, source['state'])
    return reference, model, source_path, int(source['epoch'])


def configure(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith('pair_'):
            parameter.requires_grad_(True)


def complexity_groups(indices):
    with np.load(ROOT / 'data/multitarget_lankershim_v1.npz') as data:
        features = scene_features(data['train_states'][indices, :20], data['train_mask'][indices])
    values = features['closing_strength']
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return {
        'low': np.flatnonzero(values <= q1),
        'medium': np.flatnonzero((values > q1) & (values <= q2)),
        'high': np.flatnonzero(values > q2),
    }


def training_order(groups, total, epoch):
    generator = torch.Generator().manual_seed(SEED + 320000 + epoch)
    counts = {'high': total // 2, 'medium': total // 4}
    counts['low'] = total - counts['high'] - counts['medium']
    parts = [
        torch.as_tensor(groups[key])[torch.randint(len(groups[key]), (counts[key],), generator=generator)]
        for key in ('high', 'medium', 'low')
    ]
    order = torch.cat(parts)
    return order[torch.randperm(len(order), generator=generator)].cuda()


def subset(bank, positions):
    index = torch.as_tensor(positions, device='cuda')
    return {name: tensor.index_select(0, index) for name, tensor in bank.items()}


def relative_loss(output, future, mask):
    prediction = output['source_future_position'].permute(0, 2, 1, 3)
    target = future[..., :2].permute(0, 2, 1, 3)
    # Edge orientation is receiver i, sender j: p_j - p_i.
    source_relative = prediction[:, None, :] - prediction[:, :, None]
    target_relative = target[:, None, :] - target[:, :, None]
    target_delta = target_relative - source_relative
    valid = (mask[:, :, None] & mask[:, None, :]) & output['adjacency']
    n = mask.shape[1]
    valid = valid & ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]
    valid = valid[:, :, :, None, None].expand_as(target_delta)
    losses = [F.smooth_l1_loss(delta[valid], target_delta[valid]) for delta in output['pair_relative_deltas']]
    return torch.stack(losses).mean()


def selection_score(model, selection, selection_high):
    overall, _ = retained.evaluate(model, selection)
    high, _ = retained.evaluate(model, selection_high)
    a = overall['aggregate']
    h = high['aggregate']
    score = a['ade_m'] + 0.35 * a['fde_m'] + 0.15 * (h['ade_m'] + 0.35 * h['fde_m'])
    return score, overall, high


def train(model, train_bank, train_groups, selection, selection_high, output_dir, epochs):
    configure(model)
    named = dict(model.named_parameters())
    decoder = [p for n, p in named.items() if n.startswith(('pair_decoders.', 'pair_relation_projection.')) and p.requires_grad]
    temporal = [p for n, p in named.items() if n.startswith('pair_hidden_projection.') and p.requires_grad]
    scalar = [model.pair_scale_raw]
    optimizer = torch.optim.AdamW([
        {'params': decoder, 'lr': 1.0e-4},
        {'params': temporal, 'lr': 7.5e-5},
        {'params': scalar, 'lr': 2.0e-4},
    ], weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=5e-6)

    best, initial, initial_high = selection_score(model, selection, selection_high)
    best_epoch = 0
    checkpoint = output_dir / 'temporal_pair_selected.pt'
    torch.save({'state': future_compact_state(model), 'epoch': 0, 'metrics': initial, 'high_metrics': initial_high}, checkpoint)
    with (output_dir / 'training.jsonl').open('x') as log:
        for epoch in range(1, epochs + 1):
            started = time.time()
            utils.set_seed(SEED + 310000 + epoch)
            model.train()
            noise = torch.Generator(device='cuda').manual_seed(SEED + 310000 + epoch)
            order = training_order(train_groups, len(train_bank['history']), epoch)
            totals = {'coordinate': 0.0, 'relative': 0.0, 'refinement_l1': 0.0}
            count = 0
            for offset in range(0, len(order), 18):
                indices = order[offset:offset + 18]
                snr = (5, 10, 15, 20)[(offset // 18 + epoch) % 4]
                history, future, mask = retained.make_batch(train_bank, indices, snr, noise)
                optimizer.zero_grad(set_to_none=True)
                output = model(history, mask)
                coordinate = prediction_loss(output, future, mask, True)
                relative = relative_loss(output, future, mask)
                refinement_l1 = output['pair_refinement'].abs().mean()
                loss = coordinate + 0.30 * relative + 0.002 * refinement_l1
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 3.0, error_if_nonfinite=True
                )
                optimizer.step()
                size = len(indices)
                totals['coordinate'] += float(coordinate.detach()) * size
                totals['relative'] += float(relative.detach()) * size
                totals['refinement_l1'] += float(refinement_l1.detach()) * size
                count += size
            scheduler.step()
            score, metrics, high_metrics = selection_score(model, selection, selection_high)
            if score < best:
                best = score
                best_epoch = epoch
                torch.save({'state': future_compact_state(model), 'epoch': epoch, 'metrics': metrics, 'high_metrics': high_metrics}, checkpoint)
            record = {
                'epoch': epoch,
                'coordinate_loss': totals['coordinate'] / count,
                'relative_loss': totals['relative'] / count,
                'refinement_l1_m': totals['refinement_l1'] / count,
                'metrics': metrics,
                'high_metrics': high_metrics,
                'score': score,
                'best_epoch': best_epoch,
                'seconds': time.time() - started,
            }
            log.write(json.dumps(record) + '\n')
            log.flush()
            utils.atomic_json(output_dir / 'progress.json', {'status': 'training', 'last': record})
            print(json.dumps(record), flush=True)
    selected = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restore(model, selected['state'])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('quantum', 'classical'), required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == 'jscn'
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    utils.set_seed(SEED)
    retained.NOISE = utils.load_snr_noise_map(ROOT / 'results/multitarget_snr/snr_calibration.json')
    output_dir = BASE / args.output_name
    output_dir.mkdir(exist_ok=False)
    train_bank, selection, dev, train_indices, selection_indices, dev_indices = load_banks()
    train_groups = complexity_groups(train_indices)
    selection_groups = complexity_groups(selection_indices)
    selection_high = subset(selection, selection_groups['high'])
    dev_groups = complexity_groups(dev_indices)
    selected_set = set(int(x) for x in selection_indices.tolist())
    confirmation_positions = np.asarray([
        int(p) for p in dev_groups['high'] if int(dev_indices[int(p)]) not in selected_set
    ])
    confirmation = subset(dev, confirmation_positions)

    reference, model, source_path, source_epoch = build(args.arm)
    reference = reference.cuda().eval()
    model = model.cuda().eval()
    history, mask = selection['history'][:2], selection['mask'][:2]
    with torch.no_grad():
        difference = float((reference(history, mask)['future_position'] - model(history, mask)['future_position']).abs().max())
    assert difference < 2e-6, difference
    configure(model)
    model.train()
    probe = model(history, mask)
    probe['future_position'].sum().backward()
    gradient = float(model.pair_decoders[0][-1].weight.grad.abs().max())
    assert np.isfinite(gradient) and gradient > 0, gradient
    utils.atomic_json(output_dir / 'smoke.json', {
        'epoch0_exact_max_abs_m': difference,
        'pair_decoder_gradient': gradient,
        'source': str(source_path),
        'source_epoch': source_epoch,
    })
    if args.smoke_only:
        utils.atomic_json(output_dir / 'completed.json', {'status': 'smoke_completed'})
        return

    utils.atomic_json(output_dir / 'protocol.json', {
        'seed': SEED,
        'arm': args.arm,
        'interface': 'edge QGNN latent + effective alpha*gate predicts supervised future pair-relative residuals from per-step LLM states; relative residuals analytically refine coordinates',
        'training': 'only new pair interface; frozen source QGNN/GNN and LLM',
        'sampling': '50% high, 25% medium, 25% low by outcome-blind history closing strength',
        'selection': 'fixed 256 overall primary plus 0.15 weight for its outcome-blind closing-strength high third',
        'confirmation_scenes': int(len(confirmation_positions)),
        'epochs': args.epochs,
        'no_test': True,
    })
    source_confirmation, _ = retained.evaluate(reference, confirmation)
    selected = train(model, train_bank, train_groups, selection, selection_high, output_dir, args.epochs)
    learned_confirmation, _ = retained.evaluate(model, confirmation)
    model.pair_relation_mode = 'uniform'
    uniform_confirmation, _ = retained.evaluate(model, confirmation)
    model.pair_relation_mode = 'off'
    off_confirmation, _ = retained.evaluate(model, confirmation)
    model.pair_relation_mode = 'learned'
    full_dev, arrays = retained.evaluate(model, dev, collect=True)
    np.savez_compressed(output_dir / 'full_dev_predictions.npz', **arrays, indices=dev_indices)
    summary = {
        'arm': args.arm,
        'selected_epoch': int(selected['epoch']),
        'selection': selected['metrics'],
        'selection_high': selected['high_metrics'],
        'source_confirmation': source_confirmation,
        'confirmation_learned': learned_confirmation,
        'confirmation_uniform': uniform_confirmation,
        'confirmation_off': off_confirmation,
        'full_dev_secondary': full_dev,
        'no_test': True,
    }
    utils.atomic_json(output_dir / 'summary.json', summary)
    utils.atomic_json(output_dir / 'completed.json', {'status': 'completed', 'summary': summary})
    utils.atomic_json(output_dir / 'progress.json', {'status': 'completed', 'summary': summary})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
