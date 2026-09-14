"""Single-seed development experiment for QGNN-conditioned intent tokens."""
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))

import experiment_utils as utils
import run as retained
from future_token_model import FutureTokenCoreGraphLLM, future_compact_state, restore_future_compact
from intent_token_planner_model import IntentTokenPlannerFutureLLM
from physics_aligned_dual_model_effective import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss

SEED = 2026
GRAPH_DIR = BASE / 'physics_aligned_dual_seed2026_v1'
SEGMENTS = ((0, 7), (7, 14), (14, 20))


def restore(model, state):
    missing, extra = model.load_state_dict(state, strict=False)
    assert not extra, extra
    allowed = [name for name in missing if
               (((name.startswith('gpt2.') or name.startswith('future_gpt2.')) and 'lora_' not in name)
                or name.startswith('intent_'))]
    assert len(allowed) == len(missing), missing


def build(arm):
    graph_name = f'physics_{arm}_dual'
    source_path = BASE / f"future_token_{'qgnn' if arm == 'quantum' else 'classical'}_llm_seed2026_v1" / 'future_token_qgnn_llm_selected.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    graph_state = torch.load(GRAPH_DIR / f'{graph_name}_graph_selected.pt', map_location='cpu', weights_only=True)['state']
    graph = build_physics_aligned_dual_graph(graph_name); graph.load_state_dict(graph_state)
    reference = FutureTokenCoreGraphLLM(graph); restore_future_compact(reference, source['state'])
    graph = build_physics_aligned_dual_graph(graph_name); graph.load_state_dict(graph_state)
    model = IntentTokenPlannerFutureLLM(graph); restore(model, source['state'])
    return reference, model, source_path, int(source['epoch'])


def configure(model, joint):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith('intent_'): parameter.requires_grad_(True)
    if not joint:
        model.intent_stage_strength.requires_grad_(False)
        for parameter in model.intent_local_gate.parameters(): parameter.requires_grad_(False)


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def intent_labels(history, future, mask):
    """Create early/middle/late semantic labels only from training futures."""
    history_nt = history.permute(0, 2, 1, 3)
    future_nt = future.permute(0, 2, 1, 3)
    start_velocity = history_nt[:, :, -1, 2:4]
    start_position = history_nt[:, :, -1, :2]
    labels = {'longitudinal': [], 'lateral': [], 'interaction': []}
    for start, end in SEGMENTS:
        if start > 0:
            start_velocity = future_nt[:, :, start - 1, 2:4]
            start_position = future_nt[:, :, start - 1, :2]
        end_velocity = future_nt[:, :, end - 1, 2:4]
        end_position = future_nt[:, :, end - 1, :2]
        speed_delta = end_velocity.norm(dim=-1) - start_velocity.norm(dim=-1)
        longitudinal = torch.ones_like(speed_delta, dtype=torch.long)
        longitudinal = torch.where(speed_delta < -0.25, torch.zeros_like(longitudinal), longitudinal)
        longitudinal = torch.where(speed_delta > 0.25, torch.full_like(longitudinal, 2), longitudinal)
        heading_delta = wrap_angle(
            torch.atan2(end_velocity[..., 1], end_velocity[..., 0])
            - torch.atan2(start_velocity[..., 1], start_velocity[..., 0])
        )
        lateral = torch.ones_like(longitudinal)
        lateral = torch.where(heading_delta < -0.035, torch.zeros_like(lateral), lateral)
        lateral = torch.where(heading_delta > 0.035, torch.full_like(lateral, 2), lateral)
        start_pair = start_position[:, None] - start_position[:, :, None]
        end_pair = end_position[:, None] - end_position[:, :, None]
        n = mask.shape[1]
        valid = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]
        start_distance = start_pair.norm(dim=-1).masked_fill(~valid, 1e4).min(-1).values
        end_distance = end_pair.norm(dim=-1).masked_fill(~valid, 1e4).min(-1).values
        distance_delta = end_distance - start_distance
        interaction = torch.ones_like(longitudinal)
        interaction = torch.where(distance_delta < -0.75, torch.zeros_like(interaction), interaction)
        interaction = torch.where(distance_delta > 0.75, torch.full_like(interaction, 2), interaction)
        labels['longitudinal'].append(longitudinal)
        labels['lateral'].append(lateral)
        labels['interaction'].append(interaction)
    return {name: torch.stack(values, dim=2) for name, values in labels.items()}


def intent_loss(output, labels, mask):
    valid = mask[:, :, None].expand(-1, -1, len(SEGMENTS))
    losses = [F.cross_entropy(output['intent_logits'][name][valid], labels[name][valid]) for name in labels]
    return torch.stack(losses).mean()


def candidate_loss(output, future, mask):
    target = future[..., :2] - output['source_future_position']
    error = F.smooth_l1_loss(output['intent_candidate_correction'], target, reduction='none').sum(-1)
    return (error * mask[:, None, :]).sum() / (mask.sum().clamp_min(1) * error.shape[1])


def late_loss(output, future, mask):
    distance = (output['future_position'][:, 14:] - future[:, 14:, :, :2]).norm(dim=-1)
    return (distance * mask[:, None, :]).sum() / (mask.sum().clamp_min(1) * distance.shape[1])


def accuracy(output, labels, mask):
    valid = mask[:, :, None].expand(-1, -1, len(SEGMENTS))
    return {name: float((output['intent_logits'][name].argmax(-1)[valid] == labels[name][valid]).float().mean()) for name in labels}


def train(model, bank, selection, output_dir, epochs, warmup):
    configure(model, joint=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=5e-6)
    initial, _ = retained.evaluate(model, selection)
    best = initial['aggregate']['ade_m'] + 0.50 * initial['aggregate']['fde_m']; best_epoch = 0
    checkpoint = output_dir / 'intent_planner_selected.pt'
    torch.save({'state': future_compact_state(model), 'epoch': 0, 'metrics': initial}, checkpoint)
    with (output_dir / 'training.jsonl').open('x') as log:
        for epoch in range(1, epochs + 1):
            if epoch == warmup + 1:
                configure(model, joint=True)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=2e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup, eta_min=5e-6)
            started = time.time(); utils.set_seed(SEED + 410000 + epoch); model.train()
            generator = torch.Generator(device='cuda').manual_seed(SEED + 410000 + epoch)
            order = torch.randperm(len(bank['history']), generator=torch.Generator().manual_seed(SEED + 410000 + epoch)).cuda()
            totals = {'coordinate': 0.0, 'late': 0.0, 'intent': 0.0, 'candidate': 0.0}; count = 0
            acc_sum = {'longitudinal': 0.0, 'lateral': 0.0, 'interaction': 0.0}
            for offset in range(0, len(order), 24):
                index = order[offset:offset + 24]; snr = (5, 10, 15, 20)[(offset // 24 + epoch) % 4]
                history, future, mask = retained.make_batch(bank, index, snr, generator)
                labels = intent_labels(history, future, mask)
                optimizer.zero_grad(set_to_none=True); output = model(history, mask)
                semantic = intent_loss(output, labels, mask); candidate = candidate_loss(output, future, mask)
                coordinate = prediction_loss(output, future, mask, True); late = late_loss(output, future, mask)
                if epoch <= warmup: loss = semantic + 0.35 * candidate
                else: loss = coordinate + 0.35 * late + 0.20 * semantic + 0.15 * candidate
                assert torch.isfinite(loss); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0, error_if_nonfinite=True)
                optimizer.step(); size = len(index); count += size
                for key, value in [('coordinate', coordinate), ('late', late), ('intent', semantic), ('candidate', candidate)]: totals[key] += float(value.detach()) * size
                batch_accuracy = accuracy(output, labels, mask)
                for key in acc_sum: acc_sum[key] += batch_accuracy[key] * size
            scheduler.step(); metrics, _ = retained.evaluate(model, selection)
            score = metrics['aggregate']['ade_m'] + 0.50 * metrics['aggregate']['fde_m']
            if epoch > warmup and score < best:
                best = score; best_epoch = epoch
                torch.save({'state': future_compact_state(model), 'epoch': epoch, 'metrics': metrics}, checkpoint)
            record = {'epoch': epoch, 'phase': 'intent_warmup' if epoch <= warmup else 'joint',
                      **{key: value / count for key, value in totals.items()},
                      'intent_accuracy': {key: value / count for key, value in acc_sum.items()},
                      'stage_strength': torch.tanh(model.intent_stage_strength).detach().cpu().tolist(),
                      'metrics': metrics, 'score': score, 'best_epoch': best_epoch, 'seconds': time.time() - started}
            log.write(json.dumps(record) + '\n'); log.flush(); print(json.dumps(record), flush=True)
            utils.atomic_json(output_dir / 'progress.json', {'status': 'training', 'last': record})
    selected = torch.load(checkpoint, map_location='cpu', weights_only=True); restore(model, selected['state']); return selected


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=('quantum', 'classical'), required=True)
    parser.add_argument('--output-name', required=True); parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2); parser.add_argument('--smoke-only', action='store_true'); args = parser.parse_args()
    assert Path.cwd() == ROOT and platform.node() == 'jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4); utils.set_seed(SEED); retained.NOISE = utils.load_snr_noise_map(ROOT / 'results/multitarget_snr/snr_calibration.json')
    output_dir = BASE / args.output_name; output_dir.mkdir(exist_ok=False)
    bank, selection, dev, train_indices, selection_indices, dev_indices = load_banks()
    reference, model, source_path, source_epoch = build(args.arm); reference = reference.cuda().eval(); model = model.cuda().eval()
    history, future, mask = selection['history'][:2], selection['future'][:2], selection['mask'][:2]
    with torch.no_grad(): difference = float((reference(history, mask)['future_position'] - model(history, mask)['future_position']).abs().max())
    assert difference < 2e-6
    configure(model, joint=False); model.train(); output = model(history, mask); labels = intent_labels(history, future, mask)
    (intent_loss(output, labels, mask) + candidate_loss(output, future, mask)).backward()
    gradients = {'longitudinal_head': float(model.intent_longitudinal_head.weight.grad.abs().max()),
                 'residual_head': float(model.intent_residual_head[-1].weight.grad.abs().max())}
    assert all(np.isfinite(x) and x > 0 for x in gradients.values())
    utils.atomic_json(output_dir / 'smoke.json', {'epoch0_exact_max_abs_m': difference, 'gradients': gradients,
                                                   'source': str(source_path), 'source_epoch': source_epoch})
    if args.smoke_only: utils.atomic_json(output_dir / 'completed.json', {'status': 'smoke_completed'}); return
    utils.atomic_json(output_dir / 'protocol.json', {'seed': SEED, 'arm': args.arm, 'segments': SEGMENTS,
        'route': 'QGNN effective relations route per-step LLM states; semantic longitudinal/lateral/interaction tokens plan early-middle-late residuals',
        'intent_labels': {'longitudinal_speed_delta_mps': [-0.25, 0.25], 'lateral_heading_delta_rad': [-0.035, 0.035],
                          'nearest_distance_delta_m': [-0.75, 0.75]},
        'selection': 'fixed 256 development scenes, ADE+0.50FDE; epoch0 included', 'epochs': args.epochs,
        'warmup': args.warmup, 'no_test': True})
    selected = train(model, bank, selection, output_dir, args.epochs, args.warmup)
    learned, arrays = retained.evaluate(model, dev, collect=True)
    model.intent_relation_mode = 'uniform'; uniform, _ = retained.evaluate(model, dev)
    model.intent_relation_mode = 'off'; off, _ = retained.evaluate(model, dev)
    model.intent_relation_mode = 'learned'
    np.savez_compressed(output_dir / 'full_dev_predictions.npz', **arrays, indices=dev_indices)
    summary = {'arm': args.arm, 'selected_epoch': int(selected['epoch']), 'selection': selected['metrics'],
               'full_dev_learned': learned, 'full_dev_uniform': uniform, 'full_dev_relation_off': off, 'no_test': True}
    utils.atomic_json(output_dir / 'summary.json', summary); utils.atomic_json(output_dir / 'completed.json', {'status': 'completed', 'summary': summary})
    utils.atomic_json(output_dir / 'progress.json', {'status': 'completed', 'summary': summary}); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
