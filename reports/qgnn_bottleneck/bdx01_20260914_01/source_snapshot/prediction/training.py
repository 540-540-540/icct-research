"""Shared F04/F05 training; model inputs and supervision remain separate."""
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from frontend.symbol_dataset import SharedPredictionInputs
from prediction.temporal import masked_trajectory_loss
from prediction.evaluation_cache import evaluation_graph_features
import statistics

ROOT = Path(__file__).resolve().parents[1]
SNRS = (5, 10, 15, 20)
FIELDS = ('state_hat', 'standardized_state', 'track_exists', 'detected')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Dataset:
    def __init__(self, split, root=None):
        if split not in ('train', 'V_select', 'V_confirm'):
            raise ValueError('Training/development loader refuses locked test data')
        self.split = split
        root = Path(root or ROOT / 'data/f01d')
        self.loaders = {s: SharedPredictionInputs(root / 'inputs' / f'{split}_snr_{s}.npz') for s in SNRS}
        self.n = len(self.loaders[20])
        self.normalization = self.loaders[20].normalization
        self.input_hashes = {str(s): x.input_hash for s, x in self.loaders.items()}
        with np.load(root / 'labels' / f'{split}.npz', allow_pickle=False) as f:
            if set(f.files) != {'future_position', 'label_valid'}:
                raise ValueError('Unexpected supervision fields')
            self.labels = {k: f[k] for k in f.files}
        y, valid = self.labels['future_position'], self.labels['label_valid']
        if y.shape != (self.n, 20, 8, 2) or valid.shape != y.shape[:-1] or valid.dtype != bool:
            raise ValueError('Invalid supervision shape/type')
        if not np.isfinite(y[valid]).all():
            raise ValueError('Nonfinite valid supervision')
        ref = self.loaders[20].arrays
        for loader in self.loaders.values():
            if len(loader) != self.n or not np.array_equal(ref['timestamp'], loader.arrays['timestamp']):
                raise ValueError('SNR origin alignment failed')
            if not np.array_equal(ref['track_exists'], loader.arrays['track_exists']):
                raise ValueError('SNR input eligibility differs')

    def __len__(self):
        return self.n

    def batch(self, ids, snrs, device):
        ids = np.asarray(ids, dtype=np.int64)
        snrs = np.broadcast_to(snrs, ids.shape)
        samples = [self.loaders[int(s)][int(i)] for i, s in zip(ids, snrs)]
        inputs = {k: torch.as_tensor(np.stack([x[k] for x in samples]), device=device) for k in FIELDS}
        truth = {k: torch.as_tensor(v[ids], device=device) for k, v in self.labels.items()}
        return inputs, truth


def scene_metrics(prediction, future_position, label_valid, origin_eligible):
    if prediction.shape != future_position.shape or label_valid.shape != prediction.shape[:-1]:
        raise ValueError('Metric shape mismatch')
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite model output; no scenes may be dropped')
    valid = label_valid.bool() & origin_eligible[:, None].bool()
    distance = torch.linalg.vector_norm(torch.where(valid[..., None], prediction - future_position, 0), dim=-1)
    counts = valid.sum(1)
    targets = counts > 0
    ade_count = targets.sum(1)
    fde_count = valid[:, -1].sum(1)
    target_ade = distance.sum(1) / counts.clamp_min(1)
    ade = (target_ade * targets).sum(1) / ade_count.clamp_min(1)
    fde = distance[:, -1].sum(1) / fde_count.clamp_min(1)
    return dict(scene_ade=ade, scene_fde=fde, ade_count=ade_count, fde_count=fde_count,
                loss=(ade + .5 * fde).mean(), skip_optimizer=bool((ade_count == 0).all()))


def balanced_schedule(n, seed, epoch, indices=None):
    ids = np.arange(n) if indices is None else np.asarray(indices, dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or len(ids) == 0 or ids.min() < 0 or ids.max() >= n:
        raise ValueError('Origin manifest must contain unique valid origins')
    # Fixed balanced assignment, plus a cyclic rotation: every origin sees all SNRs in four epochs.
    base_order = np.random.default_rng(seed).permutation(np.sort(ids))
    offset = np.empty(n, dtype=np.int64)
    offset[base_order] = np.arange(len(ids)) % 4
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 710])).permutation(ids)
    return order, np.asarray(SNRS)[(offset[order] + epoch) % 4]


def mutable_state(model):
    # Frozen pretrained GPT-2 weights are a local immutable dependency, not rewritten every batch.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if '.llm.' not in k or 'lora_' in k}


def load_weights(model, state):
    expected = set(mutable_state(model))
    if set(state) != expected:
        raise ValueError(f'Checkpoint model keys mismatch: missing={expected-set(state)}, extra={set(state)-expected}')
    model.load_state_dict(state, strict=False)


def rng_state(device=None):
    device = torch.device(device or 'cpu')
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                original_device=str(device))


def restore_rng(state, device=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        device = torch.device(device or state['original_device'])
        if device.type != 'cuda':
            raise ValueError('A CUDA training checkpoint requires CUDA to preserve RNG behavior')
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def configure_phase(model, phase, graph_lr):
    if phase not in ('warmup', 'adapt', 'joint'):
        raise ValueError('Unknown phase')
    groups = {}
    for name, param in model.named_parameters():
        base = '.llm.' in name and 'lora_' not in name
        graph = name.startswith('graph.') or 'graph_projection.' in name
        active = not base and (phase == 'joint' or (phase == 'adapt' and graph) or (phase == 'warmup' and not graph))
        param.requires_grad_(active)
        if active:
            lr = graph_lr if name.startswith('graph.') else (3e-5 if 'lora_' in name else 1e-4)
            wd = 0. if param.ndim < 2 or (name.startswith('graph.') and hasattr(model.graph, 'forward_history')) else .01
            groups.setdefault((lr, wd), []).append(param)
    return torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()])


def forward(model, inputs, phase='joint'):
    if phase != 'warmup':
        return model(**inputs)
    dim = model.temporal.adapter.graph_projection.in_features
    zeros = inputs['state_hat'].new_zeros((*inputs['state_hat'].shape[:-1], dim))
    return model.temporal(**inputs, graph_features=zeros)


@torch.no_grad()
def evaluate(model, dataset, device=None, micro_batch=1, indices=None, phase='joint'):
    if dataset.split not in ('V_select', 'V_confirm'):
        raise ValueError('Evaluation requires an explicitly permitted development split')
    device = device or next(model.parameters()).device
    ids = np.arange(dataset.n) if indices is None else np.asarray(indices, dtype=np.int64)
    was_training = model.training
    model.eval()
    by_snr, rows = {}, []
    for snr in SNRS:
        cached_features = None
        cache_stats = None
        if phase != 'warmup' and hasattr(model.graph, 'forward_history'):
            arrays = dataset.loaders[snr].arrays
            cached_features, cache_stats = evaluation_graph_features(
                model.graph, arrays['state_hat'][ids], arrays['track_exists'][ids])
        ade_sum = fde_sum = 0.
        ade_n = fde_n = 0
        for start in range(0, len(ids), micro_batch):
            part = ids[start:start+micro_batch]
            inputs, truth = dataset.batch(part, snr, device)
            output = (model.temporal(**inputs, graph_features=cached_features[start:start+len(part)])
                      if cached_features is not None else forward(model, inputs, phase))
            metric = scene_metrics(**output, **truth)
            a, f = metric['scene_ade'].cpu().tolist(), metric['scene_fde'].cpu().tolist()
            ac, fc = metric['ade_count'].cpu().tolist(), metric['fde_count'].cpu().tolist()
            for i, av, fv, an, fn in zip(part, a, f, ac, fc):
                rows.append(dict(origin=int(i), snr_db=snr, ADE=av if an else None, FDE=fv if fn else None,
                                 ade_targets=an, fde_targets=fn))
                ade_sum += av if an else 0.
                fde_sum += fv if fn else 0.
                ade_n += bool(an)
                fde_n += bool(fn)
        if not ade_n or not fde_n:
            raise ValueError('Development condition has no scoreable ADE/FDE scenes')
        a, f = ade_sum/ade_n, fde_sum/fde_n
        by_snr[str(snr)] = dict(ADE=a, FDE=f, J=a+.5*f, ade_sum=ade_sum, fde_sum=fde_sum,
                                ade_scenes=ade_n, fde_scenes=fde_n, total_scenes=len(ids), graph_evaluation_cache=cache_stats)
    ade = sum(r['ADE'] for r in by_snr.values()) / 4
    fde = sum(r['FDE'] for r in by_snr.values()) / 4
    model.train(was_training)
    return dict(ADE=ade, FDE=fde, J=ade+.5*fde, by_snr=by_snr, per_scene=rows, split=dataset.split)


def _fit_impl(model, train, validation, run_dir, *, seed, epochs, patience, graph_lr,
        phase='joint', batch_size=16, micro_batch=1, indices=None, snr_mode='balanced',
        resume=False, contract=None):
    if train.split != 'train' or validation.split != 'V_select':
        raise ValueError('Only train labels may optimize; only V_select may select checkpoints')
    if min(epochs, patience, batch_size, micro_batch) < 1 or snr_mode not in ('balanced', 'nominal'):
        raise ValueError('Invalid training schedule')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    manifest = np.arange(train.n) if indices is None else np.asarray(indices, dtype=np.int64)
    balanced_schedule(train.n, seed, 0, manifest)
    config = dict(seed=seed, epochs=epochs, patience=patience, graph_lr=graph_lr, phase=phase,
                  batch_size=batch_size, micro_batch=micro_batch, indices=manifest.tolist(), snr_mode=snr_mode,
                  input_hashes=train.input_hashes, validation_hashes=validation.input_hashes, contract=contract)
    optimizer = configure_phase(model, phase, graph_lr)
    current = dict(epoch=0, cursor=0, best_J=None, stale=0, history=[], optimizer_steps=0,
                   loss_sum=0., seen=0, elapsed_seconds=0., gradient_norm=None, graph_gradient_norm=None, graph_signal_seen=False, step_seconds=[])
    latest, best = run_dir/'latest.pt', run_dir/'best.pt'
    if resume and latest.exists():
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume contract differs from checkpoint')
        load_weights(model, checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current = checkpoint['progress']
        restore_rng(checkpoint['rng'], device)
    elif latest.exists():
        raise FileExistsError(f'{latest} exists; use resume or a new run directory')
    else:
        seed_all(seed)
    write_json(run_dir/'config.json', config)
    write_json(run_dir/'parameters.json', [{'name': n, 'shape': list(p.shape), 'trainable': p.requires_grad}
                                         for n,p in model.named_parameters()])
    def save(path):
        save_checkpoint(path, dict(config=config, model=mutable_state(model), optimizer=optimizer.state_dict(),
                                   rng=rng_state(device), progress=current))
    if not latest.exists():
        save(latest)
    while current['epoch'] < epochs and current['stale'] < patience:
        epoch = current['epoch']
        order, snrs = balanced_schedule(train.n, seed, epoch, manifest)
        if snr_mode == 'nominal':
            snrs[:] = 20
        model.train()
        for cursor in range(current['cursor'], len(order), batch_size):
            tick = time.perf_counter()
            ids, conditions = order[cursor:cursor+batch_size], snrs[cursor:cursor+batch_size]
            optimizer.zero_grad(set_to_none=True)
            any_supervision, weighted_loss = False, 0.
            for m in range(0, len(ids), micro_batch):
                part, condition = ids[m:m+micro_batch], conditions[m:m+micro_batch]
                inputs, truth = train.batch(part, condition, device)
                output = forward(model, inputs, phase)
                metric = masked_trajectory_loss(**output, **truth)
                any_supervision |= not metric['skip_optimizer']
                (metric['loss'] * len(part)/len(ids)).backward()
                weighted_loss += float(metric['loss'].detach()) * len(part)
            if any_supervision:
                graph_norms = [p.grad.detach().norm() for n,p in model.named_parameters() if n.startswith('graph.') and p.grad is not None]
                current['graph_gradient_norm'] = float(torch.linalg.vector_norm(torch.stack(graph_norms))) if graph_norms else 0.
                current['graph_signal_seen'] |= current['graph_gradient_norm'] > 0
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                current['gradient_norm'] = float(norm)
                optimizer.step()
                current['optimizer_steps'] += 1
                current['step_seconds'].append(time.perf_counter()-tick)
            current.update(cursor=cursor+len(ids), loss_sum=current['loss_sum']+weighted_loss,
                           seen=current['seen']+len(ids), elapsed_seconds=current['elapsed_seconds']+time.perf_counter()-tick)
            save(latest)
            print(f'[{run_dir.name}] {phase} epoch {epoch+1}/{epochs} origins {current["cursor"]}/{len(order)} loss {weighted_loss/len(ids):.6f}', flush=True)
        tick = time.perf_counter()
        metrics = evaluate(model, validation, device, micro_batch, phase=phase)
        current['elapsed_seconds'] += time.perf_counter()-tick
        write_json(run_dir/f'validation_epoch_{epoch+1:02d}.json', metrics)
        score = metrics['J']
        improved = current['best_J'] is None or score < current['best_J']
        current['stale'] = 0 if improved else current['stale']+1
        if improved:
            current['best_J'] = score
        current['history'].append(dict(epoch=epoch+1, loss=current['loss_sum']/max(1,current['seen']),
                                       ADE=metrics['ADE'], FDE=metrics['FDE'], J=score,
                                       gradient_norm=current['gradient_norm'], graph_gradient_norm=current['graph_gradient_norm'], elapsed_seconds=current['elapsed_seconds']))
        current.update(epoch=epoch+1, cursor=0, loss_sum=0., seen=0)
        if improved:
            save(best)
        save(latest)
        write_json(run_dir/'history.json', current['history'])
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    load_weights(model, checkpoint['model'])
    best_row = min(current['history'], key=lambda r:r['J'])
    summary = dict(status='complete', best_checkpoint=str(best), best_metrics=best_row, history=current['history'],
                   optimizer_steps=current['optimizer_steps'], elapsed_seconds=current['elapsed_seconds'],
                   stopped_early=current['stale'] >= patience,
                   sufficiency_unresolved=current['epoch'] == epochs and current['stale'] == 0,
                   training_origins=len(manifest), phase=phase,
                   measured_step_seconds=statistics.median(current['step_seconds']) if current['step_seconds'] else None,
                   parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   numerical_gate=current['optimizer_steps'] > 0 and (phase == 'warmup' or current['graph_signal_seen']),
                   graph_gradient_norm=current['graph_gradient_norm'])
    write_json(run_dir/'summary.json', summary)
    return summary


def load_best(model, summary):
    checkpoint = torch.load(summary['best_checkpoint'], map_location='cpu', weights_only=False)
    load_weights(model, checkpoint['model'])
    return model


def copy_own_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = {k:v for k,v in checkpoint['model'].items() if k.startswith('temporal.') and 'graph_projection.' not in k}
    target = {k:v for k,v in mutable_state(model).items() if k.startswith('temporal.') and 'graph_projection.' not in k}
    if set(state) != set(target):
        raise ValueError('Own-vehicle temporal checkpoint has incompatible keys')
    model.load_state_dict(state, strict=False)


def fit(model, train, validation, run_dir, **kwargs):
    """Own one phase directory exclusively, including resume and checkpoint writes."""
    import fcntl
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Training is already active in {run_dir}; duplicate resume refused') from exc
        try:
            return _fit_impl(model, train, validation, run_dir, **kwargs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
