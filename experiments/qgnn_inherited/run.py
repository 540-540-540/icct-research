"""Prepare by default; --run is the user's explicit 100-epoch launch."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import statistics
import tempfile
from datetime import timedelta

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from experiments.qgat_factorial import run_trial as shared
from experiments.qgnn_inherited.graph import InheritedQGNNGraph

training = shared.training
REPORT = ROOT / 'reports/qgnn_inherited100'
RESULTS = ROOT / 'results/qgnn_inherited100/seed2026/qgnn'
BASE_OPTIMIZER = shared.ORIGINAL_OPTIMIZER


def configuration():
    cfg = json.loads((ROOT / 'configs/qgnn_inherited100.json').read_text(encoding='utf-8-sig'))
    fixed = dict(seed=2026, graph_seed_offset=200000, temporal_seed_offset=100000,
                 epochs=100, patience=101, train_origins=5549, validation_origins=400,
                 batch_size=16, micro_batch=16, validation_micro_batch=16, graph_lr=.0003,
                 snr_db=[5, 10, 15, 20], device='cuda:0', devices=['cuda:0', 'cuda:1'],
                 per_device_batch=8, cuda_rng_offsets=[0, 1000000], allowed_splits=['train', 'V_select'],
                 baseline='results/qgat_extend100/seed2026/gnn', expected_optimizer_steps=34700)
    for key, value in fixed.items():
        if cfg[key] != value:
            raise ValueError(f'Frozen experiment setting changed: {key}')
    return cfg


class Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        graph_seed = cfg['seed'] + cfg['graph_seed_offset']
        training.seed_all(graph_seed)
        self.graph = InheritedQGNNGraph(seed=graph_seed)
        training.seed_all(cfg['seed'] + cfg['temporal_seed_offset'])
        self.temporal = shared.TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        args = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*args, self.graph(*args))


def optimizer(model, phase, graph_lr):
    opt = BASE_OPTIMIZER(model, phase, graph_lr)
    cores = {id(layer.core.weights) for layer in model.graph.graph_layers}
    for group in list(opt.param_groups):
        quantum = [p for p in group['params'] if id(p) in cores]
        if quantum and group['weight_decay']:
            group['params'] = [p for p in group['params'] if id(p) not in cores]
            opt.add_param_group(dict(params=quantum, lr=group['lr'], weight_decay=0.))
    return opt


def source_file_maps(obj, field='files'):
    if isinstance(obj, dict):
        if isinstance(obj.get(field), dict):
            yield obj[field]
        for value in obj.values():
            yield from source_file_maps(value, field)


def prepare(cfg):
    """Check current dependencies against the completed GNN's sealed protocol."""
    baseline = ROOT / cfg['baseline']
    old = json.loads((baseline / 'config.json').read_text())
    summary = json.loads((baseline / 'summary.json').read_text())
    if summary['status'] != 'complete' or summary['optimizer_steps'] != 34700:
        raise ValueError('The reused GNN must have completed all 100 epochs')
    if [r['epoch'] for r in summary['history']] != list(range(1, 101)):
        raise ValueError('Incomplete GNN epoch history')
    for key in ('seed', 'epochs', 'patience', 'batch_size', 'graph_lr'):
        if old[key] != cfg[key]:
            raise ValueError(f'GNN schedule mismatch: {key}')
    if old['phase'] != 'joint' or old['snr_mode'] != 'balanced':
        raise ValueError('GNN phase/SNR mismatch')
    files = ['prediction/training.py', 'prediction/temporal.py', 'prediction/classical.py',
             'prediction/evaluation_cache.py', 'frontend/symbol_dataset.py', 'frontend/echo_source.py',
             'code/00_remote_shared_dependencies/target_interaction_graph.py',
             'models/gpt2/config.json', 'models/gpt2/model.safetensors', 'data/f01d/normalization.json']
    for split in cfg['allowed_splits']:
        files.append(f'data/f01d/labels/{split}.npz')
        files.extend(f'data/f01d/inputs/{split}_snr_{s}.npz' for s in cfg['snr_db'])
    frozen = {}
    for mapping in source_file_maps(old['contract']):
        for key in files:
            if key in mapping:
                if key in frozen and frozen[key] != mapping[key]:
                    raise ValueError(f'Ambiguous baseline dependency: {key}')
                frozen[key] = mapping[key]
    current = {key: shared.digest(ROOT / key) for key in files}
    if current != frozen:
        raise ValueError('Current shared code/data differs from the completed GNN protocol')
    train, valid = shared.dataset('train'), shared.dataset('V_select')
    if train.n != cfg['train_origins'] or valid.n != cfg['validation_origins']:
        raise ValueError('Origin count mismatch')
    if old['indices'] != list(range(train.n)) or old['input_hashes'] != train.input_hashes:
        raise ValueError('Training origins/input provenance mismatch')
    if old['validation_hashes'] != dict(valid.input_hashes, selected_origins=list(range(valid.n))):
        raise ValueError('Validation origins/input provenance mismatch')
    own = ['experiments/qgnn_inherited/graph.py', 'experiments/qgnn_inherited/run.py',
           'experiments/qgnn_inherited/check_graph.py', 'configs/qgnn_inherited100.json',
           'code/02_pennylane_qgnn_core/physics_aligned_dual_model.py',
           'experiments/qgnn_inherited/distributed.py',
           'experiments/qgat_factorial/run_trial.py']
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pennylane', 'transformers')}
    recorded_versions = list(source_file_maps(old['contract'], 'versions'))
    if not recorded_versions or any(record != versions for record in recorded_versions):
        raise ValueError('Runtime versions differ from the reused GNN protocol')
    source = dict(configuration=cfg, shared_files=current,
                  implementation={p: shared.digest(ROOT / p) for p in own},
                  versions=versions,
                  baseline_summary=shared.digest(baseline / 'summary.json'))
    return train, valid, source


def self_check(cfg, train, valid, source, scratch):
    """Validate and time equal work without performing an optimizer update."""
    import torch.distributed as dist
    from experiments.qgnn_inherited import distributed as runtime
    from experiments.qgnn_inherited.check_graph import run_checks
    rank = dist.get_rank()
    device = f'cuda:{rank}'
    graph_checks = runtime.rank_zero_call(run_checks)
    model = Predictor(cfg)
    from experiments.qgat_candidate.run_formal import Predictor as BaselinePredictor
    reference = BaselinePredictor('gnn', cfg['seed'], cfg)
    left = {k: v for k, v in training.mutable_state(model).items() if k.startswith('temporal.')}
    right = {k: v for k, v in training.mutable_state(reference).items() if k.startswith('temporal.')}
    assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
    del reference, left, right
    model.to(device).eval()
    opt = optimizer(model, 'joint', cfg['graph_lr'])
    for name, p in model.named_parameters():
        if p.requires_grad:
            group, = [g for g in opt.param_groups if any(q is p for q in g['params'])]
            expected_lr = .0003 if name.startswith('graph.') else (.00003 if 'lora_' in name else .0001)
            expected_wd = 0. if p.ndim < 2 or name.endswith('.core.weights') else .01
            assert group['lr'] == expected_lr and group['weight_decay'] == expected_wd
    del opt
    original_loss = training.masked_trajectory_loss
    original_evaluate = training.evaluate
    references = {}
    if rank == 0:
        for length in (16, 13):
            inputs, truth = train.batch(shared.origins(train.n, length), np.resize(cfg['snr_db'], length), device)
            model.zero_grad(set_to_none=True)
            output = model(**inputs)
            original_loss(**output, **truth)['loss'].backward()
            references[length] = ({k: v.detach().cpu() for k, v in output.items()},
                                  {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None})
    dist.barrier()
    ids = shared.origins(train.n, 16)
    inputs, truth = train.batch(ids, cfg['snr_db'] * 4, device)
    # Measure single GPU while the other rank is idle, avoiding resource contention.
    single_times = []
    if rank == 0:
        model.train()
        for repeat in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            original_loss(**model(**inputs), **truth)['loss'].backward()
            torch.cuda.synchronize()
            if repeat:
                single_times.append(time.perf_counter() - started)
    dist.barrier()
    validation_subset = shared.Subset(valid, shared.origins(valid.n, 64))
    serial_validation = None
    single_validation_times = []
    if rank == 0:
        for repeat in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            serial_validation = original_evaluate(model, validation_subset, device, micro_batch=16)
            torch.cuda.synchronize()
            if repeat:
                single_validation_times.append(time.perf_counter() - started)
    dist.barrier()
    model.zero_grad(set_to_none=True)
    parallel = runtime.install(model, cfg)
    training.configure_phase = optimizer
    gradient_errors = {}
    for length in (16, 13):
        x, y = train.batch(shared.origins(train.n, length), np.resize(cfg['snr_db'], length), device)
        parallel.eval()
        model.zero_grad(set_to_none=True)
        part = runtime.batch_slice(length)
        output = parallel(**{k: v[part] for k, v in x.items()})
        training.masked_trajectory_loss(**output, **y)['loss'].backward()
        outputs = [None, None]
        dist.all_gather_object(outputs, {k: v.detach().cpu() for k, v in output.items()})
        if rank == 0:
            expected, grads = references[length]
            for key in expected:
                torch.testing.assert_close(torch.cat([v[key] for v in outputs]), expected[key], atol=2e-5, rtol=2e-5)
            error = 0.
            for name, p in model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(p.grad.cpu(), grads[name], atol=3e-5, rtol=3e-4, msg=name)
                    error = max(error, float((p.grad.cpu() - grads[name]).abs().max()))
            gradient_errors[str(length)] = error
    model.zero_grad(set_to_none=True)
    before = training.mutable_state(model)
    torch.cuda.reset_peak_memory_stats(rank)
    model.train()
    training.seed_all(cfg['seed'])
    rng = training.rng_state(device)
    output = training.forward(model, inputs)
    metric = training.masked_trajectory_loss(**output, **truth)
    assert not metric['skip_optimizer'] and torch.isfinite(metric['loss'])
    metric['loss'].backward()
    norms = [float(layer.core.weights.grad.norm()) for layer in model.graph.graph_layers]
    assert all(np.isfinite(v) and v > 0 for v in norms)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    training.restore_rng(rng, device)
    with torch.no_grad():
        replay = training.forward(model, inputs)
    for key in output:
        torch.testing.assert_close(output[key], replay[key], atol=0., rtol=0.)
    measured = []
    for repeat in range(4):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        training.masked_trajectory_loss(**training.forward(model, inputs), **truth)['loss'].backward()
        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        if repeat:
            measured.append(float(seconds))
    assert all(torch.equal(before[k], v) for k, v in training.mutable_state(model).items())
    validation_times = []
    for repeat in range(3):
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        parallel_validation = training.evaluate(model, validation_subset, device, micro_batch=16)
        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        if repeat:
            validation_times.append(float(seconds))
    if rank == 0:
        for key in ('ADE', 'FDE', 'J'):
            assert abs(serial_validation[key] - parallel_validation[key]) < 1e-6
        ordered = lambda result: sorted(result['per_scene'], key=lambda r: (r['snr_db'], r['origin']))
        assert len(serial_validation['per_scene']) == len(parallel_validation['per_scene']) == 256
        for a, b in zip(ordered(serial_validation), ordered(parallel_validation)):
            for key in ('origin', 'snr_db', 'ade_targets', 'fde_targets'):
                assert a[key] == b[key]
            for key in ('ADE', 'FDE'):
                assert (a[key] is None and b[key] is None) or abs(a[key] - b[key]) < 1e-6
    saved = dict(model=training.mutable_state(model), optimizer=optimizer(model, 'joint', cfg['graph_lr']).state_dict(),
                 rng=training.rng_state(device), optimizer_steps=0)
    training.save_checkpoint(scratch / 'zero-update.pt', saved)
    restored = torch.load(scratch / 'zero-update.pt', map_location='cpu', weights_only=False)
    assert restored['optimizer_steps'] == 0 and not restored['optimizer']['state']
    assert all(torch.equal(restored['model'][k], v) for k, v in saved['model'].items())
    training.restore_rng(restored['rng'], device)
    visits = np.zeros((train.n, 4), dtype=int)
    for epoch in range(100):
        ids, snrs = training.balanced_schedule(train.n, cfg['seed'], epoch)
        assert len(np.unique(ids)) == train.n
        for j, snr in enumerate(training.SNRS):
            visits[ids[snrs == snr], j] += 1
    assert np.all(visits == 25)
    memory = [None, None]
    dist.all_gather_object(memory, torch.cuda.max_memory_allocated(rank))
    timing = dict(single_gpu=statistics.median(single_times) if rank == 0 else None,
                  two_gpu=statistics.median(measured))
    result = dict(passed=True, source=source, graph_checks=graph_checks,
                  temporal_initialization_matches_GNN=True, quantum_gradient_norms=norms,
                  peak_memory_bytes=memory, checked_global_batch=16, per_gpu_batch=8,
                  gradient_errors=gradient_errors, both_process_rng_states_replay_exactly=True,
                  median_forward_backward_seconds=timing,
                  measured_two_gpu_speedup=timing['single_gpu']/timing['two_gpu'] if rank == 0 else None,
                  validation_64_origin_seconds=dict(single_gpu=statistics.median(single_validation_times) if rank == 0 else None,
                                                    two_gpu=statistics.median(validation_times)),
                  validation_parallel_equivalence=True, zero_update_checkpoint_roundtrip=True,
                  optimizer_steps_executed=0, parameters_unchanged=True,
                  confirmation_opened=False, test_opened=False)
    training.write_json(REPORT / 'checks.json', result)
    if rank == 0:
        print(json.dumps({k: v for k, v in result.items() if k != 'source'}), flush=True)


def compare(cfg, summary):
    baseline = ROOT / cfg['baseline']
    old = json.loads((baseline / 'summary.json').read_text())
    comparison = {}
    for label, a, b in [('best_J', summary['best_metrics'], old['best_metrics']),
                        ('epoch100', summary['history'][-1], old['history'][-1])]:
        qa = json.loads((RESULTS / f"validation_epoch_{a['epoch']:02d}.json").read_text())
        gb = json.loads((baseline / f"validation_epoch_{b['epoch']:02d}.json").read_text())
        for row, validation in ((a, qa), (b, gb)):
            if any(row[key] != validation[key] for key in ('ADE', 'FDE', 'J')):
                raise ValueError('Summary metrics do not match the selected validation checkpoint')
        coverage = lambda rows: sorted((r['origin'], r['snr_db'], r['ade_targets'], r['fde_targets']) for r in rows)
        if coverage(qa['per_scene']) != coverage(gb['per_scene']):
            raise ValueError('Comparison evaluation coverage differs')
        comparison[label] = dict(qgnn=a, gnn=b, relative_improvement_percent={
            key: 100 * (b[key] - a[key]) / b[key] for key in ('ADE', 'FDE', 'J')})
    training.write_json(REPORT / 'comparison.json', dict(comparison=comparison, limitation=cfg['limitation']))


def distributed_entry(rank, rendezvous, cfg, source, check, resume):
    import torch.distributed as dist
    from experiments.qgnn_inherited import distributed as runtime
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=5), device_id=torch.device(f'cuda:{rank}'))
    try:
        train, valid, actual = prepare(cfg)
        if actual != source:
            raise ValueError('Sources changed between preparation and worker startup')
        if check:
            from urllib.parse import urlparse
            self_check(cfg, train, valid, source, Path(urlparse(rendezvous).path).parent)
            return
        if rank:
            sys.stdout = open(os.devnull, 'w')
        training.configure_phase = optimizer
        model = Predictor(cfg).to(f'cuda:{rank}')
        optimizer(model, 'joint', cfg['graph_lr'])
        runtime.install(model, cfg)
        contract = dict(source=source, mode='inherited-qgnn-100', confirmation_opened=False, test_opened=False)
        training.write_json(REPORT / 'protocol.json', contract)
        summary = training.fit(model, train, valid, RESULTS, seed=cfg['seed'], epochs=100, patience=101,
                               graph_lr=cfg['graph_lr'], phase='joint', batch_size=16,
                               micro_batch=cfg['micro_batch'], snr_mode='balanced',
                               resume=resume, contract=contract)
        if (summary['optimizer_steps'] != cfg['expected_optimizer_steps']
                or [r['epoch'] for r in summary['history']] != list(range(1, 101))
                or not summary['numerical_gate']):
            raise ValueError('Training did not complete the required 100 epochs/34700 updates')
        compare(cfg, summary)
        if rank == 0:
            print(json.dumps(dict(status='complete', comparison=str(REPORT/'comparison.json'))), flush=True)
    finally:
        dist.destroy_process_group()


def launch(cfg, source, check=False, resume=False):
    import fcntl
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = ROOT / '.codex-work/qgnn-inherited-runtime'
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            rendezvous = Path(folder, 'rendezvous').as_uri()
            torch.multiprocessing.spawn(distributed_entry, args=(rendezvous, cfg, source, check, resume),
                                        nprocs=2, join=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--self-check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = configuration()
    if torch.cuda.device_count() < 2:
        raise RuntimeError('This manually launched QGNN experiment requires both GPUs visible')
    train, valid, source = prepare(cfg)
    del train, valid
    if args.self_check:
        launch(cfg, source, check=True)
        return
    if not args.run:
        checks = json.loads((REPORT/'checks.json').read_text()) if (REPORT/'checks.json').exists() else {}
        print(json.dumps(dict(ready_for_manual_launch=checks.get('passed', False) and checks.get('source') == source,
                              formal_training_started=False, optimizer_steps_executed=0,
                              existing_checkpoint=(RESULTS/'latest.pt').exists())), flush=True)
        return
    checks = json.loads((REPORT / 'checks.json').read_text())
    if not checks['passed'] or checks['source'] != source:
        raise ValueError('Checks missing or stale: rerun --self-check before manual launch')
    if args.resume and not (RESULTS / 'latest.pt').is_file():
        raise FileNotFoundError('No QGNN checkpoint to resume')
    if not args.resume and (RESULTS / 'latest.pt').exists():
        raise FileExistsError('QGNN checkpoint exists; explicitly use --run --resume')
    contract = dict(source=source, mode='inherited-qgnn-100', confirmation_opened=False, test_opened=False)
    if (REPORT / 'protocol.json').exists() and json.loads((REPORT / 'protocol.json').read_text()) != contract:
        raise ValueError('Existing QGNN protocol differs; refusing to overwrite it')
    launch(cfg, source, resume=args.resume)


if __name__ == '__main__':
    main()
