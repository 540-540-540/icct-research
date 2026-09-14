"""Paired original D / relation D experiment; use --run explicitly for 40 epochs."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_factorial import run_trial as trial
from experiments.qgat_factorial.graph import FactorialGraph
from experiments.qgat_relation.graph import RelationGraph

training = trial.training
CELLS = ('original_D', 'relation_D')
REPORT = ROOT / 'reports/qgat_relation'
TEMP = ROOT / '.codex-work/qgat-relation-prep-20260913'


def configuration():
    cfg = json.loads((ROOT / 'configs/qgat_relation.json').read_text(encoding='utf-8-sig'))
    expected = dict(seed=2026, cells=list(CELLS), devices=dict(zip(CELLS, ('cuda:0', 'cuda:1'))),
                    train_origins=5549, validation_origins=400, epochs=40, patience=41,
                    batch_size=16, micro_batch=1, graph_lr=3e-4, graph_seed_offset=200000,
                    temporal_seed_offset=100000, snr_db=list(training.SNRS),
                    allowed_splits=['train', 'V_select'], edge_weight_decay=.01,
                    theta_weight_decay=0., smoke_train_origins=17, smoke_validation_origins=2)
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'Paired relation contract changed: {key}')
    return cfg


class Predictor(nn.Module):
    def __init__(self, cell, cfg):
        super().__init__()
        if cell not in CELLS:
            raise ValueError('Unsupported model')
        training.seed_all(cfg['seed'] + cfg['graph_seed_offset'])
        self.graph = FactorialGraph(2, 45) if cell == 'original_D' else RelationGraph()
        training.seed_all(cfg['seed'] + cfg['temporal_seed_offset'])
        self.temporal = trial.TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        args = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*args, self.graph(*args))


def provenance(cfg):
    files = ['experiments/qgat_relation/run.py', 'experiments/qgat_relation/graph.py',
             'configs/qgat_relation.json', 'experiments/qgat_factorial/run_trial.py',
             'experiments/qgat_factorial/graph.py', 'experiments/qgat_candidate/graph.py',
             'prediction/training.py', 'prediction/temporal.py', 'prediction/classical.py',
             'prediction/evaluation_cache.py', 'code/00_remote_shared_dependencies/target_interaction_graph.py',
             'frontend/symbol_dataset.py', 'frontend/echo_source.py',
             'models/gpt2/config.json', 'models/gpt2/model.safetensors', 'data/f01d/normalization.json']
    for split in ('train', 'V_select'):
        files.append(f'data/f01d/labels/{split}.npz')
        files.extend(f'data/f01d/inputs/{split}_snr_{s}.npz' for s in training.SNRS)
    return dict(configuration=cfg, files={p: trial.digest(ROOT / p) for p in files},
                versions={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pennylane', 'transformers')})


def selected_origins(data, count, smoke):
    if smoke:
        return trial.origins(data.n, count)
    if data.n != count:
        raise ValueError(f'Expected all {count} {data.split} origins; found {data.n}')
    return np.arange(data.n, dtype=np.int64)


def self_check(cfg):
    torch.set_num_threads(1)
    train, valid = trial.dataset('train'), trial.dataset('V_select')
    ids = selected_origins(train, cfg['train_origins'], False)
    selected_origins(valid, cfg['validation_origins'], False)
    graphs, temporal, predictions, counts = {}, {}, {}, {}
    inputs, _ = train.batch([0], [20], 'cpu')
    for cell in CELLS:
        model = Predictor(cell, cfg).eval()
        graphs[cell] = {k: v.detach().clone() for k, v in model.graph.state_dict().items()}
        temporal[cell] = {k: v for k, v in training.mutable_state(model).items() if k.startswith('temporal.')}
        opt = trial.make_optimizer(model, 'joint', cfg['graph_lr'])
        assert all(g['weight_decay'] == 0 for g in opt.param_groups if any(p is model.graph.theta for p in g['params']))
        if cell == 'relation_D':
            groups = [g for g in opt.param_groups if any(p is model.graph.edge_encoder.weight for p in g['params'])]
            assert len(groups) == 1 and groups[0]['lr'] == cfg['graph_lr'] and groups[0]['weight_decay'] == .01
        counts[cell] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            predictions[cell] = model(**inputs)
        del model, opt
    original, relation = (graphs[c] for c in CELLS)
    assert set(relation) - set(original) == {'edge_encoder.weight'}
    assert all(torch.equal(v, relation[k]) for k, v in original.items())
    assert relation['edge_encoder.weight'].shape == (4, 7) and not relation['edge_encoder.weight'].count_nonzero()
    assert temporal[CELLS[0]].keys() == temporal[CELLS[1]].keys()
    assert all(torch.equal(v, temporal[CELLS[1]][k]) for k, v in temporal[CELLS[0]].items())
    assert counts['relation_D'] - counts['original_D'] == 28
    for key, value in predictions['original_D'].items():
        torch.testing.assert_close(value, predictions['relation_D'][key], atol=2e-6, rtol=2e-6)
    exposure = np.zeros((train.n, 4), int)
    for epoch in range(cfg['epochs']):
        order, snrs = training.balanced_schedule(train.n, cfg['seed'], epoch, ids)
        assert len(np.unique(order)) == len(ids) and set(order) == set(ids)
        amounts = [int((snrs == s).sum()) for s in training.SNRS]
        assert max(amounts) - min(amounts) <= 1
        for j, s in enumerate(training.SNRS):
            exposure[order[snrs == s], j] += 1
        batches = [order[i:i+16] for i in range(0, len(order), 16)]
        assert len(batches) == 347 and len(batches[-1]) == 13
    assert np.all(exposure == 10)
    for split in ('V_confirm', 'test'):
        try:
            trial.dataset(split)
        except ValueError:
            pass
        else:
            raise AssertionError('Forbidden split admitted')
    result = dict(passed=True, source=provenance(cfg), trainable_parameters=counts,
                  checks=['shared graph tensors identical', 'temporal tensors identical',
                          '28 edge weights initially zero', 'zero-edge end-to-end real-input predictions match',
                          'theta decay zero; edge decay .01 and lr .0003',
                          '40 complete origin traversals, each origin sees each SNR 10 times',
                          '347 batches per epoch, tail 13', 'confirmation/test rejected before dataset access'],
                  expected_steps_per_cell=13880, optimizer_steps_executed=0)
    training.write_json(REPORT / 'checks.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source'}), flush=True)


def check_only(cfg):
    train, valid = trial.dataset('train'), trial.dataset('V_select')
    selected_origins(train, 5549, False)
    selected_origins(valid, 400, False)
    source = provenance(cfg)
    path = REPORT / 'seed2026/protocol.json'
    if path.exists() and json.loads(path.read_text())['source'] != source:
        raise ValueError('Existing run source contract mismatch')
    check_path = REPORT / 'checks.json'
    checked = json.loads(check_path.read_text()) if check_path.exists() else {}
    result = dict(ready=torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                  source_self_check_current=checked.get('passed', False) and checked.get('source') == source,
                  seed=2026, cells=list(CELLS), epochs=40, patience=41,
                  train_origins=train.n, V_select_origins=valid.n, steps_per_epoch=347,
                  last_batch=13, expected_steps_per_cell=13880, optimizer_steps_executed=0,
                  confirmation_opened=False, test_opened=False)
    training.write_json(REPORT / 'readiness.json', result)
    print(json.dumps(result), flush=True)


def worker(args, cfg, run_dir, report_dir):
    cell = args.worker
    path = report_dir / 'jobs' / f'{cell}.json'
    status = dict(cell=cell, status='initializing', pid=os.getpid(), device=cfg['devices'][cell], started_at=time.time())
    training.write_json(path, status)
    try:
        contract = json.loads((report_dir / 'protocol.json').read_text())
        if contract['source'] != provenance(cfg):
            raise ValueError('Source contract changed')
        train, valid = trial.dataset('train'), trial.dataset('V_select')
        train_ids = selected_origins(train, cfg['smoke_train_origins'] if args.smoke else 5549, args.smoke)
        val_ids = selected_origins(valid, cfg['smoke_validation_origins'] if args.smoke else 400, args.smoke)
        valid = trial.Subset(valid, val_ids)
        model = Predictor(cell, cfg).to(cfg['devices'][cell])
        state = training.mutable_state(model)
        status.update(status='training',
                      initial_temporal_sha256=trial.fingerprint({k: v for k, v in state.items() if k.startswith('temporal.')}),
                      initial_common_graph_sha256=trial.fingerprint({k: v for k, v in model.graph.state_dict().items() if not k.startswith('edge_encoder.')}),
                      train_origins=train_ids.tolist(), V_select_origins=val_ids.tolist())
        del state
        initial_theta = model.graph.theta.detach().cpu().clone()
        edge_gradient = dict(observations=0, max_norm=0., all_finite=True)
        def observe_edge(gradient):
            edge_gradient['observations'] += 1
            edge_gradient['all_finite'] &= bool(torch.isfinite(gradient).all())
            edge_gradient['max_norm'] = max(edge_gradient['max_norm'], float(gradient.norm()))
        if cell == 'relation_D':
            model.graph.edge_encoder.weight.register_hook(observe_edge)
        training.write_json(path, status)
        training.configure_phase = trial.make_optimizer
        started = time.perf_counter()
        result = training.fit(model, train, valid, run_dir / cell, seed=2026,
                              epochs=1 if args.smoke else 40, patience=41, graph_lr=cfg['graph_lr'],
                              phase='joint', batch_size=16, micro_batch=1, indices=train_ids,
                              snr_mode='balanced', resume=args.resume, contract=contract)
        if not result['numerical_gate']:
            raise RuntimeError('Numerical or graph gradient gate failed')
        expected_steps = 2 if args.smoke else 13880
        if result['optimizer_steps'] != expected_steps or len(result['history']) != (1 if args.smoke else 40):
            raise RuntimeError('Incomplete fixed-epoch optimizer schedule')
        if cell == 'relation_D':
            weight = model.graph.edge_encoder.weight.detach().cpu()
            status.update(edge_gradient_this_process=edge_gradient, edge_best_maxabsdelta=float(weight.abs().max()),
                          edge_best_changed_elements=int(weight.count_nonzero()))
            if not torch.isfinite(weight).all() or not weight.count_nonzero() or not edge_gradient['all_finite']:
                raise RuntimeError('Edge parameter learning gate failed')
            if not args.resume and not edge_gradient['max_norm'] > 0:
                raise RuntimeError('New edge has no training gradient')
        status.update(status='complete', wall_seconds=time.perf_counter()-started, finished_at=time.time(), training=result,
                      best_sha256=trial.digest(run_dir / cell / 'best.pt'), latest_sha256=trial.digest(run_dir / cell / 'latest.pt'),
                      theta_best_maxabsdelta=float((model.graph.theta.detach().cpu()-initial_theta).abs().max()))
        training.write_json(path, status)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        training.write_json(path, status)
        raise


def summarize(states, cfg, run_dir, smoke):
    epoch = 1 if smoke else 40
    final, best, minima, trends = {}, {}, {}, {}
    for cell, state in states.items():
        history = state['training']['history']
        if len(history) != epoch or history[-1]['epoch'] != epoch:
            raise ValueError('Incomplete common final epoch')
        best_epoch = state['training']['best_metrics']['epoch']
        final[cell] = json.loads((run_dir / cell / f'validation_epoch_{epoch:02d}.json').read_text())
        best[cell] = json.loads((run_dir / cell / f'validation_epoch_{best_epoch:02d}.json').read_text())
        minima[cell] = {metric: dict(row := min(history, key=lambda r: r[metric]),
                                    weights_retained=row['epoch'] in (best_epoch, epoch)) for metric in ('ADE', 'FDE', 'J')}
        trends[cell] = dict(last_J_change=history[-1]['J']-history[-2]['J'] if epoch > 1 else None,
                            last_five_epochs=history[-5:], final_epoch_is_best_J=best_epoch == epoch,
                            convergence_established=False)
        for scores in (final[cell], best[cell]):
            for row in scores['per_scene']:
                row['source_origin'] = state['V_select_origins'][row['origin']]
    for scores in (final, best):
        denominator = lambda c: [(r['snr_db'], r['source_origin'], r['ade_targets'], r['fde_targets']) for r in scores[c]['per_scene']]
        assert denominator(CELLS[0]) == denominator(CELLS[1])
    for field in ('initial_temporal_sha256', 'initial_common_graph_sha256'):
        assert len({s[field] for s in states.values()}) == 1
    assert states[CELLS[0]]['train_origins'] == states[CELLS[1]]['train_origins']
    def contrast(scores):
        difference = lambda a, b: {metric: b[metric]-a[metric] for metric in ('ADE', 'FDE', 'J')}
        return dict(relation_minus_original=difference(scores[CELLS[0]], scores[CELLS[1]]),
                    by_snr={str(s): difference(scores[CELLS[0]]['by_snr'][str(s)], scores[CELLS[1]]['by_snr'][str(s)]) for s in training.SNRS})
    return dict(status='complete', smoke_only=smoke, seed=2026, limitation=cfg['limitation'], jobs=states,
                final_epoch=epoch, final_epoch_metrics=final, best_J_epoch_metrics=best,
                individual_metric_minima=minima, final_epoch_contrast=contrast(final), best_J_contrast=contrast(best),
                training_trends=trends, paired_initialization_verified=True, common_denominators_verified=True,
                confirmation_opened=False, test_opened=False, GNN_retrained=False)


def coordinate(args, cfg, run_dir, report_dir):
    source = provenance(cfg)
    checked = json.loads((REPORT / 'checks.json').read_text())
    if not checked['passed'] or checked['source'] != source:
        raise ValueError('Run --self-check on the current source first')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA devices required by paired contract')
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = dict(source=source, mode='smoke-17-tail1' if args.smoke else 'paired-full-data-fixed-40',
                        confirmation_opened=False, test_opened=False)
        path = report_dir / 'protocol.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('Run exists; use --resume explicitly')
            if json.loads(path.read_text()) != protocol:
                raise ValueError('Resume contract mismatch')
        else:
            if args.resume:
                raise FileNotFoundError('No existing run to resume')
            training.write_json(path, protocol)
        coordinator = dict(status='running', pid=os.getpid(), started_at=time.time(), devices=cfg['devices'])
        training.write_json(report_dir / 'coordinator.json', coordinator)
        active, logs, states = {}, [], {}
        try:
            for cell in CELLS:
                status_path = report_dir / 'jobs' / f'{cell}.json'
                if args.resume and status_path.exists():
                    state = json.loads(status_path.read_text())
                    if state.get('status') == 'complete':
                        for name in ('best', 'latest'):
                            if state[f'{name}_sha256'] != trial.digest(run_dir / cell / f'{name}.pt'):
                                raise ValueError('Completed checkpoint changed')
                        states[cell] = state
                        continue
                log_path = report_dir / 'logs' / f'{cell}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('a' if args.resume else 'w', buffering=1)
                logs.append(log)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--smoke' if args.smoke else '--run', '--worker', cell]
                if args.resume:
                    cmd.append('--resume')
                active[cell] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                 env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                print(f'Started {cell} on {cfg["devices"][cell]}: {log_path}', flush=True)
            while active:
                time.sleep(1)
                for cell, child in list(active.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    del active[cell]
                    if code:
                        raise RuntimeError(f'{cell} failed; inspect its log')
                    states[cell] = json.loads((report_dir / 'jobs' / f'{cell}.json').read_text())
                    if states[cell]['status'] != 'complete':
                        raise RuntimeError(f'{cell} did not complete')
                    print(f'Completed {cell}', flush=True)
            if source != provenance(cfg):
                raise ValueError('Source changed during run')
            training.write_json(report_dir / 'summary.json', summarize(states, cfg, run_dir, args.smoke))
            coordinator['status'] = 'complete'
        except BaseException as exc:
            for child in active.values():
                if child.poll() is None:
                    child.terminate()
            for child in active.values():
                child.wait()
            coordinator.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            coordinator['finished_at'] = time.time()
            training.write_json(report_dir / 'coordinator.json', coordinator)
            for log in logs:
                log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ('check-only', 'self-check', 'smoke', 'run'):
        mode.add_argument('--' + flag, action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker', choices=CELLS)
    args = p.parse_args()
    if (args.resume or args.worker) and not (args.run or args.smoke):
        p.error('--resume/--worker require --run or --smoke')
    cfg = configuration()
    if args.self_check:
        self_check(cfg)
        return
    if args.check_only:
        check_only(cfg)
        return
    run_dir = TEMP / 'smoke-results' if args.smoke else ROOT / 'results/qgat_relation/seed2026'
    report_dir = TEMP / 'smoke-reports' if args.smoke else REPORT / 'seed2026'
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    if args.worker:
        worker(args, cfg, run_dir, report_dir)
    else:
        coordinate(args, cfg, run_dir, report_dir)


if __name__ == '__main__':
    main()
