"""User-launched, fresh 100-epoch comparison of score-only and score+value QGAT."""
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

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation import run as paired
from experiments.qgat_relation.graph import RelationGraph
from experiments.qgat_value.graph import RelationValueGraph

trial, training = paired.trial, paired.training
CELLS = ('relation_D', 'relation_value')
REPORT = ROOT/'reports/qgat_value'
RESULTS = ROOT/'results/qgat_value/seed2026'
TEMP = ROOT/'.codex-work/qgat-value-prep-20260913'
BASE_EVALUATE = training.evaluate


def configuration():
    cfg = json.loads((ROOT/'configs/qgat_value.json').read_text(encoding='utf-8-sig'))
    expected = dict(seed=2026, cells=list(CELLS), devices=dict(zip(CELLS, ('cuda:0', 'cuda:1'))),
                    epochs=100, patience=101, train_origins=5549, validation_origins=400,
                    snr_db=[5, 10, 15, 20], allowed_splits=['train', 'V_select'],
                    batch_size=16, micro_batch=8, validation_micro_batch=128, graph_lr=.0003,
                    graph_seed_offset=200000, temporal_seed_offset=100000,
                    theta_weight_decay=0., edge_weight_decay=.01, model_precision='float32',
                    quantum_precision='complex128', AMP=False, TF32=False, torch_compile=False,
                    cpu_threads=1, gradient_clip_norm=1., checkpoint_every_update=True,
                    expected_optimizer_steps=34700, smoke_train_origins=17, smoke_validation_origins=129)
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'Value comparison setting changed: {key}')
    return cfg


class Predictor(nn.Module):
    def __init__(self, cell, cfg):
        super().__init__()
        if cell not in CELLS:
            raise ValueError('Unknown paired model')
        training.seed_all(cfg['seed'] + cfg['graph_seed_offset'])
        self.graph = RelationGraph() if cell == 'relation_D' else RelationValueGraph()
        training.seed_all(cfg['seed'] + cfg['temporal_seed_offset'])
        self.temporal = trial.TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        args = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*args, self.graph(*args))


def provenance(cfg):
    code = ['experiments/qgat_value/run.py', 'experiments/qgat_value/graph.py',
            'experiments/qgat_value/check_graph.py', 'experiments/qgat_value/check_training.py',
            'experiments/qgat_relation/run.py', 'experiments/qgat_relation/graph.py',
            'experiments/qgat_factorial/run_trial.py', 'experiments/qgat_factorial/graph.py',
            'experiments/qgat_candidate/graph.py', 'prediction/training.py', 'prediction/temporal.py',
            'prediction/evaluation_cache.py', 'prediction/classical.py', 'frontend/symbol_dataset.py',
            'frontend/echo_source.py', 'code/00_remote_shared_dependencies/target_interaction_graph.py']
    assets = ['models/gpt2/config.json', 'models/gpt2/model.safetensors', 'data/f01d/normalization.json']
    for split in ('train', 'V_select'):
        assets.append(f'data/f01d/labels/{split}.npz')
        assets.extend(f'data/f01d/inputs/{split}_snr_{s}.npz' for s in training.SNRS)
    records = {}
    for relative in assets:
        info = (ROOT/relative).stat()
        records[relative] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    return dict(configuration=cfg, code_text={p:(ROOT/p).read_text() for p in code},
                asset_modification_records=records,
                versions={p:importlib.metadata.version(p) for p in ('torch', 'numpy', 'pennylane', 'transformers')},
                asset_record_scope='Size/mtime modification detection, not content identity. Existing Dataset input hashes and strict checkpoint contracts remain active.')


def graph_check_current():
    path = REPORT/'graph_checks.json'
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    sources = report.get('graph_sources', {})
    return bool(report.get('passed') and sources and
                all((ROOT/p).read_text() == text for p, text in sources.items()))


def guard_resume(run_dir, resume):
    run_dir = Path(run_dir)
    latest, best = run_dir/'latest.pt', run_dir/'best.pt'
    if not resume:
        if any((run_dir/name).exists() for name in ('latest.pt', 'best.pt', 'history.json', 'config.json')):
            raise FileExistsError('Fresh run already has records; use this run\'s --resume explicitly')
        return None
    if not latest.is_file():
        raise FileNotFoundError('This run has no latest checkpoint; refusing a fresh-start fallback')
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    progress = saved['progress']
    if (progress['history'] or progress['best_J'] is not None or progress['epoch'] > 0) and not best.is_file():
        raise FileNotFoundError('A validated run must retain its best checkpoint')
    if best.is_file():
        chosen = torch.load(best, map_location='cpu', weights_only=False)
        if chosen['config'] != saved['config']:
            raise ValueError('Best/latest checkpoint configurations differ')
    return progress


class SmokeInterrupted(Exception):
    pass


def worker(args, cfg, source, run_root, report):
    cell = args.worker
    physical = cfg['devices'][cell].split(':')[1]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != physical or torch.cuda.device_count() != 1:
        raise ValueError('Each worker must see only its assigned physical GPU')
    torch.cuda.set_device(0)
    contract = json.loads((report/'protocol.json').read_text())
    if contract['source'] != source:
        raise ValueError('Worker source contract mismatch')
    path, run_dir = report/'jobs'/f'{cell}.json', run_root/cell
    status = dict(status='initializing', cell=cell, pid=os.getpid(), device=cfg['devices'][cell],
                  logical_device='cuda:0', seed=cfg['seed'], started_at=time.time())
    training.write_json(path, status)
    try:
        starting = guard_resume(run_dir, args.resume)
        initial_steps = starting['optimizer_steps'] if starting else 0
        train, valid = trial.dataset('train'), trial.dataset('V_select')
        ids = paired.selected_origins(train, 17 if args.smoke else 5549, args.smoke)
        val_ids = paired.selected_origins(valid, 129 if args.smoke else 400, args.smoke)
        valid = trial.Subset(valid, val_ids)
        batches, observations = [], dict(count=0, finite=True, max_norm=0.)
        if args.smoke:
            original_batch = valid.batch
            def recorded_batch(ids, *rest, **kwargs):
                batches.append(len(ids))
                return original_batch(ids, *rest, **kwargs)
            valid.batch = recorded_batch
        def evaluate(model, dataset, device, micro_batch=1, **kwargs):
            return BASE_EVALUATE(model, dataset, device, micro_batch=128, **kwargs)
        training.evaluate = evaluate
        training.configure_phase = trial.make_optimizer
        observed_devices = set()
        def make_model():
            model = Predictor(cell, cfg).to('cuda:0')
            model.graph.register_forward_hook(lambda module, inputs, output: observed_devices.add(output.device.index))
            if cell == 'relation_value':
                def observe(gradient):
                    observations['count'] += 1
                    observations['finite'] &= bool(torch.isfinite(gradient).all())
                    observations['max_norm'] = max(observations['max_norm'], float(gradient.norm()))
                model.graph.value_edge_encoder.weight.register_hook(observe)
            return model
        model = make_model()
        expected_epochs, expected_steps = (1, 2) if args.smoke else (100, 34700)
        options = dict(seed=2026, epochs=expected_epochs, patience=101, graph_lr=.0003,
                       phase='joint', batch_size=16, micro_batch=8, indices=ids,
                       snr_mode='balanced', contract=contract)
        status.update(status='training', train_origins=ids.tolist(), V_select_origins=val_ids.tolist(),
                      historical_trained_checkpoint_loaded=False)
        training.write_json(path, status)
        resume = args.resume
        if args.smoke and not args.resume:
            # Exercise a real, legal interruption before best.pt exists; only in isolated TEMP.
            save = training.save_checkpoint
            def interrupt_after_first_update(target, payload):
                save(target, payload)
                if Path(target).name == 'latest.pt' and payload['progress']['optimizer_steps'] == 1:
                    raise SmokeInterrupted()
            training.save_checkpoint = interrupt_after_first_update
            try:
                training.fit(model, train, valid, run_dir, resume=False, **options)
                raise AssertionError('Smoke interruption did not fire')
            except SmokeInterrupted:
                progress = guard_resume(run_dir, True)
                assert progress['epoch'] == 0 and progress['optimizer_steps'] == 1
                assert progress['cursor'] == 16 and not (run_dir/'best.pt').exists()
                status['prevalidation_interruption_recovered'] = True
            finally:
                training.save_checkpoint = save
            del model
            model = make_model()
            resume = True
        result = training.fit(model, train, valid, run_dir, resume=resume, **options)
        latest = torch.load(run_dir/'latest.pt', map_location='cpu', weights_only=False)
        assert result['optimizer_steps'] == expected_steps and len(result['history']) == expected_epochs
        assert [r['epoch'] for r in result['history']] == list(range(1, expected_epochs+1))
        assert result['numerical_gate'] and not result['stopped_early']
        assert all(torch.isfinite(v).all() for v in latest['model'].values())
        states = latest['optimizer']['state'].values()
        assert all(int(row['step']) == expected_steps for row in states)
        assert all(torch.isfinite(v).all() for row in states for v in row.values() if isinstance(v, torch.Tensor))
        added = result['optimizer_steps'] - initial_steps
        assert observed_devices <= {0} and (not added or observed_devices == {0})
        if cell == 'relation_value':
            value = latest['model']['graph.value_edge_encoder.weight']
            assert value.count_nonzero() and observations['finite']
            assert not added or observations['max_norm'] > 0
            status['value_gradient_this_process'] = observations
            status['value_latest_changed_elements'] = int(value.count_nonzero())
        if args.smoke and added:
            assert batches == [128, 1]*4, batches
        status.update(status='complete', finished_at=time.time(), training=result,
                      additional_optimizer_steps_this_process=added,
                      observed_logical_cuda_devices=sorted(observed_devices),
                      validation_batches=batches if args.smoke else None,
                      complete_adam_state_steps_verified=True,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        training.write_json(path, status)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        training.write_json(path, status)
        raise


def summarize(states, cfg, run_root, smoke):
    final, best = {}, {}
    for cell, state in states.items():
        epoch = state['training']['history'][-1]['epoch']
        chosen = state['training']['best_metrics']['epoch']
        final[cell] = json.loads((run_root/cell/f'validation_epoch_{epoch:02d}.json').read_text())
        best[cell] = json.loads((run_root/cell/f'validation_epoch_{chosen:02d}.json').read_text())
    for group in (final, best):
        identities = [[(r['origin'], r['snr_db'], r['ade_targets'], r['fde_targets'])
                       for r in group[cell]['per_scene']] for cell in CELLS]
        assert identities[0] == identities[1]
        assert len(identities[0]) == (129 if smoke else 400)*4
    delta = lambda group: {key:group['relation_value'][key]-group['relation_D'][key] for key in ('ADE', 'FDE', 'J')}
    reference = json.loads((ROOT/cfg['historical_gnn_reference']/'summary.json').read_text())
    assert reference['status'] == 'complete' and len(reference['history']) == 100
    return dict(status='complete', smoke_only=smoke, seed=2026, jobs=states,
                final_epoch_metrics=final, best_J_epoch_metrics=best,
                final_epoch_value_minus_current=delta(final), best_J_value_minus_current=delta(best),
                common_denominators_verified=True, paired_fresh_initialization_verified=True,
                historical_gnn_reference=dict(path=cfg['historical_gnn_reference'],
                    best_J_epoch_metrics=reference['best_metrics'], final_epoch_metrics=reference['history'][-1],
                    role='Historical strong reference; different physical batching in its first 40 epochs.'),
                confirmation_opened=False, test_opened=False, limitation=cfg['limitation'])


def coordinate(args, cfg, source, run_root, report):
    report.mkdir(parents=True, exist_ok=True)
    with (report/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = dict(source=source, mode='isolated-smoke' if args.smoke else 'paired-fresh100',
                        confirmation_opened=False, test_opened=False)
        path = report/'protocol.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('This paired run exists; pass --resume explicitly')
            if json.loads(path.read_text()) != protocol:
                raise ValueError('Paired resume contract mismatch')
        elif args.resume:
            raise FileNotFoundError('No paired run exists to resume')
        else:
            training.write_json(path, protocol)
        active, logs, states = {}, [], {}
        coordinator = dict(status='running', pid=os.getpid(), started_at=time.time(), devices=cfg['devices'])
        training.write_json(report/'coordinator.json', coordinator)
        try:
            for cell in CELLS:
                logfile = report/'logs'/f'{cell}.log'
                logfile.parent.mkdir(parents=True, exist_ok=True)
                log = logfile.open('a', buffering=1)
                logs.append(log)
                command = [sys.executable, '-u', str(Path(__file__).resolve()),
                           '--smoke' if args.smoke else '--run', '--worker', cell]
                if args.resume:
                    command.append('--resume')
                active[cell] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['devices'][cell].split(':')[1]))
                print(f'{cell}: {cfg["devices"][cell]}, {"isolated 2-update check" if args.smoke else "fresh 100 epochs"}', flush=True)
            while active:
                time.sleep(1)
                for cell, child in list(active.items()):
                    if child.poll() is None:
                        continue
                    del active[cell]
                    if child.returncode:
                        tail = (report/'logs'/f'{cell}.log').read_text().splitlines()[-16:]
                        raise RuntimeError(f'{cell} failed: '+'\n'.join(tail))
                    states[cell] = json.loads((report/'jobs'/f'{cell}.json').read_text())
                    assert states[cell]['status'] == 'complete'
            if source != provenance(cfg):
                raise ValueError('Source or asset modification record changed during the run')
            summary = summarize(states, cfg, run_root, args.smoke)
            training.write_json(report/'summary.json', summary)
            coordinator['status'] = 'complete'
            return summary
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
            training.write_json(report/'coordinator.json', coordinator)
            for log in logs:
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('check-only', 'self-check', 'smoke', 'run'):
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', choices=CELLS)
    args = parser.parse_args()
    if (args.worker or args.resume) and not (args.smoke or args.run):
        parser.error('--worker/--resume requires --smoke or --run')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.self_check:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('check_training.py'))],
                       cwd=ROOT, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        return
    cfg, source = configuration(), None
    source = provenance(cfg)
    read = lambda name: json.loads((REPORT/name).read_text()) if (REPORT/name).is_file() else {}
    checked, smoke_checked = read('checks.json'), read('smoke_checks.json')
    cpu_ok = graph_check_current() and checked.get('passed', False) and checked.get('source') == source
    smoke_ok = (smoke_checked.get('passed', False) and smoke_checked.get('source') == source
                and smoke_checked.get('completed_branch_resume_additional_steps') == dict.fromkeys(CELLS, 0))
    if args.check_only:
        status = dict(ready=cpu_ok and smoke_ok and torch.cuda.device_count()>=2,
                      source_cpu_checks_current=cpu_ok, source_gpu_smoke_current=smoke_ok,
                      cells=list(CELLS), devices=cfg['devices'], epochs=100, start_epoch=0,
                      batch_size=16, micro_batch=8, validation_micro_batch=128,
                      expected_optimizer_steps_per_model=34700, optimizer_steps_executed_by_check=0,
                      fresh_run_available=not (REPORT/'seed2026/protocol.json').exists(),
                      confirmation_opened=False, test_opened=False)
        training.write_json(REPORT/'readiness.json', status)
        print(json.dumps(status), flush=True)
        return
    if not cpu_ok or (args.run and not smoke_ok):
        raise ValueError('Complete current graph, CPU integration, and isolated smoke checks before manual training')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    run_root = TEMP/'results' if args.smoke else RESULTS
    report = TEMP/'reports' if args.smoke else REPORT/'seed2026'
    if args.worker:
        worker(args, cfg, source, run_root, report)
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('The coordinator requires two visible GPUs')
    complete_before_resume = (args.smoke and args.resume and
                             all(guard_resume(run_root/cell, True)['epoch'] >= 1 for cell in CELLS))
    summary = coordinate(args, cfg, source, run_root, report)
    if args.smoke:
        checks = dict(passed=True, source=source, smoke_summary=summary,
                      formal_training_started=False, formal_optimizer_steps_executed=0,
                      isolated_optimizer_steps_per_model=2)
        if args.resume:
            previous = read('smoke_checks.json')
            checks['initial_smoke_summary'] = previous.get('initial_smoke_summary', previous.get('smoke_summary'))
            added = {cell:s['additional_optimizer_steps_this_process'] for cell,s in summary['jobs'].items()}
            if complete_before_resume:
                assert all(n == 0 for n in added.values())
                checks['completed_branch_resume_additional_steps'] = added
            else:
                checks['interrupted_smoke_resume_additional_steps'] = added
        training.write_json(REPORT/'smoke_checks.json', checks)
        print(json.dumps({k:v for k,v in checks.items() if k not in ('source', 'smoke_summary', 'initial_smoke_summary')}), flush=True)


if __name__ == '__main__':
    main()
