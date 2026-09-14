"""Manual continuation to 100 total epochs: one GNN and one new QGAT on separate GPUs."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation import run as paired
from experiments.qgat_relation import run_gnn_fast as gnn_fast

training = paired.training
CELLS = ('gnn', 'relation_D')
REPORT = ROOT/'reports/qgat_extend100'
RESULTS = ROOT/'results/qgat_extend100/seed2026'
TEMP = ROOT/'.codex-work/qgat-extend100-prep-20260913'
BASE_EVALUATE = training.evaluate


def configuration():
    cfg = json.loads((ROOT/'configs/qgat_extend100.json').read_text(encoding='utf-8-sig'))
    expected = dict(seed=2026, start_epoch=40, epochs=100, patience=101,
                    train_origins=5549, validation_origins=400, batch_size=16, graph_lr=.0003,
                    snr_db=[5, 10, 15, 20], allowed_splits=['train', 'V_select'],
                    devices={'gnn': 'cuda:0', 'relation_D': 'cuda:1'},
                    micro_batch={'gnn': 16, 'relation_D': 8},
                    validation_micro_batch={'gnn': 32, 'relation_D': 128},
                    model_precision='float32', quantum_precision='complex128', AMP=False,
                    TF32=False, torch_compile=False, cpu_threads=1, gradient_clip_norm=1.,
                    checkpoint_every_update=True, expected_total_optimizer_steps=34700,
                    expected_additional_optimizer_steps=20820)
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'Continuation setting changed: {key}')
    return cfg


def file_record(path):
    stat = path.stat()
    return dict(path=str(path.relative_to(ROOT)), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def load_parents(cell, cfg, expected_source):
    directory = ROOT/cfg['parent_results'][cell]
    job = json.loads((ROOT/cfg['parent_reports'][cell]).read_text())
    if job['status'] != 'complete':
        raise ValueError(f'{cell}: parent training is not complete')
    latest = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    config, progress = latest['config'], latest['progress']
    if config['contract']['source'] != expected_source or best['config'] != config:
        raise ValueError(f'{cell}: parent source/config mismatch')
    fixed = dict(seed=2026, epochs=40, patience=41, batch_size=16, graph_lr=.0003,
                 phase='joint', snr_mode='balanced')
    if any(config[k] != v for k, v in fixed.items()) or config['indices'] != list(range(5549)):
        raise ValueError(f'{cell}: parent training conditions changed')
    if progress['epoch'] != 40 or progress['cursor'] != 0 or progress['optimizer_steps'] != 13880:
        raise ValueError(f'{cell}: continuation requires the complete epoch-40 latest checkpoint')
    if [r['epoch'] for r in progress['history']] != list(range(1, 41)) or len(progress['step_seconds']) != 13880:
        raise ValueError(f'{cell}: incomplete parent history')
    best_row = min(progress['history'], key=lambda r: r['J'])
    if best['progress']['epoch'] != best_row['epoch'] or progress['best_J'] != best_row['J']:
        raise ValueError(f'{cell}: retained best checkpoint does not match best-J history')
    if cell == 'gnn':
        streams = latest['rng'].get('two_gpu_rng', [])
        if len(streams) != 2 or not torch.equal(latest['rng']['cuda'], streams[0]):
            raise ValueError('GNN parent GPU0 RNG does not match the retained stream')
    elif latest['rng']['original_device'] != 'cuda:1':
        raise ValueError('New QGAT parent must originate on GPU1')
    return latest, best


def provenance(cfg):
    references = dict(gnn=gnn_fast.provenance(gnn_fast.configuration()),
                      relation_D=paired.provenance(paired.configuration()))
    parents = {}
    for cell in CELLS:
        latest, best = load_parents(cell, cfg, references[cell])
        directory = ROOT/cfg['parent_results'][cell]
        parents[cell] = dict(source=references[cell],
                             files={name: file_record(directory/name) for name in ('latest.pt', 'best.pt')},
                             completed_epoch=latest['progress']['epoch'],
                             optimizer_steps=latest['progress']['optimizer_steps'],
                             best_epoch=best['progress']['epoch'])
    return dict(configuration=cfg, parents=parents,
                runner_sha256=paired.trial.digest(Path(__file__).resolve()))


def assert_equal(left, right):
    """Check inherited state directly, including Adam moments and RNG, without hashing copies."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def inherited_payload(parent, fit_config):
    payload = copy.deepcopy(parent)
    payload['config'] = fit_config
    payload['rng'].pop('two_gpu_rng', None)
    payload['rng']['original_device'] = 'cuda:0'  # Each worker sees only its assigned physical GPU.
    for key in ('model', 'optimizer', 'progress'):
        assert_equal(payload[key], parent[key])
    for key in ('python', 'numpy', 'torch', 'cuda'):
        assert_equal(payload['rng'][key], parent['rng'][key])
    return payload


def inherit(run_dir, cell, cfg, source, fit_config):
    if run_dir.exists():
        for name in ('latest.pt', 'best.pt'):
            if not (run_dir/name).is_file():
                raise ValueError(f'{cell}: incomplete continuation directory; refusing a fresh-start fallback')
        saved = torch.load(run_dir/'latest.pt', map_location='cpu', weights_only=False)
        if saved['config'] != fit_config:
            raise ValueError(f'{cell}: continuation checkpoint config mismatch')
        return saved['progress']
    parent_dir = ROOT/cfg['parent_results'][cell]
    latest, best = load_parents(cell, cfg, source['parents'][cell]['source'])
    staging_root = TEMP/'inherit-stage'
    staging_root.mkdir(parents=True, exist_ok=True)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=cell+'-', dir=staging_root) as name:
        stage = Path(name)
        for filename, parent in (('best.pt', best), ('latest.pt', latest)):
            payload = inherited_payload(parent, fit_config)
            training.save_checkpoint(stage/filename, payload)
            restored = torch.load(stage/filename, map_location='cpu', weights_only=False)
            assert_equal(restored, payload)
        training.write_json(stage/'history.json', latest['progress']['history'])
        for epoch in {40, best['progress']['epoch']}:
            filename = f'validation_epoch_{epoch:02d}.json'
            shutil.copyfile(parent_dir/filename, stage/filename)
        training.write_json(stage/'inheritance.json', dict(parent=source['parents'][cell],
                             model_optimizer_progress_preserved=True, cpu_and_selected_cuda_rng_preserved=True,
                             rng_transition=cfg['rng_transition'][cell]))
        stage.rename(run_dir)
    return latest['progress']


def gnn_batch_check(model, train, ids, snrs):
    """The new single-GPU batch must match serial evaluation and full gradients."""
    model.eval()
    errors = {}
    for length in (16, 13):
        outputs, gradients = [], []
        for micro in (1, 16):
            model.zero_grad(set_to_none=True)
            predictions = []
            for start in range(0, length, micro):
                part = ids[start:min(start+micro, length)]
                inputs, truth = train.batch(part, snrs[start:start+len(part)], 'cuda:0')
                result = model(**inputs)
                metric = training.masked_trajectory_loss(**result, **truth)
                (metric['loss']*len(part)/length).backward()
                predictions.append({key: value.detach().cpu() for key, value in result.items()})
            outputs.append({key: torch.cat([row[key] for row in predictions]) for key in predictions[0]})
            gradients.append({key: value.grad.detach().cpu().clone() for key, value in model.named_parameters()
                              if value.requires_grad and value.grad is not None})
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=3e-5, rtol=1e-6)
        assert gradients[0].keys() == gradients[1].keys()
        for key in gradients[0]:
            torch.testing.assert_close(gradients[0][key], gradients[1][key], atol=2e-5, rtol=2e-4)
        errors[str(length)] = max(float((gradients[0][key]-gradients[1][key]).abs().max()) for key in gradients[0])
    model.zero_grad(set_to_none=True)
    return dict(passed=True, serial_vs_single_gpu16_gradient_max_abs=errors)


def worker(args, cfg, source, run_root, report):
    cell, smoke = args.worker, args.self_check
    physical = cfg['devices'][cell].split(':')[1]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != physical or torch.cuda.device_count() != 1:
        raise ValueError('Each worker must see only its assigned physical GPU')
    torch.cuda.set_device(0)
    contract = json.loads((report/'protocol.json').read_text())
    if contract['source'] != source:
        raise ValueError('Coordinator/worker source mismatch')
    status_path = report/'jobs'/f'{cell}.json'
    status = dict(status='initializing', cell=cell, pid=os.getpid(), device=cfg['devices'][cell],
                  logical_device='cuda:0', visible_physical_gpu=physical, seed=2026, started_at=time.time())
    training.write_json(status_path, status)
    try:
        train, valid = paired.trial.dataset('train'), paired.trial.dataset('V_select')
        ids = paired.selected_origins(train, 17 if smoke else 5549, smoke)
        val_ids = paired.selected_origins(valid, cfg['validation_micro_batch'][cell]+1 if smoke else 400, smoke)
        valid = paired.trial.Subset(valid, val_ids)
        epochs = 41 if smoke else cfg['epochs']
        fit_config = dict(seed=2026, epochs=epochs, patience=101, graph_lr=cfg['graph_lr'], phase='joint',
                          batch_size=16, micro_batch=cfg['micro_batch'][cell], indices=ids.tolist(),
                          snr_mode='balanced', input_hashes=train.input_hashes,
                          validation_hashes=valid.input_hashes, contract=contract)
        parent, _ = load_parents(cell, cfg, source['parents'][cell]['source'])
        if not smoke:
            for key in ('indices', 'input_hashes', 'validation_hashes'):
                if fit_config[key] != parent['config'][key]:
                    raise ValueError(f'{cell}: continuation must preserve {key}')
        run_dir = run_root/cell
        starting = inherit(run_dir, cell, cfg, source, fit_config)
        base_cfg = paired.configuration()
        model = (gnn_fast.baseline.Predictor('gnn', 2026, base_cfg) if cell == 'gnn'
                 else paired.Predictor(cell, base_cfg)).to('cuda:0')
        training.configure_phase = paired.trial.ORIGINAL_OPTIMIZER if cell == 'gnn' else paired.trial.make_optimizer
        optimizer = training.configure_phase(model, 'joint', cfg['graph_lr'])
        if optimizer.state_dict()['param_groups'] != parent['optimizer']['param_groups']:
            raise ValueError(f'{cell}: AdamW parameter groups changed')
        del optimizer
        training.load_weights(model, parent['model'])
        assert_equal(training.mutable_state(model), parent['model'])
        observed, validation_batches = set(), []
        model.graph.register_forward_hook(lambda module, inputs, output: observed.add(output.device.index))
        if smoke:
            old_batch = valid.batch
            def record_batch(origins, *rest, **kwargs):
                validation_batches.append(len(origins))
                return old_batch(origins, *rest, **kwargs)
            valid.batch = record_batch
            if cell == 'gnn' and starting['epoch'] < epochs:
                order, snrs = training.balanced_schedule(train.n, 2026, 40)
                status['single_gpu_batch_check'] = gnn_batch_check(model, train, order[:16], snrs[:16])
        def evaluate(base, validation, device, micro_batch=1, **kwargs):
            return BASE_EVALUATE(base, validation, device, micro_batch=cfg['validation_micro_batch'][cell], **kwargs)
        training.evaluate = evaluate
        gradient_check = dict(observations=0, nonzero=False, finite=True)
        if smoke:
            def inspect_gradient(gradient):
                gradient_check['observations'] += 1
                gradient_check['finite'] &= bool(torch.isfinite(gradient).all())
                gradient_check['nonzero'] |= bool(gradient.detach().abs().max() > 0)
            for parameter in model.graph.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(inspect_gradient)
        status.update(status='training', inherited_epochs=40, inherited_optimizer_steps=13880,
                      train_origins=ids.tolist(), V_select_origins=val_ids.tolist())
        training.write_json(status_path, status)
        torch.cuda.reset_peak_memory_stats(0)
        # fit reloads latest (never best), AdamW and RNG immediately before epoch 41.
        result = training.fit(model, train, valid, run_dir, seed=2026, epochs=epochs, patience=101,
                              graph_lr=cfg['graph_lr'], phase='joint', batch_size=16,
                              micro_batch=cfg['micro_batch'][cell], indices=ids,
                              snr_mode='balanced', resume=True, contract=contract)
        expected_steps = 13882 if smoke else 34700
        added_steps = result['optimizer_steps'] - starting['optimizer_steps']
        if (not result['numerical_gate'] or result['optimizer_steps'] != expected_steps
                or len(result['history']) != epochs or not result['graph_gradient_norm'] > 0):
            raise RuntimeError(f'{cell}: incomplete extension or failed numerical gate')
        assert_equal(result['history'][:40], parent['progress']['history'])
        if observed - {0} or (added_steps > 0 and observed != {0}):
            raise RuntimeError('Only the worker-assigned GPU may participate')
        if smoke:
            expected_batches = [cfg['validation_micro_batch'][cell], 1] * 4 if starting['epoch'] < epochs else []
            if validation_batches != expected_batches:
                raise RuntimeError('Validation batching did not match the profile')
            latest = torch.load(run_dir/'latest.pt', map_location='cpu', weights_only=False)
            assert latest['progress']['epoch'] == 41 and latest['progress']['cursor'] == 0
            assert len(latest['progress']['step_seconds']) == 13882
            assert 'two_gpu_rng' not in latest['rng'] and latest['rng']['original_device'] == 'cuda:0'
            assert all(torch.isfinite(v).all() for v in latest['model'].values())
            if added_steps:
                assert gradient_check['observations'] > 0 and gradient_check['finite'] and gradient_check['nonzero']
            assert len(latest['optimizer']['state']) == len(parent['optimizer']['state'])
            for key, old in parent['optimizer']['state'].items():
                assert float(latest['optimizer']['state'][key]['step']) == float(old['step'])+2
                assert all(torch.isfinite(v).all() for v in latest['optimizer']['state'][key].values() if isinstance(v, torch.Tensor))
            training.restore_rng(latest['rng'], 'cuda:0')
            draws = (torch.rand(8), torch.rand(8, device='cuda:0'))
            training.restore_rng(latest['rng'], 'cuda:0')
            assert torch.equal(draws[0], torch.rand(8)) and torch.equal(draws[1], torch.rand(8, device='cuda:0'))
            status.update(observed_validation_batch_sizes=validation_batches,
                          inherited_adam_steps_advanced_by_two=True, exact_saved_rng_restore=True,
                          new_graph_gradients=gradient_check)
        if source != provenance(cfg):
            raise ValueError('Sources or parent checkpoints changed during continuation')
        status.update(status='complete', training=result, finished_at=time.time(),
                      additional_optimizer_steps_this_process=added_steps,
                      observed_logical_cuda_devices=sorted(observed),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        training.write_json(status_path, status)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        training.write_json(status_path, status)
        raise


def summarize(states, cfg, run_root, smoke):
    final, best = {}, {}
    for cell, state in states.items():
        result = state['training']
        final_epoch, best_epoch = result['history'][-1]['epoch'], result['best_metrics']['epoch']
        final[cell] = json.loads((run_root/cell/f'validation_epoch_{final_epoch:02d}.json').read_text())
        best[cell] = json.loads((run_root/cell/f'validation_epoch_{best_epoch:02d}.json').read_text())
    if not smoke:
        for scores in (final, best):
            denominators = [[(r['origin'], r['snr_db'], r['ade_targets'], r['fde_targets']) for r in scores[c]['per_scene']] for c in CELLS]
            if denominators[0] != denominators[1]:
                raise ValueError('Model comparison requires identical validation coverage')
    contrast = lambda scores: {m: scores['relation_D'][m]-scores['gnn'][m] for m in ('ADE', 'FDE', 'J')}
    return dict(status='complete', smoke_only=smoke, seed=2026, jobs=states,
                final_epoch_metrics=final, best_J_epoch_metrics=best,
                final_epoch_qgat_minus_gnn=None if smoke else contrast(final),
                best_J_qgat_minus_gnn=None if smoke else contrast(best),
                common_denominators_verified=not smoke, confirmation_opened=False, test_opened=False,
                limitation=cfg['limitation'])


def coordinate(args, cfg, source, run_root, report):
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Both physical GPUs must be visible to the coordinator')
    report.mkdir(parents=True, exist_ok=True)
    with (report/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = dict(source=source, mode='smoke-epoch41-17origins' if args.self_check else 'paired-epoch40-to100',
                        confirmation_opened=False, test_opened=False)
        path = report/'protocol.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('This continuation exists; pass --resume explicitly')
            if json.loads(path.read_text()) != protocol:
                raise ValueError('Continuation resume contract mismatch')
        elif args.resume:
            raise FileNotFoundError('No continuation exists to resume')
        else:
            training.write_json(path, protocol)
        coordinator = dict(status='running', pid=os.getpid(), started_at=time.time(), devices=cfg['devices'])
        training.write_json(report/'coordinator.json', coordinator)
        active, logs, states = {}, [], {}
        try:
            for cell in CELLS:
                log_path = report/'logs'/f'{cell}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('a', buffering=1)
                logs.append(log)
                command = [sys.executable, '-u', str(Path(__file__).resolve()),
                           '--self-check' if args.self_check else '--run', '--worker', cell]
                if args.resume:
                    command.append('--resume')
                physical = cfg['devices'][cell].split(':')[1]
                active[cell] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=physical))
                print(f'{cell}: physical GPU {physical}, continue epoch 40 -> {41 if args.self_check else 100}', flush=True)
            while active:
                time.sleep(1)
                for cell, child in list(active.items()):
                    if child.poll() is None:
                        continue
                    del active[cell]
                    if child.returncode:
                        tail = (report/'logs'/f'{cell}.log').read_text().splitlines()[-15:]
                        raise RuntimeError(f'{cell} failed: '+ '\n'.join(tail))
                    states[cell] = json.loads((report/'jobs'/f'{cell}.json').read_text())
                    if states[cell]['status'] != 'complete':
                        raise RuntimeError(f'{cell}: incomplete worker')
            summary = summarize(states, cfg, run_root, args.self_check)
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
    for name in ('check-only', 'self-check', 'run'):
        modes.add_argument('--'+name, action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', choices=CELLS)
    args = parser.parse_args()
    if args.resume and not (args.run or args.self_check):
        parser.error('--resume requires --run or isolated --self-check')
    if args.worker and args.check_only:
        parser.error('--worker cannot accompany --check-only')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = configuration()
    source = provenance(cfg)
    checked_path = REPORT/'checks.json'
    checked = json.loads(checked_path.read_text()) if checked_path.exists() else {}
    passed = checked.get('passed', False) and checked.get('source') == source
    if args.check_only:
        print(json.dumps(dict(ready=passed and torch.cuda.device_count()>=2,
                              parent_epoch=40, target_epoch=100, global_batch_size=16,
                              devices=cfg['devices'], expected_additional_optimizer_steps=20820,
                              formal_optimizer_steps_executed=0,
                              fresh_continuation_available=not (REPORT/'seed2026/protocol.json').exists())), flush=True)
        return
    if args.run and not passed:
        raise ValueError('Complete --self-check for this source before manual launch')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    run_root = TEMP/'results' if args.self_check else RESULTS
    report = TEMP/'reports' if args.self_check else REPORT/'seed2026'
    if args.worker:
        worker(args, cfg, source, run_root, report)
        return
    summary = coordinate(args, cfg, source, run_root, report)
    if args.self_check:
        resume_args = copy.copy(args)
        resume_args.resume = True
        resumed = coordinate(resume_args, cfg, source, run_root, report)
        resume_checks = {cell: dict(additional_steps=state['additional_optimizer_steps_this_process'],
                                    observed_devices=state['observed_logical_cuda_devices'])
                         for cell, state in resumed['jobs'].items()}
        assert all(row['additional_steps'] == 0 and row['observed_devices'] == [] for row in resume_checks.values())
        checks = dict(passed=True, source=source, smoke_summary=summary,
                      formal_training_started=False, formal_optimizer_steps_executed=0,
                      smoke_additional_optimizer_steps_per_model=2,
                      original_40_epoch_history_preserved=True, parent_checkpoints_unchanged=True,
                      completed_branch_resume=resume_checks)
        training.write_json(checked_path, checks)
        print(json.dumps({k:v for k,v in checks.items() if k not in ('source', 'smoke_summary')}), flush=True)


if __name__ == '__main__':
    main()
