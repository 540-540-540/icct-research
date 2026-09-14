"""User-started GNN rerun using the measured runtime profile; old runs stay intact."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation import run_gnn as baseline

paired, training = baseline.paired, baseline.training
REPORT = ROOT / 'reports/qgat_relation_gnn_fast'
RESULTS = ROOT / 'results/qgat_relation_gnn_fast/seed2026'
TEMP = ROOT / '.codex-work/qgat-gnn-fast-prep-20260913'


def configuration():
    profile = json.loads((ROOT/'configs/qgat_runtime_profile.json').read_text(encoding='utf-8-sig'))
    shared, model = profile['shared_training'], profile['models']['gnn']
    expected = dict(global_batch_size=16, epochs=40, patience=41, seed=2026,
                    train_origins=5549, V_select_origins=400,
                    optimizer_updates_per_full_40_epochs=13880, graph_learning_rate=.0003,
                    gradient_clip_norm=1., checkpoint_every_optimizer_update=True,
                    model_precision='float32', AMP=False, TF32=False, torch_compile=False,
                    torch_cpu_threads=1, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)
    for key, value in expected.items():
        if shared[key] != value:
            raise ValueError(f'Unvalidated shared runtime setting: {key}')
    settings = dict(gpus=[0, 1], per_gpu_train_micro_batch=8, parallel_forward_batch=16,
                    accumulation_chunks_per_full_batch=1, validation_micro_batch=32,
                    cuda_rng_offsets=[0, 1000000])
    if any(model[key] != value for key, value in settings.items()):
        raise ValueError('GNN runtime settings differ from the measured profile')
    if profile['snr_db'] != [5, 10, 15, 20] or profile['model_architecture'] != 'unchanged':
        raise ValueError('Runtime profile must preserve model and SNR conditions')
    cfg = baseline.configuration()
    return dict(cfg, revision='qgat-relation-gnn-fast-v1', micro_batch=16,
                parallel_forward_micro_batch=16, per_device_micro_batch=8,
                validation_micro_batch=32, runtime_training=expected,
                limitation='Fresh seed-2026 fixed-40 GNN run; measured physical batch 8 per GPU, '
                           'global batch 16, validation batch 32. No confirmation/test access.')


def provenance(cfg):
    old_source = baseline.provenance(baseline.configuration())
    old_checks = json.loads((baseline.REPORT/'checks.json').read_text())
    if not old_checks['passed'] or old_checks['source'] != old_source:
        raise ValueError('Historical GNN dependencies no longer match their verified source')
    return dict(baseline_source=old_source, configuration=cfg,
                runner_sha256=paired.trial.digest(Path(__file__).resolve()))


def attach_parallel(model, cfg):
    parallel = baseline.attach_parallel(model)
    def evaluate(base, validation, device, micro_batch=1, **kwargs):
        return baseline.ORIGINAL_EVALUATE(base, validation, device,
                                          micro_batch=cfg['validation_micro_batch'], **kwargs)
    training.evaluate = evaluate
    return parallel


def execute(cfg, source, *, smoke=False, resume=False):
    report = TEMP/'reports' if smoke else REPORT/'seed2026'
    results = TEMP/'results' if smoke else RESULTS
    report.mkdir(parents=True, exist_ok=True)
    with (report/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = dict(source=source, mode='smoke-17-tail1' if smoke else 'GNN-fast-fixed-40',
                        confirmation_opened=False, test_opened=False)
        path = report/'protocol.json'
        if path.exists():
            if not resume:
                raise FileExistsError('This rerun already exists; use --resume explicitly')
            if json.loads(path.read_text()) != contract:
                raise ValueError('Resume contract differs from this source/profile')
        elif resume:
            raise FileNotFoundError('No run exists to resume')
        else:
            training.write_json(path, contract)
        job_path = report/'jobs/gnn.json'
        status = dict(status='initializing', cell='gnn', pid=os.getpid(), seed=2026,
                      device='cuda:0 + cuda:1', started_at=time.time(),
                      global_batch_size=16, per_device_micro_batch=8, validation_micro_batch=32)
        training.write_json(job_path, status)
        try:
            train, valid = paired.trial.dataset('train'), paired.trial.dataset('V_select')
            ids = paired.selected_origins(train, 17 if smoke else 5549, smoke)
            val_ids = paired.selected_origins(valid, 33 if smoke else 400, smoke)
            valid = paired.trial.Subset(valid, val_ids)
            validation_batches = []
            if smoke:
                original_batch = valid.batch
                def observe_batch(origins, *args, **kwargs):
                    validation_batches.append(len(origins))
                    return original_batch(origins, *args, **kwargs)
                valid.batch = observe_batch
            model = baseline.Predictor('gnn', cfg['seed'], cfg).to('cuda:0')
            fingerprint = baseline.initial_temporal(model)
            old = json.loads((baseline.REPORT/'checks.json').read_text())
            if fingerprint != old['initial_temporal_sha256']:
                raise ValueError('Temporal initialization must match the original D/GNN run')
            parallel = attach_parallel(model, cfg)
            seen_devices = set()
            model.graph.register_forward_hook(lambda module, inputs, output: seen_devices.add(output.device.index))
            status.update(status='training', initial_temporal_sha256=fingerprint,
                          train_origins=ids.tolist(), V_select_origins=val_ids.tolist())
            training.write_json(job_path, status)
            trained = training.fit(model, train, valid, results/'gnn', seed=cfg['seed'],
                                   epochs=1 if smoke else cfg['epochs'], patience=cfg['patience'],
                                   graph_lr=cfg['graph_lr'], phase='joint', batch_size=cfg['batch_size'],
                                   micro_batch=cfg['parallel_forward_micro_batch'], indices=ids,
                                   snr_mode='balanced', resume=resume, contract=contract)
            steps, epochs = (2, 1) if smoke else (13880, 40)
            if not trained['numerical_gate'] or trained['optimizer_steps'] != steps or len(trained['history']) != epochs:
                raise RuntimeError('Numerical/full-schedule gate failed')
            if not resume and seen_devices != {0, 1}:
                raise RuntimeError('Both GPUs must participate')
            if smoke and validation_batches != [32, 1] * 4:
                raise RuntimeError('Validation must actually use batches 32 and 1 at all four SNRs')
            if source != provenance(cfg):
                raise ValueError('Source changed during training')
            status.update(status='complete', finished_at=time.time(), training=trained,
                          observed_forward_cuda_devices_this_process=sorted(seen_devices))
            training.write_json(job_path, status)
            summary = dict(status='complete', smoke_only=smoke, jobs={'gnn': status},
                           best_J_metrics=trained['best_metrics'], final_epoch_metrics=trained['history'][-1],
                           expected_optimizer_steps=steps, two_gpus_one_model=True,
                           global_batch_size=16, per_device_micro_batch=8, validation_micro_batch=32,
                           temporal_initialization_matches_D=True, confirmation_opened=False, test_opened=False)
            if smoke:
                summary['observed_validation_batch_sizes'] = validation_batches
            training.write_json(report/'summary.json', summary)
            return summary
        except BaseException as exc:
            status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=f'{type(exc).__name__}: {exc}')
            training.write_json(job_path, status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('self-check', 'check-only', 'run'):
        modes.add_argument('--'+name, action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = configuration()
    source = provenance(cfg)
    check_path = REPORT/'checks.json'
    checked = json.loads(check_path.read_text()) if check_path.exists() else {}
    ready = torch.cuda.device_count() >= 2 and checked.get('passed', False) and checked.get('source') == source
    if args.check_only:
        print(json.dumps(dict(ready=ready, seed=2026, epochs=40, global_batch_size=16,
                              per_device_micro_batch=8, validation_micro_batch=32,
                              fresh_run_available=not (REPORT/'seed2026/protocol.json').exists(),
                              optimizer_steps_executed=0)), flush=True)
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA devices are required')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    if args.self_check:
        summary = execute(cfg, source, smoke=True)
        checkpoint = torch.load(TEMP/'results/gnn/latest.pt', map_location='cpu', weights_only=False)
        assert checkpoint['config']['micro_batch'] == 16
        assert checkpoint['progress']['optimizer_steps'] == 2
        assert len(checkpoint['rng']['two_gpu_rng']) == 2
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        validation = json.loads((TEMP/'results/gnn/validation_epoch_01.json').read_text())
        assert len(validation['per_scene']) == 33 * 4
        assert all(r['total_scenes'] == 33 for r in validation['by_snr'].values())
        result = dict(passed=True, source=source, smoke_summary=summary,
                      formal_optimizer_steps_executed=0, smoke_optimizer_steps_executed=2,
                      full_batch_16_and_tail_1=True, validation_batch_32_with_tail_1=True,
                      two_cuda_rng_states_saved=True, formal_training_started=False)
        training.write_json(check_path, result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source', 'smoke_summary')}), flush=True)
    else:
        if not ready:
            raise ValueError('Run --self-check before starting this source/profile')
        execute(cfg, source, resume=args.resume)


if __name__ == '__main__':
    main()
