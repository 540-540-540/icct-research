"""One GNN baseline, seed 2026, global batch 16, two GPUs with micro-batch 1 each."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation import run as paired
from experiments.qgat_candidate.run_formal import Predictor

training = paired.training
REPORT = ROOT / 'reports/qgat_relation_gnn'
TEMP = ROOT / '.codex-work/qgat-gnn-two-gpu-20260913'
ORIGINAL_SEED = training.seed_all
ORIGINAL_RNG = training.rng_state
ORIGINAL_RESTORE = training.restore_rng
ORIGINAL_EVALUATE = training.evaluate


def configuration():
    cfg = paired.configuration()
    return dict(cfg, revision='qgat-relation-gnn-two-gpu-v1', cells=['gnn'],
                devices={'gnn': ['cuda:0', 'cuda:1']},
                parallel_forward_micro_batch=2, per_device_micro_batch=1,
                validation_micro_batch=1, cuda_rng_offsets=[0, 1000000],
                limitation='One seed-2026 GNN baseline matched to the fixed-40-epoch D comparison. '
                           'Global batch remains 16; two concurrent one-origin micro-batches. '
                           'No confirmation/test access. Two GPUs train one model, not two seeds.')


def provenance(cfg):
    reference = paired.provenance(paired.configuration())
    saved = json.loads((paired.REPORT / 'seed2026/protocol.json').read_text())['source']
    if reference != saved:
        raise ValueError('Code/data dependencies changed from the completed D comparison')
    files = (Path(__file__).resolve(), ROOT / 'experiments/qgat_candidate/run_formal.py')
    return dict(reference_D_source=reference, configuration=cfg,
                additional_files={str(p.relative_to(ROOT)): paired.trial.digest(p) for p in files})


def seed_all(seed):
    ORIGINAL_SEED(seed)
    # Independent dropout streams; this does not change CPU model initialization.
    if torch.cuda.device_count() >= 2:
        with torch.cuda.device(1):
            torch.cuda.manual_seed(seed + 1000000)


def rng_state(device=None):
    state = ORIGINAL_RNG(device)
    state['two_gpu_rng'] = [torch.cuda.get_rng_state(i) for i in (0, 1)]
    return state


def restore_rng(state, device=None):
    ORIGINAL_RESTORE(state, device)
    if len(state.get('two_gpu_rng', [])) != 2:
        raise ValueError('Two-GPU checkpoint must contain both CUDA RNG states')
    for i, value in enumerate(state['two_gpu_rng']):
        torch.cuda.set_rng_state(value.cpu(), i)


def attach_parallel(model):
    """Keep the original model/state/optimizer names; only its forward is parallel."""
    parallel = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    def forward(base, inputs, phase='joint'):
        if base is not model or phase != 'joint':
            raise ValueError('GNN parallel adapter only supports this joint-training model')
        parallel.train(base.training)
        return parallel(**inputs) if base.training else base(**inputs)
    def evaluate(base, validation, device, micro_batch=1, **kwargs):
        # Preserve the D comparison's one-origin validation arithmetic.
        return ORIGINAL_EVALUATE(base, validation, device, micro_batch=1, **kwargs)
    training.forward = forward
    training.evaluate = evaluate
    training.seed_all = seed_all
    training.rng_state = rng_state
    training.restore_rng = restore_rng
    training.configure_phase = paired.trial.ORIGINAL_OPTIMIZER
    return parallel


def initial_temporal(model):
    state = training.mutable_state(model)
    return paired.trial.fingerprint({k: v for k, v in state.items() if k.startswith('temporal.')})


def self_check(cfg):
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA devices are required')
    torch.set_num_threads(1)
    train, valid = paired.trial.dataset('train'), paired.trial.dataset('V_select')
    paired.selected_origins(train, 5549, False)
    paired.selected_origins(valid, 400, False)
    model = Predictor('gnn', 2026, cfg).to('cuda:0').eval()
    original = json.loads((paired.REPORT/'seed2026/jobs/original_D.json').read_text())
    fingerprint = initial_temporal(model)
    assert fingerprint == original['initial_temporal_sha256']
    parallel = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=0).eval()
    inputs, _ = train.batch([0, 1], [5, 10], 'cuda:0')
    with torch.no_grad():
        expected = model(**inputs)
        actual = parallel(**inputs)
    errors = {}
    for key, value in expected.items():
        torch.testing.assert_close(value, actual[key], atol=2e-5, rtol=2e-5)
        if value.is_floating_point():
            errors[key] = float((value-actual[key]).abs().max())
    # Global loss weights must agree for a normal batch and the 13-origin tail.
    gradient_errors = {}
    for length in (16, 13):
        model.zero_grad(set_to_none=True)
        for origin in range(length):
            sample, truth = train.batch([origin], [20], 'cuda:0')
            metric = training.masked_trajectory_loss(**model(**sample), **truth)
            (metric['loss']/length).backward()
        expected_state = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        model.zero_grad(set_to_none=True)
        for start in range(0, length, 2):
            origins = list(range(start, min(start+2, length)))
            sample, truth = train.batch(origins, [20]*len(origins), 'cuda:0')
            metric = training.masked_trajectory_loss(**parallel(**sample), **truth)
            (metric['loss']*len(origins)/length).backward()
        difference = 0.
        for name, parameter in model.named_parameters():
            if name in expected_state:
                torch.testing.assert_close(parameter.grad, expected_state[name], atol=3e-5, rtol=3e-4)
                difference = max(difference, float((parameter.grad-expected_state[name]).abs().max()))
        gradient_errors[str(length)] = difference
    seed_all(2026)
    saved = rng_state('cuda:0')
    draws = [torch.rand(8, device=f'cuda:{i}') for i in (0, 1)]
    assert not torch.equal(draws[0].cpu(), draws[1].cpu())
    restore_rng(saved, 'cuda:0')
    assert all(torch.equal(x, torch.rand(8, device=f'cuda:{i}')) for i, x in enumerate(draws))
    parallel.train()
    saved = rng_state('cuda:0')
    with torch.no_grad():
        first = parallel(**inputs)
    advanced = rng_state('cuda:0')
    assert all(not torch.equal(a, b) for a, b in zip(saved['two_gpu_rng'], advanced['two_gpu_rng']))
    restore_rng(saved, 'cuda:0')
    with torch.no_grad():
        repeated = parallel(**inputs)
    for key in first:
        torch.testing.assert_close(first[key], repeated[key], atol=0, rtol=0)
    result = dict(passed=True, source=provenance(cfg), temporal_initialization_matches_D=True,
                  initial_temporal_sha256=fingerprint, model='unchanged GNNGraph + TrajectoryPredictor',
                  output_errors=errors, reduced_gradient_errors=gradient_errors,
                  independent_gpu_rng_and_exact_rng_restore=True, actual_dropout_forward_restore_exact=True, global_batch_size=16,
                  per_device_micro_batch=1, training_origins=5549, V_select_origins=400,
                  epochs=40, expected_optimizer_steps=13880, optimizer_steps_executed=0)
    training.write_json(REPORT/'checks.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='source'}), flush=True)


def run(args, cfg):
    report = TEMP/'reports' if args.smoke else REPORT/'seed2026'
    result_dir = TEMP/'results' if args.smoke else ROOT/'results/qgat_relation_gnn/seed2026'
    report.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    with (report/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = provenance(cfg)
        checks = json.loads((REPORT/'checks.json').read_text())
        if not checks['passed'] or checks['source'] != source:
            raise ValueError('Run --self-check on this source before training')
        contract = dict(source=source, mode='smoke-17-tail1' if args.smoke else 'GNN-two-GPU-fixed-40',
                        confirmation_opened=False, test_opened=False)
        path = report/'protocol.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('Existing GNN run; pass --resume explicitly')
            if json.loads(path.read_text()) != contract:
                raise ValueError('GNN resume source contract mismatch')
        else:
            if args.resume:
                raise FileNotFoundError('No GNN run exists to resume')
            training.write_json(path, contract)
        job_path = report/'jobs/gnn.json'
        status = dict(status='initializing', cell='gnn', pid=os.getpid(), device='cuda:0 + cuda:1',
                      seed=2026, started_at=time.time())
        training.write_json(job_path, status)
        try:
            train, valid = paired.trial.dataset('train'), paired.trial.dataset('V_select')
            ids = paired.selected_origins(train, 17 if args.smoke else 5549, args.smoke)
            val_ids = paired.selected_origins(valid, 2 if args.smoke else 400, args.smoke)
            valid = paired.trial.Subset(valid, val_ids)
            model = Predictor('gnn', 2026, cfg).to('cuda:0')
            fingerprint = initial_temporal(model)
            if fingerprint != checks['initial_temporal_sha256']:
                raise ValueError('Temporal initialization mismatch')
            parallel = attach_parallel(model)
            seen_devices = set()
            model.graph.register_forward_hook(lambda module, inputs, output: seen_devices.add(output.device.index))
            status.update(status='training', initial_temporal_sha256=fingerprint,
                          train_origins=ids.tolist(), V_select_origins=val_ids.tolist())
            training.write_json(job_path, status)
            trained = training.fit(model, train, valid, result_dir/'gnn', seed=2026,
                                   epochs=1 if args.smoke else 40, patience=41, graph_lr=cfg['graph_lr'],
                                   phase='joint', batch_size=16, micro_batch=2, indices=ids,
                                   snr_mode='balanced', resume=args.resume, contract=contract)
            if not trained['numerical_gate'] or trained['optimizer_steps'] != (2 if args.smoke else 13880):
                raise RuntimeError('GNN numerical/full-schedule gate failed')
            if not args.resume and seen_devices != {0, 1}:
                raise RuntimeError('Both GPUs must participate in a fresh run')
            if source != provenance(cfg):
                raise ValueError('Sources changed during GNN training')
            status.update(status='complete', finished_at=time.time(), training=trained,
                          observed_forward_cuda_devices_this_process=sorted(seen_devices),
                          best_sha256=paired.trial.digest(result_dir/'gnn/best.pt'),
                          latest_sha256=paired.trial.digest(result_dir/'gnn/latest.pt'))
            training.write_json(job_path, status)
            history = trained['history']
            summary = dict(status='complete', smoke_only=args.smoke, jobs={'gnn':status},
                           best_J_metrics=trained['best_metrics'], final_epoch_metrics=history[-1],
                           expected_optimizer_steps=2 if args.smoke else 13880,
                           two_gpus_one_model=True, global_batch_size=16, per_device_micro_batch=1,
                           temporal_initialization_matches_D=True, confirmation_opened=False, test_opened=False)
            training.write_json(report/'summary.json', summary)
        except BaseException as exc:
            status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=f'{type(exc).__name__}: {exc}')
            training.write_json(job_path, status)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ('self-check', 'check-only', 'smoke', 'run'):
        mode.add_argument('--'+name, action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.resume and not (args.run or args.smoke):
        p.error('--resume requires --run or --smoke')
    cfg = configuration()
    if args.self_check:
        self_check(cfg)
        return
    if args.check_only:
        source = provenance(cfg)
        check_path = REPORT/'checks.json'
        checked = json.loads(check_path.read_text()) if check_path.exists() else {}
        print(json.dumps(dict(ready=torch.cuda.device_count()>=2 and checked.get('passed',False) and checked.get('source')==source,
                              two_gpus_one_model=True, seed=2026, global_batch_size=16, per_device_micro_batch=1,
                              epochs=40, optimizer_steps_executed=0)), flush=True)
        return
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    run(args, cfg)


if __name__ == '__main__':
    main()
