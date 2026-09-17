"""Short A04 cost probe. Run manually; disposable weights, no F04/F05 training."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['both', 'qgnn', 'gnn'], default='both')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--small-head', action='store_true', help='F04 shared small temporal head cost')
    p.add_argument('--legacy', action='store_true', help='Cost only: old Q21 revision comparator')
    p.add_argument('--depth', type=int, choices=[2,3], default=3)
    p.add_argument('--eval-steps', type=int, default=0, help='No-gradient forward cost batches after training probe')
    p.add_argument('--run-dir', type=Path)
    args = p.parse_args()
    if args.small_head and (args.model == 'both' or args.legacy):
        p.error('--small-head requires one current model')
    if args.legacy and args.model != 'qgnn':
        p.error('--legacy requires --model qgnn')
    if args.eval_steps < 0:
        p.error('--eval-steps must be nonnegative')
    if min(args.steps, args.batch_size, args.micro_batch) < 1 or args.warmup < 0:
        p.error('Positive steps and batch sizes required')
    if args.batch_size % args.micro_batch:
        p.error('batch-size must be divisible by micro-batch')
    import numpy as np
    import torch
    from frontend.symbol_dataset import SharedPredictionInputs
    validation = json.loads((ROOT/'reports/theory_validation.json').read_text())
    if not validation.get('passed') or 'A04' not in validation.get('formula_version', ''):
        raise RuntimeError('Current A04 F02 validation must pass first')
    if not torch.cuda.is_available() or (args.model == 'both' and torch.cuda.device_count() < 2):
        raise RuntimeError('Requested CUDA devices are unavailable')
    loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
    with np.load(ROOT/'data/f01d/labels/train.npz', allow_pickle=False) as f:
        labels = {k: f[k] for k in ('future_position', 'label_valid')}
    if args.check_only:
        print(json.dumps(dict(ready=True, models=['QGNN', 'GNN'] if args.model == 'both' else [args.model.upper()],
                              train_samples=len(loader), effective_batch=args.batch_size,
                              micro_batch=args.micro_batch, measured_steps=args.steps,
                              warmup_steps=args.warmup, gpu_count=torch.cuda.device_count(),
                              optimizer_steps_executed=0)), flush=True)
        return
    run_dir = args.run_dir or ROOT/'reports/f03'/time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.model == 'both':
        children = []
        try:
            for name, device in [('qgnn', 'cuda:0'), ('gnn', 'cuda:1')]:
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--model', name,
                           '--device', device, '--steps', str(args.steps), '--warmup', str(args.warmup),
                           '--batch-size', str(args.batch_size), '--micro-batch', str(args.micro_batch),
                           '--run-dir', str(run_dir), '--depth', str(args.depth), '--eval-steps', str(args.eval_steps)]
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                children.append(subprocess.Popen(command, cwd=ROOT, env=env))
            codes = [child.wait() for child in children]
        except BaseException:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
            raise
        reports = {name: json.loads((run_dir/f'{name}.json').read_text())
                   for name in ('qgnn', 'gnn') if (run_dir/f'{name}.json').exists()}
        summary = dict(stage='F03-short', status='complete' if not any(codes) and len(reports)==2 else 'failed',
                       reports=reports, full_F03_budget_passed=False,
                       note='Short cost evidence only; no full training checkpoints or budget approval.')
        save(run_dir/'summary.json', summary)
        print(f'F03 short probe {summary["status"]}: {run_dir}', flush=True)
        if any(codes):
            raise SystemExit(1)
        return

    from prediction.model import build_model
    from prediction.temporal import masked_trajectory_loss
    torch.manual_seed(20260912)
    np_rng = np.random.default_rng(20260912)
    order = np_rng.permutation(len(loader))
    report = dict(stage='F03-short', model=args.model.upper(), status='running', device=args.device,
                  effective_batch=args.batch_size, micro_batch=args.micro_batch, measured_steps=args.steps,
                  warmup_steps=args.warmup, sample_seed=20260912, split='train', snr_db=20,
                  quantum_precision='float64/complex128', time_model_precision='float32',
                  disposable_weight_updates=True, checkpoint_saved=False, graph_depth=args.depth, legacy=args.legacy, small_head=args.small_head, steps=[])
    target = run_dir/f'{args.model}{"_legacy" if args.legacy else ""}.json'
    save(target, report)
    def sync():
        torch.cuda.synchronize(args.device)
    try:
        start = time.perf_counter()
        if args.small_head:
            from prediction.preflight import SmallPredictor
            model = SmallPredictor(args.model, loader.normalization, args.depth, seed=20260912).to(args.device).train()
        else:
            model = build_model(args.model, loader.normalization, args.device, depth=args.depth).train()
        if args.legacy:
            from prediction.quantum import QuantumGraph
            from prediction.temporal import TrajectoryPredictor
            model.graph = QuantumGraph(loader.normalization['quantum_scale'], depth=args.depth, legacy=True).to(args.device)
            model.temporal = TrajectoryPredictor(21).to(args.device)
        report['history_chunk_size'] = getattr(model.graph, 'history_chunk_size', None)
        groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lr = 3e-4 if name.startswith('graph.') else (3e-5 if 'lora_' in name else 1e-4)
            wd = 0. if param.ndim < 2 or (args.model == 'qgnn' and name.startswith('graph.')) else .01
            groups.setdefault((lr, wd), []).append(param)
        optimizer = torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd)
                                      for (lr, wd), ps in groups.items()])
        trainable = [p for p in model.parameters() if p.requires_grad]
        sync()
        report['model_load_seconds'] = time.perf_counter()-start
        report['trainable_parameters'] = sum(p.numel() for p in trainable)
        cursor = 0
        count = args.steps + args.warmup
        for step in range(count):
            phase = 'warmup' if step < args.warmup else 'measure'
            print(f'[{report["model"]}] {phase} {step+1}/{count} starting', flush=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(args.device)
            sync()
            started = time.perf_counter()
            timing = {k: 0. for k in ['data', 'graph_forward', 'temporal_forward', 'temporal_backward', 'graph_backward', 'optimizer']}
            losses, sample_ids, active_nodes = [], [], []
            any_supervision = False
            for micro in range(args.batch_size // args.micro_batch):
                tick = time.perf_counter()
                ids = [int(order[(cursor+i) % len(order)]) for i in range(args.micro_batch)]
                cursor += args.micro_batch
                samples = [loader[i] for i in ids]
                batch = {key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=args.device)
                         for key in ('state_hat', 'standardized_state', 'track_exists', 'detected')}
                truth = torch.as_tensor(labels['future_position'][ids], device=args.device)
                valid = torch.as_tensor(labels['label_valid'][ids], device=args.device)
                values = [batch[k] for k in ('state_hat','standardized_state','track_exists','detected')]
                sync(); timing['data'] += time.perf_counter()-tick
                tick = time.perf_counter()
                features = (model.graph.forward_history(values[0], values[2]) if args.model == 'qgnn'
                            else model.graph(*values))
                sync(); timing['graph_forward'] += time.perf_counter()-tick
                # Split the same chain rule at graph output to measure both backward portions.
                bridge = features.detach().requires_grad_(True)
                tick = time.perf_counter()
                output = model.temporal(*values, bridge)
                loss = masked_trajectory_loss(output['prediction'], truth, valid, output['origin_eligible'])
                any_supervision |= not loss['skip_optimizer']
                sync(); timing['temporal_forward'] += time.perf_counter()-tick
                tick = time.perf_counter()
                (loss['loss'] * args.micro_batch / args.batch_size).backward()
                sync(); timing['temporal_backward'] += time.perf_counter()-tick
                tick = time.perf_counter()
                if features.requires_grad and bridge.grad is not None:
                    features.backward(bridge.grad)
                sync(); timing['graph_backward'] += time.perf_counter()-tick
                losses.append(float(loss['loss'].detach()))
                sample_ids.extend(ids)
                active_nodes.extend(batch['track_exists'].sum(-1).flatten().tolist())
                print(f'[{report["model"]}] {phase} {step+1}/{count}: {micro+1}/{args.batch_size//args.micro_batch} micro-batches', flush=True)
            tick = time.perf_counter()
            if any_supervision:
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
            sync(); timing['optimizer'] = time.perf_counter()-tick
            row = dict(phase=phase, step=step+1, seconds=time.perf_counter()-started,
                       timing_seconds=timing, loss=statistics.mean(losses), sample_indices=sample_ids,
                       active_nodes_histogram={str(n): active_nodes.count(n) for n in sorted(set(active_nodes))},
                       peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
                       peak_reserved_GiB=torch.cuda.max_memory_reserved(args.device)/2**30,
                       optimizer_step=any_supervision)
            report['steps'].append(row)
            save(target, report)
            print(f'[{report["model"]}] {phase} done: {row["seconds"]:.2f}s, peak {row["peak_allocated_GiB"]:.2f}GiB', flush=True)
        model.eval()
        inference=[]
        with torch.no_grad():
            for eval_step in range(args.eval_steps):
                sync(); tick=time.perf_counter()
                ids=[int(order[(i+eval_step*args.micro_batch)%len(order)]) for i in range(args.micro_batch)]
                samples=[loader[i] for i in ids]
                values=[torch.as_tensor(np.stack([sample[k] for sample in samples]), device=args.device) for k in ('state_hat','standardized_state','track_exists','detected')]
                result=model(*values)
                if not torch.isfinite(result['prediction']).all():
                    raise RuntimeError('Nonfinite inference prediction')
                sync()
                inference.append(dict(seconds=time.perf_counter()-tick, samples=len(ids)))
        report['inference_batches']=inference
        measured = [r['seconds'] for r in report['steps'] if r['phase']=='measure']
        report.update(status='complete', median_step_seconds=statistics.median(measured),
                      mean_step_seconds=statistics.mean(measured),
                      estimated_100_step_minutes=statistics.median(measured)*100/60,
                      estimated_200_step_minutes=statistics.median(measured)*200/60,
                      full_F03_budget_passed=False)
        save(target, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(target, report)
        raise


if __name__ == '__main__':
    main()
