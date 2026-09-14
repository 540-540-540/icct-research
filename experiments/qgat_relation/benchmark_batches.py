"""Isolated physical-microbatch benchmark; never invokes formal training/evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/qgat_relation_batch_benchmark'
TEMP = ROOT / '.codex-work/qgat-batch-benchmark-20260913'
CELLS = ('gnn', 'original_D', 'relation_D')
CANDIDATES = {'gnn': (1, 2, 4, 8), 'original_D': (1, 2, 4, 8, 16), 'relation_D': (1, 2, 4, 8, 16)}
TOLERANCES = dict(prediction_atol=3e-5, prediction_rtol=1e-6,
                  gradient_atol=2e-5, gradient_rtol=2e-4, gradient_relative_l2=1e-4,
                  metric_atol=3e-5, metric_rtol=1e-6,
                  parameter_atol=2e-6, parameter_rtol=2e-5, parameter_relative_l2=1e-5,
                  optimizer_atol=2e-5, optimizer_rtol=2e-4, optimizer_relative_l2=1e-4)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def summary(values):
    return dict(median=statistics.median(values), mean=statistics.mean(values),
                min=min(values), max=max(values), stdev=statistics.stdev(values) if len(values) > 1 else 0.,
                q25=float(__import__('numpy').quantile(values, .25)),
                q75=float(__import__('numpy').quantile(values, .75)))


def setup(cell):
    import torch
    from experiments.qgat_relation import run as paired
    from experiments.qgat_relation import run_gnn as gnn
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    cfg = paired.configuration()
    device = 'cuda:0' if cell == 'gnn' else cfg['devices'][cell]
    model = (gnn.Predictor('gnn', cfg['seed'], cfg) if cell == 'gnn' else paired.Predictor(cell, cfg)).to(device)
    parallel = gnn.attach_parallel(model) if cell == 'gnn' else None
    optimizer = (paired.trial.ORIGINAL_OPTIMIZER if cell == 'gnn' else paired.trial.make_optimizer)(model, 'joint', cfg['graph_lr'])
    checkpoint_path = ROOT / ('results/qgat_relation_gnn/seed2026/gnn/latest.pt' if cell == 'gnn'
                              else f'results/qgat_relation/seed2026/{cell}/latest.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    paired.training.load_weights(model, checkpoint['model'])
    paired.training.seed_all(cfg['seed'])
    train = paired.trial.dataset('train')
    assert train.n == cfg['train_origins'] == 5549
    assert all(p.dtype == torch.float32 for p in model.parameters())
    devices = [0, 1] if cell == 'gnn' else [torch.device(device).index]
    source = dict(checkpoint=str(checkpoint_path), checkpoint_bytes=checkpoint_path.stat().st_size,
                  checkpoint_mtime_ns=checkpoint_path.stat().st_mtime_ns,
                  checkpoint_epoch=checkpoint['progress']['epoch'],
                  checkpoint_cursor=checkpoint['progress']['cursor'],
                  checkpoint_optimizer_steps=checkpoint['progress']['optimizer_steps'])
    del checkpoint
    return paired, model, parallel, optimizer, train, cfg, device, devices, source


def synchronize(devices):
    import torch
    for index in devices:
        torch.cuda.synchronize(index)


def finite_parameters(model, gradients=False):
    import torch
    tensors = [(name, p.grad if gradients else p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, value in tensors if value is None]
    bad = [name for name, value in tensors if value is not None and not bool(torch.isfinite(value).all())]
    if bad:
        raise FloatingPointError(f'Nonfinite {"gradients" if gradients else "parameters"}: {bad}')
    return missing


def benchmark(args):
    import torch
    paired, model, parallel, optimizer, train, cfg, device, devices, source = setup(args.model)
    training = paired.training
    physical = args.micro * (2 if args.model == 'gnn' else 1)
    order, snrs = training.balanced_schedule(train.n, cfg['seed'], 0)
    seen_devices = set()
    model.graph.register_forward_hook(lambda module, inputs, output: seen_devices.add(output.device.index))
    model.train()
    rows = []
    destination = TEMP / args.case_id / 'latest.pt'
    benchmark_config = dict(model=args.model, global_batch=16, micro_per_gpu=args.micro, physical_forward=physical,
                            schedule_epoch=0, seed=2026, optimizer_initialization='fresh AdamW then two warmup updates',
                            source=source)
    progress = dict(epoch=0, cursor=0, optimizer_steps=0, loss_sum=0., seen=0, history=[], step_seconds=[])
    missing_gradients = set()
    finite_parameters(model)
    for step in range(args.steps + 2):
        if step == 2:
            synchronize(devices)
            for index in devices:
                torch.cuda.reset_peak_memory_stats(index)
        ids, conditions = order[step*16:(step+1)*16], snrs[step*16:(step+1)*16]
        assert len(ids) == 16
        synchronize(devices)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        any_supervision, weighted_loss = False, 0.
        for start in range(0, len(ids), physical):
            part, condition = ids[start:start+physical], conditions[start:start+physical]
            inputs, truth = train.batch(part, condition, device)
            output = training.forward(model, inputs, 'joint')
            metric = training.masked_trajectory_loss(**output, **truth)
            any_supervision |= not metric['skip_optimizer']
            if not bool(torch.isfinite(metric['loss'])):
                raise FloatingPointError('Nonfinite loss')
            (metric['loss'] * len(part)/len(ids)).backward()
            weighted_loss += float(metric['loss'].detach()) * len(part)
        assert any_supervision, 'Every measured batch must execute a supervised optimizer update'
        graph_norms = [p.grad.detach().norm() for name, p in model.named_parameters() if name.startswith('graph.') and p.grad is not None]
        graph_norm = float(torch.linalg.vector_norm(torch.stack(graph_norms))) if graph_norms else 0.
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        synchronize(devices)
        compute_seconds = time.perf_counter() - tick
        # Diagnostic finite scans are outside throughput timing; the original clipping gate remains timed.
        missing_gradients.update(finite_parameters(model, gradients=True))
        finite_parameters(model)
        progress.update(cursor=(step+1)*16, optimizer_steps=step+1, seen=(step+1)*16,
                        loss_sum=progress['loss_sum']+weighted_loss, gradient_norm=float(norm), graph_gradient_norm=graph_norm)
        progress['step_seconds'].append(compute_seconds)
        synchronize(devices)
        io_tick = time.perf_counter()
        training.save_checkpoint(destination, dict(config=benchmark_config, model=training.mutable_state(model),
                                 optimizer=optimizer.state_dict(), rng=training.rng_state(device), progress=progress))
        synchronize(devices)
        io_seconds = time.perf_counter() - io_tick
        row = dict(step=step-2, warmup=step<2, samples=len(ids), optimizer_updates=1,
                   origin_ids=ids.tolist(), snr_db=conditions.tolist(),
                   compute_seconds=compute_seconds, checkpoint_seconds=io_seconds,
                   with_checkpoint_seconds=compute_seconds+io_seconds, loss=weighted_loss/len(ids),
                   gradient_norm=float(norm), graph_gradient_norm=graph_norm, finite=True)
        rows.append(row)
        print(json.dumps(dict(case=args.case_id, step=step+1, total=args.steps+2,
                             seconds=round(compute_seconds+io_seconds, 4), loss=row['loss'])), flush=True)
    measured = rows[2:]
    assert len(measured) >= 8
    assert seen_devices == set(devices), seen_devices
    memory = {str(index): dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(index),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(index),
                               total_bytes=torch.cuda.get_device_properties(index).total_memory) for index in devices}
    samples = sum(row['samples'] for row in measured)
    return dict(status='complete', kind='throughput', case_id=args.case_id, model=args.model,
                micro_per_gpu=args.micro, physical_forward=physical, global_batch=16,
                warmup_updates=2, measured_updates=len(measured), measured_samples=samples,
                source=source, schedule_epoch=0, optimizer_reset=True, finite=True,
                missing_gradient_names=sorted(missing_gradients), observed_cuda_devices=sorted(seen_devices),
                compute_seconds=summary([r['compute_seconds'] for r in measured]),
                checkpoint_seconds=summary([r['checkpoint_seconds'] for r in measured]),
                with_checkpoint_seconds=summary([r['with_checkpoint_seconds'] for r in measured]),
                origins_per_second=samples/sum(r['compute_seconds'] for r in measured),
                origins_per_second_with_checkpoint=samples/sum(r['with_checkpoint_seconds'] for r in measured),
                peak_memory=memory, steps=rows, temporary_checkpoint_bytes=destination.stat().st_size,
                torch_version=torch.__version__, gpu_names={str(i):torch.cuda.get_device_name(i) for i in devices},
                precision=dict(parameter_dtype='float32', quantum_state_dtype='complex128', amp=False, tf32=False,
                               float32_matmul_precision=torch.get_float32_matmul_precision(), compile=False),
                formal_optimizer_steps_executed=0, confirmation_opened=False, test_opened=False)


def quality(args):
    import copy
    import torch
    paired, model, parallel, optimizer, train, cfg, device, devices, source = setup(args.model)
    training = paired.training
    order, snrs = training.balanced_schedule(train.n, cfg['seed'], 0)
    model.eval()
    if parallel is not None:
        parallel.eval()
    seen_devices = set()
    model.graph.register_forward_hook(lambda module, inputs, output: seen_devices.add(output.device.index))
    saved = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)

    def evaluate_batch(ids, conditions, physical, serial=False):
        training.load_weights(model, saved['model'])
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        model.zero_grad(set_to_none=True)
        predictions, eligibility = [], []
        loss = 0.
        for start in range(0, len(ids), physical):
            part = ids[start:start+physical]
            inputs, truth = train.batch(part, conditions[start:start+physical], device)
            # Call DataParallel itself in eval; the formal adapter intentionally uses one GPU for validation.
            output = model(**inputs) if serial or parallel is None else parallel(**inputs)
            metric = training.masked_trajectory_loss(**output, **truth)
            weighted = metric['loss'] * len(part)/len(ids)
            weighted.backward()
            loss += float(weighted.detach())
            predictions.append(output['prediction'].detach().cpu())
            eligibility.append(output['origin_eligible'].detach().cpu())
        finite_parameters(model, gradients=True)
        grads = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        prediction = torch.cat(predictions)
        origin_eligible = torch.cat(eligibility)
        _, truth = train.batch(ids, conditions, 'cpu')
        full = training.masked_trajectory_loss(prediction=prediction, origin_eligible=origin_eligible, **truth)
        metric = training.scene_metrics(prediction=prediction, origin_eligible=origin_eligible, **truth)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        finite_parameters(model)
        parameters = {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
        optimizer_tensors = {f'{name}.{key}':value.detach().cpu().clone()
                             for name,p in model.named_parameters() for key,value in optimizer.state.get(p, {}).items()
                             if isinstance(value, torch.Tensor)}
        return dict(prediction=prediction, origin_eligible=origin_eligible, grads=grads,
                    parameters=parameters, optimizer_tensors=optimizer_tensors, clip_norm=float(norm),
                    metrics=dict(loss=float(full['loss']), accumulated_loss=loss,
                                 ADE=float(metric['scene_ade'].mean()), FDE=float(metric['scene_fde'].mean())))

    def state_difference(actual, expected, prefix):
        assert actual.keys() == expected.keys()
        maxabs = squared_delta = squared_reference = 0.
        failed = []
        for name, reference in expected.items():
            value = actual[name]
            delta = value.double()-reference.double()
            maxabs = max(maxabs, float(delta.abs().max()))
            squared_delta += float(delta.square().sum())
            squared_reference += float(reference.double().square().sum())
            if not torch.allclose(value, reference, atol=TOLERANCES[prefix+'_atol'], rtol=TOLERANCES[prefix+'_rtol']):
                failed.append(name)
        relative = (squared_delta/max(squared_reference, 1e-30))**.5
        return dict(maxabs=maxabs, relative_l2=relative, failed_tensor_names=failed, tensor_count=len(expected),
                    passed=not failed and relative <= TOLERANCES[prefix+'_relative_l2'])

    results = []
    for length in (16, 13):
        ids, conditions = (order[:16], snrs[:16]) if length == 16 else (order[-13:], snrs[-13:])
        expected = evaluate_batch(ids, conditions, 1, serial=True)
        for micro in args.quality_micros:
            seen_devices.clear()
            physical = micro * (2 if args.model == 'gnn' else 1)
            actual = evaluate_batch(ids, conditions, physical)
            prediction_delta = (actual['prediction']-expected['prediction']).abs()
            pred_pass = bool(torch.allclose(actual['prediction'], expected['prediction'],
                                            atol=TOLERANCES['prediction_atol'], rtol=TOLERANCES['prediction_rtol']))
            assert actual['grads'].keys() == expected['grads'].keys()
            maxabs, squared_delta, squared_reference, failed_names = 0., 0., 0., []
            for name, reference in expected['grads'].items():
                candidate = actual['grads'][name]
                difference = candidate.double()-reference.double()
                maxabs = max(maxabs, float(difference.abs().max()))
                squared_delta += float(difference.square().sum())
                squared_reference += float(reference.double().square().sum())
                if not torch.allclose(candidate, reference, atol=TOLERANCES['gradient_atol'], rtol=TOLERANCES['gradient_rtol']):
                    failed_names.append(name)
            relative_l2 = (squared_delta/max(squared_reference, 1e-30))**.5
            metric_deltas = {k:abs(actual['metrics'][k]-expected['metrics'][k]) for k in expected['metrics']}
            metrics_pass = all(delta <= TOLERANCES['metric_atol']+TOLERANCES['metric_rtol']*abs(expected['metrics'][k])
                               for k,delta in metric_deltas.items())
            masks_match = torch.equal(actual['origin_eligible'], expected['origin_eligible'])
            devices_pass = seen_devices == set(devices)
            parameter_check = state_difference(actual['parameters'], expected['parameters'], 'parameter')
            optimizer_check = state_difference(actual['optimizer_tensors'], expected['optimizer_tensors'], 'optimizer')
            # Keep integer-like Adam step counters from dominating the moment relative-error denominator.
            optimizer_check['by_state'] = {
                key:state_difference({n:v for n,v in actual['optimizer_tensors'].items() if n.endswith('.'+key)},
                                     {n:v for n,v in expected['optimizer_tensors'].items() if n.endswith('.'+key)}, 'optimizer')
                for key in ('exp_avg', 'exp_avg_sq', 'step')}
            optimizer_check['passed'] &= all(item['passed'] for item in optimizer_check['by_state'].values())
            passed = pred_pass and not failed_names and relative_l2 <= TOLERANCES['gradient_relative_l2'] and metrics_pass and masks_match and devices_pass and parameter_check['passed'] and optimizer_check['passed']
            row = dict(batch_length=length, real_epoch_tail=length==13, origin_ids=ids.tolist(), snr_db=conditions.tolist(),
                       micro_per_gpu=micro, physical_forward=physical, passed=passed,
                       prediction_maxabs=float(prediction_delta.max()), prediction_close=pred_pass,
                       gradient_maxabs=maxabs, gradient_relative_l2=relative_l2,
                       failed_gradient_names=failed_names, gradient_tensor_count=len(expected['grads']),
                       metrics_reference=expected['metrics'], metrics_candidate=actual['metrics'], metric_abs_differences=metric_deltas,
                       one_update_parameters=parameter_check, one_update_optimizer_state=optimizer_check,
                       clip_norm_reference=expected['clip_norm'], clip_norm_candidate=actual['clip_norm'],
                       masks_match=masks_match, observed_cuda_devices=sorted(seen_devices), finite=True)
            results.append(row)
            print(json.dumps(dict(case=args.case_id, batch=length, micro=micro, passed=passed,
                                 prediction_maxabs=row['prediction_maxabs'], gradient_relative_l2=relative_l2)), flush=True)
    return dict(status='complete', kind='quality', case_id=args.case_id, model=args.model, source=source,
                passed=all(r['passed'] for r in results), tolerances=TOLERANCES, results=results,
                optimizer_reset=False, optimizer_state_source='saved checkpoint; deep-copied before every comparison',
                optimizer_comparison='Each comparison reloads the checkpoint AdamW state before clip=1 and one update; exp_avg, exp_avg_sq and step included',
                comparison='same trained weights; eval disables dropout; serial one-origin gradient accumulation is reference',
                limitation='Training dropout changes random-number assignment when physical microbatch changes; '
                           'this checks deterministic numerical equivalence, not identical stochastic training trajectories or final ADE.',
                formal_optimizer_steps_executed=0, confirmation_opened=False, test_opened=False)


def run_case(cell, micro, steps, case_id, quality_micros=None):
    output = REPORT / 'case_logs' / f'{case_id}.json'
    if output.exists():
        return json.loads(output.read_text())
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--model', cell, '--micro', str(micro),
           '--steps', str(steps), '--case-id', case_id]
    if quality_micros:
        cmd += ['--quality-micros', *map(str, quality_micros)]
    log = REPORT / 'case_logs' / f'{case_id}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        code = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
    if not output.exists():
        write_json(output, dict(status='failed', model=cell, micro_per_gpu=micro, case_id=case_id, exit_code=code, log=str(log)))
    value = json.loads(output.read_text())
    print(json.dumps(dict(case=case_id, status=value['status'],
                         speed=value.get('origins_per_second_with_checkpoint'), passed=value.get('passed'))), flush=True)
    return value


def coordinate():
    TEMP.mkdir(parents=True, exist_ok=True)
    with (TEMP/'benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = dict(status='running', created_at=time.time(), global_batch=16, candidates=CANDIDATES,
                      formal_optimizer_steps_executed=0, confirmation_opened=False, test_opened=False,
                      train_origins=5549, formal_epochs=40, stages={}, tolerances=TOLERANCES,
                      precision='unchanged float32 model and complex128 quantum path; AMP/TF32/compile disabled')
        for cell in CELLS:
            run_case(cell, 1, 8, f'{cell}_m1_screen')
        for cell in CELLS:
            stage = dict(screen=[], repeats=[], quality=None)
            result['stages'][cell] = stage
            for micro in CANDIDATES[cell]:
                stage['screen'].append(run_case(cell, micro, 8, f'{cell}_m{micro}_screen'))
                write_json(REPORT/'raw_results.json', result)
            successful = [row for row in stage['screen'] if row['status']=='complete']
            ranked = sorted(successful, key=lambda row:row['with_checkpoint_seconds']['median'])
            if not ranked:
                continue
            selected = list(dict.fromkeys([1]+[row['micro_per_gpu'] for row in ranked[:2]]))
            for repeat in range(2):
                # Reverse the second sweep to reduce fixed-order thermal/time bias.
                for micro in selected if repeat == 0 else selected[::-1]:
                    stage['repeats'].append(run_case(cell, micro, 24, f'{cell}_m{micro}_repeat{repeat+1}'))
                    write_json(REPORT/'raw_results.json', result)
            completed = [row for row in stage['repeats'] if row['status']=='complete']
            timings = {micro:statistics.median([step['with_checkpoint_seconds'] for row in completed
                       if row['micro_per_gpu']==micro for step in row['steps'] if not step['warmup']]) for micro in selected
                       if any(row['micro_per_gpu']==micro for row in completed)}
            best = min(timings, key=timings.get)
            stage['repeated_median_seconds'] = timings
            stage['fastest_micro_per_gpu'] = best
            stage['quality'] = run_case(cell, best, 8, f'{cell}_quality', list(dict.fromkeys([1, best])))
            write_json(REPORT/'raw_results.json', result)
            print(json.dumps(dict(model_complete=cell, fastest_micro=best, repeated_median_seconds=timings,
                                 quality_passed=stage['quality'].get('passed'))), flush=True)
        result.update(status='complete', finished_at=time.time())
        write_json(REPORT/'raw_results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--model', choices=CELLS)
    parser.add_argument('--micro', type=int, default=1)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--case-id')
    parser.add_argument('--quality-micros', type=int, nargs='+')
    args = parser.parse_args()
    if args.all:
        coordinate()
        return
    if not args.model or not args.case_id or not args.case_id.replace('_','').isalnum():
        parser.error('An isolated case requires --model and a safe --case-id')
    if args.micro not in CANDIDATES[args.model] or args.steps < 8 or args.steps > 100:
        parser.error('Physical microbatch or measurement length outside the benchmark contract')
    if args.quality_micros and any(m not in CANDIDATES[args.model] for m in args.quality_micros):
        parser.error('Invalid quality microbatch')
    output = REPORT/'case_logs'/f'{args.case_id}.json'
    try:
        result = quality(args) if args.quality_micros else benchmark(args)
    except Exception as exc:
        result = dict(status='failed', case_id=args.case_id, model=args.model, micro_per_gpu=args.micro,
                      error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        write_json(output, result)
        raise
    write_json(output, result)


if __name__ == '__main__':
    main()
