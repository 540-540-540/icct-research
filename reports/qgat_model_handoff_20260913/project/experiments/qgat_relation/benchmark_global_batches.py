"""Isolated global-batch sweep; throughput evidence is not cross-batch quality evidence."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation import benchmark_batches as shared

REPORT = ROOT / 'reports/qgat_relation_batch_benchmark/global_batches'
TEMP = ROOT / '.codex-work/qgat-global-benchmark-20260913'
CELLS = shared.CELLS
BATCHES = (16, 32, 64, 128, 256)
TOLERANCES = shared.TOLERANCES


def batches(training, n, seed, size):
    epoch = 0
    while True:
        order, snrs = training.balanced_schedule(n, seed, epoch)
        for cursor in range(0, n, size):
            yield epoch, cursor, order[cursor:cursor+size], snrs[cursor:cursor+size]
        epoch += 1


def benchmark(args):
    import torch
    paired, model, parallel, optimizer, train, cfg, device, devices, source = shared.setup(args.model)
    training = paired.training
    size = args.batch
    seen_devices = set()
    model.graph.register_forward_hook(lambda module, inputs, output: seen_devices.add(output.device.index))
    model.train()
    destination = TEMP / args.case_id / 'latest.pt'
    config = dict(model=args.model, global_batch=size, physical_forward=size,
                  max_micro_per_gpu=size//2 if args.model=='gnn' else size,
                  seed=2026, source=source, optimizer_initialization='fresh AdamW then two warmup updates')
    progress = dict(epoch=0, cursor=0, optimizer_steps=0, seen=0, loss_sum=0., history=[], step_seconds=[])
    rows = []
    missing = set()
    schedule = batches(training, train.n, cfg['seed'], size)
    shared.finite_parameters(model)
    for step in range(args.steps+2):
        if step == 2:
            shared.synchronize(devices)
            for index in devices:
                torch.cuda.reset_peak_memory_stats(index)
        epoch, cursor, ids, snrs = next(schedule)
        assert 1 <= len(ids) <= size
        shared.synchronize(devices)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        inputs, truth = train.batch(ids, snrs, device)
        output = training.forward(model, inputs, 'joint')
        metric = training.masked_trajectory_loss(**output, **truth)
        assert not metric['skip_optimizer'], 'A measured step must contain supervision'
        if not bool(torch.isfinite(metric['loss'])):
            raise FloatingPointError('Nonfinite loss')
        metric['loss'].backward()
        weighted_loss = float(metric['loss'].detach())*len(ids)
        graph_norms = [p.grad.detach().norm() for name,p in model.named_parameters() if name.startswith('graph.') and p.grad is not None]
        graph_norm = float(torch.linalg.vector_norm(torch.stack(graph_norms))) if graph_norms else 0.
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        shared.synchronize(devices)
        compute_seconds = time.perf_counter()-tick
        missing.update(shared.finite_parameters(model, gradients=True))
        shared.finite_parameters(model)
        progress.update(epoch=epoch, cursor=cursor+len(ids), optimizer_steps=step+1,
                        seen=progress['seen']+len(ids), loss_sum=progress['loss_sum']+weighted_loss,
                        gradient_norm=float(norm), graph_gradient_norm=graph_norm)
        progress['step_seconds'].append(compute_seconds)
        shared.synchronize(devices)
        tick = time.perf_counter()
        training.save_checkpoint(destination, dict(config=config, model=training.mutable_state(model),
                                 optimizer=optimizer.state_dict(), rng=training.rng_state(device), progress=progress))
        shared.synchronize(devices)
        io_seconds = time.perf_counter()-tick
        row = dict(step=step-2, warmup=step<2, epoch=epoch, cursor=cursor, samples=len(ids), real_epoch_tail=len(ids)<size,
                   optimizer_updates=1, origin_ids=ids.tolist(), snr_db=snrs.tolist(), loss=weighted_loss/len(ids),
                   compute_seconds=compute_seconds, checkpoint_seconds=io_seconds, with_checkpoint_seconds=compute_seconds+io_seconds,
                   origins_per_second_with_checkpoint=len(ids)/(compute_seconds+io_seconds),
                   gradient_norm=float(norm), graph_gradient_norm=graph_norm, finite=True)
        rows.append(row)
        print(json.dumps(dict(case=args.case_id, step=step+1, total=args.steps+2, epoch=epoch,
                             samples=len(ids), seconds=round(compute_seconds+io_seconds, 4))), flush=True)
    measured = rows[2:]
    assert len(measured)>=8 and seen_devices==set(devices)
    samples = sum(row['samples'] for row in measured)
    memory = {str(i):dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(i), peak_reserved_bytes=torch.cuda.max_memory_reserved(i),
                         total_bytes=torch.cuda.get_device_properties(i).total_memory) for i in devices}
    return dict(status='complete', kind='throughput', model=args.model, case_id=args.case_id,
                global_batch=size, max_micro_per_gpu=size//2 if args.model=='gnn' else size, physical_forward=size,
                warmup_updates=2, measured_updates=len(measured), measured_samples=samples,
                effective_batch_tail=train.n%size, updates_per_epoch=math.ceil(train.n/size), expected_optimizer_updates_40epochs=math.ceil(train.n/size)*40,
                source=source, optimizer_reset=True, missing_gradient_names=sorted(missing), finite=True,
                compute_seconds=shared.summary([r['compute_seconds'] for r in measured]),
                checkpoint_seconds=shared.summary([r['checkpoint_seconds'] for r in measured]),
                with_checkpoint_seconds=shared.summary([r['with_checkpoint_seconds'] for r in measured]),
                step_throughput_with_checkpoint=shared.summary([r['origins_per_second_with_checkpoint'] for r in measured]),
                origins_per_second=samples/sum(r['compute_seconds'] for r in measured),
                origins_per_second_with_checkpoint=samples/sum(r['with_checkpoint_seconds'] for r in measured),
                peak_memory=memory, observed_cuda_devices=sorted(seen_devices), steps=rows,
                temporary_checkpoint_bytes=destination.stat().st_size,
                precision=dict(parameter_dtype='float32',quantum_state_dtype='complex128',amp=False,tf32=False,compile=False,
                               float32_matmul_precision=torch.get_float32_matmul_precision()),
                formal_optimizer_steps_executed=0, confirmation_opened=False, test_opened=False,
                limitation='Changing global batch changes optimizer updates and stochastic training; throughput is not a final-quality comparison.')


def compare_states(actual, expected, prefix):
    import torch
    assert actual.keys()==expected.keys()
    maxabs = squared_delta = squared_reference = 0.
    failed=[]
    for name,reference in expected.items():
        value=actual[name]
        delta=value.double()-reference.double()
        maxabs=max(maxabs,float(delta.abs().max()))
        squared_delta+=float(delta.square().sum())
        squared_reference+=float(reference.double().square().sum())
        if not torch.allclose(value,reference,atol=TOLERANCES[prefix+'_atol'],rtol=TOLERANCES[prefix+'_rtol']):
            failed.append(name)
    relative=(squared_delta/max(squared_reference,1e-30))**.5
    return dict(maxabs=maxabs,relative_l2=relative,failed_tensor_names=failed,tensor_count=len(expected),
                passed=not failed and relative<=TOLERANCES[prefix+'_relative_l2'])


def quality(args):
    import torch
    paired, model, parallel, optimizer, train, cfg, device, devices, source=shared.setup(args.model)
    training=paired.training
    model.eval()
    if parallel is not None:
        parallel.eval()
    order,snrs=training.balanced_schedule(train.n,cfg['seed'],0)
    saved=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
    seen_devices=set()
    model.graph.register_forward_hook(lambda module, inputs, output:seen_devices.add(output.device.index))

    def evaluate_batch(ids,conditions,serial):
        training.load_weights(model,saved['model'])
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        model.zero_grad(set_to_none=True)
        predictions,eligibility=[],[]
        loss=0.
        physical=1 if serial else args.batch
        for start in range(0,len(ids),physical):
            part=ids[start:start+physical]
            inputs,truth=train.batch(part,conditions[start:start+physical],device)
            output=model(**inputs) if serial or parallel is None else parallel(**inputs)
            metric=training.masked_trajectory_loss(**output,**truth)
            weighted=metric['loss']*len(part)/len(ids)
            weighted.backward()
            loss+=float(weighted.detach())
            predictions.append(output['prediction'].detach().cpu())
            eligibility.append(output['origin_eligible'].detach().cpu())
        shared.finite_parameters(model,gradients=True)
        grads={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        prediction,eligible=torch.cat(predictions),torch.cat(eligibility)
        _,truth=train.batch(ids,conditions,'cpu')
        full=training.masked_trajectory_loss(prediction=prediction,origin_eligible=eligible,**truth)
        metrics=training.scene_metrics(prediction=prediction,origin_eligible=eligible,**truth)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        shared.finite_parameters(model)
        parameters={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
        optimizer_tensors={f'{name}.{key}':value.detach().cpu().clone() for name,p in model.named_parameters()
                           for key,value in optimizer.state.get(p,{}).items() if isinstance(value,torch.Tensor)}
        return dict(prediction=prediction,origin_eligible=eligible,grads=grads,parameters=parameters,optimizer_tensors=optimizer_tensors,
                    clip_norm=float(norm),metrics=dict(loss=float(full['loss']),accumulated_loss=loss,
                    ADE=float(metrics['scene_ade'].mean()),FDE=float(metrics['scene_fde'].mean())))

    results=[]
    for length in (args.batch,train.n%args.batch):
        assert length>0
        tail=length!=args.batch
        ids,conditions=(order[-length:],snrs[-length:]) if tail else (order[:length],snrs[:length])
        expected=evaluate_batch(ids,conditions,True)
        seen_devices.clear()
        actual=evaluate_batch(ids,conditions,False)
        prediction_close=bool(torch.allclose(actual['prediction'],expected['prediction'],
                              atol=TOLERANCES['prediction_atol'],rtol=TOLERANCES['prediction_rtol']))
        gradient=compare_states(actual['grads'],expected['grads'],'gradient')
        parameters=compare_states(actual['parameters'],expected['parameters'],'parameter')
        optimizer_check={key:compare_states({n:v for n,v in actual['optimizer_tensors'].items() if n.endswith('.'+key)},
                          {n:v for n,v in expected['optimizer_tensors'].items() if n.endswith('.'+key)},'optimizer')
                         for key in ('exp_avg','exp_avg_sq','step')}
        differences={k:abs(actual['metrics'][k]-expected['metrics'][k]) for k in expected['metrics']}
        metrics_close=all(v<=TOLERANCES['metric_atol']+TOLERANCES['metric_rtol']*abs(expected['metrics'][k]) for k,v in differences.items())
        masks_match=torch.equal(actual['origin_eligible'],expected['origin_eligible'])
        passed=prediction_close and gradient['passed'] and parameters['passed'] and all(r['passed'] for r in optimizer_check.values()) and metrics_close and masks_match and seen_devices==set(devices)
        row=dict(global_batch=args.batch,batch_length=length,real_epoch_tail=tail,passed=passed,
                 origin_ids=ids.tolist(),snr_db=conditions.tolist(),prediction_maxabs=float((actual['prediction']-expected['prediction']).abs().max()),
                 prediction_close=prediction_close,gradients=gradient,one_update_parameters=parameters,one_update_optimizer_state=optimizer_check,
                 metrics_reference=expected['metrics'],metrics_candidate=actual['metrics'],metric_abs_differences=differences,
                 clip_norm_reference=expected['clip_norm'],clip_norm_candidate=actual['clip_norm'],masks_match=masks_match,
                 observed_cuda_devices=sorted(seen_devices),finite=True)
        results.append(row)
        print(json.dumps(dict(case=args.case_id,batch=length,passed=passed,prediction_maxabs=row['prediction_maxabs'],gradient_relative_l2=gradient['relative_l2'])),flush=True)
    return dict(status='complete',kind='quality',model=args.model,case_id=args.case_id,global_batch=args.batch,source=source,
                passed=all(r['passed'] for r in results),tolerances=TOLERANCES,results=results,
                optimizer_reset=False,optimizer_state_source='saved checkpoint; deep-copied before every comparison',
                comparison='Same global batch, same trained weights and AdamW state; dropout off; one-origin accumulation versus full physical batch.',
                limitation='This is within-global-batch numerical parity. It does not compare final quality between global batch sizes.',
                formal_optimizer_steps_executed=0,confirmation_opened=False,test_opened=False)


def run_case(cell,size,steps,case_id,quality_only=False):
    output=REPORT/'case_logs'/f'{case_id}.json'
    if output.exists():
        stored=json.loads(output.read_text())
        if stored['status']=='complete':
            source=stored['source']; stat=Path(source['checkpoint']).stat()
            if stat.st_size!=source['checkpoint_bytes'] or stat.st_mtime_ns!=source['checkpoint_mtime_ns']:
                raise ValueError(f'{case_id}: checkpoint changed; use a fresh case identifier')
            if stored['global_batch']!=size or (stored['kind']=='quality')!=quality_only:
                raise ValueError(f'{case_id}: stored case settings differ')
            if not quality_only and stored['measured_updates']!=steps:
                raise ValueError(f'{case_id}: stored measurement length differs')
        return stored
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',cell,'--batch',str(size),'--steps',str(steps),'--case-id',case_id]
    if quality_only:
        cmd.append('--quality')
    log=output.with_suffix('.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        code=subprocess.call(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                             env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    if not output.exists():
        shared.write_json(output,dict(status='failed',model=cell,global_batch=size,case_id=case_id,exit_code=code))
    value=json.loads(output.read_text())
    print(json.dumps(dict(case=case_id,status=value['status'],speed=value.get('origins_per_second_with_checkpoint'),passed=value.get('passed'))),flush=True)
    return value


def coordinate():
    TEMP.mkdir(parents=True,exist_ok=True)
    with (TEMP/'benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=dict(status='running',created_at=time.time(),train_origins=5549,formal_epochs=40,
                    planned_global_batches=[16,32,64,128],adaptive_global_batch=256,stages={},tolerances=TOLERANCES,
                    precision='FP32 and original complex128 quantum path; AMP/TF32/compile disabled',
                    formal_optimizer_steps_executed=0,confirmation_opened=False,test_opened=False,
                    limitation='Larger global batch reduces optimizer updates per epoch. Throughput gains are not quality-equivalence evidence.')
        # Finish all models' initial sweeps before spending time on long repeats.
        for cell in CELLS:
            stage=dict(screen=[],repeats=[],quality=None)
            result['stages'][cell]=stage
            for size in BATCHES[:4]:
                stage['screen'].append(run_case(cell,size,8,f'{cell}_b{size}_screen'))
                shared.write_json(REPORT/'raw_results.json',result)
            successful=[r for r in stage['screen'] if r['status']=='complete']
            top=max(successful,key=lambda r:r['origins_per_second_with_checkpoint'])
            can_grow=top['global_batch']==128 and all(v['peak_reserved_bytes']/v['total_bytes']<.85 for v in top['peak_memory'].values())
            stage['test_256_reason']='128 fastest with reserved memory below 85%' if can_grow else '128 not fastest or insufficient reserved-memory margin'
            if can_grow:
                stage['screen'].append(run_case(cell,256,8,f'{cell}_b256_screen'))
                shared.write_json(REPORT/'raw_results.json',result)
            successful=[r for r in stage['screen'] if r['status']=='complete']
            print(json.dumps(dict(screen_complete=cell,ranked=[(r['global_batch'],r['origins_per_second_with_checkpoint']) for r in sorted(successful,key=lambda r:r['origins_per_second_with_checkpoint'],reverse=True)])),flush=True)
        for cell,stage in result['stages'].items():
            successful=[r for r in stage['screen'] if r['status']=='complete']
            ranked=sorted(successful,key=lambda r:r['origins_per_second_with_checkpoint'],reverse=True)
            selected=list(dict.fromkeys([16]+[r['global_batch'] for r in ranked[:2]]))
            for repeat in range(2):
                for size in selected if repeat==0 else selected[::-1]:
                    stage['repeats'].append(run_case(cell,size,24,f'{cell}_b{size}_repeat{repeat+1}'))
                    shared.write_json(REPORT/'raw_results.json',result)
            valid_sizes=[size for size in selected if sum(r['status']=='complete' and r['global_batch']==size for r in stage['repeats'])==2]
            # An initial eight-step fit can miss a denser later batch; replace failed long-run candidates.
            for candidate in ranked:
                if len(valid_sizes)>=min(3,len(ranked)):
                    break
                size=candidate['global_batch']
                if size in selected:
                    continue
                selected.append(size)
                for repeat in range(2):
                    stage['repeats'].append(run_case(cell,size,24,f'{cell}_b{size}_repeat{repeat+1}'))
                    shared.write_json(REPORT/'raw_results.json',result)
                if sum(r['status']=='complete' and r['global_batch']==size for r in stage['repeats'])==2:
                    valid_sizes.append(size)
            speeds={size:statistics.median([step['origins_per_second_with_checkpoint'] for r in stage['repeats']
                    if r['status']=='complete' and r['global_batch']==size for step in r['steps'] if not step['warmup']]) for size in valid_sizes}
            if not speeds:
                stage['error']='No candidate completed both repeat runs'
                continue
            best=max(speeds,key=speeds.get)
            stage['repeated_median_origins_per_second_with_checkpoint']=speeds
            stage['fastest_global_batch']=best
            stage['quality']=run_case(cell,best,8,f'{cell}_b{best}_quality',True)
            shared.write_json(REPORT/'raw_results.json',result)
            print(json.dumps(dict(model_complete=cell,fastest_global_batch=best,repeated_median_speed=speeds,quality_passed=stage['quality'].get('passed'))),flush=True)
        result.update(status='complete',finished_at=time.time())
        shared.write_json(REPORT/'raw_results.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--model',choices=CELLS)
    parser.add_argument('--batch',type=int,choices=BATCHES)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--case-id')
    parser.add_argument('--quality',action='store_true')
    args=parser.parse_args()
    if args.all:
        coordinate()
        return
    if not args.model or not args.batch or not args.case_id or not args.case_id.replace('_','').isalnum() or not 8<=args.steps<=100:
        parser.error('A case needs --model, --batch, safe --case-id, and 8..100 measured steps')
    output=REPORT/'case_logs'/f'{args.case_id}.json'
    try:
        result=quality(args) if args.quality else benchmark(args)
    except Exception as exc:
        import torch
        result=dict(status='failed',model=args.model,global_batch=args.batch,case_id=args.case_id,
                    error=f'{type(exc).__name__}: {exc}',oom=isinstance(exc,torch.cuda.OutOfMemoryError),traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            result['peak_memory']={str(i):dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(i),peak_reserved_bytes=torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count())}
        shared.write_json(output,result)
        raise
    shared.write_json(output,result)


if __name__=='__main__':
    main()
