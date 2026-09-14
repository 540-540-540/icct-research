"""Isolated short quality screen for a physical micro-batch; never writes formal weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qgat_relation import run as paired
from experiments.qgat_relation import run_gnn as gnn

training = paired.training
REPORT = ROOT/'reports/qgat_relation_batch_benchmark/quality_runs'


def checkpoint_for(name):
    folder = 'qgat_relation_gnn' if name == 'gnn' else 'qgat_relation'
    cell = 'gnn' if name == 'gnn' else name
    return ROOT/'results'/folder/'seed2026'/cell/'latest.pt'


def next_batches(progress, samples, global_batch, train):
    epoch, cursor = progress['epoch'], progress['cursor']
    remaining = samples
    while remaining:
        order, snrs = training.balanced_schedule(train.n, 2026, epoch)
        while cursor < train.n and remaining:
            size = min(global_batch, train.n-cursor, remaining)
            yield epoch, order[cursor:cursor+size], snrs[cursor:cursor+size]
            cursor += size
            remaining -= size
        epoch, cursor = epoch+1, 0


def fixed_training_probe(model, train, progress, device):
    was_training = model.training
    model.eval()
    loss_sum, origins, conditions = 0., [], []
    with torch.no_grad():
        for _, ids, snrs in next_batches(progress,64,1,train):
            inputs, truth = train.batch(ids,snrs,device)
            result = training.forward(model,inputs)
            metric = training.masked_trajectory_loss(**result,**truth)
            loss_sum += float(metric['loss'])
            origins.extend(ids.tolist())
            conditions.extend(snrs.tolist())
    model.train(was_training)
    return dict(loss=loss_sum/len(origins), origins=origins, snr_db=conditions,
                interpretation='Fixed training origins, dropout disabled; training progress only')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=('gnn','original_D','relation_D'))
    p.add_argument('--micro', type=int, default=1, help='per-GPU physical micro-batch')
    p.add_argument('--global-batch', type=int, choices=(16,32,64,128,256), default=16)
    p.add_argument('--repeat', type=int, choices=(0,1,2), default=0)
    p.add_argument('--samples', type=int, choices=(512,1024,2048,4077,4096), default=1024,
                   help='Same ordered training origins for every batch configuration')
    p.add_argument('--device', type=int, choices=(0,1), default=0)
    p.add_argument('--initial-only', action='store_true')
    p.add_argument('--check-validation-batches', action='store_true')
    args = p.parse_args()
    devices = 2 if args.model=='gnn' else 1
    if args.micro not in (1,2,4,8,16,32,64,128,256) or args.micro*devices>args.global_batch:
        p.error('Physical micro-batch must fit the selected global batch')
    if args.model=='gnn' and args.device!=0:
        p.error('GNN uses both GPUs with cuda:0 as primary')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = f'cuda:{args.device}'
    path = checkpoint_for(args.model)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    identity = dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                    saved_optimizer_steps=checkpoint['progress']['optimizer_steps'])
    cfg = paired.configuration()
    model = (gnn.Predictor('gnn',2026,cfg) if args.model=='gnn' else paired.Predictor(args.model,cfg)).to(device)
    training.load_weights(model, checkpoint['model'])
    optimizer = (paired.trial.ORIGINAL_OPTIMIZER if args.model=='gnn' else paired.trial.make_optimizer)(model,'joint',cfg['graph_lr'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train, valid = paired.trial.dataset('train'), paired.trial.dataset('V_select')
    assert train.n==5549 and valid.n==400
    if args.model=='gnn':
        parallel = gnn.attach_parallel(model)
    output = REPORT/args.model/('initial.json' if args.initial_only else
                              f'batch{args.global_batch}_micro{args.micro}_samples{args.samples}_repeat{args.repeat}.json')
    if output.exists():
        raise FileExistsError(f'Quality result already exists: {output}')
    record = dict(model=args.model, per_device_micro_batch=args.micro, repeat=args.repeat,
                  initial_checkpoint=identity, global_batch_size=args.global_batch, phase='short quality screen',
                  requested_training_origins=0 if args.initial_only else args.samples,
                  precision='Original FP32 model and complex128 quantum path; AMP and TF32 disabled',
                  optimizer_state_reused=True, rng_policy='saved checkpoint RNG for repeat 0; paired independent dropout streams for repeats 1/2',
                  confirmation_opened=False, test_opened=False, formal_weights_written=False)
    record['initial_training_probe'] = fixed_training_probe(model,train,checkpoint['progress'],device)
    if not args.initial_only:
        if args.repeat==0:
            training.restore_rng(checkpoint['rng'], device)
        else:
            training.seed_all(700000+args.repeat)
        seen_devices = set()
        model.graph.register_forward_hook(lambda module, inputs, result: seen_devices.add(result.device.index))
        records = []
        model.train()
        micro = args.micro*2 if args.model=='gnn' else args.micro
        started = time.perf_counter()
        processed = 0
        for number, (epoch, ids, snrs) in enumerate(next_batches(checkpoint['progress'],args.samples,args.global_batch,train),1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum, supervised = 0., False
            for start in range(0,len(ids),micro):
                part, condition = ids[start:start+micro], snrs[start:start+micro]
                inputs, truth = train.batch(part,condition,device)
                result = training.forward(model,inputs)
                metric = training.masked_trajectory_loss(**result,**truth)
                supervised |= not metric['skip_optimizer']
                (metric['loss']*len(part)/len(ids)).backward()
                loss_sum += float(metric['loss'].detach())*len(part)
            if not supervised:
                raise RuntimeError('The paired global batch has no supervision')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            records.append(dict(step=number, epoch=epoch, origins=ids.tolist(), snr_db=snrs.tolist(),
                                loss=loss_sum/len(ids), gradient_norm=float(norm)))
            processed += len(ids)
            if number%8==0:
                print(f'{args.model} batch={args.global_batch} micro={args.micro} repeat={args.repeat} '
                      f'quality samples {processed}/{args.samples}, updates={number}',flush=True)
        assert processed==args.samples
        assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
        record.update(optimizer_steps_executed=len(records), training_origins_processed=processed, training_trace=records,
                      observed_cuda_devices=sorted(seen_devices), training_seconds=time.perf_counter()-started)
        record['final_training_probe'] = fixed_training_probe(model,train,checkpoint['progress'],device)
    else:
        record['optimizer_steps_executed']=0
    started = time.perf_counter()
    metrics = training.evaluate(model,valid,device,micro_batch=1)
    record.update(metrics=metrics, validation_seconds=time.perf_counter()-started)
    if args.check_validation_batches:
        checks = []
        reference = metrics['per_scene']
        evaluate_batched = gnn.ORIGINAL_EVALUATE if args.model=='gnn' else training.evaluate
        for validation_micro in (16,32,64,128):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            batch_sizes, original_batch = [], valid.batch
            def observed_batch(ids,*values,**keywords):
                batch_sizes.append(len(ids))
                return original_batch(ids,*values,**keywords)
            valid.batch = observed_batch
            try:
                candidate = evaluate_batched(model,valid,device,micro_batch=validation_micro)
            finally:
                valid.batch = original_batch
            torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter()-started
            actual = candidate['per_scene']
            same_coverage = len(actual)==len(reference) and all(
                all(a[k]==b[k] for k in ('origin','snr_db','ade_targets','fde_targets'))
                for a,b in zip(actual,reference))
            errors, numerical_pass = {}, True
            for k in ('ADE','FDE'):
                masks_match = [a[k] is None for a in actual]==[b[k] is None for b in reference]
                pairs = [(a[k],b[k]) for a,b in zip(actual,reference) if a[k] is not None and b[k] is not None]
                errors[k] = max((abs(a-b) for a,b in pairs),default=0.)
                numerical_pass &= masks_match and np.allclose(
                    [a for a,b in pairs],[b for a,b in pairs],atol=3e-5,rtol=1e-6)
            passed = bool(same_coverage and numerical_pass and max(batch_sizes)==validation_micro)
            checks.append(dict(micro_batch=validation_micro, seconds=elapsed, passed=passed,
                               same_scene_snr_target_denominators=same_coverage,
                               observed_batch_sizes=batch_sizes,
                               max_per_scene_error=errors,
                               metrics={k:candidate[k] for k in ('ADE','FDE','J')},
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device)))
            print(f'{args.model} validation micro={validation_micro} seconds={elapsed:.3f} parity={passed}',flush=True)
        record['validation_batch_checks'] = checks
    assert path.stat().st_size==identity['size'] and path.stat().st_mtime_ns==identity['mtime_ns']
    training.write_json(output,record)
    print(json.dumps(dict(output=str(output),ADE=metrics['ADE'],FDE=metrics['FDE'],J=metrics['J'],
                          optimizer_steps=record['optimizer_steps_executed'])),flush=True)


if __name__=='__main__':
    main()
