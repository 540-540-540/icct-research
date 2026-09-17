"""Reproducible equivalence and GPU cost checks, never optimizer steps."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from prediction.quantum import QuantumGraph
from prediction.quantum_torch_reference import QuantumGraph as Reference


def load_original():
    spec = importlib.util.spec_from_file_location('quantum_original', ROOT/'reports/quantum_speed/quantum_original.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QuantumGraph


def run(model, data, mask, checkpoint, frame_batch=None):
    x = data.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    kwargs = dict(checkpoint=checkpoint)
    if frame_batch is not None:
        kwargs['frame_batch'] = frame_batch
    y = model.forward_history(x, mask, **kwargs)
    torch.cuda.synchronize()
    forward = time.perf_counter()-start
    # Nonuniform cotangent exposes frame/slot reordering errors.
    weights = torch.linspace(-.7, .9, y.numel(), device=y.device).reshape_as(y)
    (y*weights).sum().backward()
    torch.cuda.synchronize()
    total = time.perf_counter()-start
    stats = dict(forward_s=forward, backward_s=total-forward, total_s=total,
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(model.theta.grad).all()
    stats['finite_output_and_gradients'] = True
    stats['effective_checkpoint'] = checkpoint if checkpoint is not None else data.shape[0]*data.shape[1]>20
    return (y.detach(), x.grad.detach(), model.theta.grad.detach().clone()), stats


def difference(actual, expected):
    result = {}
    for name, a, b in zip(('output','input_gradient','parameter_gradient'),actual,expected):
        result[name] = float((a-b).abs().max())
        torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-8)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['correctness','benchmark'], default='correctness')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--frame-batches', default='1,4,8')
    p.add_argument('--include-original', action='store_true')
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--synthetic-n', type=int, choices=range(1,9))
    p.add_argument('--repeat', type=int, default=1)
    args = p.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20260912)
    report = dict(mode=args.mode, device=torch.cuda.get_device_name(), cases=[])
    path = ROOT/'reports/quantum_speed'/f'{args.mode}-{time.strftime("%Y%m%d-%H%M%S")}.json'
    def record(item):
        report['cases'].append(item)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    if args.mode == 'correctness':
        for depth, legacy in [(2,False),(3,False),(3,True)]:
            model = QuantumGraph([40,40,15,15],depth=depth,legacy=legacy).cuda()
            reference = Reference([40,40,15,15],depth=depth,legacy=legacy).cuda()
            with torch.no_grad(): model.theta.add_(torch.randn_like(model.theta)*.2)
            reference.load_state_dict(model.state_dict())
            data = torch.randn(2,5,4,4,dtype=torch.float64,device='cuda')*10
            counts = torch.tensor([[0,1,4,3,2],[3,4,1,0,2]],device='cuda')
            order = torch.rand(2,5,4,device='cuda').argsort(-1)
            mask = order < counts[...,None]
            data[~mask] = float('nan')
            baseline, stats = run(reference,data,mask,False)
            for checkpoint in (False,True,None):
                actual, timing = run(model,data,mask,checkpoint,8)
                error = difference(actual,baseline)
                assert (actual[0][~mask]==0).all() and (actual[1][~mask]==0).all()
                record(dict(depth=depth,legacy=legacy,checkpoint=checkpoint,max_error=error,**timing))
        # Exercise full 16-qubit state and batched same-N differentiation.
        model = QuantumGraph([40,40,15,15]).cuda()
        original = load_original()([40,40,15,15]).cuda()
        with torch.no_grad(): model.theta.add_(torch.randn_like(model.theta)*.15)
        original.load_state_dict(model.state_dict())
        data = torch.randn(1,2,8,4,dtype=torch.float64,device='cuda')*10
        mask = torch.ones(1,2,8,dtype=torch.bool,device='cuda')
        baseline,_ = run(original,data,mask,True)
        for checkpoint in (False,True,None):
            actual,timing = run(model,data,mask,checkpoint,20)
            record(dict(depth=3,n=8,checkpoint=checkpoint,comparison='original PennyLane serial',
                        max_error=difference(actual,baseline),**timing))
        data = torch.randn(2,12,2,4,dtype=torch.float64,device='cuda')
        mask = torch.ones(2,12,2,dtype=torch.bool,device='cuda')
        baseline,_ = run(original,data,mask,True)
        actual,timing = run(model,data,mask,None,20)
        record(dict(depth=3,n=2,frames=24,checkpoint='auto-checkpoint',
                    max_error=difference(actual,baseline),**timing))
    else:
        from frontend.symbol_dataset import SharedPredictionInputs
        loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
        order = np.random.default_rng(20260912).permutation(len(loader))[:args.batch]
        report['sample_seed'] = 20260912
        report['sample_ids'] = [int(i) for i in order]
        samples = [loader[int(i)] for i in order]
        data = torch.as_tensor(np.stack([s['state_hat'] for s in samples]),device='cuda')
        mask = torch.as_tensor(np.stack([s['track_exists'] for s in samples]),device='cuda')
        if args.synthetic_n:
            data = torch.randn(args.batch,20,8,4,dtype=torch.float64,device='cuda')*10
            mask = torch.arange(8,device='cuda').expand(args.batch,20,8) < args.synthetic_n
        scales = loader.normalization
        # Only the graph is constructed: no temporal checkpoint or large allocations.
        scale = scales['quantum_scale']
        model = QuantumGraph(scale).cuda()
        with torch.no_grad(): model.theta.add_(torch.randn_like(model.theta)*.15)
        variants = [(int(v),not args.no_checkpoint) for v in args.frame_batches.split(',')]
        if args.include_original:
            original = load_original()(scale).cuda()
            original.load_state_dict(model.state_dict())
            _,timing=run(original,data,mask,True)
            record(dict(variant='original',batch=args.batch,**timing))
        for frame_batch, checkpoint in variants:
            for repeat in range(args.repeat):
                _,timing=run(model,data,mask,checkpoint,frame_batch)
                record(dict(variant='bucket',batch=args.batch,frame_batch=frame_batch,checkpoint=checkpoint,
                            repeat=repeat,synthetic_n=args.synthetic_n,
                            active_counts=mask.sum(-1).cpu().tolist(),**timing))
    report['passed']=True
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(path,flush=True)


if __name__ == '__main__': main()
