"""Independent small QGAT/GNN development trial; not F04/F05 or a main result."""
import argparse
import fcntl
import json
import re
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frontend.symbol_dataset import SharedPredictionInputs
from prediction.training import (scene_metrics, masked_trajectory_loss, configure_phase,
    mutable_state, seed_all, save_checkpoint, rng_state, write_json)


class NominalData:
    def __init__(self, split):
        if split not in ('train', 'V_select'):
            raise ValueError('This trial permits only train and V_select')
        self.split = split
        self.loader = SharedPredictionInputs(ROOT/f'data/f01d/inputs/{split}_snr_20.npz')
        with np.load(ROOT/f'data/f01d/labels/{split}.npz', allow_pickle=False) as f:
            if set(f.files) != {'future_position', 'label_valid'}:
                raise ValueError('Unexpected supervision fields')
            self.labels = {key: f[key] for key in f.files}
        if self.labels['future_position'].shape != (len(self.loader),20,8,2) or self.labels['label_valid'].shape != (len(self.loader),20,8):
            raise ValueError('Input and supervision shape mismatch')
        if self.labels['label_valid'].dtype != bool or not np.isfinite(self.labels['future_position'][self.labels['label_valid']]).all():
            raise ValueError('Invalid supervision')

    def batch(self, ids, device):
        samples = [self.loader[int(i)] for i in ids]
        inputs = {key: torch.as_tensor(np.stack([s[key] for s in samples]), device=device)
                  for key in ('state_hat','standardized_state','track_exists','detected')}
        truth = {key: torch.as_tensor(value[ids], device=device) for key,value in self.labels.items()}
        return inputs, truth


def choose_origins(n, count, seed):
    if not 1 <= count <= n:
        raise ValueError(f'Requested {count} origins, available {n}')
    return np.sort(np.random.default_rng(seed).choice(n, size=count, replace=False))


def validate_output(output, inputs):
    expected = inputs['track_exists'][:, -1].bool() & (inputs['track_exists'].sum(1) >= 3)
    if not torch.equal(output['origin_eligible'], expected):
        raise ValueError('Candidate changed the common input eligibility mask')
    if not torch.isfinite(output['prediction']).all():
        raise FloatingPointError('Nonfinite prediction; no origins may be removed')


@torch.no_grad()
def evaluate_nominal(model, data, ids, device, micro_batch):
    if data.split != 'V_select':
        raise ValueError('Only V_select may be evaluated')
    was_training = model.training
    model.eval()
    rows = []
    for start in range(0, len(ids), micro_batch):
        part = ids[start:start+micro_batch]
        inputs, truth = data.batch(part, device)
        output = model(**inputs)
        validate_output(output, inputs)
        metrics = scene_metrics(**output, **truth)
        for offset, origin in enumerate(part):
            an, fn = int(metrics['ade_count'][offset]), int(metrics['fde_count'][offset])
            rows.append(dict(origin=int(origin), ADE=float(metrics['scene_ade'][offset]) if an else None,
                             FDE=float(metrics['scene_fde'][offset]) if fn else None,
                             ade_targets=an, fde_targets=fn))
    ade = [r['ADE'] for r in rows if r['ADE'] is not None]
    fde = [r['FDE'] for r in rows if r['FDE'] is not None]
    if not ade or not fde:
        raise ValueError('No scoreable validation origins')
    a, f = statistics.mean(ade), statistics.mean(fde)
    model.train(was_training)
    return dict(ADE=a, FDE=f, J=a+.5*f, origins=len(ids), ade_scenes=len(ade),
                fde_scenes=len(fde), snr_db=20, per_scene=rows)


def copy_temporal(model, state):
    target = {k:v for k,v in mutable_state(model).items() if k.startswith('temporal.')}
    if set(target) != set(state):
        raise ValueError('Temporal architecture differs between candidates')
    model.load_state_dict(state, strict=False)
    assert all(torch.equal(v, model.state_dict()[k].detach().cpu()) for k,v in state.items())


def optimizer_for(model, graph_lr):
    optimizer = configure_phase(model, 'joint', graph_lr)
    # Partition the shared optimizer groups only when a candidate exposes quantum angle tensors.
    angle_ids = {id(p) for n,p in model.named_parameters() if n.startswith('graph.') and any(token in n.lower() for token in ('theta','angle'))}
    for group in list(optimizer.param_groups):
        angles = [p for p in group['params'] if id(p) in angle_ids]
        if angles and group['weight_decay']:
            ordinary = [p for p in group['params'] if id(p) not in angle_ids]
            group['params'] = ordinary
            optimizer.add_param_group(dict(params=angles, lr=group['lr'], weight_decay=0.))
    return optimizer


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_model(name, args, train, valid, train_ids, valid_ids, temporal, report, checkpoint_dir, report_path):
    seed_all(args.seed)
    if name == 'qgat':
        from experiments.qgat_candidate.graph import CandidatePredictor
        model = CandidatePredictor(train.loader.normalization)
    else:
        from prediction.model import GNN
        model = GNN()
    copy_temporal(model, temporal)
    model = model.to(args.device)
    optimizer = optimizer_for(model, args.graph_lr)
    parameters = [p for p in model.parameters() if p.requires_grad]
    record = dict(status='running', display_name='QGAT' if name == 'qgat' else 'GNN',
                  trainable_parameters=sum(p.numel() for p in parameters),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  temporal_initialization_identical=True, epochs=[], steps=[])
    report['models'][name] = record
    write_json(report_path, report)
    tick = time.perf_counter()
    record['initial_validation'] = evaluate_nominal(model, valid, valid_ids, args.device, args.micro_batch)
    record['initial_validation_seconds'] = time.perf_counter()-tick
    seed_all(args.seed+100)
    best_j = record['initial_validation']['J']
    best_epoch = 0
    model_dir = checkpoint_dir/name
    model_dir.mkdir(parents=True, exist_ok=True)
    def save(path, epoch):
        save_checkpoint(path, dict(model=mutable_state(model), optimizer=optimizer.state_dict(),
                                   rng=rng_state(args.device), epoch=epoch, manifest=report['manifest'],
                                   configuration=report['configuration'], experimental_only=True))
    save(model_dir/'initial.pt', 0)
    save(model_dir/'best.pt', 0)
    total_tick = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.random.default_rng(np.random.SeedSequence([args.seed,epoch,710])).permutation(train_ids)
        model.train()
        weighted_loss = 0.
        for cursor in range(0, len(order), args.batch_size):
            ids = order[cursor:cursor+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            if args.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(args.device)
            synchronize(args.device)
            tick = time.perf_counter()
            supervised = False
            batch_loss = 0.
            for start in range(0, len(ids), args.micro_batch):
                part = ids[start:start+args.micro_batch]
                inputs, truth = train.batch(part, args.device)
                output = model(**inputs)
                validate_output(output, inputs)
                loss = masked_trajectory_loss(**output, **truth)
                supervised |= not loss['skip_optimizer']
                (loss['loss'] * len(part)/len(ids)).backward()
                batch_loss += float(loss['loss'].detach()) * len(part)
            graph_grads = [p.grad.detach().norm() for p in model.graph.parameters() if p.grad is not None]
            graph_norm = float(torch.linalg.vector_norm(torch.stack(graph_grads))) if graph_grads else 0.
            full_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            if supervised:
                optimizer.step()
            synchronize(args.device)
            step = dict(epoch=epoch+1, batch=cursor//args.batch_size+1, origins=ids.tolist(),
                        seconds=time.perf_counter()-tick, loss=batch_loss/len(ids),
                        optimizer_step=supervised, graph_gradient_norm=graph_norm, gradient_norm=full_norm,
                        timing_warmup=sum(r['optimizer_step'] for r in record['steps']) < args.timing_warmup_steps,
                        allocated_peak_GiB=torch.cuda.max_memory_allocated(args.device)/2**30 if args.device.type == 'cuda' else None)
            record['steps'].append(step)
            weighted_loss += batch_loss
            write_json(report_path, report)
            print(f'[{name}] epoch {epoch+1}/{args.epochs} origins {cursor+len(ids)}/{len(order)} {step["seconds"]:.3f}s loss {step["loss"]:.5f}', flush=True)
        tick = time.perf_counter()
        scores = evaluate_nominal(model, valid, valid_ids, args.device, args.micro_batch)
        row = dict(epoch=epoch+1, train_loss=weighted_loss/len(order), validation=scores,
                   validation_seconds=time.perf_counter()-tick)
        record['epochs'].append(row)
        save(model_dir/'latest.pt', epoch+1)
        if scores['J'] < best_j:
            best_j, best_epoch = scores['J'], epoch+1
            save(model_dir/'best.pt', epoch+1)
        write_json(report_path, report)
    measured = [r['seconds'] for r in record['steps'] if r['optimizer_step'] and not r['timing_warmup']]
    record.update(status='complete', elapsed_train_validation_seconds=time.perf_counter()-total_tick,
                  measured_steps=len(measured), median_step_seconds=statistics.median(measured) if measured else None,
                  mean_step_seconds=statistics.mean(measured) if measured else None,
                  optimizer_steps=sum(r['optimizer_step'] for r in record['steps']),
                  best_epoch=best_epoch, best_J=best_j, best_checkpoint=str(model_dir/'best.pt'),
                  peak_allocated_GiB=max((r['allocated_peak_GiB'] for r in record['steps'] if r['allocated_peak_GiB'] is not None),default=None),
                  graph_gradient_nonzero_observed=any(r['graph_gradient_norm']>0 for r in record['steps']))
    write_json(report_path, report)
    del model, optimizer
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()


def self_check():
    assert np.array_equal(choose_origins(20,8,7), choose_origins(20,8,7))
    try: NominalData('V_confirm')
    except ValueError: pass
    else: raise AssertionError('V_confirm admitted')
    pred=torch.zeros(1,20,2,2);target=pred.clone();target[:,:,0,0]=2;target[:,:,1,0]=10
    valid=torch.zeros(1,20,2,dtype=torch.bool);valid[:,:,0]=True;valid[:,0,1]=True
    m=scene_metrics(pred,target,valid,torch.ones(1,2,dtype=torch.bool))
    assert m['scene_ade'].item()==6
    print(json.dumps(dict(passed=True,checks=['deterministic common origins','V_confirm refused before access','target-first evaluation ADE'],cuda_work_executed=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--train-origins',type=int,default=128)
    p.add_argument('--validation-origins',type=int,default=64)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--seed',type=int,default=20260913)
    p.add_argument('--graph-lr',type=float,default=3e-4)
    p.add_argument('--timing-warmup-steps',type=int,default=2)
    p.add_argument('--run-name',default='trial')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check:
        self_check();return
    if min(args.train_origins,args.validation_origins,args.epochs,args.batch_size,args.micro_batch)<1 or args.timing_warmup_steps<0:
        p.error('Positive dataset/epoch/batch values and nonnegative warmup steps required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):
        p.error('run-name accepts only letters, digits, underscore and hyphen')
    if args.micro_batch>args.batch_size or args.graph_lr<=0:
        p.error('Invalid micro-batch or learning rate')
    args.device=torch.device(args.device)
    train,valid=NominalData('train'),NominalData('V_select')
    train_ids=choose_origins(len(train.loader),args.train_origins,args.seed+1)
    valid_ids=choose_origins(len(valid.loader),args.validation_origins,args.seed+2)
    if args.check_only:
        print(json.dumps(dict(ready=(Path(__file__).parent/'graph.py').is_file(),train_origins=len(train_ids),
                              validation_origins=len(valid_ids),snr_db=20,epochs=args.epochs,
                              optimizer_steps_executed=0,cuda_work_executed=False,required_candidate='CandidatePredictor(normalization)')))
        return
    if args.device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    report_path=ROOT/'reports/qgat_candidate'/f'{args.run_name}.json'
    checkpoint_dir=ROOT/'checkpoints/qgat_candidate'/args.run_name
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if report_path.exists() or checkpoint_dir.exists():
            raise FileExistsError('Trial outputs already exist; use a distinct --run-name')
        configuration={k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items() if k not in ('check_only','self_check')}
        report=dict(status='running',experimental_only=True,
                    limitation='Single-seed, small-sample, 20 dB development trial; not F04/F05, not formal predictive superiority or quantum advantage.',
                    configuration=configuration,manifest=dict(train_origins=train_ids.tolist(),V_select_origins=valid_ids.tolist(),
                    input_hashes=dict(train=train.loader.input_hash,V_select=valid.loader.input_hash)),models={},
                    permitted_splits=['train','V_select'],snr_db=20,model_order=['qgat','gnn'])
        write_json(report_path,report)
        try:
            from prediction.model import GNN
            seed_all(args.seed)
            template=GNN()
            temporal={k:v for k,v in mutable_state(template).items() if k.startswith('temporal.')}
            del template
            save_checkpoint(checkpoint_dir/'common_temporal_initial.pt',dict(model=temporal,seed=args.seed,experimental_only=True))
            for name in ('qgat','gnn'):
                run_model(name,args,train,valid,train_ids,valid_ids,temporal,report,checkpoint_dir,report_path)
            q,g=report['models']['qgat'],report['models']['gnn']
            qv,gv=q['epochs'][-1]['validation'],g['epochs'][-1]['validation']
            assert [(r['origin'],r['ade_targets'],r['fde_targets']) for r in qv['per_scene']]==[(r['origin'],r['ade_targets'],r['fde_targets']) for r in gv['per_scene']]
            report['comparison']=dict(final_epoch_J_G_minus_Q=gv['J']-qv['J'],best_J_G_minus_Q=g['best_J']-q['best_J'],
                median_step_G_over_Q=g['median_step_seconds']/q['median_step_seconds'] if g['median_step_seconds'] and q['median_step_seconds'] else None,
                common_denominators_verified=True)
            report['status']='complete'
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            write_json(report_path,report)
            raise
        write_json(report_path,report)
        print(f'Experimental trial complete: {report_path}',flush=True)


if __name__=='__main__':main()
