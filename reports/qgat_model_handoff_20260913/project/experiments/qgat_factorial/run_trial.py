"""Independent paired 2x2 QGAT development experiment. No confirm/test access."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import prediction.training as training
from prediction.temporal import TrajectoryPredictor
ORIGINAL_OPTIMIZER=training.configure_phase


def make_optimizer(model,phase,graph_lr):
    opt=ORIGINAL_OPTIMIZER(model,phase,graph_lr)
    for group in list(opt.param_groups):
        if any(p is model.graph.theta for p in group['params']) and group['weight_decay']:
            group['params']=[p for p in group['params'] if p is not model.graph.theta]
            opt.add_param_group(dict(params=[model.graph.theta],lr=group['lr'],weight_decay=0.))
    return opt


class Predictor(nn.Module):
    def __init__(self,cell,cfg):
        super().__init__()
        from experiments.qgat_factorial.graph import FactorialGraph
        training.seed_all(cfg['seed']+cfg['graph_seed_offset'])
        self.graph=FactorialGraph(**cfg['cells'][cell])
        training.seed_all(cfg['seed']+cfg['temporal_seed_offset'])
        self.temporal=TrajectoryPredictor(128)
    def forward(self,state_hat,standardized_state,track_exists,detected):
        args=(state_hat,standardized_state,track_exists,detected)
        return self.temporal(*args,self.graph(*args))


def origins(n,count):
    if not 1<=count<=n:raise ValueError('Requested origin count unavailable')
    ids=np.unique(np.linspace(0,n-1,count).round().astype(np.int64))
    if len(ids)!=count:raise ValueError('Origin spread lost unique indices')
    return ids


def dataset(split):
    if split not in ('train','V_select'):raise ValueError('Factorial entry refuses confirmation and test')
    return training.Dataset(split)


class Subset:
    def __init__(self,base,ids):
        self.base,self.ids=base,ids
        self.split,self.n=base.split,len(ids)
        self.input_hashes=dict(base.input_hashes,selected_origins=ids.tolist())
    def batch(self,ids,snrs,device):return self.base.batch(self.ids[np.asarray(ids,dtype=int)],snrs,device)


def digest(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Path outside project')
    if any('V_confirm' in p or p=='test' or p.startswith('test_') for p in path.relative_to(ROOT).parts):
        raise ValueError('Forbidden data access')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fingerprint(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def provenance(cfg):
    files=[Path(__file__).resolve(),ROOT/'configs/qgat_factorial.json',ROOT/'experiments/qgat_factorial/graph.py',
        ROOT/'experiments/qgat_candidate/graph.py',ROOT/'prediction/training.py',ROOT/'prediction/temporal.py',
        ROOT/'prediction/evaluation_cache.py',ROOT/'frontend/symbol_dataset.py',ROOT/'frontend/echo_source.py',
        ROOT/'models/gpt2/config.json',ROOT/'models/gpt2/model.safetensors',ROOT/'data/f01d/normalization.json']
    for split in ('train','V_select'):
        files.append(ROOT/f'data/f01d/labels/{split}.npz')
        files.extend(ROOT/f'data/f01d/inputs/{split}_snr_{s}.npz' for s in training.SNRS)
    return dict(configuration=cfg,files={str(p.relative_to(ROOT)):digest(p) for p in files},
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','pennylane','transformers')})


def check_config(cfg):
    if cfg['cells']!={'A':{'qubits':1,'radius':None},'B':{'qubits':2,'radius':None},'C':{'qubits':1,'radius':45},'D':{'qubits':2,'radius':45}}:raise ValueError('Unexpected cells')
    if cfg['device_queues']!={'cuda:0':['A','C'],'cuda:1':['B','D']}:raise ValueError('Paired device assignment changed')
    for k,v in dict(seed=2026,train_origins=512,validation_origins=128,epochs=8,patience=9,batch_size=16,micro_batch=1,graph_lr=3e-4).items():
        if cfg[k]!=v:raise ValueError(f'Unexpected trial contract: {k}')


def self_check(cfg):
    torch.set_num_threads(1)
    graph_states={};common=None;counts={}
    for cell in cfg['cells']:
        model=Predictor(cell,cfg)
        state={k:v for k,v in training.mutable_state(model).items() if k.startswith('temporal.')}
        if common is None:common=state
        else:assert state.keys()==common.keys() and all(torch.equal(state[k],common[k]) for k in state)
        graph_states[cell]={k:v.detach().cpu().clone() for k,v in model.graph.state_dict().items()}
        opt=make_optimizer(model,'joint',cfg['graph_lr'])
        assert all(g['weight_decay']==0 for g in opt.param_groups if any(p is model.graph.theta for p in g['params']))
        counts[cell]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model,opt
    for a,b in [('A','C'),('B','D')]:assert graph_states[a].keys()==graph_states[b].keys() and all(torch.equal(graph_states[a][k],graph_states[b][k]) for k in graph_states[a])
    ids=origins(5549,512);visits={int(i):[] for i in ids}
    for epoch in range(8):
        order,snrs=training.balanced_schedule(5549,cfg['seed'],epoch,ids)
        assert set(order)==set(ids) and all(int((snrs==s).sum())==128 for s in training.SNRS)
        for i,s in zip(order,snrs):visits[int(i)].append(int(s))
    assert all(all(v.count(s)==2 for s in training.SNRS) for v in visits.values())
    for split in ('V_confirm','test'):
        try:dataset(split)
        except ValueError:pass
        else:raise AssertionError('Forbidden split admitted')
    result=dict(passed=True,cuda_work_executed=False,trainable_parameters=counts,checks=['all four actual temporal states equal',
        'A/C and B/D whole graph state tensors equal','theta decay zero','512 fixed spread origins; each SNR exactly twice in eight epochs',
        'V_confirm and test rejected before access'],optimizer_steps_per_cell=8*32,temporal_sha256=fingerprint(common))
    training.write_json(ROOT/'reports/qgat_factorial/checks.json',result);print(json.dumps(result),flush=True)


def worker(args,cfg,run_dir,report_dir):
    cell=args.worker;status=dict(cell=cell,status='initializing',pid=os.getpid(),device=args.device,seed=cfg['seed'],started_at=time.time())
    path=report_dir/'jobs'/f'{cell}.json'
    training.write_json(path,status)
    try:
        contract=json.loads((report_dir/'protocol.json').read_text())
        if contract['source']!=provenance(cfg):raise ValueError('Source contract changed')
        train,valid=dataset('train'),dataset('V_select')
        train_ids=origins(train.n,16 if args.smoke else cfg['train_origins'])
        val_ids=origins(valid.n,1 if args.smoke else cfg['validation_origins'])
        valid=Subset(valid,val_ids)
        model=Predictor(cell,cfg).to(args.device)
        state=training.mutable_state(model)
        status.update(status='training',initial_temporal_sha256=fingerprint({k:v for k,v in state.items() if k.startswith('temporal.')}),
            initial_graph_sha256=fingerprint(model.graph.state_dict()),train_origins=train_ids.tolist(),V_select_origins=val_ids.tolist())
        del state
        initial_theta=model.graph.theta.detach().cpu().clone()
        training.write_json(path,status)
        training.configure_phase=make_optimizer
        started=time.perf_counter()
        result=training.fit(model,train,valid,run_dir/cell,seed=cfg['seed'],epochs=1 if args.smoke else cfg['epochs'],
            patience=cfg['patience'],graph_lr=cfg['graph_lr'],phase='joint',batch_size=cfg['batch_size'],micro_batch=cfg['micro_batch'],
            indices=train_ids,snr_mode='balanced',resume=args.resume,contract=contract)
        if not result['numerical_gate']:raise RuntimeError('Numerical or graph-gradient gate failed')
        status.update(status='complete',wall_seconds=time.perf_counter()-started,finished_at=time.time(),training=result,
            best_sha256=digest(run_dir/cell/'best.pt'),latest_sha256=digest(run_dir/cell/'latest.pt'),
            theta_best_maxabsdelta=float((model.graph.theta.detach().cpu()-initial_theta).abs().max()),
            theta_best_changed_elements=int(torch.count_nonzero(model.graph.theta.detach().cpu()-initial_theta)))
        training.write_json(path,status)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        training.write_json(path,status);raise


def comparisons(scores):
    expressions={'B_minus_A':{'B':1,'A':-1},'D_minus_C':{'D':1,'C':-1},'C_minus_A':{'C':1,'A':-1},
        'D_minus_B':{'D':1,'B':-1},'interaction_D_minus_B_minus_C_plus_A':{'D':1,'B':-1,'C':-1,'A':1}}
    def calculate(rows):return {key:{metric:sum(weight*rows[cell][metric] for cell,weight in terms.items()) for metric in ('ADE','FDE','J')} for key,terms in expressions.items()}
    return dict(macro=calculate(scores),by_snr={str(s):calculate({c:r['by_snr'][str(s)] for c,r in scores.items()}) for s in training.SNRS},
        interpretation='Metric differences; negative indicates lower error for the positive-sign cell. Best-epoch contrasts may compare different epochs.')


def coordinate(args,cfg,run_dir,report_dir):
    report_dir.mkdir(parents=True,exist_ok=True);run_dir.mkdir(parents=True,exist_ok=True)
    active={};logs=[]
    with (report_dir/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=dict(source=provenance(cfg),mode='smoke' if args.smoke else 'development',test_opened=False,confirmation_opened=False)
        protocol_path=report_dir/'protocol.json'
        if protocol_path.exists():
            if not args.resume:raise FileExistsError('Run exists; use --resume explicitly')
            if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Resume contract changed')
        else:
            if args.resume:raise FileNotFoundError('No existing trial to resume')
            training.write_json(protocol_path,protocol)
        states={};queues={d:list(cells) for d,cells in cfg['device_queues'].items()}
        for device,cells in queues.items():
            for cell in list(cells):
                path=report_dir/'jobs'/f'{cell}.json'
                if args.resume and path.exists():
                    state=json.loads(path.read_text())
                    if state.get('status')=='complete':
                        if state['best_sha256']!=digest(run_dir/cell/'best.pt') or state['latest_sha256']!=digest(run_dir/cell/'latest.pt'):raise ValueError('Saved checkpoint changed')
                        states[cell]=state;cells.remove(cell)
        coordinator=dict(status='running',pid=os.getpid(),started_at=time.time(),queues=cfg['device_queues'])
        try:
            while active or any(queues.values()):
                for device,cells in queues.items():
                    if cells and device not in active:
                        cell=cells.pop(0);log_path=report_dir/'logs'/f'{cell}.log';log_path.parent.mkdir(parents=True,exist_ok=True)
                        log=log_path.open('a' if args.resume else 'w',buffering=1);logs.append(log)
                        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',cell,'--device',device,'--run-dir',str(run_dir),'--report-dir',str(report_dir)]
                        if args.smoke:cmd.append('--smoke')
                        if args.resume:cmd.append('--resume')
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                        active[device]=(cell,child);print(f'Started {cell} on {device}: {log_path}',flush=True)
                time.sleep(1)
                for device,(cell,child) in list(active.items()):
                    code=child.poll()
                    if code is None:continue
                    del active[device]
                    if code:raise RuntimeError(f'{cell} failed; read its log then resume')
                    states[cell]=json.loads((report_dir/'jobs'/f'{cell}.json').read_text())
                    if states[cell]['status']!='complete':raise RuntimeError(f'{cell} incomplete')
                    print(f'Completed {cell}',flush=True)
            if protocol['source']!=provenance(cfg):raise ValueError('Provenance changed during trial')
            assert len({s['initial_temporal_sha256'] for s in states.values()})==1
            assert states['A']['initial_graph_sha256']==states['C']['initial_graph_sha256']
            assert states['B']['initial_graph_sha256']==states['D']['initial_graph_sha256']
            final={};best={};trends={}
            for cell,status in states.items():
                final[cell]=json.loads((run_dir/cell/f'validation_epoch_{1 if args.smoke else 8:02d}.json').read_text())
                best_epoch=status['training']['best_metrics']['epoch']
                best[cell]=json.loads((run_dir/cell/f'validation_epoch_{best_epoch:02d}.json').read_text())
                history=status['training']['history']
                trends[cell]=dict(final_epoch_is_best=best_epoch==history[-1]['epoch'],
                    last_J_change=history[-1]['J']-history[-2]['J'] if len(history)>1 else None,
                    last_train_loss_change=history[-1]['loss']-history[-2]['loss'] if len(history)>1 else None,
                    full_training_sufficiency_established=False)
                for scores in (final[cell],best[cell]):
                    for row in scores['per_scene']:row['source_origin']=status['V_select_origins'][row['origin']]
            for scores in (final,best):
                reference=[(r['snr_db'],r['source_origin'],r['ade_targets'],r['fde_targets']) for r in scores['A']['per_scene']]
                assert all([(r['snr_db'],r['source_origin'],r['ade_targets'],r['fde_targets']) for r in scores[c]['per_scene']]==reference for c in 'BCD')
            summary=dict(status='complete',mode=protocol['mode'],limitation=cfg['limitation'],jobs=states,
                same_final_epoch=1 if args.smoke else 8,final_epoch_metrics=final,best_epoch_metrics=best,
                final_epoch_contrasts=comparisons(final),best_epoch_contrasts=comparisons(best),
                paired_initialization_verified=True,common_denominators_verified=True,training_trends=trends,
                winner_selected=False,test_opened=False,confirmation_opened=False)
            training.write_json(report_dir/'summary.json',summary);coordinator['status']='complete'
        except BaseException as exc:
            for _,child in active.values():
                if child.poll() is None:child.terminate()
            for _,child in active.values():child.wait()
            coordinator.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            coordinator['finished_at']=time.time()
            training.write_json(report_dir/'coordinator.json',coordinator)
            for log in logs:log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('check-only','self-check','smoke','resume'):p.add_argument('--'+flag,action='store_true')
    p.add_argument('--worker',choices=list('ABCD'));p.add_argument('--device',default='cuda:0')
    p.add_argument('--run-dir',type=Path);p.add_argument('--report-dir',type=Path)
    args=p.parse_args();cfg=json.loads((ROOT/'configs/qgat_factorial.json').read_text(encoding='utf-8-sig'));check_config(cfg)
    if (args.run_dir or args.report_dir) and (not args.worker or not args.run_dir or not args.report_dir):p.error('Paths are reserved for child workers')
    if args.self_check:self_check(cfg);return
    if args.check_only:
        train,valid=dataset('train'),dataset('V_select');source=provenance(cfg)
        assert len(origins(train.n,512))==512 and len(origins(valid.n,128))==128
        existing=ROOT/'reports/qgat_factorial/protocol.json'
        if existing.exists() and json.loads(existing.read_text())['source']!=source:raise ValueError('Existing provenance mismatch')
        print(json.dumps(dict(ready=torch.cuda.is_available() and torch.cuda.device_count()>=2,train_origins=512,V_select_origins=128,
            seed=2026,cells=cfg['cells'],epochs=8,steps_per_cell=256,optimizer_steps_executed=0,confirmation_opened=False,test_opened=False)),flush=True);return
    if args.smoke and not args.run_dir:
        tag=time.strftime('%Y%m%d-%H%M%S');run_dir=ROOT/'.codex-work/qgat-factorial-smoke'/tag;report_dir=ROOT/'reports/qgat_factorial/smoke'/tag
    else:run_dir=args.run_dir or ROOT/'results/qgat_factorial';report_dir=args.report_dir or ROOT/'reports/qgat_factorial'
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    if args.worker:worker(args,cfg,run_dir,report_dir)
    else:coordinate(args,cfg,run_dir,report_dir)


if __name__=='__main__':main()
