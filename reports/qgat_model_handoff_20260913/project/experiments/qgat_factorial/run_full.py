"""Full-data 20-epoch paired factorial route; separate from eight-epoch results."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.qgat_factorial import run_trial as trial
training=trial.training
ORIGINAL_PROVENANCE=trial.provenance


def full_origins(n,count):
    if n!=count:raise ValueError(f'Full-data contract expected {count}, found {n}')
    return np.arange(n,dtype=np.int64)


def provenance(cfg):
    source=ORIGINAL_PROVENANCE(cfg)
    for path in (Path(__file__).resolve(),ROOT/'configs/qgat_factorial_full.json'):
        source['files'][str(path.relative_to(ROOT))]=trial.digest(path)
    return source


def config_for(seed):
    cfg=json.loads((ROOT/'configs/qgat_factorial_full.json').read_text(encoding='utf-8-sig'))
    if seed not in cfg['supported_seeds']:raise ValueError('Unsupported seed')
    for k,v in dict(train_origins=5549,validation_origins=400,epochs=20,patience=21,batch_size=16,micro_batch=1).items():
        if cfg[k]!=v:raise ValueError(f'Full-data contract changed: {k}')
    if cfg['device_queues']!={'cuda:0':['A','C'],'cuda:1':['B','D']}:raise ValueError('Paired device assignment changed')
    cfg['seed']=seed
    return cfg


def self_check(cfg,report_dir):
    torch.set_num_threads(1)
    common=None;graphs={}
    for cell in cfg['cells']:
        model=trial.Predictor(cell,cfg)
        state={k:v for k,v in training.mutable_state(model).items() if k.startswith('temporal.')}
        if common is None:common=state
        else:assert state.keys()==common.keys() and all(torch.equal(state[k],common[k]) for k in state)
        graphs[cell]={k:v.detach().cpu().clone() for k,v in model.graph.state_dict().items()}
        opt=trial.make_optimizer(model,'joint',cfg['graph_lr'])
        assert all(g['weight_decay']==0 for g in opt.param_groups if any(p is model.graph.theta for p in g['params']))
        del model,opt
    for a,b in [('A','C'),('B','D')]:assert graphs[a].keys()==graphs[b].keys() and all(torch.equal(graphs[a][k],graphs[b][k]) for k in graphs[a])
    train,valid=trial.dataset('train'),trial.dataset('V_select')
    ids=full_origins(train.n,cfg['train_origins']);full_origins(valid.n,cfg['validation_origins'])
    exposure=np.zeros((len(ids),4),int);counts=[]
    for epoch in range(20):
        order,snrs=training.balanced_schedule(train.n,cfg['seed'],epoch,ids)
        assert len(order)==len(np.unique(order))==5549 and set(order)==set(ids)
        row=[int((snrs==s).sum()) for s in training.SNRS];assert max(row)-min(row)<=1;counts.append(row)
        for j,s in enumerate(training.SNRS):exposure[order[snrs==s],j]+=1
        batches=[order[i:i+16] for i in range(0,len(order),16)]
        assert len(batches)==347 and len(batches[-1])==13 and sum(map(len,batches))==5549
    assert np.all(exposure==5) and cfg['patience']>cfg['epochs']
    result=dict(passed=True,seed=cfg['seed'],checks=['actual common temporal tensors across four cells',
        'actual A/C and B/D graph tensors identical','theta weight decay zero','full 5549 train/400 V_select',
        'every origin sees each SNR five times; each epoch counts differ at most one','347 batches with tail 13; no discarded origins',
        'patience 21 prevents early stopping in fixed 20 epochs'],epoch_snr_counts=counts,
        steps_per_cell=6940,cuda_work_executed=False,optimizer_steps_executed=0)
    training.write_json(report_dir/'checks.json',result);print(json.dumps(result),flush=True)


def summarize(states,cfg,run_dir):
    final={};best={};individual={};trends={}
    for cell,state in states.items():
        history=state['training']['history']
        if len(history)!=20 or history[-1]['epoch']!=20:raise ValueError('A cell did not finish 20 epochs')
        final[cell]=json.loads((run_dir/cell/'validation_epoch_20.json').read_text())
        best_epoch=state['training']['best_metrics']['epoch']
        best[cell]=json.loads((run_dir/cell/f'validation_epoch_{best_epoch:02d}.json').read_text())
        individual[cell]={metric:dict(epoch=(row:=min(history,key=lambda r:r[metric]))['epoch'],
            ADE=row['ADE'],FDE=row['FDE'],J=row['J'],
            validation_record=str(run_dir/cell/f'validation_epoch_{row["epoch"]:02d}.json'),
            weights_retained=row['epoch'] in (best_epoch,20)) for metric in ('ADE','FDE','J')}
        trends[cell]=dict(last_J_change=history[-1]['J']-history[-2]['J'],
            last_loss_change=history[-1]['loss']-history[-2]['loss'],final_epoch_is_best_J=best_epoch==20)
    for scores in (final,best):
        ref=[(r['snr_db'],r['origin'],r['ade_targets'],r['fde_targets']) for r in scores['A']['per_scene']]
        assert all([(r['snr_db'],r['origin'],r['ade_targets'],r['fde_targets']) for r in scores[c]['per_scene']]==ref for c in 'BCD')
    assert len({s['initial_temporal_sha256'] for s in states.values()})==1
    assert states['A']['initial_graph_sha256']==states['C']['initial_graph_sha256']
    assert states['B']['initial_graph_sha256']==states['D']['initial_graph_sha256']
    return dict(status='complete',seed=cfg['seed'],limitation=cfg['limitation'],jobs=states,
        final_epoch=20,final_epoch_metrics=final,best_J_epoch_metrics=best,individual_metric_minima=individual,
        individual_metric_minima_note='Each row retains the full ADE/FDE/J tuple from its own epoch; no synthetic mixed-epoch score. Only best-J and latest weights are retained by the shared trainer.',
        final_epoch_contrasts=trial.comparisons(final),best_J_epoch_contrasts=trial.comparisons(best),
        training_trends=trends,paired_initialization_verified=True,common_denominators_verified=True,
        winner_selected=False,confirmation_opened=False,test_opened=False)


def coordinate(args,cfg,run_dir,report_dir):
    run_dir.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    active={};logs=[]
    with (report_dir/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=dict(source=provenance(cfg),mode='full-data-fixed-20',confirmation_opened=False,test_opened=False)
        path=report_dir/'protocol.json'
        if path.exists():
            if not args.resume:raise FileExistsError('Seed run exists; use --resume explicitly')
            if json.loads(path.read_text())!=protocol:raise ValueError('Resume contract mismatch')
        else:
            if args.resume:raise FileNotFoundError('No seed run to resume')
            training.write_json(path,protocol)
        queues={d:list(cells) for d,cells in cfg['device_queues'].items()};states={}
        for cells in queues.values():
            for cell in list(cells):
                path=report_dir/'jobs'/f'{cell}.json'
                if args.resume and path.exists():
                    state=json.loads(path.read_text())
                    if state.get('status')=='complete':
                        if state['best_sha256']!=trial.digest(run_dir/cell/'best.pt') or state['latest_sha256']!=trial.digest(run_dir/cell/'latest.pt'):raise ValueError('Completed checkpoint changed')
                        states[cell]=state;cells.remove(cell)
        coordinator=dict(status='running',pid=os.getpid(),seed=cfg['seed'],started_at=time.time(),queues=cfg['device_queues'])
        training.write_json(report_dir/'coordinator.json',coordinator)
        try:
            while active or any(queues.values()):
                for device,cells in queues.items():
                    if cells and device not in active:
                        cell=cells.pop(0);path=report_dir/'logs'/f'{cell}.log';path.parent.mkdir(parents=True,exist_ok=True)
                        log=path.open('a' if args.resume else 'w',buffering=1);logs.append(log)
                        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--seed',str(cfg['seed']),'--worker',cell,'--device',device]
                        if args.resume:cmd.append('--resume')
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                        active[device]=(cell,child);print(f'Started {cell}, seed {cfg["seed"]}, {device}: {path}',flush=True)
                time.sleep(1)
                for device,(cell,child) in list(active.items()):
                    code=child.poll()
                    if code is None:continue
                    del active[device]
                    if code:raise RuntimeError(f'{cell} failed; inspect log and resume')
                    states[cell]=json.loads((report_dir/'jobs'/f'{cell}.json').read_text())
                    if states[cell]['status']!='complete':raise RuntimeError(f'{cell} incomplete')
                    print(f'Completed {cell}',flush=True)
            if protocol['source']!=provenance(cfg):raise ValueError('Source changed during run')
            training.write_json(report_dir/'summary.json',summarize(states,cfg,run_dir));coordinator['status']='complete'
        except BaseException as exc:
            for _,child in active.values():
                if child.poll() is None:child.terminate()
            for _,child in active.values():child.wait()
            coordinator.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            coordinator['finished_at']=time.time();training.write_json(report_dir/'coordinator.json',coordinator)
            for log in logs:log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[2026,2027,2028],default=2026)
    p.add_argument('--resume',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--self-check',action='store_true')
    p.add_argument('--worker',choices=list('ABCD'));p.add_argument('--device',default='cuda:0')
    args=p.parse_args();cfg=config_for(args.seed)
    run_dir=ROOT/'results/qgat_factorial_full'/f'seed{args.seed}';report_dir=ROOT/'reports/qgat_factorial_full'/f'seed{args.seed}'
    if args.self_check:self_check(cfg,report_dir);return
    if args.check_only:
        train,valid=trial.dataset('train'),trial.dataset('V_select')
        full_origins(train.n,5549);full_origins(valid.n,400);source=provenance(cfg)
        path=report_dir/'protocol.json'
        if path.exists() and json.loads(path.read_text())['source']!=source:raise ValueError('Existing provenance mismatch')
        print(json.dumps(dict(ready=torch.cuda.is_available() and torch.cuda.device_count()>=2,seed=args.seed,
            train_origins=train.n,V_select_origins=valid.n,epochs=20,patience=21,steps_per_epoch=347,last_batch=13,
            steps_per_cell=6940,extra_seeds_started=False,optimizer_steps_executed=0,confirmation_opened=False,test_opened=False)),flush=True);return
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    if args.worker:
        # These two substitutions are local to this fresh child process; sealed files are untouched.
        trial.provenance=provenance;trial.origins=full_origins;args.smoke=False
        trial.worker(args,cfg,run_dir,report_dir)
    else:coordinate(args,cfg,run_dir,report_dir)


if __name__=='__main__':main()
