"""Independent QGAT/GNN formal route. Explicit user launch; no test access."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import prediction.training as training
from prediction.temporal import TrajectoryPredictor
from prediction.classical import GNNGraph
from experiments.qgat_candidate.graph import QGATGraph

ORIGINAL_OPTIMIZER = training.configure_phase


def candidate_optimizer(model, phase, graph_lr):
    """Process-local AdamW override: only QGAT theta gets zero matrix decay."""
    optimizer = ORIGINAL_OPTIMIZER(model, phase, graph_lr)
    theta = getattr(model.graph, 'theta', None)
    if theta is not None:
        for group in list(optimizer.param_groups):
            if any(p is theta for p in group['params']) and group['weight_decay'] != 0:
                group['params'] = [p for p in group['params'] if p is not theta]
                optimizer.add_param_group(dict(params=[theta], lr=group['lr'], weight_decay=0.))
    return optimizer


class Predictor(nn.Module):
    def __init__(self, name, seed, config):
        super().__init__()
        training.seed_all(seed+config['graph_seed_offset'])
        self.graph = QGATGraph() if name == 'qgat' else GNNGraph()
        training.seed_all(seed+config['temporal_seed_offset'])
        self.temporal = TrajectoryPredictor(128)

    def forward(self, state_hat, standardized_state, track_exists, detected):
        args = (state_hat, standardized_state, track_exists, detected)
        return self.temporal(*args, self.graph(*args))


def read_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    required = dict(models=['qgat','gnn'], seeds=[2023,2024,2025], epochs=20, patience=4,
                    minimum_epochs=4, batch_size=16, micro_batch=1, graph_lr=3e-4,
                    adapter_head_lr=1e-4,lora_lr=3e-5,weight_decay=.01,quantum_theta_weight_decay=0.,
                    gradient_clip=1.,snr_db=[5,10,15,20],test_access=False)
    for key,value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Formal contract differs for {key}; use an explicit route revision')
    return config


def digest(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Provenance path outside project')
    if any(p=='test' or p.startswith('test_') for p in path.relative_to(ROOT).parts):
        raise ValueError('This entry never opens locked test data')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def source_contract(config_path):
    files=[Path(__file__).resolve(),config_path,ROOT/'experiments/qgat_candidate/graph.py',
           ROOT/'prediction/training.py',ROOT/'prediction/temporal.py',ROOT/'prediction/classical.py',
           ROOT/'prediction/evaluation_cache.py',ROOT/'frontend/echo_source.py',
           ROOT/'code/00_remote_shared_dependencies/target_interaction_graph.py',ROOT/'frontend/symbol_dataset.py',ROOT/'models/gpt2/config.json',
           ROOT/'models/gpt2/model.safetensors',ROOT/'data/f01d/normalization.json']
    for split in ('train','V_select'):
        files.append(ROOT/f'data/f01d/labels/{split}.npz')
        files.extend(ROOT/f'data/f01d/inputs/{split}_snr_{snr}.npz' for snr in training.SNRS)
    return dict(files={str(p.relative_to(ROOT)):digest(p) for p in files},
                versions={p:importlib.metadata.version(p) for p in ('torch','pennylane','transformers','numpy')})


def verify_contract(expected, config_path):
    if source_contract(config_path) != expected:
        raise ValueError('Code, data, GPT-2, dependencies or configuration changed since run creation')


def temporal_state(model):
    return {k:v for k,v in training.mutable_state(model).items() if k.startswith('temporal.')}


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.shape).encode());h.update(str(v.dtype).encode())
        h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


class ValidationSubset:
    """Smoke-only view; production uses the entire original Dataset."""
    def __init__(self, dataset, count):
        self.dataset=dataset
        self.split=dataset.split
        self.n=min(count,dataset.n)
        self.input_hashes=dict(dataset.input_hashes, smoke_origins=list(range(self.n)))
    def batch(self, ids, snrs, device):
        return self.dataset.batch(ids,snrs,device)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


def worker(args, config, run_dir, report_dir):
    model_name,seed_text=args.worker.split('-');seed=int(seed_text)
    if model_name not in config['models'] or seed not in config['seeds']:
        raise ValueError('Unknown formal job')
    status_path=report_dir/'jobs'/f'{args.worker}.json'
    status=dict(task=args.worker,model=model_name,seed=seed,device=args.device,status='initializing',pid=os.getpid(),started_at=time.time())
    training.write_json(status_path,status)
    try:
        contract=json.loads((report_dir/'provenance.json').read_text())
        verify_contract(contract['source'],args.config)
        train,valid=training.Dataset('train'),training.Dataset('V_select')
        if args.smoke:
            valid=ValidationSubset(valid,config['smoke']['validation_origins'])
        model=Predictor(model_name,seed,config).to(args.device)
        fingerprint=tensor_digest(temporal_state(model))
        initial_theta=model.graph.theta.detach().cpu().clone() if hasattr(model.graph,'theta') else None
        status.update(status='training',temporal_initialization_sha256=fingerprint,train_origins=train.n if not args.smoke else config['smoke']['origins'])
        training.write_json(status_path,status)
        training.configure_phase=candidate_optimizer
        tick=time.perf_counter()
        result=training.fit(model,train,valid,run_dir/args.worker,seed=seed,
            epochs=config['smoke']['epochs'] if args.smoke else config['epochs'],patience=config['patience'],
            graph_lr=config['graph_lr'],phase='joint',batch_size=config['batch_size'],micro_batch=config['micro_batch'],
            indices=np.arange(config['smoke']['origins']) if args.smoke else None,
            snr_mode='balanced',resume=args.resume,contract=contract)
        status.update(status='complete',finished_at=time.time(),wall_seconds=time.perf_counter()-tick,training=result,
                      best_checkpoint_sha256=digest(Path(result['best_checkpoint'])))
        if initial_theta is not None:
            final=model.graph.theta.detach().cpu()
            status['theta_update']=dict(maxabsdelta=float((final-initial_theta).abs().max()),
                                       changed_elements=int(torch.count_nonzero(final-initial_theta)),total_elements=final.numel())
        if not result['numerical_gate']:
            status['status']='failed'
            raise RuntimeError('Numerical/gradient gate failed')
        training.write_json(status_path,status)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time())
        training.write_json(status_path,status)
        raise


def self_check(config, report_dir):
    torch.set_num_threads(1)
    model=Predictor('qgat',config['seeds'][0],config)
    common=temporal_state(model)
    optimizer=candidate_optimizer(model,'joint',config['graph_lr'])
    theta_groups=[g for g in optimizer.param_groups if any(p is model.graph.theta for p in g['params'])]
    assert len(theta_groups)==1 and theta_groups[0]['weight_decay']==0 and theta_groups[0]['lr']==3e-4
    assert all(g['weight_decay']==.01 for g in optimizer.param_groups if any(p is model.graph.encoder.weight for p in g['params']))
    del model,optimizer
    other=Predictor('gnn',config['seeds'][0],config)
    state=temporal_state(other)
    assert set(state)==set(common) and all(torch.equal(state[k],common[k]) for k in common)
    del other
    seen={i:set() for i in range(5549)}
    for epoch in range(4):
        ids,snrs=training.balanced_schedule(5549,2023,epoch)
        assert len(ids)==len(set(ids))==5549
        counts=[int((snrs==s).sum()) for s in training.SNRS]
        assert max(counts)-min(counts)<=1
        for i,s in zip(ids,snrs):seen[int(i)].add(int(s))
    assert all(v==set(training.SNRS) for v in seen.values())
    assert config['patience']+1>=config['minimum_epochs']
    import tempfile
    work=ROOT/'.codex-work/qgat-formal-seal-check'
    work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        folder=Path(tmp);states={}
        for seed in config['seeds']:
            for name in config['models']:
                job=f'{name}-{seed}';path=folder/f'{job}.pt';path.write_bytes(job.encode())
                states[job]=dict(status='complete',model=name,seed=seed,best_checkpoint_sha256=digest(path),
                    training=dict(best_checkpoint=str(path),numerical_gate=True))
        contract=dict(configuration=config,source={'fixture':True})
        sealed=seal_selected(states,folder,contract)
        assert len(sealed['checkpoints'])==6
        (folder/'qgat-2023.pt').write_bytes(b'changed')
        try:seal_selected(states,folder,contract)
        except ValueError as exc:assert 'changed' in str(exc)
        else:raise AssertionError('Modified selected checkpoint accepted')
    result=dict(passed=True,cuda_work_executed=False,checks=['quantum theta decay zero; encoder matrix decay 0.01',
        'actual QGAT/GNN mutable temporal tensors identical under same seed',
        'all 5549 origins exactly once per epoch; balanced four-SNR cycle',
        'patience four cannot stop before minimum four epochs',
        'all six checkpoint hashes sealed together; changed checkpoint rejected before confirmation'],temporal_initialization_sha256=tensor_digest(common))
    training.write_json(report_dir/'self_check.json',result)
    print(json.dumps(result),flush=True)


def seal_selected(states, report_dir, contract):
    config = contract['configuration']
    expected = {f'{m}-{s}' for s in config['seeds'] for m in config['models']}
    if set(states) != expected:
        raise ValueError('All six complete paired jobs are required before confirmation')
    checkpoints = {}
    for job, state in sorted(states.items()):
        if state.get('status') != 'complete' or not state['training']['numerical_gate']:
            raise ValueError(f'Incomplete or numerically invalid job: {job}')
        path = Path(state['training']['best_checkpoint'])
        actual = digest(path)
        if actual != state.get('best_checkpoint_sha256'):
            raise ValueError(f'Completed checkpoint changed before confirmation: {job}')
        checkpoints[job] = dict(path=str(path), sha256=actual, model=state['model'], seed=state['seed'])
    manifest = dict(source=contract['source'], checkpoints=checkpoints, test_opened=False)
    path = report_dir/'selected_checkpoints.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Previously sealed checkpoint selection changed')
    else:
        training.write_json(path,manifest)
    return manifest


def final_confirmation(args, config, run_dir, report_dir, states, selected):
    dataset=training.Dataset('V_confirm')
    confirm_hashes={f'data/f01d/labels/V_confirm.npz':digest(ROOT/'data/f01d/labels/V_confirm.npz')}
    confirm_hashes.update({f'data/f01d/inputs/V_confirm_snr_{s}.npz':digest(ROOT/f'data/f01d/inputs/V_confirm_snr_{s}.npz') for s in training.SNRS})
    rows=[]
    for job,state in states.items():
        path=report_dir/'confirmation'/f'{job}.json'
        ckpt=Path(selected['checkpoints'][job]['path'])
        if digest(ckpt) != selected['checkpoints'][job]['sha256']:
            raise ValueError(f'Sealed checkpoint changed: {job}')
        identity=dict(checkpoint_sha256=selected['checkpoints'][job]['sha256'],confirm_hashes=confirm_hashes)
        if args.resume and path.exists():
            row=json.loads(path.read_text())
            if row['identity']!=identity:raise ValueError('Confirmation dependencies changed')
        else:
            model=Predictor(state['model'],state['seed'],config).to(args.devices[0])
            training.load_best(model,state['training'])
            scores=training.evaluate(model,dataset,args.devices[0],micro_batch=config['micro_batch'])
            row=dict(task=job,model=state['model'],seed=state['seed'],identity=identity,metrics=scores,
                     role='final development confirmation only; no checkpoint or configuration selection',test_opened=False)
            training.write_json(path,row)
            del model
            torch.cuda.empty_cache()
        rows.append(row)
    return rows


def coordinate(args, config, run_dir, report_dir):
    run_dir.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    active={};logs=[]
    with (report_dir/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provenance_path=report_dir/'provenance.json'
        source=source_contract(args.config)
        contract=dict(source=source,configuration=config,mode='smoke' if args.smoke else 'formal')
        if provenance_path.exists():
            if not args.resume:raise FileExistsError('Existing formal run; pass --resume to continue it')
            if json.loads(provenance_path.read_text())!=contract:raise ValueError('Resume provenance mismatch')
        else:
            if args.resume:raise FileNotFoundError('No run exists to resume')
            training.write_json(provenance_path,contract)
        jobs=[f'{m}-{s}' for s in (config['seeds'][:1] if args.smoke else config['seeds']) for m in config['models']]
        pending=[];states={}
        for job in jobs:
            path=report_dir/'jobs'/f'{job}.json'
            if args.resume and path.exists() and json.loads(path.read_text()).get('status')=='complete':
                states[job]=json.loads(path.read_text())
                if digest(Path(states[job]['training']['best_checkpoint'])) != states[job].get('best_checkpoint_sha256'):
                    raise ValueError(f'Completed checkpoint changed before resume: {job}')
            else:pending.append(job)
        coordinator=dict(status='running',pid=os.getpid(),started_at=time.time(),jobs=jobs,devices=args.devices,smoke=args.smoke)
        training.write_json(report_dir/'coordinator.json',coordinator)
        try:
            while pending or active:
                for device in args.devices:
                    if pending and device not in active:
                        job=pending.pop(0)
                        log_path=report_dir/'logs'/f'{job}.log';log_path.parent.mkdir(parents=True,exist_ok=True)
                        log=log_path.open('a' if args.resume else 'w',buffering=1);logs.append(log)
                        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',job,'--device',device,
                             '--config',str(args.config),'--run-dir',str(run_dir),'--report-dir',str(report_dir)]
                        if args.smoke:cmd.append('--smoke')
                        if args.resume:cmd.append('--resume')
                        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                               env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                        active[device]=(job,child)
                        print(f'Started {job} on {device}; log: {log_path}',flush=True)
                time.sleep(1)
                for device,(job,child) in list(active.items()):
                    code=child.poll()
                    if code is None:continue
                    del active[device]
                    path=report_dir/'jobs'/f'{job}.json'
                    state=json.loads(path.read_text()) if path.exists() else {}
                    if code or state.get('status')!='complete':raise RuntimeError(f'{job} failed (exit {code}); inspect its log, then --resume')
                    states[job]=state
                    print(f'Completed {job}',flush=True)
            for seed in (config['seeds'][:1] if args.smoke else config['seeds']):
                if states[f'qgat-{seed}']['temporal_initialization_sha256']!=states[f'gnn-{seed}']['temporal_initialization_sha256']:
                    raise ValueError('Paired temporal initialization mismatch')
            verify_contract(contract['source'],args.config)
            selected=None if args.smoke else seal_selected(states,report_dir,contract)
            confirmations=[] if args.smoke else final_confirmation(args,config,run_dir,report_dir,states,selected)
            summary=dict(status='complete',mode='smoke' if args.smoke else 'formal',tasks=states,
                         confirmation=confirmations,paired_initialization_verified=True,test_opened=False,
                         sufficiency_unresolved=any(s['training']['sufficiency_unresolved'] for s in states.values()))
            training.write_json(report_dir/'summary.json',summary)
            coordinator.update(status='complete',finished_at=time.time())
        except BaseException as exc:
            for _,child in active.values():
                if child.poll() is None:child.terminate()
            for _,child in active.values():child.wait()
            coordinator.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time())
            raise
        finally:
            for log in logs:log.close()
            training.write_json(report_dir/'coordinator.json',coordinator)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/qgat_formal.json')
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--worker')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--report-dir',type=Path)
    args=p.parse_args();args.config=args.config.resolve()
    config=read_config(args.config)
    for device in args.devices:
        if not device.startswith('cuda:') or not device[5:].isdigit():p.error('CUDA devices must use cuda:N')
    if len(set(args.devices))!=len(args.devices):p.error('Duplicate CUDA devices')
    if bool(args.run_dir)!=bool(args.report_dir):p.error('Internal worker paths must be paired')
    if args.run_dir and not args.worker:p.error('Explicit paths are reserved for coordinator workers')
    if args.smoke and not args.run_dir:
        tag=time.strftime('%Y%m%d-%H%M%S')
        run_dir=ROOT/'.codex-work/qgat-formal-smoke'/tag
        report_dir=ROOT/'reports/qgat_formal/smoke'/tag
    else:
        run_dir=args.run_dir or ROOT/'results/qgat_formal'
        report_dir=args.report_dir or ROOT/'reports/qgat_formal'
    if args.self_check:
        self_check(config,ROOT/'reports/qgat_formal');return
    if args.check_only:
        train,valid=training.Dataset('train'),training.Dataset('V_select')
        source=source_contract(args.config)
        existing=report_dir/'provenance.json'
        if existing.exists() and json.loads(existing.read_text())['source']!=source:raise ValueError('Existing provenance differs')
        if not torch.cuda.is_available() or any(int(d[5:])>=torch.cuda.device_count() for d in args.devices):raise RuntimeError('Requested CUDA unavailable')
        print(json.dumps(dict(ready=True,training_origins=train.n,validation_origins=valid.n,
              seeds=config['seeds'],models=config['models'],snr_db=config['snr_db'],max_epochs=config['epochs'],
              patience=config['patience'],minimum_epochs=config['minimum_epochs'],devices=args.devices,
              existing_run=existing.exists(),optimizer_steps_executed=0,confirmation_opened=False,test_opened=False)),flush=True)
        return
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    if args.worker:worker(args,config,run_dir,report_dir)
    else:coordinate(args,config,run_dir,report_dir)


if __name__=='__main__':main()
