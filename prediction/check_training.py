"""Bounded CPU checks for the shared trainer: scoring, phases and interrupted resume."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction.training import (fit, seed_all, balanced_schedule, scene_metrics, configure_phase,
                                 copy_own_weights, mutable_state, Dataset)
from prediction.temporal import masked_trajectory_loss


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = nn.Linear(4, 4)
        self.temporal = nn.Module()
        self.temporal.adapter = nn.Module()
        self.temporal.adapter.graph_projection = nn.Linear(4, 4, bias=False)
        self.temporal.head = nn.Sequential(nn.Dropout(.25), nn.Linear(4, 2))
    def forward(self, state_hat, standardized_state, track_exists, detected):
        x = self.temporal.adapter.graph_projection(self.graph(standardized_state))
        prediction = self.temporal.head(x)
        return dict(prediction=prediction, origin_eligible=track_exists[:, -1])


class Data:
    def __init__(self, split, fail_after=None):
        self.split, self.n = split, 4
        self.input_hashes = {'fixture':'deterministic'}
        self.calls, self.fail_after = 0, fail_after
    def batch(self, ids, snrs, device):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError('simulated interruption')
        self.calls += 1
        ids = torch.as_tensor(ids, device=device)
        x = ids[:,None,None,None].float().expand(-1,20,8,4)/10 + .1
        m = torch.ones(x.shape[:-1], dtype=torch.bool, device=device)
        return dict(state_hat=x, standardized_state=x, track_exists=m, detected=m), dict(future_position=x[..., :2]*.5, label_valid=m)


def main():
    torch.set_num_threads(1)
    checks=[]
    for e in range(4):
        ids,snrs=balanced_schedule(11,2026,e)
        assert len(set(ids))==11
    for e in range(4):
        _, snrs = balanced_schedule(100,2026,e,indices=[2,7,8,19,41,55,97])
        counts=[sum(snrs==s) for s in (5,10,15,20)]
        assert max(counts)-min(counts)<=1
    by_origin={i:set() for i in range(11)}
    for e in range(4):
        for i,s in zip(*balanced_schedule(11,2026,e)):
            by_origin[int(i)].add(int(s))
    assert all(v=={5,10,15,20} for v in by_origin.values())
    checks.append('one real origin per epoch and four-condition cycle')
    pred=torch.zeros(1,20,2,2); truth=pred.clone(); truth[:,:,0,0]=2;truth[:,:,1,0]=10
    mask=torch.zeros(1,20,2,dtype=torch.bool);mask[:,:,0]=True;mask[:,0,1]=True
    eligible=torch.ones(1,2,dtype=torch.bool)
    assert scene_metrics(pred,truth,mask,eligible)['scene_ade'].item()==6
    assert abs(masked_trajectory_loss(pred,truth,mask,eligible)['scene_ade'].item()-50/21)<1e-6
    checks.append('evaluation target-first ADE differs correctly from training valid-point ADE')
    try: Dataset('test')
    except ValueError: pass
    else: raise AssertionError('test was allowed')
    checks.append('locked test loader rejected before filesystem access')
    model=Tiny();configure_phase(model,'adapt',3e-4)
    assert all(not p.requires_grad for p in model.temporal.head.parameters())
    output=model(**Data('train').batch([0],20,'cpu')[0]);output['prediction'].sum().backward()
    assert model.graph.weight.grad is not None and model.graph.weight.grad.abs().sum()>0
    checks.append('adapt freezes head and keeps gradients to graph through frozen head')
    work=ROOT/'.codex-work'/'check_training';work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        tmp=Path(tmp)
        opts=dict(seed=71,epochs=2,patience=3,graph_lr=3e-4,batch_size=2,micro_batch=1)
        import fcntl
        locked=tmp/'locked';locked.mkdir()
        with (locked/'.training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:fit(Tiny(),Data('train'),Data('V_select'),locked,**opts)
            except RuntimeError as e:assert 'already active' in str(e)
            else:raise AssertionError('Concurrent training writer accepted')
        checks.append('phase lock rejects concurrent training or resume without stale PID files')
        seed_all(12);complete=Tiny();summary=fit(complete,Data('train'),Data('V_select'),tmp/'clean',**opts)
        seed_all(12);partial=Tiny()
        try:fit(partial,Data('train',fail_after=2),Data('V_select'),tmp/'resume',**opts)
        except RuntimeError as exc:assert 'simulated interruption' in str(exc)
        else:raise AssertionError('No interruption')
        seed_all(999);continued=Tiny();resumed=fit(continued,Data('train'),Data('V_select'),tmp/'resume',resume=True,**opts)
        for k,v in mutable_state(complete).items():assert torch.equal(v,mutable_state(continued)[k]),k
        assert summary['best_metrics']['J']==resumed['best_metrics']['J']
        checks.append('interrupted resume exactly matches uninterrupted weights and validation J with dropout')
        try:fit(continued,Data('train'),Data('V_select'),tmp/'resume',resume=True,**dict(opts,seed=72))
        except ValueError:pass
        else:raise AssertionError('Changed contract accepted')
        checks.append('resume rejects changed seed/config')
        seed_all(91);destination=Tiny();before=destination.temporal.adapter.graph_projection.weight.detach().clone()
        copy_own_weights(destination,summary['best_checkpoint'])
        assert torch.equal(before,destination.temporal.adapter.graph_projection.weight)
        assert torch.equal(destination.temporal.head[1].weight,complete.temporal.head[1].weight)
        checks.append('own checkpoint copies common temporal weights and preserves independently initialized graph projection')
        from scripts import run_f05
        import hashlib
        original_root = run_f05.ROOT
        run_f05.ROOT = tmp/'gate'
        run_f05.ROOT.mkdir()
        code_paths = ['prediction/training.py','prediction/model.py','prediction/quantum.py',
                      'prediction/temporal.py','prediction/classical.py','prediction/evaluation_cache.py','scripts/run_f05.py']
        data_paths = [f'data/f01d/inputs/{split}_snr_{snr}.npz' for split in ('train','V_select') for snr in (5,10,15,20)]
        data_paths += ['data/f01d/labels/train.npz','data/f01d/labels/V_select.npz','data/f01d/normalization.json']
        manifest = {}
        for name in code_paths+data_paths+['checkpoints/f04.pt','models/gpt2/config.json','models/gpt2/model.safetensors','reports/f04_contract.json']:
            f=run_f05.ROOT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'fixture')
            manifest[name]=hashlib.sha256(b'fixture').hexdigest()
        decision=run_f05.ROOT/'reports/preflight_decision.json';decision.parent.mkdir(parents=True,exist_ok=True)
        decision.write_text(json.dumps({'status':'PASS'}))
        protocol=dict(status='PASS',budget_accepted=True,training=dict(batch_size=16,micro_batch=1,warmup_epochs=3,adapt_epochs=2,joint_epochs=15,patience=4),
                      models=dict(qgnn=dict(depth=3,graph_lr=3e-4),gnn=dict(depth=2,graph_lr=3e-4)),
                      provenance=dict(f04_contract='reports/f04_contract.json',f04_contract_sha256=manifest['reports/f04_contract.json'],pretrained_hashes={k:manifest[k] for k in ('models/gpt2/config.json','models/gpt2/model.safetensors')},code_hashes={k:manifest[k] for k in code_paths},data_hashes={k:manifest[k] for k in data_paths},
                                      checkpoint_hashes={'checkpoints/f04.pt':manifest['checkpoints/f04.pt']},
                                      f04_decision_sha256=hashlib.sha256(decision.read_bytes()).hexdigest()))
        protocol_path=run_f05.ROOT/'selected.yaml';protocol_path.write_text(json.dumps(protocol))
        assert run_f05.read_protocol(protocol_path)['status']=='PASS'
        (run_f05.ROOT/code_paths[0]).write_bytes(b'changed')
        try:run_f05.read_protocol(protocol_path)
        except RuntimeError as e:assert 'changed' in str(e)
        else:raise AssertionError('Code drift accepted')
        try:run_f05.file_hash('data/f01d/labels/test.npz')
        except RuntimeError:pass
        else:raise AssertionError('Provenance accessed locked test')
        run_f05.ROOT=original_root
        checks.append('F05 gate verifies sealed hashes and rejects implementation drift')
        checks.append('F05 provenance refuses locked test paths before access')

    result=dict(passed=True,checks=checks,count=len(checks))
    from prediction.training import write_json
    write_json(ROOT/'reports/training_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
