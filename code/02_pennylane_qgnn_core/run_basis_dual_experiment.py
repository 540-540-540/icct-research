"""Matched seed-2026 experiment for basis-mixing hierarchical dual QGNN."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/js_cn/sensing'); BASE=ROOT/'diagnostics/core_message_seed2026_v2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as r
import run as retained
from physics_aligned_basis_model import build_basis_dual_graph
from run_multitarget_experiment import prediction_loss

SEED=2026; ARMS=('basis_classical_dual','basis_quantum_dual'); REF=BASE/'converged_protocol_seed2026_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(m): return r.tensor_mapping_sha256({n:p.detach().cpu() for n,p in m.items()})

def banks():
    with np.load(BASE/'retained_inputs/circuit_split_indices.npz') as s: ti=s['circuit_train_indices'].copy();di=s['circuit_dev_indices'].copy()
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d: states=d['train_states'].copy();masks=d['train_mask'].copy()
    si=r.select_profile_indices(di,256)
    def bank(idx): return {'history':torch.from_numpy(states[idx,:20]).float().cuda(),'future':torch.from_numpy(states[idx,20:]).float().cuda(),'mask':torch.from_numpy(masks[idx]).bool().cuda()}
    return bank(ti),bank(si),bank(di),ti,si,di

def expected():
    out={}
    with (REF/'quantum_graph.jsonl').open() as f:
        for line in f:
            x=json.loads(line);out['graph_'+str(x['epoch'])]=x['input_sha256']
    return out

def warm_start(model):
    source=torch.load(REF/'plain_graph_selected.pt',map_location='cpu',weights_only=True)['state']
    own=model.state_dict(); used=[]
    for name,value in source.items():
        if not name.startswith('graph_layers.') and name in own and own[name].shape==value.shape:
            own[name]=value;used.append(name)
    model.load_state_dict(own,strict=True)
    return thash({n:p for n,p in model.named_parameters() if not n.startswith('graph_layers.')}),len(used)

def aux_loss(model,future,mask):
    pos=future[...,:2]
    relative=pos[:,:,:,None,:]-pos[:,:,None,:,:]
    minimum=relative.norm(dim=-1).amin(dim=1)
    target=torch.exp(-minimum/8.0)
    n=mask.shape[1]; valid=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=mask.device)[None]
    losses=[]
    for layer in model.graph_layers:
        losses.append(((torch.sigmoid(layer.last_risk_logits)-target).square()*valid).sum()/valid.sum().clamp_min(1))
    return torch.stack(losses).mean()

def train(model,arm,train_bank,selection,out,expected_hashes):
    named=dict(model.named_parameters()); common=[p for n,p in named.items() if not n.startswith('graph_layers.')]
    core_names=('core.','angle_','score.','gate.','basis_readout.','risk_head.')
    core=[p for n,p in named.items() if n.startswith('graph_layers.') and any(x in n for x in core_names)]
    # Avoid tensor equality in membership testing.
    core_ids={id(p) for p in core}; graph_other=[p for n,p in named.items() if n.startswith('graph_layers.') and id(p) not in core_ids]
    opt=torch.optim.AdamW([{'params':core,'lr':8e-4},{'params':graph_other,'lr':3e-4},{'params':common,'lr':8e-5}],weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=1e-5)
    for p in common:p.requires_grad_(False)
    best=float('inf');best_ep=0;ck=out/(arm+'_graph_selected.pt')
    with (out/(arm+'_graph.jsonl')).open('x') as log:
        for ep in range(1,41):
            if ep==5:
                for p in common:p.requires_grad_(True)
            started=time.time();r.set_seed(SEED+10000+ep);model.train();gen=torch.Generator(device='cuda').manual_seed(SEED+10000+ep)
            order=torch.randperm(len(train_bank['history']),generator=torch.Generator().manual_seed(SEED+10000+ep)).cuda();total=0.;aux_total=0.;count=0;dig=r.InputDigest()
            for k in range(0,len(order),24):
                idx=order[k:k+24];snr=(5,10,15,20)[(k//24+ep)%4]
                h,f,m=retained.make_batch(train_bank,idx,snr,gen);dig.update(h);opt.zero_grad(set_to_none=True)
                pred=model(h,m);aux=aux_loss(model,f,m);loss=prediction_loss(pred,f,m,True)+.08*aux
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);opt.step()
                total+=float(loss.detach())*len(idx);aux_total+=float(aux.detach())*len(idx);count+=len(idx)
            key='graph_'+str(ep);assert dig.hexdigest()==expected_hashes[key];sched.step();metrics,_=retained.evaluate(model,selection)
            score=metrics['aggregate']['ade_m']+.35*metrics['aggregate']['fde_m']
            if score<best:
                best=score;best_ep=ep;torch.save({'state':{n:t.detach().cpu().clone() for n,t in model.state_dict().items()},'epoch':ep,'metrics':metrics},ck)
            rec={'arm':arm,'epoch':ep,'train_loss':total/count,'aux_loss':aux_total/count,'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-started,'warmup':ep<=4}
            log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True);r.atomic_json(out/'progress.json',{'status':'training','last':rec})
    selected=torch.load(ck,map_location='cpu',weights_only=True);model.load_state_dict(selected['state']);return best_ep

def smoke(out,train_bank):
    evidence={}; outside=None; noncore=None; warm=None
    for arm in ARMS:
        r.set_seed(SEED);model=build_basis_dual_graph(arm).cuda();wh,used=warm_start(model)
        oh=thash({n:p for n,p in model.named_parameters() if not n.startswith('graph_layers.')})
        nh=thash({n:p for n,p in model.named_parameters() if n.startswith('graph_layers.') and '.core.' not in n})
        if outside is None:outside=oh;noncore=nh;warm=wh
        assert (oh,nh,wh)==(outside,noncore,warm)
        h,f,m=train_bank['history'][:2],train_bank['future'][:2],train_bank['mask'][:2]
        pred=model(h,m);aux=aux_loss(model,f,m);loss=prediction_loss(pred,f,m,True)+.08*aux;loss.backward()
        cg=[max(float(p.grad.abs().max()) for p in layer.core.parameters() if p.grad is not None) for layer in model.graph_layers]
        rg=[float(layer.risk_head.weight.grad.abs().max()) for layer in model.graph_layers]
        assert min(cg)>0 and min(rg)>0
        evidence[arm]={'loss':float(loss.detach()),'aux_loss':float(aux.detach()),'core_gradients':cg,'risk_gradients':rg,'warm_tensors':used,'parameters':sum(p.numel() for p in model.parameters())}
        del model;torch.cuda.empty_cache()
    r.atomic_json(out/'smoke.json',evidence);return evidence

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-name',default='basis_dual_seed2026_v1');ap.add_argument('--smoke-only',action='store_true');args=ap.parse_args()
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);out=BASE/args.output_name;out.mkdir(exist_ok=False)
    tr,se,dev,ti,si,di=banks();retained.NOISE=r.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    ev=smoke(out,tr)
    if args.smoke_only:r.atomic_json(out/'completed.json',{'status':'smoke_completed','evidence':ev});print(json.dumps(ev));return
    sources=[BASE/'physics_aligned_basis_model.py',BASE/'run_basis_dual_experiment.py'];hashes={str(p):sha(p) for p in sources}
    r.atomic_json(out/'protocol.json',{'seed':SEED,'arms':list(ARMS),'warm_start':'common non-graph tensors from selected plain GNN','frozen_warmup_epochs':4,'max_epochs':40,'auxiliary':'future minimum pair-distance risk, weight 0.08','message_bases':['sender','edge','difference','joint'],'no_llm':True,'no_test':True,'indices':{'train':ti.tolist(),'selection':si.tolist(),'dev':di.tolist()},'source_hashes':hashes})
    summaries={}
    for arm in ARMS:
        r.set_seed(SEED);model=build_basis_dual_graph(arm).cuda();wh,used=warm_start(model);best=train(model,arm,tr,se,out,expected())
        metrics,arrays=retained.evaluate(model,dev,collect=True);np.savez_compressed(out/(arm+'_graph_full_dev.npz'),**arrays,indices=di)
        summaries[arm]={'graph':metrics,'selected_epoch':best,'parameters':sum(p.numel() for p in model.parameters()),'warm_hash':wh};r.atomic_json(out/'summary.json',summaries)
        del model;torch.cuda.empty_cache()
    assert all(sha(p)==v for p,v in hashes.items())
    done={'status':'completed','summaries':summaries,'no_llm':True,'no_test':True,'source_hashes_verified':True};r.atomic_json(out/'completed.json',done);r.atomic_json(out/'progress.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
