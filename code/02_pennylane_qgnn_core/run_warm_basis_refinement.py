"""Short, checkpoint-preserving refinement of matched strong dual models."""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as r
import run as retained
from warm_basis_model import build_warm_basis_graph,restore_strong
from run_multitarget_experiment import prediction_loss
SEED=2026;ARMS=('warm_classical_dual','warm_quantum_dual')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def banks():
    with np.load(BASE/'retained_inputs/circuit_split_indices.npz') as s:ti=s['circuit_train_indices'].copy();di=s['circuit_dev_indices'].copy()
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:states=d['train_states'].copy();masks=d['train_mask'].copy()
    si=r.select_profile_indices(di,256)
    def b(i):return {'history':torch.from_numpy(states[i,:20]).float().cuda(),'future':torch.from_numpy(states[i,20:]).float().cuda(),'mask':torch.from_numpy(masks[i]).bool().cuda()}
    return b(ti),b(si),b(di),ti,si,di
def expected():
    z={}
    with (BASE/'converged_protocol_seed2026_v1/quantum_graph.jsonl').open() as f:
        for line in f:x=json.loads(line);z[x['epoch']]=x['input_sha256']
    return z
def aux(model,future,mask):
    rel=future[...,:2][:,:,:,None,:]-future[...,:2][:,:,None,:,:];target=torch.exp(-rel.norm(dim=-1).amin(1)/8)
    n=mask.shape[1];valid=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=mask.device)[None]
    return torch.stack([(((torch.sigmoid(x.last_risk_logits)-target)**2)*valid).sum()/valid.sum().clamp_min(1) for x in model.graph_layers]).mean()
def train(model,arm,tr,se,out,exp):
    named=dict(model.named_parameters());new_terms=('basis_','difference_value','joint_value','risk_head')
    special=[p for n,p in named.items() if any(x in n for x in new_terms)];ids={id(p) for p in special};old=[p for p in model.parameters() if id(p) not in ids]
    opt=torch.optim.AdamW([{'params':special,'lr':5e-4},{'params':old,'lr':1e-4}],weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=16,eta_min=1e-5)
    initial,_=retained.evaluate(model,se);best=initial['aggregate']['ade_m']+.35*initial['aggregate']['fde_m'];best_ep=0;ck=out/(arm+'_selected.pt')
    torch.save({'state':{n:t.detach().cpu().clone() for n,t in model.state_dict().items()},'epoch':0,'metrics':initial},ck)
    with (out/(arm+'.jsonl')).open('x') as log:
      for ep in range(1,17):
        st=time.time();r.set_seed(SEED+10000+ep);model.train();gen=torch.Generator(device='cuda').manual_seed(SEED+10000+ep);order=torch.randperm(len(tr['history']),generator=torch.Generator().manual_seed(SEED+10000+ep)).cuda();tot=at=0.;cnt=0;dig=r.InputDigest()
        for k in range(0,len(order),24):
            ix=order[k:k+24];snr=(5,10,15,20)[(k//24+ep)%4];h,f,m=retained.make_batch(tr,ix,snr,gen);dig.update(h);opt.zero_grad(set_to_none=True);pred=model(h,m);a=aux(model,f,m);loss=prediction_loss(pred,f,m,True)+.02*a;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();tot+=float(loss.detach())*len(ix);at+=float(a.detach())*len(ix);cnt+=len(ix)
        assert dig.hexdigest()==exp[ep];sched.step();metrics,_=retained.evaluate(model,se);score=metrics['aggregate']['ade_m']+.35*metrics['aggregate']['fde_m']
        if score<best:best=score;best_ep=ep;torch.save({'state':{n:t.detach().cpu().clone() for n,t in model.state_dict().items()},'epoch':ep,'metrics':metrics},ck)
        rec={'arm':arm,'epoch':ep,'loss':tot/cnt,'aux':at/cnt,'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-st};log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True);r.atomic_json(out/'progress.json',{'status':'training','last':rec})
    z=torch.load(ck,map_location='cpu',weights_only=True);model.load_state_dict(z['state']);return best_ep,z['metrics']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-name',default='warm_basis_refine_seed2026_v1');ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available();torch.set_num_threads(4)
    out=BASE/a.output_name;out.mkdir(exist_ok=False);tr,se,dev,ti,si,di=banks();retained.NOISE=r.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json');evidence={}
    for arm in ARMS:
        r.set_seed(SEED);m=build_warm_basis_graph(arm).cuda();p,n=restore_strong(m,arm);h,f,mask=tr['history'][:2],tr['future'][:2],tr['mask'][:2];o=m(h,mask);loss=prediction_loss(o,f,mask,True)+.02*aux(m,f,mask);loss.backward();gr=[max(float(q.grad.abs().max()) for q in x.core.parameters() if q.grad is not None) for x in m.graph_layers];assert min(gr)>0;evidence[arm]={'source':str(p),'loaded_tensors':n,'loss':float(loss.detach()),'core_gradients':gr};del m;torch.cuda.empty_cache()
    r.atomic_json(out/'smoke.json',evidence)
    if a.smoke_only:r.atomic_json(out/'completed.json',{'status':'smoke_completed'});print(json.dumps(evidence));return
    src=[BASE/'warm_basis_model.py',BASE/'run_warm_basis_refinement.py'];hashes={str(p):sha(p) for p in src};r.atomic_json(out/'protocol.json',{'seed':SEED,'epochs':16,'initial_checkpoint_is_candidate':True,'aux_weight':.02,'new_lr':5e-4,'old_lr':1e-4,'no_llm':True,'no_test':True,'indices':{'train':ti.tolist(),'selection':si.tolist(),'dev':di.tolist()},'hashes':hashes})
    sums={}
    for arm in ARMS:
        r.set_seed(SEED);m=build_warm_basis_graph(arm).cuda();source,n=restore_strong(m,arm);ep,sel=train(m,arm,tr,se,out,expected());metrics,arr=retained.evaluate(m,dev,collect=True);np.savez_compressed(out/(arm+'_full_dev.npz'),**arr,indices=di);sums[arm]={'graph':metrics,'selected_epoch':ep,'selection':sel,'source':str(source),'parameters':sum(p.numel() for p in m.parameters())};r.atomic_json(out/'summary.json',sums);del m;torch.cuda.empty_cache()
    assert all(sha(p)==v for p,v in hashes.items());done={'status':'completed','summaries':sums,'source_hashes_verified':True,'no_llm':True,'no_test':True};r.atomic_json(out/'completed.json',done);r.atomic_json(out/'progress.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
