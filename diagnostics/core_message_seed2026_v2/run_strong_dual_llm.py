"""Long, staged LLM refinement for the strongest matched physics-aligned dual graphs."""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as r
import run as retained
from model import CoreGraphLLM,compact_state,restore_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss
SEED=2026;ARMS=('physics_classical_dual','physics_quantum_dual');GRAPH_DIR=BASE/'physics_aligned_dual_seed2026_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def banks():
    with np.load(BASE/'retained_inputs/circuit_split_indices.npz') as s:ti=s['circuit_train_indices'].copy();di=s['circuit_dev_indices'].copy()
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:states=d['train_states'].copy();masks=d['train_mask'].copy()
    si=r.select_profile_indices(di,256)
    def b(i):return {'history':torch.from_numpy(states[i,:20]).float().cuda(),'future':torch.from_numpy(states[i,20:]).float().cuda(),'mask':torch.from_numpy(masks[i]).bool().cuda()}
    return b(ti),b(si),b(di),ti,si,di
def load_graph(arm):
    g=build_physics_aligned_dual_graph(arm).cuda();z=torch.load(GRAPH_DIR/(arm+'_graph_selected.pt'),map_location='cpu',weights_only=True);g.load_state_dict(z['state']);return g,z['epoch']
def expected():
    out={}
    with (BASE/'converged_protocol_seed2026_v1/quantum_llm.jsonl').open() as f:
        for line in f:x=json.loads(line);out[x['epoch']]=x['input_sha256']
    return out
def train(model,arm,tr,se,out,expected_hashes):
    named=dict(model.named_parameters());graph=[p for n,p in named.items() if n.startswith('graph_backbone.')]
    lora=[p for n,p in named.items() if p.requires_grad and 'lora_' in n]
    last=[p for n,p in named.items() if p.requires_grad and n.startswith('coordinate_head.4.')]
    ids={id(p) for p in graph+lora+last};heads=[p for n,p in named.items() if p.requires_grad and id(p) not in ids]
    opt=torch.optim.AdamW([{'params':graph,'lr':2e-5},{'params':heads,'lr':2e-4},{'params':last,'lr':5e-4},{'params':lora,'lr':5e-5}],weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=16,eta_min=5e-6)
    for p in graph:p.requires_grad_(False)
    initial,_=retained.evaluate(model,se);best=initial['aggregate']['ade_m']+.35*initial['aggregate']['fde_m'];best_ep=0;ck=out/(arm+'_llm_selected.pt')
    torch.save({'state':compact_state(model),'epoch':0,'metrics':initial},ck)
    with (out/(arm+'_llm.jsonl')).open('x') as log:
      for ep in range(1,17):
        if ep==5:
            for p in graph:p.requires_grad_(True)
        st=time.time();r.set_seed(SEED+20000+ep);model.train();gen=torch.Generator(device='cuda').manual_seed(SEED+20000+ep);order=torch.randperm(len(tr['history']),generator=torch.Generator().manual_seed(SEED+20000+ep)).cuda();ct=tt=0.;cnt=0;dig=r.InputDigest()
        for k in range(0,len(order),24):
            ix=order[k:k+24];snr=(5,10,15,20)[(k//24+ep)%4];h,f,m=retained.make_batch(tr,ix,snr,gen);dig.update(h);opt.zero_grad(set_to_none=True);o=model(h,m);coord=prediction_loss(o,f,m,True);tok=token_loss(o['token_logits'],model.future_token_ids(h,f),m);loss=coord+.02*tok;loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.);opt.step();ct+=float(coord.detach())*len(ix);tt+=float(tok.detach())*len(ix);cnt+=len(ix)
        digest=dig.hexdigest();
        if ep in expected_hashes:assert digest==expected_hashes[ep]
        sched.step();metrics,_=retained.evaluate(model,se);score=metrics['aggregate']['ade_m']+.35*metrics['aggregate']['fde_m']
        if score<best:best=score;best_ep=ep;torch.save({'state':compact_state(model),'epoch':ep,'metrics':metrics},ck)
        rec={'arm':arm,'epoch':ep,'coordinate_loss':ct/cnt,'token_ce':tt/cnt,'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-st,'graph_frozen':ep<=4};log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True);r.atomic_json(out/'progress.json',{'status':'training','last':rec})
    z=torch.load(ck,map_location='cpu',weights_only=True);restore_compact(model,z['state']);return best_ep,z['metrics']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-name',default='strong_dual_llm_seed2026_v1');ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0);torch.set_num_threads(4)
    out=BASE/a.output_name;out.mkdir(exist_ok=False);tr,se,dev,ti,si,di=banks();retained.NOISE=r.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json');evidence={};common=None
    for arm in ARMS:
        r.set_seed(SEED);g,gep=load_graph(arm);m=CoreGraphLLM(g).cuda();h,f,mask=tr['history'][:2],tr['future'][:2],tr['mask'][:2];o=m(h,mask);assert torch.equal(o['future_position'],o['graph_future_position']);loss=prediction_loss(o,f,mask,True)+.02*token_loss(o['token_logits'],m.future_token_ids(h,f),mask);loss.backward();ng=r.tensor_mapping_sha256({n:p.detach().cpu() for n,p in m.named_parameters() if not n.startswith('graph_backbone.')});
        if common is None:common=ng
        assert ng==common;evidence[arm]={'graph_epoch':gep,'initial_identity':True,'loss':float(loss.detach()),'non_graph_hash':ng};del m,g;torch.cuda.empty_cache()
    r.atomic_json(out/'smoke.json',evidence)
    if a.smoke_only:r.atomic_json(out/'completed.json',{'status':'smoke_completed'});print(json.dumps(evidence));return
    src=[BASE/'run_strong_dual_llm.py',BASE/'physics_aligned_dual_model.py',BASE/'model.py',ROOT/'MultiTargetTimeLLM.py'];hashes={str(p):sha(p) for p in src};r.atomic_json(out/'protocol.json',{'seed':SEED,'epochs':16,'graph_frozen_epochs':4,'graph_lr':2e-5,'head_lr':2e-4,'coordinate_final_lr':5e-4,'lora_lr':5e-5,'token_weight':.02,'epoch0_identity_fallback':True,'no_test':True,'indices':{'train':ti.tolist(),'selection':si.tolist(),'dev':di.tolist()},'hashes':hashes})
    sums={}
    for arm in ARMS:
        r.set_seed(SEED);g,gep=load_graph(arm);m=CoreGraphLLM(g).cuda();ep,sel=train(m,arm,tr,se,out,expected());metrics,arr=retained.evaluate(m,dev,collect=True);np.savez_compressed(out/(arm+'_llm_full_dev.npz'),**arr,indices=di);sums[arm]={'llm':metrics,'selected_epoch':ep,'selection':sel,'source_graph_epoch':gep,'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'total':sum(p.numel() for p in m.parameters())};r.atomic_json(out/'summary.json',sums);del m,g;torch.cuda.empty_cache()
    assert all(sha(p)==v for p,v in hashes.items());done={'status':'completed','summaries':sums,'source_hashes_verified':True,'no_test':True};r.atomic_json(out/'completed.json',done);r.atomic_json(out/'progress.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
