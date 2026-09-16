"""Coordinate-focused continuation from the best frozen-stage quantum LLM checkpoint."""
import argparse,json,platform,sys,time
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
SEED=2026;SOURCE=BASE/'strong_dual_llm_seed2026_v1';GRAPH=BASE/'physics_aligned_dual_seed2026_v1'
def banks():
    with np.load(BASE/'retained_inputs/circuit_split_indices.npz') as s:ti=s['circuit_train_indices'].copy();di=s['circuit_dev_indices'].copy()
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:states=d['train_states'].copy();masks=d['train_mask'].copy()
    si=r.select_profile_indices(di,256)
    def b(i):return {'history':torch.from_numpy(states[i,:20]).float().cuda(),'future':torch.from_numpy(states[i,20:]).float().cuda(),'mask':torch.from_numpy(masks[i]).bool().cuda()}
    return b(ti),b(si),b(di),ti,si,di
def load_model():
    g=build_physics_aligned_dual_graph('physics_quantum_dual');g.load_state_dict(torch.load(GRAPH/'physics_quantum_dual_graph_selected.pt',map_location='cpu',weights_only=True)['state']);m=CoreGraphLLM(g)
    z=torch.load(SOURCE/'physics_quantum_dual_llm_selected.pt',map_location='cpu',weights_only=True);assert z['epoch']==4;restore_compact(m,z['state']);return m,z
def train(model,tr,se,out):
    for p in model.graph_backbone.parameters():p.requires_grad_(False)
    named=dict(model.named_parameters());lora=[p for n,p in named.items() if p.requires_grad and 'lora_' in n];last=[p for n,p in named.items() if p.requires_grad and n.startswith('coordinate_head.4.')];ids={id(p) for p in lora+last};heads=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    opt=torch.optim.AdamW([{'params':heads,'lr':1e-4},{'params':last,'lr':3e-4},{'params':lora,'lr':2.5e-5}],weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=16,eta_min=5e-6)
    initial,_=retained.evaluate(model,se);best=initial['aggregate']['ade_m']+.35*initial['aggregate']['fde_m'];best_ep=0;ck=out/'physics_quantum_dual_llm_selected.pt';torch.save({'state':compact_state(model),'epoch':0,'metrics':initial},ck)
    with (out/'physics_quantum_dual_llm.jsonl').open('x') as log:
      for ep in range(1,17):
        st=time.time();r.set_seed(SEED+30000+ep);model.train();gen=torch.Generator(device='cuda').manual_seed(SEED+30000+ep);order=torch.randperm(len(tr['history']),generator=torch.Generator().manual_seed(SEED+30000+ep)).cuda();ct=tt=0.;cnt=0
        for k in range(0,len(order),24):
            ix=order[k:k+24];snr=(5,10,15,20)[(k//24+ep)%4];h,f,m=retained.make_batch(tr,ix,snr,gen);opt.zero_grad(set_to_none=True);o=model(h,m);coord=prediction_loss(o,f,m,True);tok=token_loss(o['token_logits'],model.future_token_ids(h,f),m);loss=coord+.005*tok;loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.);opt.step();ct+=float(coord.detach())*len(ix);tt+=float(tok.detach())*len(ix);cnt+=len(ix)
        sched.step();metrics,_=retained.evaluate(model,se);score=metrics['aggregate']['ade_m']+.35*metrics['aggregate']['fde_m']
        if score<best:best=score;best_ep=ep;torch.save({'state':compact_state(model),'epoch':ep,'metrics':metrics},ck)
        rec={'epoch':ep,'coordinate_loss':ct/cnt,'token_ce':tt/cnt,'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-st};log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True);r.atomic_json(out/'progress.json',{'status':'training','last':rec})
    z=torch.load(ck,map_location='cpu',weights_only=True);restore_compact(model,z['state']);return z
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-name',default='quantum_llm_frozen_cont_seed2026_v1');ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available();torch.set_num_threads(4);out=BASE/a.output_name;out.mkdir(exist_ok=False);tr,se,dev,ti,si,di=banks();retained.NOISE=r.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json');m,z=load_model();m=m.cuda();initial,_=retained.evaluate(m,se);assert abs(initial['aggregate']['ade_m']-z['metrics']['aggregate']['ade_m'])<1e-7;r.atomic_json(out/'smoke.json',{'source_epoch':4,'restored_exactly':True,'initial':initial})
    if a.smoke_only:r.atomic_json(out/'completed.json',{'status':'smoke_completed'});print(json.dumps(initial));return
    r.atomic_json(out/'protocol.json',{'seed':SEED,'source':'strong_dual_llm_seed2026_v1 quantum selected epoch 4','epochs':16,'graph_frozen':True,'token_weight':.005,'head_lr':1e-4,'coordinate_final_lr':3e-4,'lora_lr':2.5e-5,'epoch0_fallback':True,'no_test':True,'indices':{'train':ti.tolist(),'selection':si.tolist(),'dev':di.tolist()}})
    best=train(m,tr,se,out);metrics,arr=retained.evaluate(m,dev,collect=True);np.savez_compressed(out/'physics_quantum_dual_llm_full_dev.npz',**arr,indices=di);summary={'llm':metrics,'continuation_selected_epoch':best['epoch'],'selection':best['metrics'],'source_epoch':4};r.atomic_json(out/'summary.json',summary);done={'status':'completed','summary':summary,'no_test':True};r.atomic_json(out/'completed.json',done);r.atomic_json(out/'progress.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
