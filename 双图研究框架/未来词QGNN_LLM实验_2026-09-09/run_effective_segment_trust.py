"""Train alpha-times-gate QGNN routing into a three-stage LLM trust interface."""
from __future__ import annotations
import argparse,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as utils
import run as retained
from analyze_interaction_strata import scene_features
from future_token_model import FutureTokenCoreGraphLLM,future_compact_state,restore_future_compact
from interaction_trust_model import EffectiveSegmentInteractionTrustFutureLLM
from physics_aligned_dual_model_effective import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss
SEED=2026;GRAPH=BASE/'physics_aligned_dual_seed2026_v1';SEGMENTS=3
def restore(m,state):
    missing,extra=m.load_state_dict(state,strict=False);assert not extra,extra;assert all(((x.startswith('gpt2.') or x.startswith('future_gpt2.')) and 'lora_' not in x) or x.startswith('trust_') for x in missing),missing
def build(arm):
    name=f'physics_{arm}_dual';src=torch.load(BASE/f"future_token_{'qgnn' if arm=='quantum' else 'classical'}_llm_seed2026_v1"/'future_token_qgnn_llm_selected.pt',map_location='cpu',weights_only=True);gs=torch.load(GRAPH/f'{name}_graph_selected.pt',map_location='cpu',weights_only=True)['state'];g=build_physics_aligned_dual_graph(name);g.load_state_dict(gs);ref=FutureTokenCoreGraphLLM(g);restore_future_compact(ref,src['state']);g=build_physics_aligned_dual_graph(name);g.load_state_dict(gs);m=EffectiveSegmentInteractionTrustFutureLLM(g,SEGMENTS);restore(m,src['state']);return ref,m
def configure(m):
    for p in m.parameters():p.requires_grad_(False)
    for n,p in m.named_parameters():
        if n.startswith('trust_'):p.requires_grad_(True)
def groups(indices):
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:f=scene_features(d['train_states'][indices,:20],d['train_mask'][indices])
    q=np.quantile(f['composite'],[1/3,2/3]);s=f['composite'];return {'low':np.flatnonzero(s<=q[0]),'medium':np.flatnonzero((s>q[0])&(s<=q[1])),'high':np.flatnonzero(s>q[1])}
def order_for(g,total,ep):
    z=torch.Generator().manual_seed(SEED+200000+ep);c={'high':total//2,'medium':total//4};c['low']=total-c['high']-c['medium'];x=torch.cat([torch.as_tensor(g[k])[torch.randint(len(g[k]),(c[k],),generator=z)] for k in ('high','medium','low')]);return x[torch.randperm(len(x),generator=z)].cuda()
def subset(b,p):i=torch.as_tensor(p,device='cuda');return {k:v.index_select(0,i) for k,v in b.items()}
def optimal(o,f,m):
    graph=o['graph_future_position'];corr=o['ungated_future_position']-graph;target=f[...,:2]-graph;values=[]
    for ci,ti in zip(torch.tensor_split(corr,SEGMENTS,dim=1),torch.tensor_split(target,SEGMENTS,dim=1)):
        values.append(((ci*ti).sum((1,3))/ci.square().sum((1,3)).clamp_min(1e-6)).clamp(0.,2.))
    return torch.stack(values,-1),m[:,:,None].expand(-1,-1,SEGMENTS)
def train(m,b,g,sel,out,epochs):
    configure(m);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=5e-6);initial,_=retained.evaluate(m,sel);best=initial['aggregate']['ade_m']+.5*initial['aggregate']['fde_m'];best_ep=0;ck=out/'segment_trust_selected.pt';torch.save({'state':future_compact_state(m),'epoch':0,'metrics':initial},ck)
    with (out/'training.jsonl').open('x') as log:
      for ep in range(1,epochs+1):
        st=time.time();utils.set_seed(SEED+200000+ep);m.train();gen=torch.Generator(device='cuda').manual_seed(SEED+200000+ep);order=order_for(g,len(b['history']),ep);cs=os=0.;pm=torch.zeros(SEGMENTS);tm=torch.zeros(SEGMENTS);count=0
        for off in range(0,len(order),24):
            ix=order[off:off+24];h,f,mask=retained.make_batch(b,ix,(5,10,15,20)[(off//24+ep)%4],gen);opt.zero_grad(set_to_none=True);o=m(h,mask);target,v=optimal(o,f,mask);pred=o['segment_gain'];coord=prediction_loss(o,f,mask,True);aux=F.mse_loss(pred[v],target[v]);loss=coord+aux;loss.backward();torch.nn.utils.clip_grad_norm_(params,3.);opt.step();n=len(ix);count+=n;cs+=float(coord.detach())*n;os+=float(aux.detach())*n;pm+=pred.detach()[v].view(-1,SEGMENTS).mean(0).cpu()*n;tm+=target.detach()[v].view(-1,SEGMENTS).mean(0).cpu()*n
        sched.step();metrics,_=retained.evaluate(m,sel);score=metrics['aggregate']['ade_m']+.5*metrics['aggregate']['fde_m'];
        if score<best:best=score;best_ep=ep;torch.save({'state':future_compact_state(m),'epoch':ep,'metrics':metrics},ck)
        rec={'epoch':ep,'coordinate_loss':cs/count,'optimal_gain_loss':os/count,'optimal_gain_mean':(tm/count).tolist(),'predicted_gain_mean':(pm/count).tolist(),'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-st};log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
    z=torch.load(ck,map_location='cpu',weights_only=True);restore(m,z['state']);return z
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('quantum','classical'),required=True);ap.add_argument('--output-name',required=True);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available();torch.set_num_threads(4);utils.set_seed(SEED);retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json');out=BASE/a.output_name;out.mkdir(exist_ok=False);b,sel,dev,ti,si,di=load_banks();g=groups(ti);dg=groups(di);used=set(int(x) for x in si.tolist());pos=np.asarray([int(p) for p in dg['high'] if int(di[int(p)]) not in used]);confirm=subset(dev,pos);ref,m=build(a.arm);ref,m=ref.cuda().eval(),m.cuda().eval();h,mask=sel['history'][:2],sel['mask'][:2]
    with torch.no_grad():exact=float((ref(h,mask)['future_position']-m(h,mask)['future_position']).abs().max());assert exact<2e-6
    configure(m);m.train();m(h,mask)['future_position'].sum().backward();grad=float(m.trust_segment_head[-1].weight.grad.abs().max());assert grad>0;utils.atomic_json(out/'smoke.json',{'exact':exact,'segment_gradient':grad})
    if a.smoke_only:utils.atomic_json(out/'completed.json',{'status':'smoke_completed'});return
    utils.atomic_json(out/'protocol.json',{'seed':SEED,'arm':a.arm,'interface':'effective QGNN influence alpha*gate routes LLM temporal states -> early/middle/late trust gains','segments':SEGMENTS,'selection':'fixed 256 scenes ADE+.5FDE','confirmation_scenes':int(len(pos)),'no_test':True});source,_=retained.evaluate(ref,confirm);z=train(m,b,g,sel,out,a.epochs);learned,_=retained.evaluate(m,confirm);m.trust_attention_mode='uniform';uniform,_=retained.evaluate(m,confirm);m.trust_attention_mode='off';off,_=retained.evaluate(m,confirm);m.trust_attention_mode='learned';full,_=retained.evaluate(m,dev);summary={'arm':a.arm,'selected_epoch':int(z['epoch']),'selection':z['metrics'],'source_confirmation':source,'confirmation_learned':learned,'confirmation_uniform':uniform,'confirmation_off':off,'full_dev_secondary':full,'no_test':True};utils.atomic_json(out/'summary.json',summary);utils.atomic_json(out/'completed.json',{'status':'completed','summary':summary});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
