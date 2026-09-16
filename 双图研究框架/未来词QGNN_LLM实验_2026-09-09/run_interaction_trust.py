"""Train and independently validate the QGNN-conditioned LLM trust gate."""
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
from interaction_trust_model import InteractionTrustFutureLLM
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss
SEED=2026;GRAPH=BASE/'physics_aligned_dual_seed2026_v1'

def restore(model,state):
    missing,extra=model.load_state_dict(state,strict=False);assert not extra,extra
    assert all(((x.startswith('gpt2.') or x.startswith('future_gpt2.')) and 'lora_' not in x) or x.startswith('trust_') for x in missing),missing
def build(arm):
    name=f'physics_{arm}_dual';src=torch.load(BASE/f"future_token_{'qgnn' if arm=='quantum' else 'classical'}_llm_seed2026_v1"/'future_token_qgnn_llm_selected.pt',map_location='cpu',weights_only=True);gs=torch.load(GRAPH/f'{name}_graph_selected.pt',map_location='cpu',weights_only=True)['state']
    g=build_physics_aligned_dual_graph(name);g.load_state_dict(gs);ref=FutureTokenCoreGraphLLM(g);restore_future_compact(ref,src['state'])
    g=build_physics_aligned_dual_graph(name);g.load_state_dict(gs);model=InteractionTrustFutureLLM(g);restore(model,src['state']);return ref,model
def configure(m):
    for p in m.parameters():p.requires_grad_(False)
    for n,p in m.named_parameters():
        if n.startswith('trust_'):p.requires_grad_(True)
def groups(indices):
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:f=scene_features(d['train_states'][indices,:20],d['train_mask'][indices])
    q=np.quantile(f['composite'],[1/3,2/3]);s=f['composite'];return {'low':np.flatnonzero(s<=q[0]),'medium':np.flatnonzero((s>q[0])&(s<=q[1])),'high':np.flatnonzero(s>q[1])}
def order_for(g,total,ep):
    z=torch.Generator().manual_seed(SEED+180000+ep);counts={'high':total//2,'medium':total//4};counts['low']=total-counts['high']-counts['medium'];x=torch.cat([torch.as_tensor(g[k])[torch.randint(len(g[k]),(counts[k],),generator=z)] for k in ('high','medium','low')]);return x[torch.randperm(len(x),generator=z)].cuda()
def subset(b,p):
    i=torch.as_tensor(p,device='cuda');return {k:v.index_select(0,i) for k,v in b.items()}
def optimal_trust(o,f,m):
    graph=o['graph_future_position'];correction=o['ungated_future_position']-graph;target_residual=f[...,:2]-graph
    gain=(correction*target_residual).sum(-1)/correction.square().sum(-1).clamp_min(1e-6)
    gain=gain.clamp(0.,2.);valid=m[:,None,:].expand_as(gain);return gain,valid
def train(m,bank,g,selection,out,epochs):
    configure(m);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1.5e-4,weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5)
    initial,_=retained.evaluate(m,selection);best=initial['aggregate']['ade_m']+.35*initial['aggregate']['fde_m'];best_ep=0;ck=out/'trust_selected.pt';torch.save({'state':future_compact_state(m),'epoch':0,'metrics':initial},ck)
    with (out/'training.jsonl').open('x') as log:
      for ep in range(1,epochs+1):
        st=time.time();utils.set_seed(SEED+180000+ep);m.train();gen=torch.Generator(device='cuda').manual_seed(SEED+180000+ep);order=order_for(g,len(bank['history']),ep);cs=us=gm=pr=0.;count=0
        for off in range(0,len(order),24):
            ix=order[off:off+24];h,f,mask=retained.make_batch(bank,ix,(5,10,15,20)[(off//24+ep)%4],gen);opt.zero_grad(set_to_none=True);o=m(h,mask);target_gain,v=optimal_trust(o,f,mask);coord=prediction_loss(o,f,mask,True);u=F.smooth_l1_loss(o['trust_gain'][v],target_gain[v],beta=.25);loss=coord+.05*u;loss.backward();torch.nn.utils.clip_grad_norm_(params,3.);opt.step();n=len(ix);count+=n;cs+=float(coord.detach())*n;us+=float(u.detach())*n;gm+=float(m.last_trust_mean)*n;pr+=float(target_gain[v].mean())*n
        sched.step();metrics,_=retained.evaluate(m,selection);score=metrics['aggregate']['ade_m']+.35*metrics['aggregate']['fde_m']
        if score<best:best=score;best_ep=ep;torch.save({'state':future_compact_state(m),'epoch':ep,'metrics':metrics},ck)
        rec={'epoch':ep,'coordinate_loss':cs/count,'optimal_gain_loss':us/count,'optimal_gain_mean':pr/count,'predicted_gain_mean':gm/count,'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-st};log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
    z=torch.load(ck,map_location='cpu',weights_only=True);restore(m,z['state']);return z
@torch.no_grad()
def utility_metrics(m,b):
    absolute=total=0.;within=0
    for s in range(0,len(b['history']),24):
        i=torch.arange(s,min(s+24,len(b['history'])),device='cuda');h=b['history'].index_select(0,i);f=b['future'].index_select(0,i);mask=b['mask'].index_select(0,i);o=m(h,mask);target,v=optimal_trust(o,f,mask);delta=(o['trust_gain']-target).abs();absolute+=float(delta[v].sum());within+=int(((delta<=.25)&v).sum());total+=int(v.sum())
    return {'gain_mae':absolute/total,'within_0.25_rate':within/total}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('quantum','classical'),required=True);ap.add_argument('--output-name',required=True);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available();torch.set_num_threads(4);utils.set_seed(SEED);retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json');out=BASE/a.output_name;out.mkdir(exist_ok=False)
    bank,selection,dev,ti,si,di=load_banks();g=groups(ti);dg=groups(di);selected_ids=set(int(x) for x in si.tolist());pos=np.asarray([int(p) for p in dg['high'] if int(di[int(p)]) not in selected_ids]);confirm=subset(dev,pos);ref,m=build(a.arm);ref,m=ref.cuda().eval(),m.cuda().eval();h,mask=selection['history'][:2],selection['mask'][:2]
    with torch.no_grad():exact=float((ref(h,mask)['future_position']-m(h,mask)['future_position']).abs().max());assert exact<2e-6
    configure(m);m.train();o=m(h,mask);loss=o['future_position'].sum();loss.backward();grad=float(m.trust_head[-1].weight.grad.abs().max());assert grad>0;utils.atomic_json(out/'smoke.json',{'exact':exact,'trust_gradient':grad})
    if a.smoke_only:utils.atomic_json(out/'completed.json',{'status':'smoke_completed'});return
    utils.atomic_json(out/'protocol.json',{'seed':SEED,'arm':a.arm,'interface':'QGNN attention routes LLM future states; joint trust gate scales the retained LLM correction from 0 to 2 per target and time','epochs':a.epochs,'selection':'fixed 256 scenes ADE+.35FDE','confirmation_scenes':int(len(pos)),'confirmation_disjoint_from_selection':True,'no_test':True})
    source,_=retained.evaluate(ref,confirm);z=train(m,bank,g,selection,out,a.epochs);learned,_=retained.evaluate(m,confirm);um=utility_metrics(m,confirm);m.trust_attention_mode='uniform';uniform,_=retained.evaluate(m,confirm);m.trust_attention_mode='off';off,_=retained.evaluate(m,confirm);m.trust_attention_mode='learned';full,_=retained.evaluate(m,dev)
    summary={'arm':a.arm,'selected_epoch':int(z['epoch']),'selection':z['metrics'],'source_confirmation':source,'confirmation_learned':learned,'confirmation_uniform':uniform,'confirmation_off':off,'utility_metrics':um,'full_dev_secondary':full,'no_test':True};utils.atomic_json(out/'summary.json',summary);utils.atomic_json(out/'completed.json',{'status':'completed','summary':summary});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
