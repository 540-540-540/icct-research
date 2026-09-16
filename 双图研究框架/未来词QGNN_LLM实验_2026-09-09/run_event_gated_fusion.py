"""Train and validate the interaction-event-gated QGNN+LLM interface."""
from __future__ import annotations

import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/js_cn/sensing'); BASE=ROOT/'diagnostics/core_message_seed2026_v2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as utils
import run as retained
from analyze_interaction_strata import scene_features
from event_gated_fusion_model import EventGatedFutureLLM
from future_token_model import FutureTokenCoreGraphLLM,future_compact_state,restore_future_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from run_future_token_qgnn_llm import load_banks
from run_multitarget_experiment import prediction_loss

SEED=2026; GRAPH_DIR=BASE/'physics_aligned_dual_seed2026_v1'

def restore_event(model,state):
    missing,extra=model.load_state_dict(state,strict=False);assert not extra,extra
    for name in missing:
        frozen=((name.startswith('gpt2.') or name.startswith('future_gpt2.')) and 'lora_' not in name)
        assert frozen or name.startswith('event_'),name

def build_pair(arm):
    graph_name=f'physics_{arm}_dual'; source_dir=BASE/f"future_token_{'qgnn' if arm=='quantum' else 'classical'}_llm_seed2026_v1"
    source=torch.load(source_dir/'future_token_qgnn_llm_selected.pt',map_location='cpu',weights_only=True)
    gs=torch.load(GRAPH_DIR/f'{graph_name}_graph_selected.pt',map_location='cpu',weights_only=True)['state']
    g=build_physics_aligned_dual_graph(graph_name);g.load_state_dict(gs);reference=FutureTokenCoreGraphLLM(g);restore_future_compact(reference,source['state'])
    g=build_physics_aligned_dual_graph(graph_name);g.load_state_dict(gs);model=EventGatedFutureLLM(g);restore_event(model,source['state'])
    return reference,model,source,source_dir

def configure(model):
    for p in model.parameters():p.requires_grad_(False)
    for n,p in model.named_parameters():
        if n.startswith('event_'):p.requires_grad_(True)

def groups(indices):
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:f=scene_features(d['train_states'][indices,:20],d['train_mask'][indices])
    c=np.quantile(f['composite'],[1/3,2/3]);s=f['composite']
    return {'low':np.flatnonzero(s<=c[0]),'medium':np.flatnonzero((s>c[0])&(s<=c[1])),'high':np.flatnonzero(s>c[1])}

def order_for(group,total,epoch):
    gen=torch.Generator().manual_seed(SEED+130000+epoch); counts={'high':total//2,'medium':total//4};counts['low']=total-counts['high']-counts['medium'];parts=[]
    for name in ('high','medium','low'):
        src=torch.as_tensor(group[name]);parts.append(src[torch.randint(len(src),(counts[name],),generator=gen)])
    x=torch.cat(parts);return x[torch.randperm(len(x),generator=gen)].cuda()

def event_loss(logits,labels,mask):
    valid=mask[:,None,:].expand_as(labels); y=labels[valid].float(); z=logits[valid]
    positives=y.sum(); negatives=len(y)-positives; pos_weight=(negatives/positives.clamp_min(1)).clamp(1,10)
    return F.binary_cross_entropy_with_logits(z,y,pos_weight=pos_weight),float(y.mean())

@torch.no_grad()
def event_metrics(model,bank):
    model.eval();tp=fp=fn=0.;total_positive=0.;total=0.
    for start in range(0,len(bank['history']),24):
        idx=torch.arange(start,min(start+24,len(bank['history'])),device='cuda');h=bank['history'].index_select(0,idx);f=bank['future'].index_select(0,idx);m=bank['mask'].index_select(0,idx)
        o=model(h,m);labels=model.interaction_labels(h,f,m);pred=o['interaction_probability']>=.5;valid=m[:,None,:].expand_as(labels)
        tp+=float((pred&labels&valid).sum());fp+=float((pred&~labels&valid).sum());fn+=float((~pred&labels&valid).sum());total_positive+=float((labels&valid).sum());total+=float(valid.sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);return {'f1':2*precision*recall/max(precision+recall,1e-12),'precision':precision,'recall':recall,'positive_rate':total_positive/max(total,1)}

def train(model,train,group,selection,out,epochs):
    configure(model);params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    initial,_=retained.evaluate(model,selection);best=initial['aggregate']['ade_m']+.35*initial['aggregate']['fde_m'];best_ep=0;ck=out/'event_gated_selected.pt';torch.save({'state':future_compact_state(model),'epoch':0,'metrics':initial},ck)
    with (out/'training.jsonl').open('x') as log:
      for ep in range(1,epochs+1):
        st=time.time();utils.set_seed(SEED+130000+ep);model.train();gen=torch.Generator(device='cuda').manual_seed(SEED+130000+ep);order=order_for(group,len(train['history']),ep);cs=es=rs=pr=0.;count=0
        for off in range(0,len(order),24):
            ix=order[off:off+24];snr=(5,10,15,20)[(off//24+ep)%4];h,f,m=retained.make_batch(train,ix,snr,gen);opt.zero_grad(set_to_none=True);o=model(h,m);labels=model.interaction_labels(h,f,m);coord=prediction_loss(o,f,m,True);ev,pos=event_loss(o['interaction_logits'],labels,m);loss=coord+.05*ev;loss.backward();torch.nn.utils.clip_grad_norm_(params,3.,error_if_nonfinite=True);opt.step();n=len(ix);count+=n;cs+=float(coord.detach())*n;es+=float(ev.detach())*n;rs+=float(model.last_event_residual_mean)*n;pr+=pos*n
        sched.step();metrics,_=retained.evaluate(model,selection);score=metrics['aggregate']['ade_m']+.35*metrics['aggregate']['fde_m']
        if score<best:best=score;best_ep=ep;torch.save({'state':future_compact_state(model),'epoch':ep,'metrics':metrics},ck)
        rec={'epoch':ep,'coordinate_loss':cs/count,'event_bce':es/count,'event_positive_rate':pr/count,'residual_mean_m':rs/count,'metrics':metrics,'score':score,'best_epoch':best_ep,'seconds':time.time()-st};log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
    z=torch.load(ck,map_location='cpu',weights_only=True);restore_event(model,z['state']);return z

def subset(bank,pos):
    ix=torch.as_tensor(pos,device='cuda');return {k:v.index_select(0,ix) for k,v in bank.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('quantum','classical'),required=True);ap.add_argument('--output-name',required=True);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args()
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available();torch.set_num_threads(4);utils.set_seed(SEED);out=BASE/a.output_name;out.mkdir(exist_ok=False);retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    train_bank,selection,dev,ti,si,di=load_banks();group=groups(ti);devgroup=groups(di);high=devgroup['high']
    selection_scene_ids=set(int(x) for x in si.tolist())
    confirm_pos=np.asarray([int(pos) for pos in high if int(di[int(pos)]) not in selection_scene_ids],dtype=np.int64)
    if confirm_pos.size==0:raise RuntimeError('No disjoint high-complexity confirmation scenes remain')
    confirmation=subset(dev,confirm_pos)
    reference,model,source,source_dir=build_pair(a.arm);reference,model=reference.cuda().eval(),model.cuda().eval();h,f,m=selection['history'][:2],selection['future'][:2],selection['mask'][:2]
    with torch.no_grad():exact=float((reference(h,m)['future_position']-model(h,m)['future_position']).abs().max());assert exact<2e-6
    configure(model);model.train();o=model(h,m);labels=model.interaction_labels(h,f,m);loss,_=event_loss(o['interaction_logits'],labels,m);loss.backward();grad=float(model.event_classifier[-1].weight.grad.abs().max());assert grad>0
    utils.atomic_json(out/'smoke.json',{'exact':exact,'event_gradient':grad})
    if a.smoke_only:utils.atomic_json(out/'completed.json',{'status':'smoke_completed'});return
    utils.atomic_json(out/'protocol.json',{'seed':SEED,'arm':a.arm,'epochs':a.epochs,'event':'per-target per-future-step distance<=12m and closing','train_sampling':{'high':.5,'medium':.25,'low':.25},'selection':'fixed 256-scene set; ADE+.35FDE','confirmation':'all outcome-blind high-complexity dev scenes disjoint from the fixed selection bank','confirmation_scenes':int(confirm_pos.size),'no_test':True})
    source_confirm,_=retained.evaluate(reference,confirmation);selected=train(model,train_bank,group,selection,out,a.epochs);confirm,_=retained.evaluate(model,confirmation);full,_=retained.evaluate(model,dev);events=event_metrics(model,confirmation)
    model.attention_ablation_mode='uniform';uniform,_=retained.evaluate(model,confirmation);model.attention_ablation_mode='off';off,_=retained.evaluate(model,confirmation)
    summary={'arm':a.arm,'selected_epoch':int(selected['epoch']),'selection':selected['metrics'],'source_confirmation':source_confirm,'confirmation':confirm,'confirmation_uniform':uniform,'confirmation_off':off,'confirmation_event_metrics':events,'full_dev_secondary':full,'no_test':True};utils.atomic_json(out/'summary.json',summary);utils.atomic_json(out/'completed.json',{'status':'completed','summary':summary});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
