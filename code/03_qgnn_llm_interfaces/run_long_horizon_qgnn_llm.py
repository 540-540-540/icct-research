"""Matched 4-second QGNN then guarded future-token LLM experiment."""
from __future__ import annotations
import argparse,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2'
HERE=ROOT/'双图研究框架/未来词QGNN_LLM实验_2026-09-09'
sys.path[:0]=[str(HERE),str(BASE),str(ROOT)]
import experiment_utils as utils
import run as retained
from future_token_model import FutureTokenCoreGraphLLM,future_compact_state
from physics_aligned_dual_model_effective import PhysicsAlignedMessageLayer,build_physics_aligned_dual_graph
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss
from target_interaction_graph import ForecasterConfig,TargetInteractionGNN

SEED=2026;HISTORY=20;HORIZON=40
CACHE=ROOT/'data/multitarget_lankershim_h20_p40_matched_v1.npz'
SPLIT=BASE/'retained_inputs/long40_circuit_split_indices.npz'
GRAPH20=BASE/'physics_aligned_dual_seed2026_v1/physics_quantum_dual_graph_selected.pt'
LLM20=BASE/'future_token_qgnn_llm_seed2026_v1/future_token_qgnn_llm_selected.pt'


def build_long_graph():
    torch.manual_seed(SEED);config=ForecasterConfig(history_length=HISTORY,prediction_length=HORIZON)
    model=TargetInteractionGNN(config)
    torch.manual_seed(SEED+3);model.graph_layers[0]=PhysicsAlignedMessageLayer(config,'quantum','physical',SEED+103)
    torch.manual_seed(SEED+1);model.graph_layers[1]=PhysicsAlignedMessageLayer(config,'quantum','contextual',SEED+101)
    return model


def initialize_graph_from_20(model):
    source=torch.load(GRAPH20,map_location='cpu',weights_only=True)['state'];target=model.state_dict()
    loaded=[]
    for name,value in source.items():
        if name in target and target[name].shape==value.shape:target[name]=value;loaded.append(name)
    target['time_embedding.weight'][:20]=source['time_embedding.weight']
    model.load_state_dict(target);return loaded


class GuardedLongFutureLLM(FutureTokenCoreGraphLLM):
    def __init__(self,graph):
        super().__init__(graph);self.long_stage_gain=nn.Parameter(torch.zeros(4))
    def forward(self,history,target_mask):
        output=super().forward(history,target_mask);candidate=output['future_position'];base=output['graph_future_position']
        pieces=[]
        for stage,part in enumerate(torch.tensor_split(candidate-base,4,dim=1)):
            pieces.append(torch.tanh(self.long_stage_gain[stage])*part)
        correction=torch.cat(pieces,dim=1);future=base+correction
        output['llm_candidate_position']=candidate;output['future_position']=future
        output['displacement']=future-history[:,-1,None,:,:2];return output


def initialize_llm_from_20(model):
    source=torch.load(LLM20,map_location='cpu',weights_only=True)['state'];target=model.state_dict();loaded=[]
    for name,value in source.items():
        if name.startswith('graph_backbone.'):continue
        if name in target and target[name].shape==value.shape:target[name]=value;loaded.append(name)
    for name in ('future_queries',):
        value=source[name].T[None]
        target[name]=F.interpolate(value,size=HORIZON,mode='linear',align_corners=True)[0].T
    for prefix in ('coordinate_head.4',):
        weight=source[prefix+'.weight'].view(20,2,-1).permute(2,1,0)
        weight=F.interpolate(weight,size=HORIZON,mode='linear',align_corners=True).permute(2,1,0).reshape(HORIZON*2,-1)
        bias=source[prefix+'.bias'].view(20,2).T[None]
        bias=F.interpolate(bias,size=HORIZON,mode='linear',align_corners=True)[0].T.reshape(-1)
        target[prefix+'.weight']=weight;target[prefix+'.bias']=bias
    model.load_state_dict(target);return loaded


def compact_restore(model,state):
    missing,extra=model.load_state_dict(state,strict=False);assert not extra,extra
    assert all((name.startswith('gpt2.') or name.startswith('future_gpt2.')) and 'lora_' not in name for name in missing),missing


def load_banks():
    with np.load(SPLIT,allow_pickle=False) as split:train_index=split['circuit_train_indices'].copy();dev_index=split['circuit_dev_indices'].copy()
    with np.load(CACHE,allow_pickle=False) as data:states=data['train_states'].copy();masks=data['train_mask'].copy()
    selection_index=utils.select_profile_indices(dev_index,256)
    def bank(index):return {'history':torch.from_numpy(states[index,:HISTORY]).float().cuda(),
                            'future':torch.from_numpy(states[index,HISTORY:]).float().cuda(),
                            'mask':torch.from_numpy(masks[index]).bool().cuda()}
    return bank(train_index),bank(selection_index),bank(dev_index),train_index,selection_index,dev_index


@torch.no_grad()
def evaluate(model,bank,batch_size=24,collect=False):
    model.eval();result={};arrays={}
    for snr in (5,10,15,20):
        generator=torch.Generator(device='cuda').manual_seed(SEED+100000);rows=[]
        for start in range(0,len(bank['history']),batch_size):
            index=torch.arange(start,min(start+batch_size,len(bank['history'])),device='cuda')
            history,future,mask=retained.make_batch(bank,index,snr,generator);output=model(history,mask)
            distance=(output['future_position']-future[...,:2]).norm(dim=-1)
            rows.append(torch.stack([(distance.double()*mask[:,None,:]).sum((1,2)),
                                     (distance[:,-1].double()*mask).sum(1),mask.sum(1)],1).cpu())
        values=torch.cat(rows).numpy();denominator=values[:,2].sum()
        result[str(snr)]={'ade_m':float(values[:,0].sum()/(denominator*HORIZON)),
                          'fde_m':float(values[:,1].sum()/denominator)}
        if collect:arrays[str(snr)]=values
    aggregate={key:float(np.mean([x[key] for x in result.values()])) for key in ('ade_m','fde_m')}
    return {'aggregate':aggregate,'by_snr':result},arrays


def score(metrics):return metrics['aggregate']['ade_m']+0.50*metrics['aggregate']['fde_m']


def train_graph(model,train,selection,out,epochs):
    for parameter in model.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=2e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5)
    initial,_=evaluate(model,selection);best=score(initial);best_epoch=0;checkpoint=out/'long40_qgnn_selected.pt'
    torch.save({'state':{n:t.detach().cpu() for n,t in model.state_dict().items()},'epoch':0,'metrics':initial},checkpoint)
    with (out/'graph_training.jsonl').open('x') as log:
      for epoch in range(1,epochs+1):
        started=time.time();utils.set_seed(SEED+510000+epoch);model.train();generator=torch.Generator(device='cuda').manual_seed(SEED+510000+epoch)
        order=torch.randperm(len(train['history']),generator=torch.Generator().manual_seed(SEED+510000+epoch)).cuda();total=0.;count=0
        for offset in range(0,len(order),24):
            index=order[offset:offset+24];snr=(5,10,15,20)[(offset//24+epoch)%4];history,future,mask=retained.make_batch(train,index,snr,generator)
            optimizer.zero_grad(set_to_none=True);output=model(history,mask);loss=prediction_loss(output,future,mask,True)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(index);count+=len(index)
        scheduler.step();metrics,_=evaluate(model,selection);current=score(metrics)
        if current<best:best=current;best_epoch=epoch;torch.save({'state':{n:t.detach().cpu() for n,t in model.state_dict().items()},'epoch':epoch,'metrics':metrics},checkpoint)
        record={'epoch':epoch,'loss':total/count,'metrics':metrics,'score':current,'best_epoch':best_epoch,'seconds':time.time()-started}
        log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True);utils.atomic_json(out/'progress.json',{'phase':'graph','last':record})
    selected=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(selected['state']);return selected


def configure_llm(model,joint):
    for parameter in model.parameters():parameter.requires_grad_(False)
    for name,parameter in model.named_parameters():
        if name.startswith(('future_queries','future_type_embedding','token_head.','token_residual_head.','adaptive_gate.')):parameter.requires_grad_(True)
        if name.startswith('future_gpt2.') and 'lora_' in name:parameter.requires_grad_(True)
    model.fusion_strength.requires_grad_(True);model.long_stage_gain.requires_grad_(joint)


def candidate_loss(output,future,mask):
    distance=(output['llm_candidate_position']-future[...,:2]).norm(dim=-1)
    weights=torch.linspace(.5,1.7,HORIZON,device=distance.device)[None,:,None]
    return (distance*weights*mask[:,None,:]).sum()/(weights*mask[:,None,:]).sum().clamp_min(1)


def train_llm(model,train,selection,out,epochs,warmup):
    configure_llm(model,False);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1.5e-4,weight_decay=2e-4)
    initial,_=evaluate(model,selection);best=score(initial);best_epoch=0;checkpoint=out/'long40_qgnn_llm_selected.pt'
    torch.save({'state':future_compact_state(model),'epoch':0,'metrics':initial},checkpoint)
    with (out/'llm_training.jsonl').open('x') as log:
      for epoch in range(1,epochs+1):
        if epoch==warmup+1:
            configure_llm(model,True);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=8e-5,weight_decay=2e-4)
        started=time.time();utils.set_seed(SEED+520000+epoch);model.train();generator=torch.Generator(device='cuda').manual_seed(SEED+520000+epoch)
        order=torch.randperm(len(train['history']),generator=torch.Generator().manual_seed(SEED+520000+epoch)).cuda();totals={'coordinate':0.,'candidate':0.,'token':0.};count=0
        for offset in range(0,len(order),18):
            index=order[offset:offset+18];snr=(5,10,15,20)[(offset//18+epoch)%4];history,future,mask=retained.make_batch(train,index,snr,generator)
            optimizer.zero_grad(set_to_none=True);output=model(history,mask);candidate=candidate_loss(output,future,mask)
            tokens=token_loss(output['token_logits'],model.future_token_ids(history,future),mask);coordinate=prediction_loss(output,future,mask,True)
            loss=.25*candidate+.04*tokens if epoch<=warmup else coordinate+.30*candidate+.04*tokens
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.,error_if_nonfinite=True);optimizer.step();size=len(index);count+=size
            for key,value in [('coordinate',coordinate),('candidate',candidate),('token',tokens)]:totals[key]+=float(value.detach())*size
        metrics,_=evaluate(model,selection);current=score(metrics)
        if epoch>warmup and current<best:best=current;best_epoch=epoch;torch.save({'state':future_compact_state(model),'epoch':epoch,'metrics':metrics},checkpoint)
        record={'epoch':epoch,'phase':'warmup' if epoch<=warmup else 'joint',**{k:v/count for k,v in totals.items()},
                'stage_gain':torch.tanh(model.long_stage_gain).detach().cpu().tolist(),'metrics':metrics,'score':current,'best_epoch':best_epoch,'seconds':time.time()-started}
        log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True);utils.atomic_json(out/'progress.json',{'phase':'llm','last':record})
    selected=torch.load(checkpoint,map_location='cpu',weights_only=True);compact_restore(model,selected['state']);return selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='long40_qgnn_llm_seed2026_v1');parser.add_argument('--graph-epochs',type=int,default=8)
    parser.add_argument('--llm-epochs',type=int,default=8);parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);utils.set_seed(SEED);retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    out=BASE/args.output_name;out.mkdir(exist_ok=False);train,selection,dev,train_index,selection_index,dev_index=load_banks()
    old=build_physics_aligned_dual_graph('physics_quantum_dual');old.load_state_dict(torch.load(GRAPH20,map_location='cpu',weights_only=True)['state']);old=old.cuda().eval()
    graph=build_long_graph();loaded=initialize_graph_from_20(graph);graph=graph.cuda().eval()
    with torch.no_grad():difference=float((old(selection['history'][:2],selection['mask'][:2])['future_position']-graph(selection['history'][:2],selection['mask'][:2])['future_position'][:,:20]).abs().max())
    assert difference<2e-6,difference
    utils.atomic_json(out/'smoke.json',{'first20_exact_max_abs_m':difference,'compatible_graph_tensors':len(loaded),'gpu':torch.cuda.get_device_name(0)})
    del old;torch.cuda.empty_cache()
    if args.smoke_only:utils.atomic_json(out/'completed.json',{'status':'smoke_completed'});return
    utils.atomic_json(out/'protocol.json',{'seed':SEED,'history_steps':HISTORY,'prediction_steps':HORIZON,'dt_s':.1,'horizon_s':4.,
       'matched_cache':str(CACHE),'train_scenes':len(train_index),'dev_scenes':len(dev_index),'selection_scenes':len(selection_index),
       'graph_epochs':args.graph_epochs,'llm_epochs':args.llm_epochs,'llm_warmup':args.warmup,'selection':'fixed 256 ADE+0.50FDE','no_test':True})
    graph_selected=train_graph(graph,train,selection,out,args.graph_epochs);graph_full,graph_arrays=evaluate(graph,dev,collect=True)
    model=GuardedLongFutureLLM(graph);llm_loaded=initialize_llm_from_20(model);model=model.cuda()
    with torch.no_grad():guard=float((model(selection['history'][:2],selection['mask'][:2])['future_position']-graph(selection['history'][:2],selection['mask'][:2])['future_position']).abs().max())
    assert guard==0.,guard
    llm_selected=train_llm(model,train,selection,out,args.llm_epochs,args.warmup);llm_full,llm_arrays=evaluate(model,dev,collect=True)
    np.savez_compressed(out/'long40_graph_full_dev.npz',**graph_arrays,indices=dev_index);np.savez_compressed(out/'long40_llm_full_dev.npz',**llm_arrays,indices=dev_index)
    summary={'graph_selected_epoch':int(graph_selected['epoch']),'llm_selected_epoch':int(llm_selected['epoch']),
             'graph_full_dev':graph_full,'llm_full_dev':llm_full,'epoch0_llm_exact_to_graph_m':guard,'compatible_llm_tensors':len(llm_loaded),'no_test':True}
    utils.atomic_json(out/'summary.json',summary);utils.atomic_json(out/'completed.json',{'status':'completed','summary':summary});utils.atomic_json(out/'progress.json',{'status':'completed','summary':summary});print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
