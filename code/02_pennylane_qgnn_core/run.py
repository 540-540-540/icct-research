import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,sys,json,time,math,platform,hashlib,random
from pathlib import Path
import numpy as np
import torch
from model import build_graph,CoreGraphLLM,compact_state,restore_compact
ROOT=Path('/home/js_cn/sensing')
import experiment_utils as r
from run_multitarget_experiment import prediction_loss
from run_multitarget_graph_llm import token_loss
from evaluate_association_downstream_forecasting import reconstruct_histories,forecast_metrics,load_checkpoint
from measurement_track_association import AssociationGraphConfig,MeasurementTrackAssociationNet
from measurement_track_dataset import AssociationSimulationConfig,load_snr_noise
BASE=Path(__file__).resolve().parent
ARMS=('plain','classical','quantum'); SEED=2026

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(x): print(json.dumps(x,ensure_ascii=False),flush=True)
def js(p,x): r.atomic_json(p,x)
def graph_state(m): return {n:t.detach().cpu().clone() for n,t in m.state_dict().items()}
def digest_shared(m):
    return r.tensor_mapping_sha256({n:p.detach().cpu() for n,p in m.named_parameters() if not n.startswith('graph_layers.1.')})
def make_batch(bank,idx,snr,gen):
    h,f,m=[bank[k].index_select(0,idx) for k in ('history','future','mask')]
    v=NOISE[snr]
    obs=r.add_noise(h,m,v['position_sigma_m'],v['velocity_sigma_mps'],gen)
    return obs,f,m

@torch.no_grad()
def evaluate(model,bank,batch_size=24,collect=False):
    model.eval(); result={}; arrays={}
    for snr in (5,10,15,20):
        gen=torch.Generator(device='cuda').manual_seed(SEED+100000)
        acc=[]
        for start in range(0,len(bank['history']),batch_size):
            idx=torch.arange(start,min(start+batch_size,len(bank['history'])),device='cuda')
            h,f,m=make_batch(bank,idx,snr,gen)
            o=model(h,m); d=(o['future_position']-f[...,:2]).norm(dim=-1)
            assert torch.isfinite(d).all()
            # Float64 scene sums and counts preserve target-weighted aggregation.
            sum_t=(d.double()*m[:,None,:]).sum((1,2))
            sum_f=(d[:,-1].double()*m).sum(1)
            count=m.sum(1)
            acc.append(torch.stack([sum_t,sum_f,count],1).cpu())
        a=torch.cat(acc).numpy(); den=a[:,2].sum()
        result[str(snr)]=dict(ade_m=float(a[:,0].sum()/(den*20)),fde_m=float(a[:,1].sum()/den))
        if collect: arrays[str(snr)]=a
    agg={k:float(np.mean([x[k] for x in result.values()])) for k in ('ade_m','fde_m')}
    return dict(aggregate=agg,by_snr=result),arrays

def score(ev): return ev['aggregate']['ade_m']+.35*ev['aggregate']['fde_m']

def train_stage(model,arm,phase,epochs,train,dev,out,expected_inputs):
    is_llm=phase=='llm'; named=dict(model.named_parameters())
    if is_llm:
        groups=[dict(params=[p for n,p in named.items() if p.requires_grad and n.startswith('graph_backbone.')],lr=1e-4),
                dict(params=[p for n,p in named.items() if p.requires_grad and not n.startswith('graph_backbone.') and 'lora_' not in n],lr=3e-4),
                dict(params=[p for n,p in named.items() if p.requires_grad and 'lora_' in n],lr=7.5e-5)]
    else: groups=[dict(params=list(model.parameters()),lr=1e-3)]
    opt=torch.optim.AdamW(groups,weight_decay=2e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    ev,_=evaluate(model,dev); best=score(ev); best_ep=0
    state=compact_state if is_llm else graph_state
    ck=out/(arm+'_'+phase+'_selected.pt')
    torch.save(dict(state=state(model),epoch=0,metrics=ev),ck)
    start_state={n:p.detach().cpu().clone() for n,p in named.items() if p.requires_grad}
    gradmax={n:0. for n,p in named.items() if p.requires_grad}
    records=[dict(epoch=0,metrics=ev)]
    phase_offset=20000 if is_llm else 10000
    with (out/(arm+'_'+phase+'.jsonl')).open('x') as log:
        for ep in range(1,epochs+1):
            started=time.time(); r.set_seed(SEED+phase_offset+ep); model.train()
            gen=torch.Generator(device='cuda').manual_seed(SEED+phase_offset+ep)
            order=torch.randperm(len(train['history']),generator=torch.Generator().manual_seed(SEED+phase_offset+ep)).to('cuda')
            loss_sum=0.; count=0; dig=r.InputDigest()
            for k in range(0,len(order),24):
                idx=order[k:k+24]; snr=(5,10,15,20)[(k//24+ep)%4]
                h,f,m=make_batch(train,idx,snr,gen);dig.update(h)
                opt.zero_grad(set_to_none=True)
                pred=model(h,m)
                loss=prediction_loss(pred,f,m,graph_weighting=True)
                if is_llm: loss=loss+.035*token_loss(pred['token_logits'],model.future_token_ids(h,f),m)
                assert torch.isfinite(loss), (arm,phase,ep,k)
                loss.backward()
                for n,p in named.items():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(),n
                        if n in gradmax: gradmax[n]=max(gradmax[n],float(p.grad.abs().max()))
                torch.nn.utils.clip_grad_norm_([p for p in named.values() if p.requires_grad],3.,error_if_nonfinite=True)
                opt.step(); loss_sum+=float(loss.detach())*len(idx);count+=len(idx)
            assert count==len(order)
            key=phase+'_'+str(ep);d=dig.hexdigest()
            if key in expected_inputs: assert expected_inputs[key]==d
            else: expected_inputs[key]=d
            schedule.step(); ev,_=evaluate(model,dev)
            if score(ev)<best:
                best=score(ev);best_ep=ep
                torch.save(dict(state=state(model),epoch=ep,metrics=ev),ck)
            rec=dict(arm=arm,phase=phase,epoch=ep,train_loss=loss_sum/count,metrics=ev,best_epoch=best_ep,seconds=time.time()-started,input_sha256=d)
            records.append(rec);log.write(json.dumps(rec)+'\n');log.flush();emit(rec)
            js(out/'progress.json',dict(status='training',arm=arm,phase=phase,epoch=ep,epochs=epochs,last=rec))
    changes={n:float((p.detach().cpu()-start_state[n]).abs().max()) for n,p in named.items() if n in start_state}
    selected=torch.load(ck,map_location='cpu',weights_only=True)
    if is_llm:restore_compact(model,selected['state'])
    else:model.load_state_dict(selected['state'],strict=True)
    recheck,_=evaluate(model,dev)
    assert all(abs(recheck['aggregate'][k]-selected['metrics']['aggregate'][k])<1e-7 for k in ('ade_m','fde_m'))
    summary=dict(selected_epoch=best_ep,metrics=recheck,gradient_max=gradmax,parameter_max_change=changes,
                 never_received_gradient=[n for n,v in gradmax.items() if v==0],records=records,checkpoint_sha256=sha(ck))
    js(out/(arm+'_'+phase+'_audit.json'),summary)
    return summary

def smoke(out):
    evidence={}; shared=None; matched=None
    h=TRAIN['history'][:2];f=TRAIN['future'][:2];m=TRAIN['mask'][:2]
    for arm in ARMS:
        g=build_graph(arm).cuda().eval()
        d=digest_shared(g)
        if shared is None: shared=d
        assert d==shared
        if arm!='plain':
            common=r.tensor_mapping_sha256({n:p.cpu() for n,p in g.graph_layers[1].named_parameters() if not n.startswith('core.')})
            if matched is None:matched=common
            assert common==matched
        o=g(h,m);loss=prediction_loss(o,f,m,True);loss.backward()
        assert torch.isfinite(loss)
        if arm=='quantum':
            assert g.graph_layers[1].core.weights.grad.abs().max()>0
            # Actual historical input sensitivity and permutation equivariance.
            perm=torch.arange(h.shape[2]-1,-1,-1,device='cuda')
            with torch.no_grad():
                op=g(h[:,:,perm],m[:,perm])['future_position']
            assert torch.allclose(op,o['future_position'][:,:,perm],atol=2e-5,rtol=2e-5)
            from target_interaction_graph import build_edge_features
            edges,dist=build_edge_features(h[:,-1,:,:2],h[:,-1,:,2:]);adj=m[:,:,None]&m[:,None,:]&(dist<=45)
            x=torch.randn(2,h.shape[2],128,device='cuda',requires_grad=True)
            y,_=g.graph_layers[1](x,edges,adj)
            assert torch.autograd.grad(y.square().sum(),x)[0].abs().max()>0
        r.set_seed(SEED);llm=CoreGraphLLM(g).cuda().eval()
        lo=llm(h,m);assert torch.equal(lo['future_position'],lo['graph_future_position'])
        lloss=prediction_loss(lo,f,m,True)+.035*token_loss(lo['token_logits'],llm.future_token_ids(h,f),m)
        llm.zero_grad(set_to_none=True);lloss.backward()
        if arm=='quantum':assert llm.graph_backbone.graph_layers[1].core.weights.grad.abs().max()>0
        cs=compact_state(llm)
        # Same-class restoration checks exact forward reconstruction.
        with torch.no_grad(): before=llm(h,m)['future_position'].clone()
        restore_compact(llm,cs)
        with torch.no_grad(): assert torch.equal(before,llm(h,m)['future_position'])
        evidence[arm]=dict(graph_parameters=sum(p.numel() for p in g.parameters()),
            graph_shared_initial_hash=d,llm_trainable=sum(p.numel() for p in llm.parameters() if p.requires_grad),
            core_parameters=sum(p.numel() for p in g.graph_layers[1].core.parameters()) if arm!='plain' else None,
            gradient_finite=True,llm_path_connected=True,restore_exact=True)
        del g,llm;torch.cuda.empty_cache()
    js(out/'smoke.json',evidence);emit(dict(smoke='passed',evidence=evidence))

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--smoke',action='store_true');args=args.parse_args()
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);r.set_seed(SEED)
    out=BASE/('smoke_v2' if args.smoke else 'development');out.mkdir(exist_ok=False)
    cache=ROOT/'data/multitarget_lankershim_v1.npz'
    with np.load(BASE/'retained_inputs/circuit_split_indices.npz',allow_pickle=False) as split:
        train_idx=split['circuit_train_indices'].copy();dev_idx=split['circuit_dev_indices'].copy()
    with np.load(cache,allow_pickle=False) as z:
        states=z['train_states'].copy();masks=z['train_mask'].copy()
    select_idx=r.select_profile_indices(dev_idx,256)
    def bank(idx):
        return dict(history=torch.from_numpy(states[idx,:20]).float().cuda(),future=torch.from_numpy(states[idx,20:]).float().cuda(),mask=torch.from_numpy(masks[idx]).bool().cuda())
    TRAIN=bank(train_idx);DEV=bank(select_idx);FULL=bank(dev_idx)
    calibration=ROOT/'results/multitarget_snr/snr_calibration.json'
    NOISE=r.load_snr_noise_map(calibration)
    sources=[BASE/'model.py',BASE/'run.py',ROOT/'target_interaction_graph.py',ROOT/'MultiTargetTimeLLM.py',
        BASE/'quantum_core.py',ROOT/'run_multitarget_experiment.py',ROOT/'run_multitarget_graph_llm.py',
        ROOT/'evaluate_association_downstream_forecasting.py',ROOT/'measurement_track_dataset.py',ROOT/'measurement_track_association.py',
        ROOT/'run_measurement_track_experiment.py',calibration,ROOT/'results/measurement_track_association/association_gnn.pt']
    hashes={str(p):sha(p) for p in sources}
    if args.smoke:
        # Eval-mode gradient probes need the native GRU backward, not cuDNN's training-only backward.
        with torch.backends.cudnn.flags(enabled=False): smoke(out)
        sys.exit(0)
    assert (BASE/'smoke_v2/smoke.json').exists()
    js(out/'protocol.json',dict(seed=SEED,arms=ARMS,graph_epochs=12,llm_epochs=6,batch=24,precision='FP32',
        train_indices=train_idx.tolist(),selection_indices=select_idx.tolist(),full_dev_indices=dev_idx.tolist(),
        selection='four-SNR macro ADE + 0.35 FDE; checkpoints 0..budget; no early stopping',
        initialization='graph from scratch; identical common tensors; GPT2 same pretrained checkpoint, fresh equal LoRA/heads; all graph params trained',
        message='replace second layer; receiver/sender historical nodes + 7 physical edge features -> 6 angles -> quantum/MLP -> 12 -> sole 4-head attention scores and message gates; sender value and edge value preserve message content',
        parameter_matching='same encoder, score/gate readout, value and edge-value paths; classical MLP core 84 params, quantum 54; not exact runtime/parameter parity',
        association='same existing classic association GNN; fixed 64 circuit-dev scenes, 4 SNR, 10% misses and 1.5 clutter; no test',
        limitations='development only, single seed, bounded budget, known identity training histories, fixed target membership and noisy true initialization in association eval',
        gpu=torch.cuda.get_device_name(0),python=sys.executable,hashes=hashes))
    expected_inputs={};summaries={};llm_init=None
    for arm in ARMS:
        g=build_graph(arm).cuda()
        graph_result=train_stage(g,arm,'graph',12,TRAIN,DEV,out,expected_inputs)
        graph_full,arr=evaluate(g,FULL,collect=True);np.savez_compressed(out/(arm+'_graph_full_dev.npz'),**arr,indices=dev_idx)
        r.set_seed(SEED);model=CoreGraphLLM(g).cuda()
        init=r.tensor_mapping_sha256({n:p.detach().cpu() for n,p in model.named_parameters() if not n.startswith('graph_backbone.')})
        if llm_init is None:llm_init=init
        assert llm_init==init
        frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        frozen_hash=r.tensor_mapping_sha256(frozen);del frozen
        llm_result=train_stage(model,arm,'llm',6,TRAIN,DEV,out,expected_inputs)
        assert frozen_hash==r.tensor_mapping_sha256({n:p.detach().cpu() for n,p in model.named_parameters() if not p.requires_grad})
        llm_full,arr=evaluate(model,FULL,collect=True);np.savez_compressed(out/(arm+'_llm_full_dev.npz'),**arr,indices=dev_idx)
        summaries[arm]=dict(graph=graph_full,llm=llm_full,graph_selected_epoch=graph_result['selected_epoch'],llm_selected_epoch=llm_result['selected_epoch'],
            llm_non_graph_initial_hash=init,frozen_gpt_verified=True)
        js(out/'summary.json',summaries)
        del model,g;torch.cuda.empty_cache()
    # Evaluate the identical reconstructed unknown-identity measurement history for all predictors.
    assoc=load_checkpoint(MeasurementTrackAssociationNet(AssociationGraphConfig(hidden_dim=96,graph_layers=2)),ROOT/'results/measurement_track_association/association_gnn.pt').cuda().eval()
    ai=r.select_profile_indices(dev_idx,64);sim=AssociationSimulationConfig(seed=SEED)
    reconstructed={};simnoise=load_snr_noise(str(calibration))
    for snr in (5,10,15,20):
        js(out/'progress.json',dict(status='association_reconstruction',snr=snr))
        reconstructed[snr]=reconstruct_histories('association_gnn',assoc,states[ai],masks[ai],simnoise,snr,sim,torch.device('cuda'),64)
        emit(dict(association_snr=snr,rmse=reconstructed[snr]['history_position_rmse_m']))
    del assoc
    assoc_hash={str(s):hashlib.sha256(x['history'].tobytes()).hexdigest() for s,x in reconstructed.items()}
    downstream={}
    for arm in ARMS:
        r.set_seed(SEED);model=CoreGraphLLM(build_graph(arm)).cuda()
        ck=torch.load(out/(arm+'_llm_selected.pt'),map_location='cpu',weights_only=True);restore_compact(model,ck['state']);model.eval()
        downstream[arm]={str(s):forecast_metrics(model,x['history'],states[ai,20:],masks[ai],torch.device('cuda'),24) for s,x in reconstructed.items()}
        emit(dict(downstream_arm=arm,metrics=downstream[arm]))
        del model;torch.cuda.empty_cache()
    js(out/'association_downstream.json',dict(results=downstream,indices=ai.tolist(),input_sha256=assoc_hash,
        reconstruction={str(s):{k:v for k,v in x.items() if k!='history'} for s,x in reconstructed.items()}))
    assert all(sha(p)==h for p,h in hashes.items())
    js(out/'completed.json',dict(status='completed',seed=SEED,summaries=summaries,source_hashes_verified=True,no_test=True))
    js(out/'progress.json',dict(status='completed'))
    emit(dict(status='COMPLETED'))
