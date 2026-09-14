import argparse, copy, hashlib, json, platform, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from model import build, Circuit, OLD
from measurement_track_dataset import AssociationFrameDataset, AssociationSimulationConfig, simulate_measurements, build_association_features
from run_measurement_track_experiment import decode_hungarian_baseline, make_single_batch

BASE=Path(__file__).resolve().parents[1]
CACHE='/home/js_cn/sensing/data/multitarget_lankershim_v1.npz'
CAL='/home/js_cn/sensing/双图研究框架/01_目标交互图_已完成/results_snapshot/snr_calibration.json'
KEYS=('track_features','measurement_features','edge_features','candidate_mask','track_mask','measurement_mask')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def state_sha(m):
    h=hashlib.sha256()
    for k,v in sorted(state(m).items()): h.update(k.encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()
def materialize(ds):
    samples=[ds[i] for i in range(len(ds))]
    return {k:torch.stack([b[k] for b in samples]).cuda() for k in (*KEYS,'labels')}
def take(data,ix): return {k:v[ix] for k,v in data.items()}
@torch.no_grad()
def ce(model,data):
    total=count=0
    for i in range(0,len(data['labels']),64):
        b=take(data,slice(i,i+64)); y=model({k:b[k] for k in KEYS})['logits']
        total+=float(torch.nn.functional.cross_entropy(y.flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum'))
        count+=int((b['labels']!=-100).sum())
    return total/count

def circuit_check():
    torch.manual_seed(16); c=Circuit().double(); x=torch.randn(3,4,dtype=torch.double,requires_grad=True)
    s=c.state(x); assert torch.allclose(s.square().sum(-1),torch.ones(3,dtype=torch.double),atol=1e-12)
    # Independent full 16x16 Kronecker matrices, little-endian qubit order.
    refs=[]
    for b in range(3):
        v=torch.zeros(16,dtype=torch.double); v[0]=1.
        for l in range(2):
            for q in range(4):
                a=x[b,q]+c.theta[l,q]; co=torch.cos(a/2); si=torch.sin(a/2)
                gate=torch.stack([torch.stack([co,-si]),torch.stack([si,co])])
                mat=torch.ones(1,1,dtype=torch.double)
                for bit in reversed(range(4)): mat=torch.kron(mat,gate if bit==q else torch.eye(2,dtype=torch.double))
                v=mat@v
            for q in range(4):
                signs=torch.tensor([-1. if ((i>>q)&1) and ((i>>((q+1)%4))&1) else 1. for i in range(16)],dtype=torch.double)
                v=v*signs
        refs.append(v)
    ref=torch.stack(refs); assert torch.allclose(s,ref,atol=1e-12)
    loss=c(x).square().sum(); loss.backward(); assert c.theta.grad.abs().max()>0 and x.grad.abs().max()>0
    assert torch.autograd.gradcheck(c,(x.detach().requires_grad_(),),eps=1e-6,atol=1e-4)
    return dict(norm_and_matrix_verified=True,input_gradcheck=True,parameter_gradient_nonzero=True)

def decode(p,f):
    tracks=np.flatnonzero(f['track_mask']); measurements=np.flatnonzero(f['measurement_mask'])
    assignment=np.full(8,12,dtype=np.int64)
    if not len(tracks): return assignment
    real=-np.log(np.maximum(p[np.ix_(tracks,measurements)],1e-12))
    real=np.where(f['candidate_mask'][np.ix_(tracks,measurements)],real,1e6)
    dummy=np.broadcast_to(-np.log(np.maximum(p[tracks,-1:],1e-12)),(len(tracks),len(tracks)))
    rr,cc=linear_sum_assignment(np.concatenate([real,dummy],1))
    for a,b in zip(rr,cc):
        if b<len(measurements): assignment[tracks[a]]=measurements[b]
    return assignment

@torch.no_grad()
def track(model,states,masks,noise,snr,sim):
    floor=np.array([sim.covariance_floor_position_m**2]*2+[sim.covariance_floor_velocity_mps**2]*2,dtype=np.float32)
    process=np.array([sim.process_position_sigma_m**2]*2+[sim.process_velocity_sigma_mps**2]*2,dtype=np.float32)
    recovery=np.array([sim.reinitialization_position_sigma_m**2]*2+[sim.reinitialization_velocity_sigma_mps**2]*2,dtype=np.float32)
    rows=[]; digest=hashlib.sha256()
    for scene in range(len(states)):
        valid=masks[scene]; rng=np.random.default_rng(sim.seed+7000000+scene)
        x=states[scene,0].copy(); x[:,:2]+=rng.normal(0,sim.process_position_sigma_m,x[:,:2].shape)
        x[:,2:]+=rng.normal(0,sim.process_velocity_sigma_mps,x[:,2:].shape)
        cov=np.tile(np.maximum(process,floor),(8,1)); misses=np.zeros(8); conf=np.ones(8)
        last=np.full(8,-1); ever=np.zeros(8,dtype=bool); prior=np.zeros(8,dtype=bool)
        row=dict(correct=0,wrong_associations=0,detected=0,assigned=0,owner_transitions=0,fragmentations=0,squared_error=0.,positions=0)
        for t in range(1,20):
            rng=np.random.default_rng(sim.seed+8000000+snr*100000+scene*101+t)
            measurement=simulate_measurements(states[scene,t],valid,snr,noise,sim,rng)
            for k in sorted(measurement): digest.update(k.encode()); digest.update(np.asarray(measurement[k]).tobytes())
            predicted=x.copy(); predicted[:,:2]+=x[:,2:]*sim.dt; pcov=np.maximum(cov+process,floor)
            f=build_association_features(predicted,pcov,valid,np.full(8,t),misses,conf,measurement,snr,sim)
            if model is None:
                assignment=decode_hungarian_baseline(f['edge_features'],f['candidate_mask'],valid,f['measurement_mask'],12,rejection_mahalanobis_sq=16.)
            else:
                p=model(make_single_batch(f,torch.device('cuda')))['probabilities'][0].cpu().numpy(); assignment=decode(p,f)
            current=np.zeros(8,dtype=bool); owners=measurement['owner']
            for i in np.flatnonzero(valid):
                row['detected']+=int(np.any(owners==i)); a=assignment[i]
                if a<len(owners):
                    owner=int(owners[a]); row['assigned']+=1; row['correct']+=int(owner==i); row['wrong_associations']+=int(owner!=i)
                    current[i]=owner==i
                    if owner>=0:
                        row['owner_transitions']+=int(last[i]>=0 and last[i]!=owner); last[i]=owner
                    gain=pcov[i]/np.maximum(pcov[i]+measurement['covariance'][a],1e-6)
                    x[i]=predicted[i]+gain*(measurement['state'][a]-predicted[i]); cov[i]=np.maximum((1-gain)*pcov[i],floor)
                    misses[i]=0; conf[i]=min(1.,conf[i]+.08)
                else:
                    x[i]=predicted[i]; cov[i]=pcov[i]; misses[i]+=1; conf[i]*=.82
                    if misses[i]>=sim.reinitialize_after_misses: cov[i]=np.maximum(cov[i],recovery); conf[i]=min(conf[i],.25)
                row['fragmentations']+=int(current[i] and ever[i] and not prior[i]); ever[i]|=current[i]
            prior=current
            row['squared_error']+=float(np.sum((x[valid,:2]-states[scene,t,valid,:2])**2)); row['positions']+=int(valid.sum())
        rows.append(row)
    totals={k:sum(row[k] for row in rows) for k in rows[0]}
    totals['association_f1']=2*totals['correct']/max(totals['assigned']+totals['detected'],1)
    totals['position_rmse_m']=float(np.sqrt(totals['squared_error']/totals['positions']))
    return dict(aggregate=totals,scenes=rows,measurement_sha256=digest.hexdigest())

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    assert platform.node()=='jscn' and 'RTX 4090' in torch.cuda.get_device_name(0)
    out=BASE/('smoke' if args.smoke else 'development'); out.mkdir(exist_ok=False)
    checks=circuit_check(); torch.set_num_threads(4)
    source=OLD.parent/'checkpoints/association_gnn.pt'; initial=torch.load(source,map_location='cpu',weights_only=True)
    hashes={str(p):sha(p) for p in [source,Path(__file__),BASE/'code/model.py',BASE/'验证方案.md',OLD/'measurement_track_dataset.py',OLD/'measurement_track_association.py',OLD/'run_measurement_track_experiment.py']}
    if not args.smoke:
        passed=json.loads((BASE/'smoke/completed.json').read_text()); assert passed['source_sha256']==hashes and passed['engineering_passed']
    sim=AssociationSimulationConfig(seed=2026)
    train_ds=AssociationFrameDataset(CACHE,'train',CAL,sim,max_samples=128 if args.smoke else 12000)
    val_ds=AssociationFrameDataset(CACHE,'val',CAL,sim,max_samples=64 if args.smoke else 4000)
    train=materialize(train_ds); val=materialize(val_ds)
    save(out/'protocol.json',dict(source_sha256=hashes,seed=2026,epochs=2 if args.smoke else 4,train_frames=len(train_ds),selection_frames=len(val_ds),test_loaded=False,circuit_checks=checks,cache_sha256=sha(CACHE),calibration_sha256=sha(CAL)))
    models={}; training={}; reference=None; reference_input=None
    frozen=build('plain',initial).eval(); models['original_gnn']=frozen
    with torch.no_grad(): initial_prediction=frozen({k:val[k][:32] for k in KEYS})['logits']
    for kind in ('plain','mlp','quantum'):
        model=build(kind,initial).eval()
        with torch.no_grad(): assert torch.equal(model({k:val[k][:32] for k in KEYS})['logits'],initial_prediction)
        starting=state(model); best=copy.deepcopy(starting); best_ce=ce(model,val); best_epoch=0
        extra=[p for n,p in model.named_parameters() if any(t in n for t in ('.encoder.','.mapping.','.readout.'))]
        extra_ids={id(p) for p in extra}; original=[p for p in model.parameters() if id(p) not in extra_ids]
        optimizer=torch.optim.AdamW([dict(params=original,lr=1e-4),dict(params=extra,lr=3e-4)],weight_decay=1e-4)
        records=[]; digest=hashlib.sha256(); qgrad=0.
        for epoch in range(1,(2 if args.smoke else 4)+1):
            order=torch.randperm(len(train_ds),generator=torch.Generator().manual_seed(2026+epoch)); digest.update(order.numpy().tobytes())
            started=time.time(); total=0; count=0
            for ix in order.split(64):
                b=take(train,ix.cuda()); optimizer.zero_grad(set_to_none=True)
                logits=model({k:b[k] for k in KEYS})['logits']
                loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),b['labels'].flatten(),ignore_index=-100)
                assert torch.isfinite(loss); loss.backward()
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
                        if n.endswith('mapping.theta'): qgrad=max(qgrad,float(p.grad.abs().max()))
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); optimizer.step()
                n=int((b['labels']!=-100).sum()); total+=float(loss)*n; count+=n
            metric=ce(model,val)
            if metric<best_ce: best_ce=metric; best_epoch=epoch; best=state(model)
            row=dict(arm=kind,epoch=epoch,training_ce=total/count,validation_ce=metric,seconds=time.time()-started); records.append(row)
            print(json.dumps(row),flush=True); save(out/(kind+'_training.json'),records)
        if kind=='quantum':
            assert qgrad>0 and not torch.equal(state(model)['layers.1.edge_message.mapping.theta'],starting['layers.1.edge_message.mapping.theta'])
        if reference_input is None: reference_input=digest.hexdigest()
        assert reference_input==digest.hexdigest()
        model.load_state_dict(best,strict=True); p=out/(kind+'_selected.pt'); torch.save(dict(state=best,arm=kind,selected_epoch=best_epoch),p)
        restored=build(kind,initial).eval(); restored.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['state'],strict=True)
        assert state_sha(restored)==state_sha(model)
        models[kind]=restored; training[kind]=dict(selected_epoch=best_epoch,validation_ce=best_ce,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(p),quantum_gradient_max=qgrad)
    # Predetermined val development scenes, no original test loading.
    indices=np.linspace(len(val_ds.states)//2,len(val_ds.states)-1,2 if args.smoke else 24,dtype=int)
    states=val_ds.states[indices]; masks=val_ds.mask[indices]
    results={}; conditions=[(.1,1.5)] if args.smoke else [(.1,1.5),(.3,1.5),(.1,5.)]
    for miss,false in conditions:
        config=replace(sim,miss_probability=miss,false_alarm_rate=false)
        for snr in ([5] if args.smoke else [5,10,15,20]):
            key='%s_%s_%s'%(miss,false,snr); results[key]={}; paired=None
            for arm,m in [('hungarian',None),*models.items()]:
                result=track(m,states,masks,val_ds.noise_map,snr,config)
                if paired is None: paired=result['measurement_sha256']
                assert paired==result['measurement_sha256']; results[key][arm]=result
                print(json.dumps(dict(condition=key,arm=arm,metrics=result['aggregate'])),flush=True)
            save(out/'tracking_progress.json',results)
    assert all(sha(p)==h for p,h in hashes.items())
    save(out/'completed.json',dict(engineering_passed=True,source_sha256=hashes,training=training,results=results,scene_indices=indices.tolist(),test_loaded=False,downstream_forecasting_run=False))
    lines=['# 量子关联图开发验证结果','','本轮为关联与递归状态重建开发筛选，尚未运行经典目标交互图及 LLM 的下游预测。','',
       '| 漏检率_虚警率_SNR | 方法 | 关联 F1 | 错误关联数 | owner 变化数 | 位置 RMSE (m) |','|---|---|---:|---:|---:|---:|']
    for key,arms in results.items():
        for arm,data in arms.items():
            d=data['aggregate']; lines.append('| %s | %s | %.6f | %d | %d | %.6f |'%(key,arm,d['association_f1'],d['wrong_associations'],d['owner_transitions'],d['position_rmse_m']))
    lines+=['','关联 F1 不是标准 IDF1；owner 变化数不是标准 MOT IDSW。当前为固定目标、带噪首帧初始化及仿真量测。',
       '全部方案应用相同稳定化滤波；因此不能直接与旧报告数值混合比较。单种子、小预算、既有 val 场景不构成独立测试证据。','']
    (out/'结果与分析.md').write_text('\n'.join(lines),encoding='utf-8'); print('COMPLETED',flush=True)

if __name__=='__main__': main()
