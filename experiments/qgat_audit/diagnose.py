"""Fixed-checkpoint development diagnostics; no optimizer step, no test access."""
import sys,json,math,time,statistics
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from prediction.training import Dataset,scene_metrics,masked_trajectory_loss,load_weights,seed_all
from experiments.qgat_candidate.run_formal import Predictor,read_config,verify_contract
from experiments.qgat_candidate.graph import one_qubit_state


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def describe(x):
    x=np.asarray(x,dtype=float).ravel()
    if not len(x):return None
    assert np.isfinite(x).all()
    return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),p10=float(np.quantile(x,.1)),
                median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()),count=int(x.size))


def quantum_statistics(graph,values,stats):
    physical,z,m,d=values;m=m.bool();clean=torch.where(m[...,None],z,0)
    pre=graph.encoder(torch.cat((clean,m[...,None].to(z),d[...,None].to(z)),-1))
    bounded=torch.tanh(pre);angles=math.pi*bounded.double()
    e=one_qubit_state(angles,angles.new_zeros(3))
    q=one_qubit_state(angles,graph.theta[0].double());k=one_qubit_state(angles,graph.theta[1].double())
    gamma=torch.einsum('...jd,...id->...ij',k.conj(),q)
    pairs=m[..., :,None]&m[...,None,:]
    weights=gamma.abs().square()*pairs
    total=weights.sum(-1);alpha=weights/total.clamp_min(1e-12)[...,None]
    count=m.sum(-1).clamp_min(1);many=m&(count[...,None]>1)
    uniform=m[...,None,:].double()/count[...,None,None]
    entropy=-(alpha*alpha.clamp_min(1e-12).log()).sum(-1)/count.double().log().clamp_min(1)[...,None]
    # Exact normalized entropy for count>=2, avoiding log(1) only on unused rows.
    entropy=-(alpha*alpha.clamp_min(1e-12).log()).sum(-1)/count.double().clamp_min(2).log()[...,None]
    distances=torch.cdist(physical[...,:2].float(),physical[...,:2].float())
    raw=graph.features(z,m,d)
    for key,value in dict(success_probability=(total/count[...,None])[m],
            attention_entropy_normalized=entropy[many],self_attention=alpha.diagonal(dim1=-2,dim2=-1)[m],
            self_uniform=(torch.ones_like(total)/count[...,None])[m],
            attention_l1_from_uniform=(alpha-uniform).abs().sum(-1)[m],
            far_attention_mass=(alpha*(distances>45)).sum(-1)[m],
            far_uniform_mass=(uniform*(distances>45)).sum(-1)[m],
            tanh_absolute=bounded[m].abs(),raw_features=raw[m]).items():
        stats.setdefault(key,[]).append(value.detach().cpu().numpy())
    stats.setdefault('fallback_count',0);stats['fallback_count']+=int(((total/count[...,None]<=1e-12)&m).sum())
    # All-neighbor pure-state fidelity measures how distinguishable encoded cars are.
    fidelity=torch.einsum('...jd,...id->...ij',e.conj(),e).abs().square()
    offdiag=pairs&~torch.eye(m.shape[-1],device=m.device,dtype=torch.bool)
    stats.setdefault('encoding_neighbor_fidelity',[]).append(fidelity[offdiag].cpu().numpy())


def group(name):
    if name.startswith('graph.theta'):return 'quantum_theta'
    if name.startswith('graph.encoder'):return 'angle_encoder'
    if name.startswith('graph.readout'):return 'graph_readout'
    if name.startswith('graph.'):return 'classical_graph'
    if 'graph_projection' in name:return 'graph_projection'
    if 'lora_A' in name:return 'lora_A'
    if 'lora_B' in name:return 'lora_B'
    if 'head.' in name:return 'prediction_head'
    return 'state_marker_adapter'


def main():
    torch.set_num_threads(1);device='cuda:0'
    config=read_config(ROOT/'configs/qgat_formal.json')
    contract=json.loads((ROOT/'reports/qgat_formal/provenance.json').read_text())
    verify_contract(contract['source'],ROOT/'configs/qgat_formal.json')
    summary=json.loads((ROOT/'reports/qgat_formal/summary.json').read_text())
    data=Dataset('V_select');train=Dataset('train')
    ids=np.unique(np.linspace(0,len(data)-1,64).round().astype(int))
    train_ids=np.random.default_rng(9381).choice(len(train),16,replace=False)
    report=dict(scope='fixed best checkpoint diagnostic on a prespecified time-spread V_select subset',
        V_select_origins=ids.tolist(),gradient_train_origins=train_ids.tolist(),snrs=[5,10,15,20],
        modes=['baseline','zero_graph','self_only'],optimizer_steps_executed=0,test_opened=False,
        limitation='Fixed-head perturbations create distribution shift and are not independently retrained ablations.',jobs={})
    target=ROOT/'reports/qgat_audit/diagnostics.json'
    for job,s in sorted(summary['tasks'].items()):
        tick=time.perf_counter();model=Predictor(s['model'],s['seed'],config).to(device).eval()
        checkpoint=torch.load(s['training']['best_checkpoint'],map_location='cpu',weights_only=False)
        load_weights(model,checkpoint['model'])
        archived=json.loads((ROOT/'results/qgat_formal'/job/f"validation_epoch_{s['training']['best_metrics']['epoch']:02d}.json").read_text())
        expected={(x['origin'],x['snr_db']):x for x in archived['per_scene']}
        rows={k:[] for k in report['modes']};stats={};baseline_error=0.;effect=[]
        with torch.no_grad():
            for snr in report['snrs']:
                for i in ids:
                    inp,truth=data.batch([i],snr,device);values=tuple(inp[k] for k in ('state_hat','standardized_state','track_exists','detected'))
                    x,z,m,d=values
                    features=model.graph(*values)
                    # Only change the graph's source set; temporal inputs remain identical.
                    alone=(x.reshape(-1,1,4),z.reshape(-1,1,4),m.reshape(-1,1),d.reshape(-1,1))
                    self_features=model.graph(*alone).reshape_as(features)
                    predictions={}
                    for mode,f in [('baseline',features),('zero_graph',torch.zeros_like(features)),('self_only',self_features)]:
                        output=model.temporal(**inp,graph_features=f)
                        assert torch.isfinite(output['prediction']).all()
                        metric=scene_metrics(**output,**truth)
                        row=dict(origin=int(i),snr=snr,ADE=float(metric['scene_ade'][0]),FDE=float(metric['scene_fde'][0]),
                            ade_count=int(metric['ade_count'][0]),fde_count=int(metric['fde_count'][0]))
                        rows[mode].append(row);predictions[mode]=output['prediction']
                    baseline_error=max(baseline_error,abs(rows['baseline'][-1]['ADE']-expected[(int(i),snr)]['ADE']),
                        abs(rows['baseline'][-1]['FDE']-expected[(int(i),snr)]['FDE']))
                    effect.append(dict(origin=int(i),snr=snr,graph_self_feature_rms=float((features-self_features)[m].square().mean().sqrt()),
                        graph_token_rms=float(model.temporal.adapter.graph_projection(features)[m].square().mean().sqrt()),
                        state_token_rms=float(model.temporal.adapter.state_projection(z)[m].square().mean().sqrt())))
                    if s['model']=='qgat':quantum_statistics(model.graph,values,stats)
        assert baseline_error<1e-5,f'Archived baseline mismatch {baseline_error}'
        scores={}
        for mode,rr in rows.items():
            by={}
            for snr in report['snrs']:
                a=[x['ADE'] for x in rr if x['snr']==snr and x['ade_count']];f=[x['FDE'] for x in rr if x['snr']==snr and x['fde_count']]
                by[str(snr)]=dict(ADE=statistics.mean(a),FDE=statistics.mean(f));by[str(snr)]['J']=by[str(snr)]['ADE']+.5*by[str(snr)]['FDE']
            scores[mode]={k:statistics.mean(v[k] for v in by.values()) for k in ('ADE','FDE','J')}
            scores[mode]['by_snr']=by
        for mode in rows:
            assert [(x['origin'],x['snr'],x['ade_count'],x['fde_count']) for x in rows[mode]]==[(x['origin'],x['snr'],x['ade_count'],x['fde_count']) for x in rows['baseline']]
        representation={}
        if stats:
            for key,v in stats.items():
                if key=='fallback_count':representation[key]=v;continue
                arr=np.concatenate(v,axis=0)
                if key=='raw_features':
                    cov=np.cov(arr,rowvar=False);eig=np.linalg.eigvalsh(cov).clip(0)
                    representation['readout_feature_std']=arr.std(0).tolist()
                    representation['readout_covariance_eigenvalues']=eig.tolist()
                    representation['readout_covariance_effective_rank']=float(eig.sum()**2/(eig@eig))
                else:representation[key]=describe(arr)
        # One diagnostic gradient batch, no parameter update. Eval disables dropout.
        seed_all(9381);model.zero_grad(set_to_none=True)
        for index,i in enumerate(train_ids):
            inp,truth=train.batch([i],report['snrs'][index%4],device)
            output=model(**inp);loss=masked_trajectory_loss(**output,**truth)
            (loss['loss']/len(train_ids)).backward()
        grads={}
        for name,p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                g=grads.setdefault(group(name),dict(sum_squared=0.,nonzero=0,numel=0))
                g['sum_squared']+=float(p.grad.double().square().sum());g['nonzero']+=int(torch.count_nonzero(p.grad));g['numel']+=p.numel()
        for g in grads.values():g['l2_norm']=math.sqrt(g.pop('sum_squared'))
        if hasattr(model.graph,'theta'):grads['theta_matrix']=model.graph.theta.grad.detach().cpu().tolist()
        report['jobs'][job]=dict(scores=scores,baseline_archived_max_error=baseline_error,
            quantum_statistics=representation,gradient_probe=grads,
            graph_path={k:describe([x[k] for x in effect]) for k in ('graph_self_feature_rms','graph_token_rms','state_token_rms')},
            per_scene=rows,seconds=time.perf_counter()-tick)
        write(target,report)
        print(job,json.dumps(scores),flush=True)
        del model,checkpoint;torch.cuda.empty_cache()
    report['completed']=True;write(target,report)


if __name__=='__main__':main()
