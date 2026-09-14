"""Locked official-test evaluation, paired bootstrap and paper-ready report."""
import hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as r
import run as retained
from model import build_graph,CoreGraphLLM,restore_compact
from physics_aligned_dual_model import build_physics_aligned_dual_graph
from analyze_interaction_strata import scene_features,aggregate
SEED=2026;GRAPH=BASE/'physics_aligned_dual_seed2026_v1';LLM_CLASSIC=BASE/'strong_dual_llm_seed2026_v1';LLM_QUANTUM=BASE/'quantum_llm_frozen_cont_seed2026_v1';CONV=BASE/'converged_protocol_seed2026_v1'
OUT=BASE/'final_qgnn_paper_evaluation_seed2026_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_state(model,path):model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);return model
def candidates():
    yield 'plain_gnn',load_state(build_graph('plain'),CONV/'plain_graph_selected.pt')
    for arm in ('physics_classical_dual','physics_quantum_dual'):
        yield arm,load_state(build_physics_aligned_dual_graph(arm),GRAPH/(arm+'_graph_selected.pt'))
    for arm in ('physics_classical_dual','physics_quantum_dual'):
        g=load_state(build_physics_aligned_dual_graph(arm),GRAPH/(arm+'_graph_selected.pt'));m=CoreGraphLLM(g)
        source=LLM_CLASSIC if arm=='physics_classical_dual' else LLM_QUANTUM
        restore_compact(m,torch.load(source/(arm+'_llm_selected.pt'),map_location='cpu',weights_only=True)['state']);yield arm+'_llm',m
def metric_from_arrays(arrays,ix):
    vals=[]
    for snr in ('5','10','15','20'):
        a=arrays[snr][ix];den=a[:,2].sum();vals.append((a[:,0].sum()/(den*20),a[:,1].sum()/den))
    return np.mean(vals,axis=0)
def paired_bootstrap(reference,candidate,reps=2000):
    rng=np.random.default_rng(SEED);n=len(reference['5']);values=[]
    for _ in range(reps):
        ix=rng.integers(0,n,n);ref=metric_from_arrays(reference,ix);cand=metric_from_arrays(candidate,ix);values.append(100*(ref-cand)/ref)
    x=np.asarray(values)
    return {k:{'median_improvement_percent':float(np.median(x[:,j])),'ci95_percent':[float(v) for v in np.quantile(x[:,j],[.025,.975])],'bootstrap_probability_improvement':float((x[:,j]>0).mean())} for j,k in enumerate(('ade','fde'))}
def main():
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available() and (LLM_CLASSIC/'summary.json').exists() and (LLM_QUANTUM/'completed.json').exists()
    OUT.mkdir(exist_ok=False);retained.NOISE=r.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:
        states=d['test_states'].copy();masks=d['test_mask'].copy()
    bank={'history':torch.from_numpy(states[:,:20]).float().cuda(),'future':torch.from_numpy(states[:,20:]).float().cuda(),'mask':torch.from_numpy(masks).bool().cuda()}
    summaries={};arrays={}
    for name,model in candidates():
        r.set_seed(SEED);model=model.cuda();start=time.time();metrics,arr=retained.evaluate(model,bank,collect=True);seconds=time.time()-start
        summaries[name]={'metrics':metrics,'parameters':sum(p.numel() for p in model.parameters()),'evaluation_seconds':seconds};arrays[name]=arr
        np.savez_compressed(OUT/(name+'.npz'),**arr,indices=np.arange(len(states)));r.atomic_json(OUT/'progress.json',{'status':'evaluating','completed':list(summaries),'last':name});print(json.dumps({name:summaries[name]}),flush=True);del model;torch.cuda.empty_cache()
    comparisons={
      'quantum_llm_vs_classical_llm':paired_bootstrap(arrays['physics_classical_dual_llm'],arrays['physics_quantum_dual_llm']),
      'quantum_llm_vs_plain_gnn':paired_bootstrap(arrays['plain_gnn'],arrays['physics_quantum_dual_llm']),
      'quantum_graph_vs_classical_graph':paired_bootstrap(arrays['physics_classical_dual'],arrays['physics_quantum_dual'])}
    features=scene_features(states[:,:20],masks);score=features['composite'];cuts=np.quantile(score,[1/3,2/3]);groups={'low':score<=cuts[0],'medium':(score>cuts[0])&(score<=cuts[1]),'high':score>cuts[1]};strata={}
    for group,sel in groups.items():
      strata[group]={}
      for snr in ('5','10','15','20'):
        c=aggregate(arrays['physics_classical_dual_llm'][snr],sel);q=aggregate(arrays['physics_quantum_dual_llm'][snr],sel);strata[group][snr]={'classical':c,'quantum':q,'quantum_improvement_percent':{'ade':100*(c['ade_m']-q['ade_m'])/c['ade_m'],'fde':100*(c['fde_m']-q['fde_m'])/c['fde_m']}}
    final={'status':'completed','seed':SEED,'split':'official test (historically used by earlier project phases; not claimed as never-seen blind test)','summaries':summaries,'paired_bootstrap_2000':comparisons,'complexity_strata':strata,'no_multiple_seeds':True}
    r.atomic_json(OUT/'final_results.json',final)
    def m(name,key):return summaries[name]['metrics']['aggregate'][key+'_m']
    qa,qf=m('physics_quantum_dual_llm','ade'),m('physics_quantum_dual_llm','fde');ca,cf=m('physics_classical_dual_llm','ade'),m('physics_classical_dual_llm','fde');pa,pf=m('plain_gnn','ade'),m('plain_gnn','fde')
    report=f"""# 双层QGNN＋LLM最终单种子结果（seed 2026）\n\n## 官方测试集四SNR宏平均\n\n| 模型 | ADE (m) | FDE (m) |\n|---|---:|---:|\n| Plain GNN | {pa:.6f} | {pf:.6f} |\n| 双层经典图模型 | {m('physics_classical_dual','ade'):.6f} | {m('physics_classical_dual','fde'):.6f} |\n| 双层QGNN | {m('physics_quantum_dual','ade'):.6f} | {m('physics_quantum_dual','fde'):.6f} |\n| 双层经典图模型＋LLM | {ca:.6f} | {cf:.6f} |\n| 双层QGNN＋LLM | {qa:.6f} | {qf:.6f} |\n\nQGNN＋LLM相对Plain GNN：ADE改善 {(pa-qa)/pa*100:.3f}%，FDE改善 {(pf-qf)/pf*100:.3f}%。  \nQGNN＋LLM相对匹配经典＋LLM：ADE改善 {(ca-qa)/ca*100:.3f}%，FDE改善 {(cf-qf)/cf*100:.3f}%。\n\n## 证据边界\n\n结果为单种子；置信区间来自同一模型下的配对场景bootstrap，不能替代多种子训练。该官方测试划分在项目早期实验中曾被使用，因此不表述为从未查看的盲测。完整分层和bootstrap见 `final_results.json`。\n"""
    (OUT/'论文结果汇总.md').write_text(report,encoding='utf-8');r.atomic_json(OUT/'completed.json',{'status':'completed','final_results_sha256':sha(OUT/'final_results.json'),'report_sha256':sha(OUT/'论文结果汇总.md')});r.atomic_json(OUT/'progress.json',{'status':'completed'});print(report,flush=True)
if __name__=='__main__':main()
