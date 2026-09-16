"""Quantify the QGNN -> LLM -> effective interface chain on hard dev scenes."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2';sys.path.insert(0,str(BASE))
from analyze_interaction_strata import scene_features
SNRS=('5','10','15','20');SEED=2026;REPS=5000
PATHS={
 'plain_gnn':BASE/'converged_protocol_seed2026_v1/plain_graph_full_dev.npz',
 'qgnn':BASE/'physics_aligned_dual_seed2026_v1/physics_quantum_dual_graph_full_dev.npz',
 'qgnn_llm':BASE/'future_token_qgnn_llm_seed2026_v1/future_token_qgnn_llm_full_dev.npz',
 'final_qgnn_llm':BASE/'effective_segment_qgnn_llm_seed2026_v1/full_dev_predictions.npz',
 'final_classical_llm':BASE/'effective_segment_classical_llm_seed2026_v1/full_dev_predictions.npz'}
def load(path):
    z=np.load(path);return {s:z[s] for s in SNRS},z['indices']
def metric(a,idx):
    values=[]
    for s in SNRS:
        x=a[s][idx];d=x[:,2].sum();values.append([x[:,0].sum()/(d*20),x[:,1].sum()/d])
    return np.asarray(values).mean(0)
def compare(ref,cand,idx,rng):
    p=100*(metric(ref,idx)-metric(cand,idx))/metric(ref,idx);draw=np.empty((REPS,2))
    for k in range(REPS):
        z=idx[rng.integers(0,len(idx),len(idx))];r,c=metric(ref,z),metric(cand,z);draw[k]=100*(r-c)/r
    return {n:{'point_percent':float(p[j]),'ci95':[float(x) for x in np.quantile(draw[:,j],[.025,.975])],'p_improve':float((draw[:,j]>0).mean())} for j,n in enumerate(('ade','fde'))}
def main():
    models={};indices=None
    for n,p in PATHS.items():
        a,i=load(p);indices=i if indices is None else indices;assert np.array_equal(i,indices),n;models[n]=a
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:f=scene_features(d['train_states'][indices,:20],d['train_mask'][indices])
    groups={'all':np.arange(len(indices))}
    for key in ('composite','closing_strength','maneuver'):
        for frac,label in ((.2,'top20'),(.1,'top10')):groups[f'{key}_{label}']=np.flatnonzero(f[key]>=np.quantile(f[key],1-frac))
    pairs={'final_vs_plain_gnn':('plain_gnn','final_qgnn_llm'),'llm_interface_vs_qgnn':('qgnn','final_qgnn_llm'),'new_interface_vs_original_qgnn_llm':('qgnn_llm','final_qgnn_llm'),'quantum_vs_matched_classical':('final_classical_llm','final_qgnn_llm'),'original_llm_vs_qgnn':('qgnn','qgnn_llm')}
    rng=np.random.default_rng(SEED);out={'seed':SEED,'repetitions':REPS,'groups':{}}
    for gn,idx in groups.items():
        out['groups'][gn]={'scenes':int(len(idx)),'metrics':{n:{'ade_m':float(metric(a,idx)[0]),'fde_m':float(metric(a,idx)[1])} for n,a in models.items()},'comparisons':{n:compare(models[r],models[c],idx,rng) for n,(r,c) in pairs.items()}}
    target=BASE/'effective_segment_qgnn_llm_seed2026_v1/contribution_chain.json';target.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
