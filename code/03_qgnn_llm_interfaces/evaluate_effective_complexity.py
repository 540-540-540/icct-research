"""Outcome-blind hard-scene analysis for the matched effective-relation models."""
from __future__ import annotations
import json,platform,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2';HERE=ROOT/'双图研究框架/未来词QGNN_LLM实验_2026-09-09'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE))
import experiment_utils as utils
import run as retained
from analyze_interaction_strata import scene_features
from run_effective_segment_trust import build,restore
from run_future_token_qgnn_llm import load_banks
SNRS=('5','10','15','20');SEED=2026;REPS=5000
def metrics(data,idx):
    vals=[]
    for s in SNRS:
        x=data[s][idx];d=x[:,2].sum();vals.append([x[:,0].sum()/(d*20),x[:,1].sum()/d])
    return np.asarray(vals).mean(0)
def boot(c,q,idx,rng):
    point=100*(metrics(c,idx)-metrics(q,idx))/metrics(c,idx);draw=np.empty((REPS,2))
    for k in range(REPS):
        z=idx[rng.integers(0,len(idx),len(idx))];a,b=metrics(c,z),metrics(q,z);draw[k]=100*(a-b)/a
    return {name:{'point_percent':float(point[j]),'ci95':[float(x) for x in np.quantile(draw[:,j],[.025,.975])],'probability_improvement':float((draw[:,j]>0).mean())} for j,name in enumerate(('ade','fde'))}
def main():
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available();retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json');_,_,dev,_,_,di=load_banks();arrays={}
    for arm in ('quantum','classical'):
        _,m=build(arm);run=BASE/f'effective_segment_{"qgnn" if arm=="quantum" else "classical"}_llm_seed2026_v1';z=torch.load(run/'segment_trust_selected.pt',map_location='cpu',weights_only=True);restore(m,z['state']);m=m.cuda().eval();_,a=retained.evaluate(m,dev,collect=True);arrays[arm]=a;np.savez_compressed(run/'full_dev_predictions.npz',**a,indices=di)
    with np.load(ROOT/'data/multitarget_lankershim_v1.npz') as d:f=scene_features(d['train_states'][di,:20],d['train_mask'][di])
    groups={'all':np.arange(len(di))}
    for key in ('composite','targets','close_pairs','conflict_pairs','closing_strength','maneuver'):
        for fraction,label in ((.2,'top20'),(.1,'top10')):
            cut=np.quantile(f[key],1-fraction);groups[f'{key}_{label}']=np.flatnonzero(f[key]>=cut)
    rng=np.random.default_rng(SEED);result={'seed':SEED,'bootstrap_repetitions':REPS,'selection_is_outcome_blind':True,'groups':{}}
    for name,idx in groups.items():
        q,c=metrics(arrays['quantum'],idx),metrics(arrays['classical'],idx);result['groups'][name]={'scenes':int(len(idx)),'quantum':{'ade_m':float(q[0]),'fde_m':float(q[1])},'classical':{'ade_m':float(c[0]),'fde_m':float(c[1])},'quantum_vs_classical':boot(arrays['classical'],arrays['quantum'],idx,rng)}
    out=BASE/'effective_segment_qgnn_llm_seed2026_v1';utils.atomic_json(out/'outcome_blind_hard_strata.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
