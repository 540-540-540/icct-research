"""Paired block bootstrap over scene/time blocks; one trained seed only."""
import json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;D=B/'development'
indices=np.load(D/'quantum_llm_full_dev.npz')['indices']
with np.load('/home/js_cn/sensing/data/multitarget_lankershim_v1.npz',allow_pickle=False) as z:
    starts=z['train_start_index'][indices]
blocks=(starts//200).astype(int)
unique=np.unique(blocks);rng=np.random.default_rng(20260906)

def load(arm,phase,snr): return np.load(D/f'{arm}_{phase}_full_dev.npz')[str(snr)]
def summarize(ref,phase):
    out={}
    for snr in (5,10,15,20):
        q=load('quantum',phase,snr);c=load(ref,phase,snr)
        # columns: sum over all 20 steps, final sum, valid targets
        assert np.array_equal(q[:,2],c[:,2])
        scene_q=np.c_[q[:,0]/(20*q[:,2]),q[:,1]/q[:,2]]
        scene_c=np.c_[c[:,0]/(20*c[:,2]),c[:,1]/c[:,2]]
        weights=q[:,2]
        wins=(scene_q<scene_c).mean(0)
        delta=[]
        for _ in range(10000):
            chosen=rng.choice(unique,len(unique),replace=True)
            rows=np.concatenate([np.flatnonzero(blocks==x) for x in chosen])
            w=weights[rows]
            mq=(scene_q[rows]*w[:,None]).sum(0)/w.sum()
            mc=(scene_c[rows]*w[:,None]).sum(0)/w.sum()
            delta.append(100*(mc-mq)/mc)
        delta=np.asarray(delta)
        point=100*((scene_c*weights[:,None]).sum(0)-(scene_q*weights[:,None]).sum(0))/(scene_c*weights[:,None]).sum(0)
        out[str(snr)]={'reduction_percent':point.tolist(),'block_bootstrap_95_percentile_ci':np.percentile(delta,[2.5,97.5],axis=0).T.tolist(),
                       'scene_win_fraction_unweighted':wins.tolist()}
    return out
result={'training_seed':2026,'block_width_frames':200,'unique_blocks':int(len(unique)),
        'warning':'CIs condition on this one trained seed and quantify paired scene/time-block variation only; they do not quantify training-seed uncertainty.',
        'comparisons':{}}
for phase in ('graph','llm'):
    for ref in ('plain','classical'):result['comparisons'][phase+'_vs_'+ref]=summarize(ref,phase)
(D/'paired_block_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
