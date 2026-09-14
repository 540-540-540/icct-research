"""Measure when the retained LLM correction helps the QGNN prediction."""
from __future__ import annotations

import json, platform, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/js_cn/sensing');BASE=ROOT/'diagnostics/core_message_seed2026_v2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment_utils as utils
import run as retained
from run_event_gated_fusion import build_pair
from run_future_token_qgnn_llm import load_banks

SEED=2026

def accum(pred,target,mask,store):
    err=torch.linalg.vector_norm(pred-target[...,:2],dim=-1)
    valid=mask[:,None,:].expand_as(err)
    store['ade_sum']+=float(err[valid].sum());store['ade_n']+=int(valid.sum())
    store['fde_sum']+=float(err[:,-1][mask].sum());store['fde_n']+=int(mask.sum())

def main():
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available()
    retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    _,_,dev,_,_,_=load_banks();model,_,_,_=build_pair('quantum');model=model.cuda().eval()
    scales=torch.tensor([0.,.5,1.,1.5,2.],device='cuda')
    totals={str(float(x)):{'ade_sum':0.,'ade_n':0,'fde_sum':0.,'fde_n':0} for x in scales}
    oracle_step={'ade_sum':0.,'ade_n':0,'fde_sum':0.,'fde_n':0};oracle_target={'ade_sum':0.,'ade_n':0,'fde_sum':0.,'fde_n':0}
    helpful=valid_count=helpful_final=final_count=0
    for snr in (5,10,15,20):
      gen=torch.Generator(device='cuda').manual_seed(SEED+170000+snr)
      for off in range(0,len(dev['history']),24):
        ix=torch.arange(off,min(off+24,len(dev['history'])),device='cuda');h,f,m=retained.make_batch(dev,ix,snr,gen)
        with torch.no_grad():o=model(h,m)
        graph=o['graph_future_position'];base=o['future_position'];correction=base-graph
        candidates=graph[None]+scales[:,None,None,None,None]*correction[None]
        target=f[...,:2];errors=torch.linalg.vector_norm(candidates-target[None],dim=-1)
        valid=m[:,None,:].expand_as(errors[0])
        for k,s in enumerate(scales):accum(candidates[k],f,m,totals[str(float(s))])
        helpful+=int(((errors[2]<errors[0])&valid).sum());valid_count+=int(valid.sum())
        helpful_final+=int(((errors[2][:,-1]<errors[0][:,-1])&m).sum());final_count+=int(m.sum())
        best_step=errors.argmin(dim=0);chosen=torch.gather(candidates,0,best_step[None,...,None].expand(1,*best_step.shape,2)).squeeze(0);accum(chosen,f,m,oracle_step)
        mean_target=(errors*valid[None]).sum(dim=2)/valid.sum(dim=1).clamp_min(1)[None]
        best_target=mean_target.argmin(dim=0);chosen_t=torch.gather(candidates,0,best_target[None,:,None,:,None].expand(1,-1,candidates.shape[2],-1,2)).squeeze(0);accum(chosen_t,f,m,oracle_target)
    def done(x):return {'ade_m':x['ade_sum']/x['ade_n'],'fde_m':x['fde_sum']/x['fde_n']}
    report={'fixed_scales':{k:done(v) for k,v in totals.items()},'oracle_per_step':done(oracle_step),'oracle_per_target':done(oracle_target),'llm_helpful_rate_per_step':helpful/valid_count,'llm_helpful_rate_final':helpful_final/final_count,'snr_macro_protocol':True}
    out=BASE/'llm_correction_oracle_seed2026_v1';out.mkdir(exist_ok=False);utils.atomic_json(out/'summary.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
