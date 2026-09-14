"""Diagnostic upper bound: replace predicted intent probabilities with true labels."""
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/js_cn/sensing'); BASE=ROOT/'diagnostics/core_message_seed2026_v2'
HERE=ROOT/'双图研究框架/未来词QGNN_LLM实验_2026-09-09'
sys.path[:0]=[str(HERE),str(BASE),str(ROOT)]
import experiment_utils as utils
import run as retained
from run_future_token_qgnn_llm import load_banks
from run_intent_token_planner import build, restore, intent_labels, SEED


@torch.no_grad()
def evaluate_mode(model, bank, mode, batch_size=24):
    model.eval(); by_snr={}
    for snr in (5,10,15,20):
        generator=torch.Generator(device='cuda').manual_seed(SEED+100000); rows=[]
        for start in range(0,len(bank['history']),batch_size):
            index=torch.arange(start,min(start+batch_size,len(bank['history'])),device='cuda')
            history,future,mask=retained.make_batch(bank,index,snr,generator)
            if mode=='oracle':
                labels=intent_labels(history,future,mask)
                model.intent_probability_override=torch.cat([
                    F.one_hot(labels[name],3).float() for name in ('longitudinal','lateral','interaction')
                ],dim=-1)
            elif mode=='uniform':
                shape=(len(index),mask.shape[1],3,9)
                model.intent_probability_override=torch.full(shape,1/3,device='cuda')
            else:model.intent_probability_override=None
            output=model(history,mask); distance=(output['future_position']-future[...,:2]).norm(dim=-1)
            rows.append(torch.stack([(distance.double()*mask[:,None,:]).sum((1,2)),
                                     (distance[:,-1].double()*mask).sum(1),mask.sum(1)],1).cpu())
        values=torch.cat(rows).numpy(); denominator=values[:,2].sum()
        by_snr[str(snr)]={'ade_m':float(values[:,0].sum()/(denominator*20)),
                          'fde_m':float(values[:,1].sum()/denominator)}
    model.intent_probability_override=None
    aggregate={key:float(np.mean([value[key] for value in by_snr.values()])) for key in ('ade_m','fde_m')}
    return {'aggregate':aggregate,'by_snr':by_snr}


def main():
    assert Path.cwd()==ROOT and platform.node()=='jscn' and torch.cuda.is_available()
    retained.NOISE=utils.load_snr_noise_map(ROOT/'results/multitarget_snr/snr_calibration.json')
    _,_,dev,_,_,_=load_banks(); results={}
    for arm in ('quantum','classical'):
        _,model,_,_=build(arm); model=model.cuda()
        checkpoint=torch.load(BASE/f'intent_token_{"qgnn" if arm=="quantum" else "classical"}_seed2026_v1'/'intent_planner_selected.pt',map_location='cpu',weights_only=True)
        restore(model,checkpoint['state'])
        results[arm]={mode:evaluate_mode(model,dev,mode) for mode in ('predicted','uniform','oracle')}
        del model;torch.cuda.empty_cache()
    utils.atomic_json(BASE/'intent_token_oracle_seed2026_v1.json',{'diagnostic_only':True,'uses_future_labels':True,'results':results})
    print(json.dumps(results))


if __name__=='__main__':main()
