"""Validation-only paired SNR robustness evaluation for frozen RAJ checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.raj_qgnn.model import build_model
from scripts.train_raj_qgnn import evaluate


def load_model(kind,checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config']
    model=build_model(kind,cfg['seed'],cfg['depth'],correction_cap_m=cfg['correction_cap'],
                      classical_hidden=cfg.get('classical_hidden',42)).to(device)
    missing,unexpected=model.load_state_dict(saved['model_state'],strict=False)
    bad=[k for k in missing if not (k.startswith('llm.gpt2.') and 'lora_' not in k)]
    if unexpected or bad:raise ValueError(f'checkpoint mismatch unexpected={unexpected} missing={bad}')
    return model


def metrics(rows,ids):
    chosen=[rows[i] for i in ids]
    ade=float(np.mean([r['ADE'] for r in chosen]));fde=float(np.mean([r['FDE'] for r in chosen]))
    return {'ADE':ade,'FDE':fde,'J':ade+.5*fde,'scenes':len(chosen)}


def gains(q,c):
    return {k:100*(c[k]-q[k])/c[k] for k in ('ADE','FDE','J')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--quantum-run');p.add_argument('--classical-run');p.add_argument('--output',default='cross_snr_analysis.json')
    args=p.parse_args();device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    campaign=ROOT/args.campaign
    qrun=ROOT/args.quantum_run if args.quantum_run else campaign/'4096_quantum'
    crun=ROOT/args.classical_run if args.classical_run else campaign/'4096_matched_classical'
    checkpoints={'quantum':qrun/'best.pt','matched_classical':crun/'best.pt'}
    models={k:load_model(k,v,device) for k,v in checkpoints.items()}
    # Freeze input-only interaction strata once at the training SNR, then reuse scene indices at every SNR.
    reference=SinDPredictionDataset('val',0.,ROOT,True)
    loader=DataLoader(reference,batch_size=args.batch_size,shuffle=False,num_workers=0)
    _,reference_rows=evaluate(models['quantum'],loader,device)
    risk=np.asarray([r['max_pair_risk'] for r in reference_rows])
    fixed_ids={'all':list(range(len(risk))),
               'top20':np.flatnonzero(risk>=np.quantile(risk,.8)).tolist(),
               'top10':np.flatnonzero(risk>=np.quantile(risk,.9)).tolist()}
    result={'protocol':{'split':'validation','train_snr_db':0,'test_set_used':False,
                        'strata':'fixed once from 0 dB history-only max_pair_risk'},'snr':{}}
    for snr in (-10.,-5.,0.,5.,10.):
        dataset=SinDPredictionDataset('val',snr,ROOT,True)
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
        rows={};summary={}
        for kind,model in models.items():summary[kind],rows[kind]=evaluate(model,loader,device)
        strata={}
        for name,ids in fixed_ids.items():
            q=metrics(rows['quantum'],ids);c=metrics(rows['matched_classical'],ids)
            strata[name]={'quantum':q,'classical':c,'gain_pct':gains(q,c)}
        result['snr'][str(int(snr))]={'overall_gain_pct':gains(summary['quantum'],summary['matched_classical']),
                                     'strata':strata}
        print(snr,result['snr'][str(int(snr))]['overall_gain_pct'],flush=True)
    out=campaign/args.output;out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
