"""Validation-only evaluator for Graph + Motion-Token GPT-2 models."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from frontend.automatum_prediction_dataset import AutomatumPredictionDataset
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.q0.graph_motion_llm import build_graph_motion_llm
from prediction.q0.motion_token_llm import tokenizer_config_for_dataset
from prediction.q0.features import physical_edge_features
from prediction.q0.normalization import load_normalization,normalization_path

def scene_errors(pred,future,mask):
    d=torch.linalg.vector_norm(pred-future[...,:2],dim=-1)
    d=torch.where(mask[:,None,:],d,torch.zeros_like(d))
    n=mask.sum(1).clamp_min(1)
    return d.sum((1,2))/(n*d.shape[1]), d[:,-1].sum(1)/n

def interaction_stats(history,mask):
    edge,pair=physical_edge_features(history,mask)
    last=edge[:,-1]
    dist,closing,dcpa=last[...,4],last[...,5],last[...,7]
    inf=torch.full_like(dist,float('inf'))
    return {
      'vehicle_count':mask.sum(1),
      'close_pairs_20m':(((dist<20)&pair).sum((1,2))//2),
      'close_pairs_30m':(((dist<30)&pair).sum((1,2))//2),
      'closing_pairs_30m_gt_0p5':((((dist<30)&(closing>.5)&pair).sum((1,2)))//2),
      'min_pair_distance_m':torch.where(pair,dist,inf).amin((1,2)),
      'min_cpa_distance_m':torch.where(pair,dcpa,inf).amin((1,2)),
      'max_closing_rate_mps':torch.where(pair,closing,torch.full_like(closing,-float('inf'))).amax((1,2)),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graph',choices=['nograph','mpnn','routed_mpnn','pair_triplet'],required=True)
    p.add_argument('--dataset',choices=['automatum','sind'],default='sind')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--snr',type=float,default=0.)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--output',required=True)
    a=p.parse_args()

    device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    stats=load_normalization(normalization_path(ROOT,a.snr,a.dataset))
    token_cfg=tokenizer_config_for_dataset(a.dataset)
    model=build_graph_motion_llm(a.graph,stats,init_seed=a.seed,tokenizer_config=token_cfg).to(device)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    missing,unexpected=model.load_state_dict(payload['model_state'],strict=False)
    if unexpected or any(not k.startswith('llm.gpt2.') for k in missing):
        raise ValueError(f'mismatch missing={missing} unexpected={unexpected}')
    model.eval()

    dataset_cls = AutomatumPredictionDataset if a.dataset == 'automatum' else SinDPredictionDataset
    ds=dataset_cls('val',a.snr,ROOT,True)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0)
    rows=[]; cur=0
    with torch.no_grad():
      for b in dl:
        h=b['history_state'].to(device).float(); m=b['vehicle_mask'].to(device).bool(); f=b['future_state'].to(device).float()
        o=model(h,m); ade,fde=scene_errors(o['prediction'],f,m); phy=interaction_stats(h,m)
        for j in range(h.shape[0]):
          r={'dataset_index':cur+j,'scene_id':int(b['scene_id'][j]),'start_frame':int(b['start_frame'][j]),
             'ADE':float(ade[j].cpu()),'FDE':float(fde[j].cpu())}
          for k,v in phy.items():
            x=v[j].cpu(); r[k]=float(x) if x.dtype.is_floating_point else int(x)
          rows.append(r)
        cur += h.shape[0]

    summary={'model':f'{a.graph}_motion_llm','dataset':a.dataset,'snr_db':a.snr,'checkpoint_validation':payload.get('validation'),
      'scenes':len(rows),'ADE':sum(r['ADE'] for r in rows)/len(rows),
      'FDE':sum(r['FDE'] for r in rows)/len(rows),'test_set_used':False}
    out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

