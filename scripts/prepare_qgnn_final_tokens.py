"""Fit a central-fine/tail-coarse motion vocabulary on train sensing only."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.q0.motion_token_llm import AutomatumMotionTokenizer

def main():
    torch.set_num_threads(4)
    ds=SinDPredictionDataset('train',0.,ROOT,False)
    tokenizer=AutomatumMotionTokenizer()
    indices=ds._history_indices
    valid=ds.vehicle_mask[:,None,:].repeat(19,axis=1).reshape(-1)
    values=[[],[]]; per_snr=[]
    for snr_index,snr in enumerate(ds.snr_levels_db):
        h=ds.state_hat[snr_index,np.maximum(indices,0)]
        h=np.where(ds.vehicle_mask[:,None,:,None],h,0).astype('float32')
        f,l=tokenizer.local_deltas(torch.from_numpy(h[...,:2]),torch.from_numpy(h[...,2:]))
        f=f.numpy().reshape(-1)[valid];l=l.numpy().reshape(-1)[valid]
        values[0].append(f);values[1].append(l)
        per_snr.append({'snr_db':float(snr),'transitions':len(f),'forward_quantiles':np.quantile(f,[0,.001,.01,.5,.99,.999,1]).tolist(),'lateral_quantiles':np.quantile(l,[0,.001,.01,.5,.99,.999,1]).tolist()})
        print('fit snr',snr,'transitions',len(f),flush=True)
    centers=[]
    for axis,(low,high) in enumerate([(-1.2,2.2),(-1.3,1.3)]):
        all_values=np.concatenate(values[axis]);step=(high-low)/30
        left=max(low-float(all_values.min())+step,5*step)
        right=max(float(all_values.max())-high+step,5*step)
        table=np.concatenate((low-np.geomspace(step,left,5)[::-1],np.linspace(low,high,31),high+np.geomspace(step,right,5)))
        centers.append(table.tolist())
    result={'revision':'PHQGNN-MOTION-41x41-V1','source_split':'train','snrs_db':ds.snr_levels_db.tolist(),'samples':len(ds),'central_bins_per_axis':31,'tail_bins_per_side':5,'forward_centers':centers[0],'lateral_centers':centers[1],'embedding':'bilinear 4-token interpolation + continuous delta/overflow adapter','test_set_used':False,'statistics':per_snr}
    dest=ROOT/'configs/qgnn_final_tokens.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print('saved',dest,flush=True)
if __name__=='__main__': main()
