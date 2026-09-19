"""Train-only output-range audit and train/validation five-SNR token coverage."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from frontend.sind_prediction_dataset import SinDPredictionDataset
from prediction.qgnn_final.model import NonuniformTokenizer

def main():
    torch.set_num_threads(2)
    tok=NonuniformTokenizer(json.loads((ROOT/'configs/qgnn_final_tokens.json').read_text()))
    report={'test_set_used':False,'token_coverage':[],'train_CV_residual':[]}
    for split in ['train','val']:
        ds=SinDPredictionDataset(split,0.,ROOT,False)
        dt=(ds.history_timestamp[:,-1]-ds.history_timestamp[:,0])/19
        report[split+'_dt_seconds']={'min':float(dt.min()),'max':float(dt.max()),'mean':float(dt.mean())}
        for k,snr in enumerate(ds.snr_levels_db):
            hist=ds.state_hat[k,np.maximum(ds._history_indices,0)]
            hist=np.where(ds.vehicle_mask[:,None,:,None],hist,0).astype('float32')
            f,l=tok.local_deltas(torch.from_numpy(hist[...,:2]),torch.from_numpy(hist[...,2:]))
            valid=torch.from_numpy(np.broadcast_to(ds.vehicle_mask[:,None,:],f.shape).copy())
            fv,lv=f[valid],l[valid]
            ft,lt=tok.axis_tables(fv.device,fv.dtype)
            ids,weights,overflow=tok.corners(fv,lv)
            tf,tl=tok._tables(fv.device,fv.dtype)
            reconstructed=torch.stack(((tf[ids]*weights).sum(-1),(tl[ids]*weights).sum(-1)),-1)+overflow
            error=float((reconstructed-torch.stack((fv,lv),-1)).abs().max())
            coverage={'split':split,'snr_db':float(snr),'transitions':len(fv),'old_central_outside_fraction':float(((fv< -1.2)|(fv>2.2)|(lv< -1.3)|(lv>1.3)).float().mean()),'new_outer_outside_fraction':float((overflow.abs().sum(-1)>0).float().mean()),'interpolation_plus_overflow_max_error_m':error}
            report['token_coverage'].append(coverage);print('COVERAGE',json.dumps(coverage),flush=True)
            if split=='train':
                last=hist[:,-1];steps=np.arange(1,21)[None,:,None,None]*dt[:,None,None,None]
                cv=last[:,None,:,:2]+steps*last[:,None,:,2:]
                residual=ds.future[...,:2]-cv
                mask=np.broadcast_to(ds.vehicle_mask[:,None,:,None],residual.shape)
                values=np.abs(residual[mask]);frame_mask=np.broadcast_to(ds.vehicle_mask[:,None,:],residual.shape[:-1])
                floors={}
                for cap in [4.,8.,12.,16.]:
                    remaining=residual-np.clip(residual,-cap,cap)
                    distance=np.sqrt(np.square(remaining).sum(-1))*frame_mask
                    counts=ds.vehicle_mask.sum(-1).clip(1)
                    floors[str(cap)]={'ADE_lower_bound_m':float((distance.mean(1).sum(-1)/counts).mean()),'FDE_lower_bound_m':float((distance[:,-1].sum(-1)/counts).mean()),'outside_any_coordinate_fraction':float((np.abs(residual).max(-1)[frame_mask]>cap).mean())}
                record={'snr_db':float(snr),'abs_coordinate_residual_quantiles':np.quantile(values,[.5,.95,.99,.995,.999,1.]).tolist(),'hard_cap_floors':floors}
                report['train_CV_residual'].append(record);print('CAP_AUDIT',json.dumps(record),flush=True)
            dest=ROOT/'reports/qgnn/interface_audit.json';temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(dest)
if __name__=='__main__': main()
