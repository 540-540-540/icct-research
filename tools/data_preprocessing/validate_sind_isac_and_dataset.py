#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from frontend.controlled_isac.sind_frontend import sense_vehicle
from frontend.controlled_isac.automatum_measurement import setup_from_config, calibration_from_config
from frontend.sind_prediction_dataset import SinDPredictionDataset

CFG=json.loads((ROOT/'configs/sind_controlled_isac.json').read_text())
SETUP=setup_from_config(CFG)
CAL=calibration_from_config(CFG)
SNR=np.array([-10.,-5.,0.,5.,10.],dtype=np.float32)
OUT=ROOT/'reports/sind/sind_isac_validation.json'
checks=[]

def ck(name,ok,detail=None):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})
    if not ok: print('FAIL',name,detail,flush=True)

def history_keys(samples):
    keys=set()
    for sc,st,ids,m in zip(samples['scene_id'],samples['start_frame'],
                           samples['vehicle_ids'],samples['vehicle_mask']):
        for t in range(20):
            for vid in ids[m]:
                keys.add((int(sc),int(st)+t,int(vid)))
    return keys
summary={}
for split in ['train','val','test']:
    cp=ROOT/f'data/sind/isac/{split}/sensing_cache.npz'
    sp=ROOT/f'data/sind/splits/{split}/samples.npz'
    tp=ROOT/f'data/sind/splits/{split}/trajectories.csv'
    c=np.load(cp,allow_pickle=False); s=np.load(sp,allow_pickle=False)
    ck(f'{split}.snr_levels',np.array_equal(c['snr_levels_db'],SNR),c['snr_levels_db'].tolist())
    M=len(c['scene_id'])
    ck(f'{split}.shape',c['state_hat'].shape==(5,M,4) and
       c['n_bs'].shape==(5,M) and c['rank'].shape==(5,M))
    ck(f'{split}.finite',np.isfinite(c['state_hat']).all() and
       np.isfinite(c['condition_ratio']).all())
    keys=list(zip(c['scene_id'].astype(int),c['frame'].astype(int),c['vehicle_id'].astype(int)))
    ck(f'{split}.cache_key_unique',len(keys)==len(set(keys)))
    needed=history_keys(s)
    ck(f'{split}.cache_coverage',set(keys)==needed,{'cache':len(keys),'needed':len(needed)})
    ck(f'{split}.n_bs_ge2',bool((c['n_bs']>=2).all()),int(c['n_bs'].min()))
    ck(f'{split}.rank2',bool((c['rank']>=2).all()),float((c['rank']>=2).mean()))

    d=pd.read_csv(tp,usecols=['scene_id','vehicle_id','frame_id','timestamp','x','y','vx','vy'])
    lut={(int(r.scene_id),int(r.frame_id),int(r.vehicle_id)):
         (float(r.timestamp),float(r.x),float(r.y),float(r.vx),float(r.vy))
         for r in d.itertuples(index=False)}
    gt=np.array([lut[k][1:] for k in keys],dtype=float)
    ts=np.array([lut[k][0] for k in keys],dtype=float)
    ck(f'{split}.timestamp_alignment',float(np.max(np.abs(ts-c['timestamp'])))<1e-12,
       float(np.max(np.abs(ts-c['timestamp']))))
    pos=[];vel=[]
    for j in range(5):
        pe=np.linalg.norm(c['state_hat'][j,:,:2].astype(float)-gt[:,:2],axis=1)
        ve=np.linalg.norm(c['state_hat'][j,:,2:].astype(float)-gt[:,2:],axis=1)
        pos.append(float(np.sqrt(np.mean(pe**2))))
        vel.append(float(np.sqrt(np.mean(ve**2))))
    ck(f'{split}.position_monotonic',all(pos[i]>pos[i+1] for i in range(4)),pos)
    ck(f'{split}.velocity_monotonic',all(vel[i]>vel[i+1] for i in range(4)),vel)

    # Deterministic frontend equality on evenly-spaced 200 states for all 5 SNRs.
    ix=np.unique(np.linspace(0,M-1,min(200,M),dtype=int))
    max_diff=0.0
    for i in ix:
        sc=int(c['scene_id'][i]);fr=int(c['frame'][i]);vid=int(c['vehicle_id'][i])
        _,x,y,vx,vy=lut[(sc,fr,vid)]
        for j,snr in enumerate(SNR.tolist()):
            e=sense_vehicle(sc,fr,vid,[x,y],[vx,vy],snr,CAL,SETUP)
            got=np.array([e['x_hat'],e['y_hat'],e['vx_hat'],e['vy_hat']],dtype=np.float32)
            max_diff=max(max_diff,float(np.max(np.abs(got-c['state_hat'][j,i]))))
    ck(f'{split}.frontend_exact',max_diff==0.0,max_diff)

    # Model-facing loader smoke at all five SNRs.
    loader_meta=[]
    reference_future=None
    for snr in SNR.tolist():
        ds=SinDPredictionDataset(split=split,snr_db=snr,root_dir=ROOT,return_tensors=False)
        ck(f'{split}.loader_len_{snr}',len(ds)==len(s['scene_id']),len(ds))
        sample_ids=np.unique(np.linspace(0,len(ds)-1,min(16,len(ds)),dtype=int))
        futures=[]
        for idx in sample_ids:
            item=ds[int(idx)]
            ck(f'{split}.loader_shape_{snr}_{idx}',
               item['history_state'].shape==(20,8,4) and
               item['future_state'].shape==(20,8,4) and
               item['vehicle_mask'].shape==(8,) and
               item['history_timestamp'].shape==(20,))
            ck(f'{split}.loader_finite_{snr}_{idx}',
               np.isfinite(item['history_state']).all() and np.isfinite(item['future_state']).all())
            ck(f'{split}.loader_padding_{snr}_{idx}',
               bool((item['history_state'][:,~item['vehicle_mask'],:]==0).all() and
                    (item['future_state'][:,~item['vehicle_mask'],:]==0).all()))
            futures.append(item['future_state'])
        futures=np.stack(futures)
        if reference_future is None: reference_future=futures
        else: ck(f'{split}.future_snr_invariant_{snr}',np.array_equal(reference_future,futures))
        loader_meta.append({'snr':snr,'samples_checked':len(sample_ids)})
    summary[split]={
        'unique_history_states':M,
        'position_rmse_m_by_snr':dict(zip(SNR.astype(float).astype(str),pos)),
        'velocity_rmse_mps_by_snr':dict(zip(SNR.astype(float).astype(str),vel)),
        'n_bs_distribution':{str(k):int((c['n_bs'][0]==k).sum()) for k in [1,2,3]},
        'rank2_fraction':float((c['rank'][0]>=2).mean()),
        'condition_ratio_median':float(np.median(c['condition_ratio'][0])),
        'loader':loader_meta,
    }

ck('geometry_search_train_only',
   json.loads((ROOT/'reports/sind/sind_geometry_search.json').read_text()).get('scope')=='train_only')
ck('config_test_not_tuned',CFG['dataset']['test_scope'].startswith('never used'))

status=all(x['pass'] for x in checks)
payload={'status':'PASS' if status else 'FAIL','checks':checks,'summary':summary}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
print(payload['status'],len(checks),sum(x['pass'] for x in checks),flush=True)
if not status: raise SystemExit(2)
