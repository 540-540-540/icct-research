#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib import format as npy_format

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from frontend.controlled_isac.sind_frontend import sense_vehicle, STATE_FIELDS
from frontend.controlled_isac.automatum_measurement import setup_from_config, calibration_from_config

CONFIG=ROOT/'configs/sind_controlled_isac.json'
SNR=np.array([-10.,-5.,0.,5.,10.],dtype=np.float32)
CACHE_KEYS=['scene_id','frame','timestamp','vehicle_id','snr_levels_db','state_hat',
            'n_bs','condition_ratio','rank']

_WORKER_SETUP=None
_WORKER_CAL=None

def _worker_init(setup,cal):
    global _WORKER_SETUP,_WORKER_CAL
    _WORKER_SETUP=setup
    _WORKER_CAL=cal

def _sense_chunk(records):
    k=len(records); L=len(SNR)
    hat=np.empty((L,k,4),np.float32)
    nbs=np.empty((L,k),np.uint8)
    cond=np.empty((L,k),np.float32)
    rank=np.empty((L,k),np.uint8)
    for q,rec in enumerate(records):
        _,scene,frame,vehicle,_,x,y,vx,vy=rec
        for j,snr in enumerate(SNR.tolist()):
            e=sense_vehicle(scene,frame,vehicle,[x,y],[vx,vy],snr,_WORKER_CAL,_WORKER_SETUP)
            hat[j,q]=[e['x_hat'],e['y_hat'],e['vx_hat'],e['vy_hat']]
            nbs[j,q]=e['n_bs']; cond[j,q]=e['condition_ratio']; rank[j,q]=e['rank']
    start=int(records[0][0])
    return start,hat,nbs,cond,rank

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def write_npz(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED) as zf:
        for k in CACHE_KEYS:
            i=zipfile.ZipInfo(k+'.npy',date_time=(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_STORED
            with zf.open(i,'w') as f:npy_format.write_array(f,np.ascontiguousarray(a[k]),allow_pickle=False)
def unique_history_keys(samples):
    z=np.load(samples,allow_pickle=False)
    keys=set()
    for sc,st,ids,m in zip(z['scene_id'],z['start_frame'],z['vehicle_ids'],z['vehicle_mask']):
        for t in range(20):
            for vid in ids[m]:
                keys.add((int(sc),int(st)+t,int(vid)))
    return sorted(keys),len(z['scene_id'])

def trajectory_lookup(path):
    d=pd.read_csv(path,usecols=['scene_id','vehicle_id','frame_id','timestamp','x','y','vx','vy'])
    return {(int(r.scene_id),int(r.frame_id),int(r.vehicle_id)):
            (float(r.timestamp),float(r.x),float(r.y),float(r.vx),float(r.vy))
            for r in d.itertuples(index=False)}

def build(split,cfg,setup,cal,workers):
    base=ROOT/'data/sind/splits'/split
    samples=base/'samples.npz'; traj=base/'trajectories.csv'
    keys,nsamp=unique_history_keys(samples); lut=trajectory_lookup(traj)
    missing=[k for k in keys if k not in lut]
    if missing: raise RuntimeError(f'{split}: missing {len(missing)} history keys')
    M=len(keys); L=len(SNR)
    sc=np.empty(M,np.uint8);fr=np.empty(M,np.int32);ts=np.empty(M,np.float64);vid=np.empty(M,np.int32)
    hat=np.empty((L,M,4),np.float32);nbs=np.empty((L,M),np.uint8)
    cond=np.empty((L,M),np.float32);rank=np.empty((L,M),np.uint8)
    gt=np.empty((M,4),np.float64)
    records=[]
    for i,(scene,frame,vehicle) in enumerate(keys):
        timestamp,x,y,vx,vy=lut[(scene,frame,vehicle)]
        sc[i]=scene;fr[i]=frame;vid[i]=vehicle;ts[i]=timestamp;gt[i]=[x,y,vx,vy]
        records.append((i,scene,frame,vehicle,timestamp,x,y,vx,vy))

    workers=max(1,min(int(workers),M))
    chunk_size=max(500,min(4000,(M+workers*8-1)//(workers*8)))
    chunks=[records[i:i+chunk_size] for i in range(0,M,chunk_size)]
    print(f'{split}: {M} unique states, workers={workers}, chunks={len(chunks)}, chunk_size={chunk_size}',flush=True)
    done=0
    with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init,initargs=(setup,cal)) as pool:
        futures=[pool.submit(_sense_chunk,c) for c in chunks]
        for fut in as_completed(futures):
            start,hc,nbc,cc,rc=fut.result()
            k=hc.shape[1]
            hat[:,start:start+k]=hc
            nbs[:,start:start+k]=nbc
            cond[:,start:start+k]=cc
            rank[:,start:start+k]=rc
            done+=k
            if done==M or done//20000 != (done-k)//20000:
                print(split,done,'/',M,flush=True)
    metrics={}
    pos_rmse=[];vel_rmse=[]
    for j,snr in enumerate(SNR.tolist()):
        pe=np.linalg.norm(hat[j,:,:2].astype(float)-gt[:,:2],axis=1)
        ve=np.linalg.norm(hat[j,:,2:].astype(float)-gt[:,2:],axis=1)
        pr=float(np.sqrt(np.mean(pe**2)));vr=float(np.sqrt(np.mean(ve**2)))
        pos_rmse.append(pr);vel_rmse.append(vr)
        metrics[str(snr)]={'position_rmse_m':pr,'velocity_rmse_mps':vr,
                           'position_mae_m':float(pe.mean()),'velocity_mae_mps':float(ve.mean()),
                           'position_p95_m':float(np.quantile(pe,.95)),
                           'velocity_p95_mps':float(np.quantile(ve,.95)),
                           'n_bs_distribution':{str(k):int((nbs[j]==k).sum()) for k in [1,2,3]},
                           'rank2_fraction':float((rank[j]>=2).mean()),
                           'condition_ratio_median':float(np.median(cond[j]))}
    pos_mono=all(pos_rmse[i]>pos_rmse[i+1] for i in range(L-1))
    vel_mono=all(vel_rmse[i]>vel_rmse[i+1] for i in range(L-1))
    arrays={'scene_id':sc,'frame':fr,'timestamp':ts,'vehicle_id':vid,'snr_levels_db':SNR,
            'state_hat':hat,'n_bs':nbs,'condition_ratio':cond,'rank':rank}
    out=ROOT/'data/sind/isac'/split/'sensing_cache.npz';write_npz(out,arrays)
    manifest={'split':split,'revision':cfg['revision'],'samples':nsamp,'unique_history_states':M,
              'state_fields':list(STATE_FIELDS),'cache_sha256':sha(out),
              'samples_sha256':sha(samples),'trajectories_sha256':sha(traj),
              'snr_levels_db':SNR.tolist(),'metrics':metrics,
              'position_strict_monotonic':pos_mono,'velocity_strict_monotonic':vel_mono,
              'geometry_test_tuned':False}
    write_json(manifest,ROOT/'data/sind/isac'/split/'sensing_manifest.json')
    write_json(manifest,ROOT/'reports/sind'/f'{split}_sensing_manifest.json')
    print(split,json.dumps({'M':M,'pos':pos_rmse,'vel':vel_rmse,'mono':[pos_mono,vel_mono]}),flush=True)
    return manifest
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',default='all',choices=['all','train','val','test'])
    ap.add_argument('--workers',type=int,default=min(12,max(1,(os.cpu_count() or 2)-2)))
    args=ap.parse_args()
    cfg=json.loads(CONFIG.read_text());setup=setup_from_config(cfg);cal=calibration_from_config(cfg)
    splits=['train','val','test'] if args.split=='all' else [args.split]
    out={s:build(s,cfg,setup,cal,args.workers) for s in splits}
    write_json({'revision':cfg['revision'],'workers':args.workers,'splits':out},
               ROOT/'reports/sind/sind_isac_cache_report.json')

if __name__=='__main__': main()
