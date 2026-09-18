#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/sind'
BUILD=json.loads((ROOT/'reports/sind/sind_dataset_build.json').read_text())
OUT=ROOT/'reports/sind/sind_dataset_validation.json'
H=20; F=20; TOTAL=40; MAXN=8

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

checks=[]
def ck(name,ok,detail=None):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})
    if not ok: print('FAIL',name,detail,flush=True)

def relation_matrix(S):
    dp=S[None,:,:2]-S[:,None,:2]
    dv=S[None,:,2:]-S[:,None,2:]
    dist=np.linalg.norm(dp,axis=2)
    closing=np.maximum(0.0,-np.sum(dp*dv,axis=2)/np.maximum(dist,1e-9))
    vv=np.sum(dv*dv,axis=2)
    tcpa=np.where(vv>1e-9,-np.sum(dp*dv,axis=2)/np.maximum(vv,1e-9),np.inf)
    valid_cpa=(tcpa>0)&(tcpa<=4)
    safe_t=np.where(valid_cpa,tcpa,0.0)
    dc_calc=np.linalg.norm(dp+dv*safe_t[:,:,None],axis=2)
    dcpa=np.where(valid_cpa,dc_calc,np.inf)
    active=(dist<=30)&((closing>.5)|(valid_cpa&(dcpa<=10)))
    np.fill_diagonal(active,False)
    return dist,closing,tcpa,dcpa,active
def recompute_selection(ids,S):
    dist,closing,tcpa,dcpa,active=relation_matrix(S)
    degree=active.sum(1)
    cand=np.flatnonzero(degree>=2)
    if len(cand)==0: return None
    close20=(dist<=20).sum(1)-1
    closing_mass=(closing*active).sum(1)
    masked=np.where(active,dcpa,np.inf)
    min_dcpa=masked.min(1)
    focal=min(cand.tolist(),key=lambda i:(-int(degree[i]),-int(close20[i]),
                                          -float(closing_mass[i]),float(min_dcpa[i]),int(ids[i])))
    if len(ids)<=MAXN:
        chosen=np.arange(len(ids),dtype=int); mode=0
    else:
        order=[]
        for j in range(len(ids)):
            if j==focal: continue
            order.append((0 if active[focal,j] else 1,
                          float(dcpa[focal,j]) if np.isfinite(dcpa[focal,j]) else 1e9,
                          -float(closing[focal,j]),float(dist[focal,j]),int(ids[j]),j))
        chosen=np.asarray([focal]+[x[-1] for x in sorted(order)[:MAXN-1]],dtype=int)
        mode=1
    chosen=np.asarray(sorted(chosen.tolist(),key=lambda i:int(ids[i])),dtype=int)
    return ids[chosen].astype(np.int32),mode,int(ids[focal])

def build_grid(df):
    fmax=int(df.frame_id.max()); vmax=int(df.vehicle_id.max())
    grid=np.full((fmax+1,vmax+1,4),np.nan,np.float32)
    fr=df.frame_id.to_numpy(np.int32); vi=df.vehicle_id.to_numpy(np.int32)
    grid[fr,vi,0]=df.x.to_numpy(np.float32)
    grid[fr,vi,1]=df.y.to_numpy(np.float32)
    grid[fr,vi,2]=df.vx.to_numpy(np.float32)
    grid[fr,vi,3]=df.vy.to_numpy(np.float32)
    clock=np.full(fmax+1,np.nan,np.float64)
    fc=df[['frame_id','timestamp']].drop_duplicates().sort_values('frame_id')
    clock[fc.frame_id.to_numpy(np.int32)]=fc.timestamp.to_numpy(np.float64)
    spans=df.groupby('vehicle_id').frame_id.agg(['min','max'])
    return grid,clock,spans
split_agents={}; split_keys={}
for split in ['train','val','test']:
    z=np.load(DATA/'splits'/split/'samples.npz',allow_pickle=False)
    req={'history','future','vehicle_ids','vehicle_mask','num_vehicles','scene_id',
         'start_frame','start_timestamp','history_timestamp','original_num_vehicles',
         'selection_mode','focal_vehicle_id'}
    ck(f'{split}.keys',set(z.files)==req,sorted(z.files))
    n=len(z['history'])
    ck(f'{split}.shapes',z['history'].shape==(n,20,8,4) and
       z['future'].shape==(n,20,8,4) and z['history_timestamp'].shape==(n,20))
    ck(f'{split}.finite',np.isfinite(z['history']).all() and
       np.isfinite(z['future']).all() and np.isfinite(z['history_timestamp']).all())
    mask=z['vehicle_mask']; nums=z['num_vehicles'].astype(int)
    ck(f'{split}.mask_count',np.array_equal(mask.sum(1),nums))
    ck(f'{split}.n_bounds',bool(((nums>=2)&(nums<=8)).all()))
    pad=~mask
    ck(f'{split}.padding_zero',bool(np.all(z['history'][pad[:,None,:].repeat(20,1)]==0.0) and
                                    np.all(z['future'][pad[:,None,:].repeat(20,1)]==0.0)))
    ck(f'{split}.id_padding',bool((z['vehicle_ids'][pad]==-1).all()))
    dt=np.diff(z['history_timestamp'],axis=1)
    ck(f'{split}.time_increasing',bool((dt>0).all()))
    ck(f'{split}.nominal_10hz',bool((np.abs(dt-.1)<=.005).all()),
       {'median':float(np.median(dt)),'min':float(dt.min()),'max':float(dt.max())})
    ck(f'{split}.start_ts',bool(np.array_equal(z['start_timestamp'],z['history_timestamp'][:,0])))
    keys=set(zip(z['scene_id'].astype(int).tolist(),z['start_frame'].astype(int).tolist()))
    ck(f'{split}.sample_key_unique',len(keys)==n)
    split_keys[split]=keys
    agents=set()
    for sc,ids,m in zip(z['scene_id'],z['vehicle_ids'],mask):
        agents.update((int(sc),int(v)) for v in ids[m])
    split_agents[split]=agents

    df=pd.read_csv(DATA/'splits'/split/'trajectories.csv')
    max_state=0.0; max_time=0.0; selection_bad=0; focal_bad=0; orig_bad=0
    for sid in sorted(df.scene_id.unique()):
        sdf=df[df.scene_id==sid]
        grid,clock,spans=build_grid(sdf)
        ix=np.flatnonzero(z['scene_id'].astype(int)==int(sid))
        if len(ix)==0: continue
        starts=z['start_frame'][ix].astype(int)
        ids=z['vehicle_ids'][ix].astype(int)
        masks=mask[ix]
        safe=np.where(masks,ids,0)
        hfr=starts[:,None]+np.arange(H)[None,:]
        ffr=starts[:,None]+H+np.arange(F)[None,:]
        exp_h=grid[hfr[:,:,None],safe[:,None,:],:]
        exp_f=grid[ffr[:,:,None],safe[:,None,:],:]
        active4=masks[:,None,:,None]
        max_state=max(max_state,float(np.nanmax(np.abs(np.where(active4,exp_h,0.0)-z['history'][ix]))))
        max_state=max(max_state,float(np.nanmax(np.abs(np.where(active4,exp_f,0.0)-z['future'][ix]))))
        exp_ts=clock[hfr]
        max_time=max(max_time,float(np.nanmax(np.abs(exp_ts-z['history_timestamp'][ix]))))

        guard=set(BUILD['split_boundaries'][str(int(sid))]['leakage_guard_vehicle_ids'])
        span_ids=spans.index.to_numpy(int); mn=spans['min'].to_numpy(int); mx=spans['max'].to_numpy(int)
        for local,k in enumerate(ix):
            st=int(z['start_frame'][k])
            elig=span_ids[(mn<=st)&(mx>=st+TOTAL-1)]
            elig=np.asarray([v for v in elig if int(v) not in guard],dtype=int)
            orig_bad += int(len(elig)!=int(z['original_num_vehicles'][k]))
            S=grid[st+H-1,elig,:].astype(float)
            recomputed=recompute_selection(elig,S)
            if recomputed is None:
                selection_bad+=1; continue
            chosen,mode,focal=recomputed
            got=z['vehicle_ids'][k][z['vehicle_mask'][k]].astype(np.int32)
            selection_bad += int(not np.array_equal(chosen,got) or mode!=int(z['selection_mode'][k]))
            focal_bad += int(focal!=int(z['focal_vehicle_id'][k]))
    ck(f'{split}.source_state_exact',max_state==0.0,max_state)
    ck(f'{split}.source_timestamp_exact',max_time<1e-12,max_time)
    ck(f'{split}.original_n_exact',orig_bad==0,orig_bad)
    ck(f'{split}.selection_exact',selection_bad==0,selection_bad)
    ck(f'{split}.focal_exact',focal_bad==0,focal_bad)
    print(split,'validated',n,flush=True)
for a,b in [('train','val'),('train','test'),('val','test')]:
    ck(f'agent_leakage_{a}_{b}',len(split_agents[a]&split_agents[b])==0,len(split_agents[a]&split_agents[b]))
    ck(f'sample_key_overlap_{a}_{b}',len(split_keys[a]&split_keys[b])==0,len(split_keys[a]&split_keys[b]))

for sid,s in BUILD['scenes'].items():
    ck(f'source_hash_scene{sid}',sha256(ROOT/s['source_file'])==s['sha256'])

status=all(x['pass'] for x in checks)
summary={
    'checks':len(checks),
    'passed':sum(x['pass'] for x in checks),
    'sample_counts':{s:BUILD['splits'][s]['samples'] for s in ['train','val','test']},
    'agent_overlap':{a+'_'+b:len(split_agents[a]&split_agents[b])
                     for a,b in [('train','val'),('train','test'),('val','test')]}
}
payload={'status':'PASS' if status else 'FAIL','checks':checks,'summary':summary}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
print(payload['status'],flush=True)
if not status: raise SystemExit(2)
