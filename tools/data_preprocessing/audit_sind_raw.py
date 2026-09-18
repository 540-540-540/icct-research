#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/sind/raw'
OUT=ROOT/'reports/sind/sind_raw_audit.json'
FILES={'changchun':RAW/'changchun/Veh_smoothed_tracks.csv',
       'xian':RAW/'xian/Veh_smoothed_tracks.csv'}
FORMAL={'car','bus','truck'}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def quant(a):
    a=np.asarray(a,float)
    return {'min':float(np.min(a)),'p50':float(np.median(a)),
            'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),
            'max':float(np.max(a))}
def audit(path):
    d=pd.read_csv(path)
    req=['track_id','frame_id','timestamp_ms','agent_type','x','y','vx','vy']
    out={'sha256':sha(path),'bytes':path.stat().st_size,'rows_all':len(d),
         'tracks_all':int(d.track_id.nunique()),'types_all':d.agent_type.value_counts().to_dict(),
         'missing_required':int(d[req].isna().sum().sum()),
         'duplicate_track_frame':int(d.duplicated(['track_id','frame_id']).sum())}
    f=d[d.agent_type.isin(FORMAL)].copy().sort_values(['track_id','frame_id'])
    out['formal']={'rows':len(f),'tracks':int(f.track_id.nunique()),
                   'types':f.agent_type.value_counts().to_dict(),
                   'x_range_m':[float(f.x.min()),float(f.x.max())],
                   'y_range_m':[float(f.y.min()),float(f.y.max())],
                   'speed_mps':quant(np.hypot(f.vx,f.vy))}
    clock=f[['frame_id','timestamp_ms']].drop_duplicates().sort_values('frame_id')
    dt=np.diff(clock.timestamp_ms.to_numpy(float))/1000.0
    out['timebase']={'frame_min':int(clock.frame_id.min()),'frame_max':int(clock.frame_id.max()),
                     'frames':len(clock),'median_dt_s':float(np.median(dt)),
                     'min_dt_s':float(dt.min()),'max_dt_s':float(dt.max()),
                     'nominal_hz':10.0,'resampled':False}
    g=f.groupby('track_id').agg(first=('frame_id','min'),last=('frame_id','max'),
                                n=('frame_id','size'))
    contiguous=(g['last']-g['first']+1==g['n'])
    duration=(g['n']-1)*np.median(dt)
    out['formal']['track_frames']=quant(g['n'])
    out['formal']['track_duration_s']=quant(duration)
    out['formal']['noncontiguous_tracks']=int((~contiguous).sum())
    step_speeds=[];vel_mid=[];fd_err=[];large_jump=0
    for _,q in f.groupby('track_id'):
        if len(q)<2:continue
        q=q.sort_values('frame_id')
        t=q.timestamp_ms.to_numpy(float)/1000.0
        dd=np.diff(q[['x','y']].to_numpy(float),axis=0)
        dtt=np.diff(t)
        ss=np.linalg.norm(dd,axis=1)/dtt
        vv=q[['vx','vy']].to_numpy(float)
        vm=0.5*(vv[:-1]+vv[1:])
        fd=dd/dtt[:,None]
        err=np.linalg.norm(fd-vm,axis=1)
        step_speeds.extend(ss.tolist())
        vel_mid.extend(np.linalg.norm(vm,axis=1).tolist())
        fd_err.extend(err.tolist())
        large_jump+=int((ss>50).sum())
    out['formal']['finite_difference_step_speed_mps']=quant(step_speeds)
    out['formal']['velocity_mid_speed_mps']=quant(vel_mid)
    out['formal']['fd_vs_vmid_error_mps']=quant(fd_err)
    out['formal']['step_speed_gt50_count']=large_jump
    out['formal']['nonfinite_state_count']=int((~np.isfinite(f[['x','y','vx','vy']].to_numpy(float))).sum())
    return out

payload={'dataset':'SinD public subset raw audit',
         'formal_types':sorted(FORMAL),
         'scenes':{name:audit(path) for name,path in FILES.items()}}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
