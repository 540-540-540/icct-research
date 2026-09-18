#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/'data/sind/splits/train/trajectories.csv'
OUT=ROOT/'reports/sind/sind_geometry_search.json'
RMIN=5.0; RMAX=150.0; HALF=math.radians(70.0); HEIGHT=5.0

def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi

def visibility(points,stations,bores):
    delta=points[:,None,:]-stations[None,:,:]
    rho=np.linalg.norm(delta,axis=2)
    r=np.sqrt(rho*rho+HEIGHT*HEIGHT)
    ang=np.arctan2(delta[:,:,1],delta[:,:,0])
    rel=np.vectorize(wrap)(ang-bores[None,:])
    return (r>=RMIN)&(r<=RMAX)&(np.abs(rel)<=HALF)

def velocity_geometry(points,stations,vis):
    vals=[]
    for p,row in zip(points,vis):
        idx=np.nonzero(row)[0]
        if len(idx)<2:
            vals.append(0.0); continue
        dirs=[]
        for k in idx:
            d=p-stations[k]; rr=math.sqrt(float(d@d)+HEIGHT*HEIGHT)
            dirs.append(d/rr)
        A=np.asarray(dirs)
        s=np.linalg.svd(A,compute_uv=False)
        vals.append(float(s[-1]/max(s[0],1e-12)))
    return np.asarray(vals)
d=pd.read_csv(TRAIN,usecols=['scene_id','frame_id','x','y'])
res={}
for sid,g in d.groupby('scene_id'):
    # geometry calibration uses deterministic spatial subsampling of TRAIN only
    q=g.iloc[::max(1,len(g)//30000)][['x','y']].to_numpy(float)
    lo=np.quantile(q,.02,axis=0); hi=np.quantile(q,.98,axis=0)
    center=np.median(q,axis=0)
    span=max(float(hi[0]-lo[0]),float(hi[1]-lo[1]))
    best=None; rows=[]
    for radius in [45.,55.,65.,75.]:
        for rot in np.arange(0.,120.,10.):
            angles=np.radians(rot+np.array([0.,120.,240.]))
            stations=center[None,:]+radius*np.c_[np.cos(angles),np.sin(angles)]
            bores=np.arctan2(center[1]-stations[:,1],center[0]-stations[:,0])
            vis=visibility(q,stations,bores)
            count=vis.sum(1)
            geom=velocity_geometry(q,stations,vis)
            score=(float((count>=2).mean()),float((count>=3).mean()),
                   float(np.median(geom[count>=2])) if np.any(count>=2) else 0.,
                   -float(np.mean(np.maximum(0,2-count))))
            row={'radius':radius,'rotation_deg':float(rot),'score':score,
                 'stations':stations.tolist(),'boresights_deg':np.degrees(bores).tolist(),
                 'coverage_ge1':float((count>=1).mean()),'coverage_ge2':float((count>=2).mean()),
                 'coverage_ge3':float((count>=3).mean()),
                 'median_condition_ratio_ge2':float(np.median(geom[count>=2])) if np.any(count>=2) else 0.,
                 'min_condition_ratio_ge2':float(np.min(geom[count>=2])) if np.any(count>=2) else 0.}
            rows.append(row)
            if best is None or score>tuple(best['score']): best=row
    res[str(int(sid))]={'train_points_sampled':len(q),'center_xy_m':center.tolist(),
                        'p02_xy':lo.tolist(),'p98_xy':hi.tolist(),'best':best,
                        'candidates':rows}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps({'scope':'train_only','scenes':res},indent=2)+'\n')
print(json.dumps({k:v['best'] for k,v in res.items()},indent=2))
