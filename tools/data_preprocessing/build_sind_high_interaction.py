#!/usr/bin/env python3
"""Build the frozen SinD high-interaction subset for ICCT.

Sources: public SinD GitHub LFS examples, Changchun + Xi'an.
State: [x,y,vx,vy], source timestamps audited as stable nominal ~10 Hz.
Split: per-scene chronological 8:1:1 with 40-frame gaps.
Window: 20 history + 20 future. Keep only windows whose last history frame
contains a vehicle with >=2 active neighbors. N<=8 keeps all eligible vehicles;
N>8 emits exactly one history-only interaction-centered local graph.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib import format as npy_format

NOMINAL_HZ=10.0
NOMINAL_DT=1.0/NOMINAL_HZ
DT_TOLERANCE_S=0.005
CONSISTENCY_ERROR_MAX_MPS=5.0
HIST=20; FUT=20; TOTAL=40; MAXN=8
SCENES={0:'changchun',1:'xian'}
SOURCE_REPO='https://github.com/SOTIF-AVLab/SinD'
SOURCE_REPO_COMMIT='930e4dea78d924c6e9a58ff8e378331f93bba8ec'
EXPECTED_SOURCE_SHA={
    'changchun':'f3011d7dc1786f940981a9d49c7f83c7860beda9d14ed0e06c995f7b7e590692',
    'xian':'1bf5d8450577249beb9c97bc210e7510babf0796ba817f9a88c64d19a5a5e4a7',
}
FOUR_WHEEL={'car','bus','truck'}
NPZ_KEYS=['history','future','vehicle_ids','vehicle_mask','num_vehicles','scene_id',
          'start_frame','start_timestamp','history_timestamp','original_num_vehicles',
          'selection_mode','focal_vehicle_id']

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def load_raw(raw_root):
    frames=[]; meta={}
    for sid,city in SCENES.items():
        p=Path(raw_root)/city/'Veh_smoothed_tracks.csv'
        actual_sha=sha256(p)
        if actual_sha!=EXPECTED_SOURCE_SHA[city]:
            raise ValueError(f'{city}: raw SHA mismatch: {actual_sha}')
        raw=pd.read_csv(p)
        d=raw[raw.agent_type.isin(FOUR_WHEEL)].copy()
        keep=['track_id','frame_id','timestamp_ms','agent_type','x','y','vx','vy']
        d=d[keep].sort_values(['track_id','frame_id']).reset_index(drop=True)
        assert d[keep].notna().all().all()
        assert d.duplicated(['track_id','frame_id']).sum()==0
        bad_tracks=[]
        for track_id,tg in d.groupby('track_id',sort=False):
            if len(tg)<2:
                continue
            tt=tg.timestamp_ms.to_numpy(float)/1000.0
            xy=tg[['x','y']].to_numpy(float)
            vv=tg[['vx','vy']].to_numpy(float)
            dd=np.diff(tt)
            fd=np.diff(xy,axis=0)/dd[:,None]
            midpoint=.5*(vv[:-1]+vv[1:])
            err=np.linalg.norm(fd-midpoint,axis=1)
            if float(np.max(err))>CONSISTENCY_ERROR_MAX_MPS:
                bad_tracks.append(int(track_id))
        d=d[~d.track_id.isin(bad_tracks)].copy()
        d=d.sort_values(['frame_id','track_id']).reset_index(drop=True)
        frame_clock=d[['frame_id','timestamp_ms']].drop_duplicates().sort_values('frame_id')
        assert frame_clock.frame_id.is_monotonic_increasing
        ts=frame_clock.timestamp_ms.to_numpy(dtype=float)/1000.0
        dt=np.diff(ts)
        assert (dt>0).all()
        # The task is nominally 10 Hz. Preserve source timestamps when the recording
        # is already a stable ~10 Hz sequence; do not relabel it to a fixed dt.
        if np.max(np.abs(dt-NOMINAL_DT))>DT_TOLERANCE_S:
            raise ValueError(f'{city}: timestamp cadence is not safely compatible with nominal 10 Hz')
        g=d.groupby('track_id').frame_id.agg(['min','max','count'])
        assert ((g['max']-g['min']+1)==g['count']).all()
        d['original_track_id']=d.track_id.astype(np.int64)
        ids={old:i+1 for i,old in enumerate(sorted(d.track_id.unique()))}
        d['vehicle_id']=d.track_id.map(ids).astype(np.int32)
        d['scene_id']=sid
        d['timestamp']=d.timestamp_ms/1000.0
        frames.append(d[['scene_id','vehicle_id','original_track_id','frame_id','timestamp',
                         'x','y','vx','vy','agent_type']])
        meta[str(sid)]={'city':city,'source_file':str(p),'sha256':actual_sha,
                        'raw_rows_all_types':int(len(raw)),'rows':len(d),
                        'vehicles':int(d.vehicle_id.nunique()),'frame_range':[int(d.frame_id.min()),int(d.frame_id.max())],
                        'x_range':[float(d.x.min()),float(d.x.max())],'y_range':[float(d.y.min()),float(d.y.max())],
                        'types':d.agent_type.value_counts().to_dict(),
                        'quality_filter':{'rule':'exclude full track if max adjacent position-velocity consistency error > 5 m/s',
                                          'threshold_mps':CONSISTENCY_ERROR_MAX_MPS,
                                          'excluded_original_track_ids':bad_tracks,
                                          'excluded_track_count':len(bad_tracks)},
                        'timebase':{'nominal_hz':NOMINAL_HZ,'timestamp_source':'timestamp_ms',
                                    'median_dt_s':float(np.median(dt)),'min_dt_s':float(dt.min()),
                                    'max_dt_s':float(dt.max()),'resampled':False}}
    return pd.concat(frames,ignore_index=True),meta

def crossing_count(spans,g,gap=TOTAL):
    before=spans['min']<g
    after=spans['max']>=g+gap
    return int((before&after).sum())

def choose_boundaries(d):
    out={}
    for sid in sorted(d.scene_id.unique()):
        s=d[d.scene_id==sid]
        f0,f1=int(s.frame_id.min()),int(s.frame_id.max())
        spans=s.groupby('vehicle_id').frame_id.agg(['min','max'])
        F=f1-f0+1; usable=F-2*TOTAL
        train=int(round(.8*usable)); val=int(round(.1*usable))
        ng1=f0+train; ng2=ng1+TOTAL+val
        counts=s.groupby('frame_id').size().reindex(range(f0,f1+1),fill_value=0).to_numpy()
        prefix=np.concatenate([[0],np.cumsum(counts)])
        def gap_rows(g):
            a=g-f0; b=a+TOTAL
            return int(prefix[b]-prefix[a])
        cross_cache={}
        def cross(g):
            if g not in cross_cache:
                cross_cache[g]=crossing_count(spans,g)
            return cross_cache[g]
        best=None
        g1s=range(max(f0+TOTAL,ng1-300),min(f1-2*TOTAL,ng1+300)+1)
        g2_lo=max(f0+2*TOTAL,ng2-300); g2_hi=min(f1-TOTAL+1,ng2+300)
        for g1 in g1s:
            train_slots=g1-f0
            for g2 in range(max(g1+2*TOTAL,g2_lo),g2_hi+1):
                slots=(train_slots,g2-(g1+TOTAL),f1-(g2+TOTAL)+1)
                ratios=np.asarray(slots,dtype=float)/usable
                if np.max(np.abs(ratios-np.array([.8,.1,.1])))>.02:
                    continue
                key=(cross(g1)+cross(g2),gap_rows(g1)+gap_rows(g2),
                     abs(g1-ng1)+abs(g2-ng2),g1,g2)
                if best is None or key<best[0]:
                    best=(key,g1,g2,slots,ratios)
        if best is None:
            raise RuntimeError(f'no split for scene {sid}')
        _,g1,g2,slots,ratios=best
        guard1=set(spans.index[(spans['min']<g1)&(spans['max']>=g1+TOTAL)].astype(int).tolist())
        guard2=set(spans.index[(spans['min']<g2)&(spans['max']>=g2+TOTAL)].astype(int).tolist())
        leakage_guard=sorted(guard1|guard2)
        out[int(sid)]={'f0':f0,'f1':f1,'g1':g1,'g2':g2,'slots':list(map(int,slots)),
                       'ratios':ratios.tolist(),'crossing_total':best[0][0],
                       'gap_rows_total':best[0][1],
                       'leakage_guard_vehicle_ids':leakage_guard,
                       'leakage_guard_count':len(leakage_guard)}
    return out
def pair_features(a,b):
    dp=b[:2]-a[:2]; dv=b[2:]-a[2:]
    dist=float(np.linalg.norm(dp))
    closing=max(0.0,-float(dp@dv)/max(dist,1e-9))
    vv=float(dv@dv)
    if vv>1e-9:
        tc=-float(dp@dv)/vv
        dc=float(np.linalg.norm(dp+dv*tc)) if 0<tc<=4 else float('inf')
    else:
        tc,dc=float('inf'),float('inf')
    return dist,closing,tc,dc

def active_relation(feature):
    dist,cl,t,dc=feature
    return dist<=30 and (cl>.5 or (0<t<=4 and dc<=10))

def select_nodes(ids,states):
    """Vectorized history-only selector; scientific ordering is unchanged."""
    n=len(ids)
    pos=states[:,:2].astype(float,copy=False)
    vel=states[:,2:].astype(float,copy=False)
    dp=pos[None,:,:]-pos[:,None,:]
    dv=vel[None,:,:]-vel[:,None,:]
    dist=np.linalg.norm(dp,axis=2)
    dot=np.sum(dp*dv,axis=2)
    closing=np.maximum(0.0,-dot/np.maximum(dist,1e-9))
    vv=np.sum(dv*dv,axis=2)
    tcpa=np.where(vv>1e-9,-dot/np.maximum(vv,1e-9),np.inf)
    valid_cpa=(tcpa>0)&(tcpa<=4)
    safe_t=np.where(valid_cpa,tcpa,0.0)
    dc_calc=np.linalg.norm(dp+dv*safe_t[:,:,None],axis=2)
    dcpa=np.where(valid_cpa,dc_calc,np.inf)

    active=(dist<=30)&((closing>.5)|(valid_cpa&(dcpa<=10)))
    np.fill_diagonal(active,False)
    active_degree=active.sum(axis=1)
    close20=(dist<=20).sum(axis=1)-1
    closing_mass=(closing*active).sum(axis=1)
    min_dcpa=np.where(active,dcpa,np.inf).min(axis=1)

    candidates=np.flatnonzero(active_degree>=2)
    if candidates.size==0:
        return None
    focal=min(candidates.tolist(),key=lambda i:(-int(active_degree[i]),-int(close20[i]),
                                                -float(closing_mass[i]),float(min_dcpa[i]),int(ids[i])))
    if n<=MAXN:
        chosen=np.arange(n,dtype=int)
        mode=0
    else:
        ranked=[]
        for j in range(n):
            if j==focal:
                continue
            ranked.append((0 if active[focal,j] else 1,
                           float(dcpa[focal,j]) if np.isfinite(dcpa[focal,j]) else 1e9,
                           -float(closing[focal,j]),float(dist[focal,j]),int(ids[j]),j))
        chosen=np.asarray([focal]+[x[-1] for x in sorted(ranked)[:MAXN-1]],dtype=int)
        mode=1
    chosen=np.asarray(sorted(chosen.tolist(),key=lambda i:int(ids[i])),dtype=int)
    return chosen,mode,int(ids[focal])

def split_of(frame,b):
    if b['f0']<=frame<b['g1']:
        return 'train'
    if b['g1']+TOTAL<=frame<b['g2']:
        return 'val'
    if b['g2']+TOTAL<=frame<=b['f1']:
        return 'test'
    return None
def build_samples(d,bounds):
    outputs={s:{k:[] for k in NPZ_KEYS} for s in ['train','val','test']}
    traj={s:[] for s in ['train','val','test']}
    for sid,b in bounds.items():
        s=d[d.scene_id==sid].copy()
        leakage_guard=set(int(x) for x in b['leakage_guard_vehicle_ids'])
        masks={
            'train':(s.frame_id>=b['f0'])&(s.frame_id<b['g1']),
            'val':(s.frame_id>=b['g1']+TOTAL)&(s.frame_id<b['g2']),
            'test':(s.frame_id>=b['g2']+TOTAL)&(s.frame_id<=b['f1'])}
        for split,mask in masks.items():
            kept=mask & (~s.vehicle_id.isin(leakage_guard))
            traj[split].append(s[kept])

        f0,f1=int(s.frame_id.min()),int(s.frame_id.max())
        vmax=int(s.vehicle_id.max())
        states=np.full((f1+1,vmax+1,4),np.nan,dtype=np.float32)
        fr=s.frame_id.to_numpy(np.int32)
        vi=s.vehicle_id.to_numpy(np.int32)
        states[fr,vi,0]=s.x.to_numpy(np.float32)
        states[fr,vi,1]=s.y.to_numpy(np.float32)
        states[fr,vi,2]=s.vx.to_numpy(np.float32)
        states[fr,vi,3]=s.vy.to_numpy(np.float32)

        first=np.full(vmax+1,f1+1,dtype=np.int32)
        last=np.full(vmax+1,-1,dtype=np.int32)
        spans=s.groupby('vehicle_id').frame_id.agg(['min','max'])
        span_ids=spans.index.to_numpy(np.int32)
        first[span_ids]=spans['min'].to_numpy(np.int32)
        last[span_ids]=spans['max'].to_numpy(np.int32)
        valid_vehicle=np.zeros(vmax+1,dtype=bool)
        valid_vehicle[span_ids]=True
        if leakage_guard:
            valid_vehicle[np.asarray(sorted(leakage_guard),dtype=np.int32)]=False

        frame_clock=np.full(f1+1,np.nan,dtype=np.float64)
        fc=s[['frame_id','timestamp']].drop_duplicates().sort_values('frame_id')
        frame_clock[fc.frame_id.to_numpy(np.int32)]=fc.timestamp.to_numpy(np.float64)
        assert np.isfinite(frame_clock[f0:f1+1]).all()

        accepted=0
        for f in range(b['f0'],b['f1']-TOTAL+2):
            split=split_of(f,b)
            if split is None or split_of(f+TOTAL-1,b)!=split:
                continue
            eligible=valid_vehicle & (first<=f) & (last>=f+TOTAL-1)
            ids=np.flatnonzero(eligible).astype(np.int32)
            if ids.size<2:
                continue

            end_states=states[f+HIST-1,ids,:]
            selected=select_nodes(ids,end_states)
            if selected is None:
                continue
            pick,mode,focal=selected
            selected_ids=ids[pick]
            n=len(selected_ids)
            q=states[f:f+TOTAL][:,selected_ids,:]
            if q.shape!=(TOTAL,n,4) or not np.isfinite(q).all():
                raise RuntimeError(f'scene {sid} frame {f}: invalid dense state slice {q.shape}')

            h=np.zeros((HIST,MAXN,4),np.float32)
            u=np.zeros((FUT,MAXN,4),np.float32)
            h[:,:n]=q[:HIST]
            u[:,:n]=q[HIST:]
            vids=np.full(MAXN,-1,np.int32); vids[:n]=selected_ids
            vm=np.zeros(MAXN,bool); vm[:n]=True
            history_ts=frame_clock[f:f+HIST].copy()
            vals={'history':h,'future':u,'vehicle_ids':vids,'vehicle_mask':vm,
                  'num_vehicles':n,'scene_id':sid,'start_frame':f,
                  'start_timestamp':float(history_ts[0]),'history_timestamp':history_ts,
                  'original_num_vehicles':int(ids.size),
                  'selection_mode':mode,'focal_vehicle_id':focal}
            for k,v in vals.items():
                outputs[split][k].append(v)
            accepted+=1
            if accepted%5000==0:
                print(f'scene {sid}: accepted {accepted} high-interaction windows',flush=True)
        print(f'scene {sid}: dense-array build accepted {accepted} windows',flush=True)
    return outputs,traj
def stack_payload(buf):
    spec={
        'history':((0,HIST,MAXN,4),np.float32),
        'future':((0,FUT,MAXN,4),np.float32),
        'vehicle_ids':((0,MAXN),np.int32),
        'vehicle_mask':((0,MAXN),np.bool_),
        'num_vehicles':((0,),np.uint8),
        'scene_id':((0,),np.uint8),
        'start_frame':((0,),np.int32),
        'start_timestamp':((0,),np.float64),
        'history_timestamp':((0,HIST),np.float64),
        'original_num_vehicles':((0,),np.uint8),
        'selection_mode':((0,),np.uint8),
        'focal_vehicle_id':((0,),np.int32)}
    out={}
    for k,(shape,dtype) in spec.items():
        if not buf[k]:
            out[k]=np.zeros(shape,dtype=dtype)
        elif k in ['history','future','vehicle_ids','vehicle_mask','history_timestamp']:
            out[k]=np.stack(buf[k]).astype(dtype)
        else:
            out[k]=np.asarray(buf[k],dtype=dtype)
    return out

def write_npz(path,arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as zf:
        for k in NPZ_KEYS:
            info=zipfile.ZipInfo(k+'.npy',date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_STORED
            with zf.open(info,'w') as f:
                npy_format.write_array(f,np.ascontiguousarray(arrays[k]),allow_pickle=False)
def interaction_stats(arr):
    W=len(arr['history'])
    out={'windows':W,'n_dist':{},'original_n_gt8':int((arr['original_num_vehicles']>8).sum())}
    n=arr['num_vehicles']
    out['n_dist']={str(k):int((n==k).sum()) for k in range(2,9)}
    out['n_ge_fraction']={str(k):float((n>=k).mean()) for k in [4,5,6]}
    pair_total=0
    vals={r:0 for r in [10,15,20,25,30,45]}
    close={c:0 for c in [0,.5,1.]}
    cpaw={d:0 for d in [2,5,10]}
    active={k:0 for k in [2,3,4]}
    targets=0
    for h,nn in zip(arr['history'],n):
        s=h[-1,:int(nn)].astype(float)
        targets+=len(s); deg=np.zeros(len(s),int)
        anyc={d:False for d in cpaw}
        for i in range(len(s)):
            for j in range(i+1,len(s)):
                dist,cl,t,dc=pair_features(s[i],s[j]); pair_total+=1
                for r in vals:
                    if dist<=r: vals[r]+=1
                for c in close:
                    if cl>c: close[c]+=1
                if active_relation((dist,cl,t,dc)):
                    deg[i]+=1; deg[j]+=1
                if t<=4:
                    for dd in cpaw:
                        if dc<=dd: anyc[dd]=True
        for k in active:
            active[k]+=int((deg>=k).sum())
        for dd in cpaw:
            cpaw[dd]+=int(anyc[dd])
    out['pair_total']=pair_total
    out['radius_pairs_per_window']={str(k):v/W for k,v in vals.items()}
    out['closing_pairs_per_window']={str(k):v/W for k,v in close.items()}
    out['cpa_window_fraction']={str(k):v/W for k,v in cpaw.items()}
    out['active_neighbor_target_fraction']={str(k):v/targets for k,v in active.items()}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-root',default='data/sind/raw')
    ap.add_argument('--out-root',default='data/sind')
    ap.add_argument('--report',default='reports/sind/sind_dataset_build.json')
    ap.add_argument('--force',action='store_true',
                    help='rebuild even when the completion marker already exists')
    args=ap.parse_args()

    root=Path(args.out_root)
    root.mkdir(parents=True,exist_ok=True)
    done_marker=root/'.BUILD_COMPLETE'
    if done_marker.exists() and not args.force:
        print(f'SKIP: frozen build already complete at {done_marker}')
        return
    lock_handle=(root/'.build.lock').open('w')
    try:
        fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another SinD builder owns data/sind/.build.lock')
    if args.force:
        done_marker.unlink(missing_ok=True)

    print('STAGE load_raw',flush=True)
    d,source_meta=load_raw(args.raw_root)
    print('STAGE choose_boundaries',flush=True)
    bounds=choose_boundaries(d)
    print('STAGE build_samples',flush=True)
    outputs,traj=build_samples(d,bounds)
    print('STAGE build_samples DONE', {k:len(v['history']) for k,v in outputs.items()}, flush=True)
    root=Path(args.out_root)
    mapping=(d[['scene_id','vehicle_id','original_track_id','agent_type']]
             .drop_duplicates().sort_values(['scene_id','vehicle_id']))
    mapping_path=root/'vehicle_id_mapping.csv'
    mapping.to_csv(mapping_path,index=False)
    stats={'revision':'SIND-HIGH-INTERACTION-V1',
           'source':{'repository':SOURCE_REPO,'commit':SOURCE_REPO_COMMIT,
                     'raw_sha256':EXPECTED_SOURCE_SHA},
           'vehicle_id_mapping':{'path':'data/sind/vehicle_id_mapping.csv',
                                 'rows':int(len(mapping)),'sha256':sha256(mapping_path)},
           'timebase':{'nominal_hz':NOMINAL_HZ,'nominal_dt_s':NOMINAL_DT,
                       'rule':'preserve audited source timestamp_ms; no fixed-dt relabeling'},
           'scenes':source_meta,'split_boundaries':bounds,'splits':{}}
    for split in ['train','val','test']:
        print('STAGE write',split,flush=True)
        td=pd.concat(traj[split],ignore_index=True).sort_values(['scene_id','frame_id','vehicle_id'])
        tp=root/'splits'/split/'trajectories.csv'
        tp.parent.mkdir(parents=True,exist_ok=True)
        cols=['scene_id','vehicle_id','frame_id','timestamp','x','y','vx','vy','agent_type']
        td[cols].to_csv(tp,index=False)
        arr=stack_payload(outputs[split])
        sp=root/'splits'/split/'samples.npz'
        write_npz(sp,arr)
        stats['splits'][split]={
            'trajectories_rows':len(td),
            'samples':len(arr['history']),
            'trajectories_sha256':sha256(tp),
            'samples_sha256':sha256(sp),
            'interaction':interaction_stats(arr)}
        print(split,stats['splits'][split]['samples'],
              stats['splits'][split]['interaction'],flush=True)
    print('STAGE write_report',flush=True)
    write_json(stats,args.report)
    done_marker.write_text(
        json.dumps({'status':'COMPLETE','report':str(args.report),
                    'report_sha256':sha256(args.report)},indent=2)+'\n',
        encoding='utf-8')
    print('WROTE',args.report)
    print('FROZEN',done_marker)

if __name__=='__main__':
    main()
