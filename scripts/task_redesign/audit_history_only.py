#!/usr/bin/env python3
"""Isolated history-only cohort audit. No model training or future-state targets.
Run with the ICCT environment from this worktree. Public raw data are private/ignored.
Test rows are rejected by frame metadata before numerical state conversion.
"""
from __future__ import annotations
import csv, json, math, hashlib, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from frontend.controlled_isac.sind_frontend import sense_vehicle
from frontend.controlled_isac.automatum_measurement import setup_from_config,calibration_from_config
OUT=ROOT/'reports/task_redesign'
DT=100/999

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def pair_metrics(state, radius=50., dcpa_limit=5., closing_limit=.5):
    p,v=state[:,:2],state[:,2:]
    r=p[None]-p[:,None]; u=v[None]-v[:,None]
    d=np.linalg.norm(r,axis=-1); a=np.sum(u*u,axis=-1); b=np.sum(r*u,axis=-1)
    closing=-b/np.maximum(d,1e-6)
    tcpa=np.divide(-b,a,out=np.full_like(a,np.inf),where=a>1e-8)
    safe=np.where(np.isfinite(tcpa),tcpa,0)
    dcpa=np.linalg.norm(r+u*safe[...,None],axis=-1)
    cpa=(closing>=closing_limit)&(tcpa>0)&(tcpa<=40*DT)&(dcpa<=dcpa_limit)
    speed=np.linalg.norm(v,axis=-1); heading=v/np.maximum(speed[:,None],1e-6)
    lon=np.sum(r*heading[:,None],axis=-1)
    lat=np.abs(r[...,0]*heading[:,None,1]-r[...,1]*heading[:,None,0])
    aligned=heading@heading.T>=math.cos(math.pi/6)
    following=(speed[:,None]>=.5)&(speed[None]>=.5)&aligned&(lon>0)&(lat<=2.5)&(lon/np.maximum(speed[:,None],.5)<=2)
    near=(d<=radius)&~np.eye(len(state),dtype=bool)
    edge=near&(cpa|following)
    c=d*d-25; disc=b*b-a*c
    ttc=np.full_like(d,np.inf)
    valid=(a>1e-8)&(b<0)&(disc>=0)
    ttc[valid]=(-b[valid]-np.sqrt(disc[valid]))/a[valid]
    ttc[c<=0]=0; np.fill_diagonal(ttc,np.inf)
    return edge,near,cpa&near,following&near,d,closing,tcpa,dcpa,ttc

def summarise(rows):
    if not rows: return {'targets':0}
    result={'targets':len(rows),'origins':len({(r['scene'],r['t0']) for r in rows}),
       'by_scene':dict(Counter(str(r['scene']) for r in rows)),
       'interacting_neighbor_count':dict(Counter(str(r['k']) for r in rows)),
       'context_neighbor_count':dict(Counter(str(r['n50']) for r in rows)),
       'critical_targets':sum(r['k']>=1 for r in rows),
       'critical_origins':len({(r['scene'],r['t0']) for r in rows if r['k']>=1}),
       'multi2_targets':sum(r['k']>=2 for r in rows),
       'cpa_targets':sum(r['cpa']>0 for r in rows),'following_targets':sum(r['following']>0 for r in rows),
       'no_future_observations':sum(r['future_points']==0 for r in rows),
       'endpoint_observed':sum(r['endpoint'] for r in rows),
       'critical_endpoint_observed':sum(r['endpoint'] and r['k']>=1 for r in rows),
       'critical_time_blocks_30s':len({(r['scene'],int(r['t0']*DT//30)) for r in rows if r['k']>=1}),
       'critical_tracks':len({(r['scene'],r['id']) for r in rows if r['k']>=1})}
    for key in ['n_active','n50','min_dist','max_closing','min_dcpa','min_tcpa','min_ttc']:
        a=np.array([r[key] for r in rows]); a=a[np.isfinite(a)]
        result[key+'_quantiles']={str(q):float(np.quantile(a,q)) for q in [0,.1,.5,.9,1]} if len(a) else None
    return result

def main():
    start=time.time(); cfg=json.loads((ROOT/'configs/sind_controlled_isac.json').read_text())
    setup=setup_from_config(cfg); cal=calibration_from_config(cfg)
    old=json.loads((ROOT/'reports/sind/sind_dataset_build.json').read_text())
    prereg=json.loads((OUT/'design_preregistration_20260920.json').read_text())
    rows_all=[]; details={}; sens={}; sources={}
    for sid,city in [(0,'changchun'),(1,'xian')]:
        bound=old['split_boundaries'][str(sid)]; g1,g2=bound['g1'],bound['g2']
        p=ROOT/'data/task_redesign/raw'/city/'Veh_smoothed_tracks.csv'
        actual=sha(p); expected=old['source']['raw_sha256'][city]; assert actual==expected
        sources[city]={'sha256':actual,'bytes':p.stat().st_size}
        meta={}; records={'train':{},'val':{}}; skipped_test=0
        with p.open(newline='') as f:
            rd=csv.reader(f); cols=next(rd); ix={c:i for i,c in enumerate(cols)}
            for z in rd:
                if z[ix['agent_type']] not in {'car','truck','bus'}: continue
                ident=int(z[ix['track_id']]); frame=int(z[ix['frame_id']])
                m=meta.setdefault(ident,[frame,frame,set()]); m[0]=min(m[0],frame);m[1]=max(m[1],frame)
                split='train' if frame<g1 else ('val' if g1+60<=frame<g2 else ('test' if frame>=g2+60 else 'guard'))
                m[2].add(split)
                if split not in records:
                    skipped_test+=int(split=='test'); continue
                # Only train/val history-source states are converted. No future labels are assembled.
                state=[float(z[ix[k]]) for k in ['x','y','vx','vy']]
                records[split].setdefault(ident,[]).append((frame,float(z[ix['timestamp_ms']])/1000,state))
        bad=old['scenes'][str(sid)]['quality_filter']['excluded_original_track_ids']
        legacy_ids=sorted(set(meta)-set(bad)); ids={k:i+1 for i,k in enumerate(legacy_ids)}
        for k in bad:
            if k in meta: ids[k]=len(ids)+1
        purge={k for k,m in meta.items() if len(m[2]&{'train','val','test'})>1}
        details[city]={'bounds':{'train':[0,g1-1],'val':[g1+60,g2-1]},'purged_original_ids':sorted(purge),'test_rows_state_not_parsed':skipped_test,'new_history_quality_policy':'finite complete history only; no full-track dynamics filters','retained_previously_bad_ids':[k for k in bad if k not in purge]}
        for split,data in records.items():
            lo,hi=details[city]['bounds'][split]; tracks={}
            for ident,rr in data.items():
                if ident in purge: continue
                rr.sort(key=lambda x:x[0]); tracks[ident]=(np.array([r[0] for r in rr]),np.array([r[1] for r in rr]),np.array([r[2] for r in rr]))
            stage=[]; nbscount=Counter(); seconds=[]
            for t0 in range(19,hi-19,10):
                if t0-19<lo: continue
                hh=[]; kept=[]
                for ident,(fr,ts,states) in tracks.items():
                    q=np.searchsorted(fr,t0-19); end=q+20
                    if end>len(fr) or fr[q]!=t0-19 or fr[end-1]!=t0: continue
                    h=states[q:end]
                    if not np.isfinite(h).all() or not np.all(np.diff(fr[q:end])==1): continue
                    kept.append(ident); hh.append(h); seconds.extend(np.diff(ts[q:end]).tolist())
                if not kept: continue
                observed=[]
                for ident,h in zip(kept,hh):
                    e=sense_vehicle(sid,t0,ids[ident],h[-1,:2],h[-1,2:],0.,cal,setup)
                    observed.append([e[k] for k in ['x_hat','y_hat','vx_hat','vy_hat']]); nbscount[e['n_bs']]+=1
                observed=np.array(observed); edge,near,cpa,fol,d,cl,tc,dc,ttc=pair_metrics(observed)
                np.fill_diagonal(d,np.inf)
                for i,ident in enumerate(kept):
                    dd=near[i]; r={'scene':sid,'split':split,'t0':t0,'id':ident,'n_active':len(kept),'n50':int(dd.sum()),'k':int(edge[i].sum()),'cpa':int(cpa[i].sum()),'following':int(fol[i].sum()),'min_dist':float(d[i].min()),'max_closing':float(cl[i,dd].max()) if dd.any() else 0.,'min_dcpa':float(dc[i,dd&(tc[i]>0)&(tc[i]<=40*DT)].min()) if np.any(dd&(tc[i]>0)&(tc[i]<=40*DT)) else float('inf'),'min_tcpa':float(tc[i,dd&(tc[i]>0)].min()) if np.any(dd&(tc[i]>0)) else float('inf'),'min_ttc':float(ttc[i,dd].min()) if dd.any() else float('inf')}
                    fr=tracks[ident][0]
                    for horizon in [20,30,40]:
                        if t0+horizon>hi: continue
                        count=int(np.searchsorted(fr,t0+horizon,side='right')-np.searchsorted(fr,t0,side='right'))
                        endpoint=bool(np.any(fr==t0+horizon))
                        stage.append(dict(r,horizon=horizon,future_points=count,endpoint=endpoint))
                if split=='train' and t0+40<=hi:
                    for key,rad,dcv,clv in [('base',50,5,.5),('D4',50,4,.5),('D6',50,6,.5),('R40',40,5,.5),('R60',60,5,.5),('C03',50,5,.3),('C07',50,5,.7)]:
                        ee=pair_metrics(observed,rad,dcv,clv)[0]; box=sens.setdefault(key,{'targets':0,'critical':0,'multi2':0}); box['targets']+=len(kept);box['critical']+=int((ee.sum(1)>=1).sum());box['multi2']+=int((ee.sum(1)>=2).sum())
            rows_all.extend(stage)
            details[city][split]={'tracks':len(tracks),'sensed_origin_states_by_n_bs':dict(nbscount),'median_source_dt_s':float(np.median(seconds)) if seconds else None}
            print(city,split,'completed',len(stage),'horizon-target records',flush=True)
    summary={'schema':'history_only_task_audit_v1','base_commit':prereg['base_commit'],'prereg_sha256':sha(OUT/'design_preregistration_20260920.json'),'raw_sources':sources,'details':details,'split_horizon':{},'train_only_sensitivity':sens,'scope':{'formal_test_state_parsed':False,'formal_test_labels_opened':False,'formal_validation_future_state_targets_built':False,'any_future_state_targets_built':False,'future_availability_metadata_only':True,'model_training':False,'selector_input':'frozen same-protocol 0dB sensing of last historical state','historical_source_smoothing_causality':'unverified','legacy_sensing_function_and_ID_mapping_reused':True,'bitwise_cache_parity_measured':False},'elapsed_s':time.time()-start}
    for split in ['train','val']:
        summary['split_horizon'][split]={str(h):summarise([r for r in rows_all if r['split']==split and r['horizon']==h]) for h in [20,30,40]}
        summary['split_horizon'][split]['40_by_scene']={str(s):summarise([r for r in rows_all if r['split']==split and r['horizon']==40 and r['scene']==s]) for s in [0,1]}
    # Metadata-only membership manifests keep censored targets. No future coordinates or errors.
    manifest=OUT/'history_only_membership_20260920.csv'
    fields=['scene','split','t0','id','horizon','n_active','n50','k','cpa','following','future_points','endpoint']
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows_all)
    summary['membership_sha256']=sha(manifest)
    (OUT/'history_only_audit_20260920.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({s:summary['split_horizon'][s]['40'] for s in ['train','val']},indent=2),flush=True)
if __name__=='__main__': main()
