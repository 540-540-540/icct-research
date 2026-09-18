"""F01-A source preparation. CPU only; outputs are simulator sources, never model inputs."""
from pathlib import Path
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from station_geometry import station_positions, station_boresights, visibility
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/f01a'
DATA=ROOT/'data/f01_source'
C=json.loads((ROOT/'configs/f01a.json').read_text())
SPLITS=list(C['split_fractions'])

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def causal_velocity(times,xy):
    n=len(times); vel=np.zeros((n,2)); count=np.zeros(n,dtype=np.int8)
    start=0
    for i in range(n):
        if i and times[i]-times[i-1]!=100: start=i
        lo=max(start,i-4); count[i]=i-lo+1
        if count[i]>=3:
            t=(times[lo:i+1]-times[i])/1000
            t=t-t.mean()
            vel[i]=(t[:,None]*xy[lo:i+1]).sum(0)/np.dot(t,t)
    return vel,count

def choose(frame,seed):
    if frame.empty: return []
    rng=np.random.default_rng(seed)
    frame=frame.sort_values('key')
    anchors=rng.choice(len(frame),size=min(3,len(frame)),replace=False)
    xy=frame[['x','y']].to_numpy(); keys=frame.key.to_numpy()
    result=[]
    for a in anchors:
        dist=np.linalg.norm(xy-xy[a],axis=1)
        ix=[j for j in np.lexsort((keys,dist)) if j!=a and dist[j]<=45][:7]
        result.append((int(keys[a]),[int(keys[a])]+[int(keys[j]) for j in ix]))
    return result

def retain_disjoint_episodes(candidates):
    """Chronological split priority; reject a whole cohort, never trim neighbors."""
    owner={}; kept=[]; rejected=[]
    for split in SPLITS:
        ordered=sorted((e for e in candidates if e['split']==split),key=lambda e:(e['start_ms'],e['anchor_key']))
        for e in ordered:
            conflicts={str(g):owner[g] for g in e['source_groups'] if g in owner and owner[g]!=split}
            if conflicts:
                rejected.append({'episode_id':e['episode_id'],'split':split,'start_ms':e['start_ms'],'anchor_key':e['anchor_key'],'conflicting_source_group_owners':conflicts})
                continue
            kept.append(e)
            for g in e['source_groups']: owner[g]=split
    return kept,rejected,owner

def main():
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data/Lankershim_Vehicle_Trajectories.csv')
    offsets=np.sort((d.Global_Time-d.Frame_ID*100).unique())
    d['segment']=np.searchsorted(offsets,d.Global_Time-d.Frame_ID*100)
    d['key']=pd.factorize(pd.MultiIndex.from_frame(d[['segment','Vehicle_ID']]),sort=True)[0]
    identities=d[['key','segment','Vehicle_ID']].drop_duplicates().sort_values('key')
    t0=int(d.Global_Time.min()); tend=int(d.Global_Time.max())+100
    cuts=np.rint((t0+(tend-t0)*np.cumsum([0]+list(C['split_fractions'].values())))/100).astype(np.int64)*100
    cuts[0]=t0; cuts[-1]=tend
    d['split']='guard'
    intervals={}
    for i,s in enumerate(SPLITS):
        a=int(cuts[i]+(5000 if i else 0)); b=int(cuts[i+1]-(5000 if i<3 else 0))
        intervals[s]=[a,b]
        d.loc[(d.Global_Time>=a)&(d.Global_Time<b),'split']=s
    # Conservative cross-record grouping: repeated nearby global geometry; dimensions can change between annotations.
    # This is a leakage exclusion group, NOT a claim of exact recovered physical identity.
    parent=np.arange(len(identities))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=int(parent[a])
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    matches=Counter()
    for sa in range(len(offsets)):
        for sb in range(sa+1,len(offsets)):
            a=d[d.segment==sa]; b=d[d.segment==sb]
            left={int(t):g for t,g in a.groupby('Global_Time')}
            for t,g in b.groupby('Global_Time'):
                if int(t) not in left: continue
                f=left[int(t)]
                near=cKDTree(f[['Global_X','Global_Y']].to_numpy()*0.3048).query_ball_point(g[['Global_X','Global_Y']].to_numpy()*0.3048,r=3)
                for j,ii in enumerate(near):
                    bj=g.iloc[j]
                    for k in ii:
                        ai=f.iloc[k]
                        matches[(int(ai.key),int(bj.key))]+=1
    linked=[]
    for (a,b),n in matches.items():
        if n>=10:
            union(a,b); linked.append({'key_a':a,'key_b':b,'nearby_frames':n})
    d['source_group']=np.array([find(i) for i in range(len(identities))])[d.key.to_numpy()]
    identities['source_group']=[find(int(k)) for k in identities.key]
    # At shared wall-clock times, the earliest recording exclusively owns simulation generation.
    d['owned']=False; prior_end=None; segments=[]
    for seg,g in d.groupby('segment'):
        first=int(g.Global_Time.min()); last=int(g.Global_Time.max())+100
        start=first if prior_end is None else max(first,prior_end)
        d.loc[g.index,'owned']=g.Global_Time>=start
        segments.append({'segment':int(seg),'offset_ms':int(offsets[seg]),'raw_start_ms':first,'raw_end_exclusive_ms':last,'owned_start_ms':start,'owned_end_exclusive_ms':last})
        prior_end=max(prior_end or last,last)
    # Source identity embargo uses all legal-time raw observations, even in overlap.
    group_splits=d[d.split!='guard'].groupby('source_group').split.nunique()
    crossing=set(map(int,group_splits[group_splits>1].index))
    # Absolute local metres for causal cohort selection; translation is fitted after retention.
    d['x']=d.Local_X*0.3048; d['y']=d.Local_Y*0.3048
    d=d.sort_values(['key','Global_Time']).reset_index(drop=True)
    d['vx']=0.; d['vy']=0.; d['past_frames']=np.int8(0)
    for key,g in d.groupby('key',sort=False):
        vel,count=causal_velocity(g.Global_Time.to_numpy(),g[['x','y']].to_numpy())
        d.loc[g.index,['vx','vy']]=vel; d.loc[g.index,'past_frames']=count
    available=d[d.owned&(d.split!='guard')]
    frames={int(t):g for t,g in available.groupby('Global_Time')}
    episodes=[]; discarded=Counter(); attempted=Counter(); rejected=[]
    for seg in segments:
        for s,(low,high) in intervals.items():
            low=max(low,seg['owned_start_ms']); high=min(high,seg['owned_end_exclusive_ms'])
            # Fixed 20s grid nested inside fixed 60s absolute source blocks.
            for start in range(t0,tend,20000):
                end=start+20000
                if start<low or end>high: continue
                frame=frames.get(start)
                if frame is None: continue
                frame=frame[frame.segment==seg['segment']]
                for anchor,keys in choose(frame,C['seed']+(start-t0)//100+seg['segment']*100000):
                    attempted[s]+=1
                    groups=[find(k) for k in keys]
                    block=(start-t0)//60000
                    assert (end-1-t0)//60000==block
                    # first history sample begins after 1s burn-in; last future sample <= episode end-0.1s
                    origins=list(range(start+2900,end-2000,500 if s=='train' else 1000))
                    episodes.append({'episode_id':f's{seg["segment"]}_t{start}_a{anchor}','split':s,'segment':seg['segment'],'start_ms':start,'end_exclusive_ms':end,'source_block':int(block),'anchor_key':anchor,'source_keys':keys,'source_groups':groups,'candidate_prediction_origins_ms':origins,'exposure_status':C['exposure_status']})
    candidates=episodes
    episodes,rejected,source_owners=retain_disjoint_episodes(candidates)
    discarded=Counter(e['split'] for e in rejected)
    # Fit geometry only on actually retained training cohort rows, de-duplicated across anchors.
    # Excluded/held-out vehicle rows cannot contribute even indirectly through this fit.
    fit_mask=np.zeros(len(d),dtype=bool)
    for e in episodes:
        if e['split']=='train':
            fit_mask |= d.key.isin(e['source_keys']).to_numpy() & (d.Global_Time.to_numpy()>=e['start_ms']) & (d.Global_Time.to_numpy()<e['end_exclusive_ms'])
    train=d.loc[fit_mask]
    bounds=train[['Local_X','Local_Y']].quantile([.05,.95]).to_numpy()*0.3048
    center=bounds.mean(0); W,L=bounds[1]-bounds[0]
    stations=station_positions(W,L)
    d['x']-=center[0]; d['y']-=center[1]
    # Global time-block round-robin cap for train; no replacement or duplication.
    pools=defaultdict(list)
    for e in episodes:
        if e['split']=='train':
            for t in e['candidate_prediction_origins_ms']: pools[e['source_block']].append((e['episode_id'],t))
    rng=np.random.default_rng(C['seed'])
    for pool in pools.values(): rng.shuffle(pool)
    chosen=set()
    while len(chosen)<6000 and any(pools.values()):
        for b in sorted(pools):
            if pools[b] and len(chosen)<6000: chosen.add(pools[b].pop())
    for e in episodes:
        origins=e.pop('candidate_prediction_origins_ms')
        e['prediction_grid_ms']=[t for t in origins if e['split']!='train' or (e['episode_id'],t) in chosen]
    # Source-side state, not anonymous estimates: physically kept outside future model input cache.
    arr=np.empty(len(d),dtype=[('source_key','i8'),('source_group','i8'),('segment','i2'),('time_ms','i8'),('x','f8'),('y','f8'),('vx','f8'),('vy','f8'),('past_frames','i1'),('owned','?')])
    for k,col in [('source_key','key'),('source_group','source_group'),('segment','segment'),('time_ms','Global_Time'),('x','x'),('y','y'),('vx','vx'),('vy','vy'),('past_frames','past_frames'),('owned','owned')]: arr[k]=d[col].to_numpy()
    np.save(DATA/'source_states.npy',arr,allow_pickle=False)
    identities.to_csv(DATA/'source_identity.csv',index=False)
    (OUT/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    # Geometric eligibility only: this is not detector recall.
    geom={}
    for s in SPLITS:
        rows=d[(d.split==s)&d.owned]
        vis=visibility(rows[['x','y']].to_numpy(),stations)
        geom[s]={'rows':len(rows),'at_least_one_station':float(vis.any(1).mean()),'at_least_two_stations':float((vis.sum(1)>=2).mean()),'all_three_stations':float(vis.all(1).mean())}
    write('geometry.json',{'fit_split':'retained training episode source rows only; de-duplicated across anchors','training_rows':len(train),'source_local_bounds_5_95_m':bounds.tolist(),'source_origin_m':center.tolist(),'W_m':float(W),'L_m':float(L),'stations_xy_m':stations.tolist(),'boresight_rad':station_boresights(stations).tolist(),'boresight_deg':[0.,180.,0.],'revision':'A01','station_height_m':6,'vehicle_height_m':1,'fov_half_angle_deg':70,'range_m':[10,300],'coverage_is_geometry_not_detection':geom})
    write('split_manifest.json',{'status':'built_pending_review','revision':'A01','config':'configs/f01a.json','time_origin_ms':t0,'end_exclusive_ms':tend,'raw_cutpoints_ms':cuts.tolist(),'guarded_intervals_ms':intervals,'segments':segments,'overlap_policy':'earliest segment owns each timestamp; later overlapping rows retained only as source audit','source_group_policy':'conservative cross-segment potential same vehicle; >=10 frames within 3m global position, no size/class hard gate; conservative possible identity, not exact matching','source_group_links':linked,'raw_cross_split_source_groups_diagnostic_only':sorted(crossing),'source_group_owner':{str(k):v for k,v in source_owners.items()},'identity_policy':'chronological whole-episode first-claim: train, V_select, V_confirm, test; retained cohorts pairwise disjoint','candidate_episodes':dict(attempted),'discarded_entire_episodes_cross_identity':dict(discarded),'episodes_by_split':dict(Counter(e['split'] for e in episodes)),'prediction_grid_count_by_split':{s:sum(len(e['prediction_grid_ms']) for e in episodes if e['split']==s) for s in SPLITS},'prediction_grid_is_not_yet_legal_track_origins':True,'exposure_status':C['exposure_status']})
    active=d.past_frames>=3
    speed=np.hypot(d.loc[active,'vx'],d.loc[active,'vy'])
    write('units_report.json',{'position_conversion':'Local_X/Y ft * 0.3048, subtract training-only origin','time':'Global_Time ms; cadence 100ms; no resampling','velocity':'least squares over at most 5 contiguous past/current frames; min3; reset after gap','raw_v_Vel_used_for_simulation':False,'speed_m_s_quantiles':dict(zip(['0','50','95','99','100'],map(float,np.quantile(speed,[0,.5,.95,.99,1])))),'inactive_source_rows':int((~active).sum()),'source_states_role':'simulator/label side only; NOT state_hat and forbidden to prediction loader'})
    # Executable causal and split checks.
    tt=np.arange(10,dtype=np.int64)*100; xx=np.column_stack([3*tt/1000,7*tt/1000])
    v,n=causal_velocity(tt,xx); assert np.allclose(v[2:],[3,7])
    changed=xx.copy(); changed[6:]+=10000
    assert np.array_equal(causal_velocity(tt,changed)[0][:6],v[:6])
    # Real-track prefix must exactly agree with full-series processing.
    for _,g in list(d.groupby('key',sort=False))[:12]:
        tt_real=g.Global_Time.to_numpy(); xy_real=g[['x','y']].to_numpy(); ncut=min(20,len(g))
        vv,_=causal_velocity(tt_real,xy_real); vp,_=causal_velocity(tt_real[:ncut],xy_real[:ncut])
        assert np.array_equal(vv[:ncut],vp)
    newborn=pd.DataFrame({'key':[0],'x':[0.],'y':[0.],'past_frames':[1]})
    assert choose(newborn,2026)==[(0,[0])]
    for e in episodes:
        a,b=intervals[e['split']]
        assert a<=e['start_ms'] and e['end_exclusive_ms']<=b
        assert len(e['source_keys'])<=8 and len(set(e['source_keys']))==len(e['source_keys'])
        assert all(source_owners[g]==e['split'] for g in e['source_groups'])
        f=frames[e['start_ms']].set_index('key')
        pos=f.loc[e['source_keys'],['x','y']].to_numpy()
        assert np.all(np.linalg.norm(pos-pos[0],axis=1)<=45+1e-9)
        for t in e['prediction_grid_ms']:
            assert t-1900>=e['start_ms']+1000 and t+2000<e['end_exclusive_ms']
    split_groups={s:set(g for e in episodes if e['split']==s for g in e['source_groups']) for s in SPLITS}
    for i,s in enumerate(SPLITS):
        for other in SPLITS[i+1:]: assert not split_groups[s]&split_groups[other]
    sample=[{'episode_id':'a','split':'train','start_ms':0,'anchor_key':0,'source_groups':[1]},
            {'episode_id':'b','split':'V_select','start_ms':1,'anchor_key':1,'source_groups':[1,2]},
            {'episode_id':'c','split':'V_confirm','start_ms':2,'anchor_key':2,'source_groups':[2]}]
    retained,removed,owned=retain_disjoint_episodes(sample)
    assert [e['episode_id'] for e in retained]==['a','c'] and len(removed)==1 and owned[2]=='V_confirm'
    assert not set(train.source_group).intersection(g for g,split in source_owners.items() if split!='train')
    write('rejected_episodes.json',rejected)
    write('checks.json',{'status':'passed','checks':['rejected cohorts do not claim source identity','geometry fit excludes nontraining source groups','newborn eligible for cohort before echo activation','real track prefix invariance','constant velocity exact','future perturbation leaves past velocity unchanged','all episode support inside guarded split','no statistical block crossing','<=8 fixed source slots and <=45m initial neighbors','cross-split source groups disjoint','burnin/history/future inside episode','training origin cap <=6000'],'full_frontend_causality':'pending F01-D','model_training_started':False})
    print(json.dumps({'episodes':Counter(e['split'] for e in episodes),'removed':discarded,'geometry':geom,'source_groups_linked':len(linked)},indent=2))
if __name__=='__main__': main()

