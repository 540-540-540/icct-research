"""Read-only training-cache diagnosis; reference states used only for audit."""
import json
from collections import Counter,defaultdict
import numpy as np
from .echo_source import SourceEpisodes,ROOT
from .tracker import CentralTracker,measurement
from .run_frontend import current_truth,bind,load,dump

def stats(values):
    a=np.asarray(values,float)
    return {'n':len(a),'p50':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())} if len(a) else {'n':0}

def diagnose(index,source):
    ep=source.episodes[index]
    config=load(ROOT/'reports/f01c/C01_before_R_floor/configs/frontend_config.json');config['q_a']=4.
    tracker=CentralTracker(source.stations,source.bores,config)
    observations=[json.loads(x) for x in (ROOT/f'data/f01_frontend/detections/episode_{index:03d}.jsonl').read_text().splitlines()]
    events=defaultdict(list);births={};false=defaultdict(list);correct=defaultdict(list)
    for cycle,ob in enumerate(observations):
        sid=ob['station_id'];ns=ob['measurement_time_ns'];dets=ob['detections']
        truth,slots,_=source.at_time(ep,ns)
        ztruth=np.array([measurement(x,source.stations[sid],source.bores[sid])[0] for x in truth]).reshape(-1,3)
        refs=[]
        for j,d in enumerate(dets):
            z=np.asarray(d['z']);dz=z-ztruth
            if len(truth):
                dz[:,1]=(dz[:,1]+np.pi)%(2*np.pi)-np.pi
                norm=np.linalg.norm(dz/np.array([1.5,np.deg2rad(5),1.]),axis=1)
                nearest=int(np.argmin(norm));r=dz[nearest]
                near={'source_slot':int(slots[nearest]),'residual_r_theta_vr':r.tolist(),'normalized_distance':float(norm[nearest])}
            else:near={'source_slot':None,'normalized_distance':None}
            grid=np.array(d.get('quality',{}).get('grid_index',[0,0,0]))
            neighbors=[]
            for k,other in enumerate(dets):
                if k==j:continue
                delta=np.abs(grid-np.array(other.get('quality',{}).get('grid_index',[0,0,0])))
                if delta[0]<=2 and delta[1]<=2 and delta[2]<=4:
                    neighbors.append({'detection_index':k,'grid_delta':delta.tolist()})
            refs.append(dict(time_ns=ns,station_id=sid,detection_index=j,z=z.tolist(),quality=d.get('quality'),nearest_reference=near,nearby_peak_candidates=neighbors))
        update=tracker.update_station(dets,sid,ns)
        matched=set()
        for m in update['matches']:
            j=m['detection_index'];matched.add(j);events[m['key']].append(refs[j])
        remaining=[j for j in range(len(dets)) if j not in matched and dets[j]['z'][0]>5]
        for key,j in zip(update['birth_keys'],remaining):
            events[key].append(refs[j]);births[key]=refs[j]
        if cycle%3!=2:continue
        deadline=ob['deadline_ns'];rows=tracker.finish_cycle(deadline)
        true,slots,_=current_truth(source,ep,deadline//1_000_000)
        pool=[t for t in rows if t['confirmed'] and t['exists']]
        x=np.array([t['state_hat'] for t in pool]).reshape(-1,4)
        dist=np.linalg.norm(x[:,None,:2]-true[None,:,:2],axis=2)
        pairs=bind(dist,5.);good={i:j for i,j in pairs}
        for i,t in enumerate(pool):
            if i in good:
                correct[t['key']].append({'time_ns':deadline,'source_slot':int(slots[good[i]])})
                continue
            j=int(np.argmin(dist[i])) if len(true) else None
            refdist=float(dist[i,j]) if j is not None else None
            nearest_good=min(good,key=lambda k:float(np.linalg.norm(x[i,:2]-x[k,:2]))) if good else None
            neargood=float(np.linalg.norm(x[i,:2]-x[nearest_good,:2])) if nearest_good is not None else None
            false[t['key']].append(dict(time_ns=deadline,nearest_reference_distance_m=refdist,
                nearest_reference_velocity_error_mps=float(np.linalg.norm(x[i,2:]-true[j,2:])) if j is not None else None,
                nearest_reference_slot=int(slots[j]) if j is not None else None,
                nearest_correct_track_distance_m=neargood,nearest_correct_track_velocity_difference_mps=float(np.linalg.norm(x[i,2:]-x[nearest_good,2:])) if nearest_good is not None else None,detected=t['detected'],misses=t['misses'],age=t['age']))
    keys=[];allfalse=[f for v in false.values() for f in v]
    for key,frames in false.items():
        ev=events[key];corr=correct[key];hist=tracker.history[key]
        slotcounts=Counter(e['nearest_reference']['source_slot'] for e in ev)
        row=dict(key=key,false_confirmed_frames=len(frames),correctly_matched_frames=len(corr),
                 confirmed_frame_count=sum(r['exists'] and r['confirmed'] for r in hist),
                 lifetime_cycles=len(hist),hit_station_counts=dict(Counter(e['station_id'] for e in ev)),
                 hit_cycle_count=len(set((e['time_ns']-ep['start_ms']*1_000_000)//100_000_000 for e in ev)),
                 hit_nearest_reference_slot_counts=dict(slotcounts),
                 birth=births[key],false_frame_details=frames,
                 false_nearest_reference_distance_m=stats([f['nearest_reference_distance_m'] for f in frames if f['nearest_reference_distance_m'] is not None]),
                 false_velocity_error_mps=stats([f['nearest_reference_velocity_error_mps'] for f in frames if f['nearest_reference_velocity_error_mps'] is not None]),
                 false_nearest_correct_track_distance_m=stats([f['nearest_correct_track_distance_m'] for f in frames if f['nearest_correct_track_distance_m'] is not None]),
                 false_nearest_correct_track_velocity_difference_mps=stats([f['nearest_correct_track_velocity_difference_mps'] for f in frames if f['nearest_correct_track_velocity_difference_mps'] is not None]),
                 correct_source_slot_changes=sum(a['source_slot']!=b['source_slot'] for a,b in zip(corr,corr[1:])),
                 observed_peak_neighbor_hit_count=sum(bool(e['nearby_peak_candidates']) for e in ev),
                 measurement_events=ev)
        keys.append(row)
    categories=Counter()
    for f in allfalse:
        categories['duplicate_within_5m' if f['nearest_reference_distance_m'] is not None and f['nearest_reference_distance_m']<=5 else 'outside_5m_or_reference_absent']+=1
        if not f['detected']:categories['coasting_without_current_detection']+=1
    return dict(episode_index=index,split=ep['split'],false_frame_categories=dict(categories),
        false_confirmed_frames=len(allfalse),false_keys=sorted(keys,key=lambda r:-r['false_confirmed_frames']))

def main():
    source=SourceEpisodes()
    # Frozen C01 pre-floor ranking: false origins 62,29,24 respectively.
    indices=[62,158,125]
    results=[diagnose(i,source) for i in indices]
    report=dict(scope='Read-only C01 cached detections replay q_a=4; training reference only for diagnosis; no filtering or metric changes',
                selected_by='Three training episodes with most false confirmed track-origin counts',episodes=results)
    dump(ROOT/'reports/f01c/false_track_diagnosis.json',report)
    for r in results:
        print(r['episode_index'],r['false_frame_categories'])
        for k in r['false_keys'][:6]:
            print({field:k[field] for field in ('key','false_confirmed_frames','correctly_matched_frames','lifetime_cycles','hit_station_counts','hit_cycle_count','observed_peak_neighbor_hit_count','false_nearest_reference_distance_m','false_velocity_error_mps','false_nearest_correct_track_distance_m')})
if __name__=='__main__':main()
