"""F01-C orchestration. Truth is confined to training calibration and evaluation."""
from pathlib import Path
import argparse,json,time,hashlib,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import linear_sum_assignment
from .echo_source import SourceEpisodes,ROOT

OUT=ROOT/'reports/f01c'
DATA=ROOT/'data/f01_frontend'

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load(path): return json.loads(path.read_text())

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def bind(cost,gate):
    """Maximum-cardinality valid evaluation assignment, then minimum cost."""
    cost=np.asarray(cost,float)
    if not cost.size:return []
    n,m=cost.shape
    penalty=(n+m+1)*(gate+1)
    aug=np.full((n,m+n),penalty)
    aug[:,:m]=np.where(cost<=gate,cost,penalty*100)
    rows,cols=linear_sum_assignment(aug)
    return [(int(i),int(j)) for i,j in zip(rows,cols) if j<m and cost[i,j]<=gate]

def current_truth(source,episode,time_ms):
    # Current-row existence is independent of future labels and detector success.
    states=[];slots=[];velocity_valid=[]
    for slot,key in enumerate(episode['source_keys']):
        tr=source.tracks[key]; j=np.searchsorted(tr['time_ms'],time_ms)
        if j<len(tr) and int(tr[j]['time_ms'])==time_ms:
            row=tr[j];states.append([row['x'],row['y'],row['vx'],row['vy']]);slots.append(slot);velocity_valid.append(bool(row['past_frames']>=3))
    return np.array(states,float).reshape(-1,4),slots,velocity_valid

def worker_device():
    from multiprocessing import current_process
    identity=current_process()._identity
    return 'cuda:'+str(((identity[-1]-1) if identity else 0)%2)

def quality_db(d):
    q=d.get('quality',{})
    return float(d.get('peak_to_noise_db',q.get('peak_to_noise_db',0.)))

def quality_bin(d):return int(np.searchsorted([12.,20.],quality_db(d),side='right'))

def calibration_episode(index):
    from .detector import detect
    source=SourceEpisodes();ep=source.episodes[index]
    manifest=load(OUT/'development_manifest.json'); detector_config=load(ROOT/'configs/detector.json');detector_config['device']=worker_device()
    rows=[];visible=hits=overflow=0
    for elapsed in manifest['train_calibration_elapsed_ms']:
        for b in range(3):
            obs,audit=source.observation(index,ep['start_ms']+elapsed,b)
            detections,diagnostics=detect(obs,detector_config)
            overflow+=int(diagnostics.get('overflow_count',diagnostics.get('overflow',0)))
            mask=np.asarray(audit['visible'],bool)
            truths=np.array([audit['range_m'],audit['theta_rad'],audit['radial_velocity_mps']]).T[mask]
            visible+=len(truths)
            if not len(truths) or not detections:continue
            z=np.array([d['z'] for d in detections]);res=z[:,None,:]-truths[None,:,:];res[:,:,1]=wrap(res[:,:,1])
            costs=np.sum((res/np.array([1.5,np.deg2rad(5.),1.]))**2,axis=2)
            pairs=bind(costs,11.345)
            hits+=len(pairs)
            for i,j in pairs:rows.append(dict(residual=res[i,j].tolist(),quality_bin=quality_bin(detections[i]),peak_to_noise_db=quality_db(detections[i]),station_id=b,episode_index=index,elapsed_ms=elapsed))
    return dict(episode_index=index,visible=visible,hits=hits,overflow=overflow,residuals=rows)

def calibrate(workers):
    manifest=load(OUT/'development_manifest.json')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(calibration_episode,manifest['train_calibration_episodes']))
    residuals=[row for result in results for row in result['residuals']]
    if len(residuals)<30:raise RuntimeError('Insufficient training calibration residuals')
    floors=np.array([.35,np.deg2rad(5.),.2])**2
    def covariance(rows):
        e=np.array([row['residual'] for row in rows]);return np.diag(np.maximum(np.mean(e*e,axis=0),floors)).tolist()
    global_R=covariance(residuals);bins=[]
    for b in range(3):
        selected=[row for row in residuals if row['quality_bin']==b]
        bins.append(dict(bin=b,count=len(selected),R=covariance(selected) if len(selected)>=30 else global_R,fallback_global=len(selected)<30))
    visible=sum(r['visible'] for r in results);hits=sum(r['hits'] for r in results)
    detector_config=load(ROOT/'configs/detector.json')
    config=dict(frontend_revision='C04',q_a=4.,association_mode='jpda',p_D=(hits+1)/(visible+2),R_global=global_R,R_quality_bins=bins,quality_bin_edges_db=[12.,20.],detector_config=detector_config,
        calibration=dict(split='train',visible=visible,matched=hits,p_D_estimator='Beta(1,1) posterior mean',residual_estimator='diagonal second moment including mean bias; floors 0.35m,5deg,0.2m/s; C04 conservative angular floor, empirical range/Doppler; declared JPDA mixture',matching_gate='3D conservative initial R Mahalanobis gate 11.345; residual censoring and unresolved peaks remain limitations'))
    # Tracker expects clutter_intensity; detector calibration owns its value.
    for key in ('clutter_intensity','lambda_c'):
        if key in detector_config:config['clutter_intensity']=detector_config[key]
    dump(ROOT/'configs/frontend_config.json',config)
    dump(OUT/'R_calibration.json',dict(config=config,per_episode=[{k:v for k,v in r.items() if k!='residuals'} for r in results],residuals=residuals))
    print(json.dumps({'calibration_matches':hits,'visible':visible,'p_D':config['p_D'],'R_bins':bins}),flush=True)

def generate_episode(index):
    from .detector import detect
    source=SourceEpisodes();ep=source.episodes[index]; config=load(ROOT/'configs/frontend_config.json')
    config['detector_config']['device']=worker_device()
    directory=DATA/'detections';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'episode_{index:03d}.jsonl';tmp=path.with_suffix('.jsonl.part')
    started=time.perf_counter();count=overflow=0
    with tmp.open('w') as f:
        for obs,_ in source.iter_episode(index):
            dets,diag=detect(obs,config['detector_config'])
            for d in dets:d['R']=config['R_quality_bins'][quality_bin(d)]['R']
            record=dict(deadline_ns=int(obs['deadline_ns']),station_id=int(obs['station_id']),measurement_time_ns=int(obs['measurement_time_ns']),detections=dets,diagnostics=diag)
            f.write(json.dumps(record,allow_nan=False)+'\n');count+=len(dets)
            overflow+=int(diag.get('overflow_count',diag.get('overflow',0)))
    tmp.replace(path)
    result=dict(episode_index=index,split=ep['split'],cubes=199*3,detections=count,overflow_count=overflow,elapsed_seconds=time.perf_counter()-started)
    dump(DATA/'generation'/f'episode_{index:03d}.json',result)
    return result

def generate(workers,indices=None):
    manifest=load(OUT/'development_manifest.json')
    indices=indices or manifest['train_tracking_audit_episodes']+manifest['development_episodes']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(generate_episode,i):i for i in indices}
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)


def track_episode(index,q_a):
    from .tracker import CentralTracker
    source=SourceEpisodes();ep=source.episodes[index];config=load(ROOT/'configs/frontend_config.json');config['q_a']=q_a
    tracker=CentralTracker(source.stations,source.bores,config)
    rows=[];score=[];previous_binding={};previous_key={};id_switches=0;truth_switches=0; nis=[]; frame_previous={};frame_switches=0;frame_true=0;frame_matched=0;frame_false=0;frame_confirmed=0;frame_count=0; jpda_components=[]
    observations=[json.loads(line) for line in (DATA/'detections'/f'episode_{index:03d}.jsonl').read_text().splitlines()]
    assert len(observations)==597
    grid=set(ep['prediction_grid_ms'])
    for k,observation in enumerate(observations):
        # R is a frozen training-calibration product, independent of the sample truth.
        for detection in observation['detections']:detection['R']=config['R_quality_bins'][quality_bin(detection)]['R']
        update=tracker.update_station(observation['detections'],observation['station_id'],observation['measurement_time_ns'])
        jpda_components.extend(update.get('jpda_diagnostics',{}).get('components',[]))
        nis.extend((float(m['nis']),float(m.get('association_probability',1.))) for m in update.get('matches',[]) if 'nis' in m and m.get('association_probability',1.)>0)
        if k%3!=2:continue
        time_ns=observation['deadline_ns']; records=tracker.finish_cycle(time_ns)
        rows.append(dict(time_ns=time_ns,tracks=records))
        if time_ns//1_000_000>=ep['start_ms']+1000:
            frame_state,frame_slots,_=current_truth(source,ep,time_ns//1_000_000)
            frame_pool=[t for t in records if t['confirmed'] and t['exists']]
            frame_estimate=np.array([t['state_hat'] for t in frame_pool],float).reshape(-1,4)
            frame_pairs=bind(np.linalg.norm(frame_estimate[:,None,:2]-frame_state[None,:,:2],axis=2),5.)
            frame_count+=1;frame_true+=len(frame_state);frame_matched+=len(frame_pairs);frame_confirmed+=len(frame_pool);frame_false+=len(frame_pool)-len(frame_pairs)
            for i,j in frame_pairs:
                identity=ep['source_keys'][frame_slots[j]]; key=frame_pool[i]['key']
                if identity in frame_previous and frame_previous[identity]!=key:frame_switches+=1
                frame_previous[identity]=key
        if time_ns//1_000_000 not in grid:continue
        selected=tracker.select()
        true,slots,velocity_valid=current_truth(source,ep,time_ns//1_000_000)
        states=np.array([s['state_hat'] for s in selected],float).reshape(-1,4)
        distances=np.linalg.norm(states[:,None,:2]-true[None,:,:2],axis=2)
        pairs=bind(distances,5.)
        pool=[t for t in records if t['confirmed'] and t['exists']]
        pool_states=np.array([t['state_hat'] for t in pool],float).reshape(-1,4)
        pool_pairs=bind(np.linalg.norm(pool_states[:,None,:2]-true[None,:,:2],axis=2),5.)
        matches=[]
        for i,j in pairs:
            key=int(selected[i]['key']);slot=int(slots[j]);identity=ep['source_keys'][slot]
            if key in previous_binding and previous_binding[key]!=identity:id_switches+=1
            previous_binding[key]=identity
            if identity in previous_key and previous_key[identity]!=key:truth_switches+=1
            previous_key[identity]=key
            matches.append(dict(key=key,source_slot=slot,position_sq_error=float(np.sum((states[i,:2]-true[j,:2])**2)),velocity_sq_error=float(np.sum((states[i,2:]-true[j,2:])**2)) if velocity_valid[j] else None))
        stationary=np.linalg.norm(true[:,2:],axis=1)<=.5
        score.append(dict(stationary_true_count=int(stationary.sum()),stationary_matched_count=sum(bool(stationary[j]) for _,j in pairs),time_ns=time_ns,true_count=len(true),selected_count=len(selected),confirmed_pool_count=len(pool),false_confirmed_pool=len(pool)-len(pool_pairs),pool_matched=len(pool_pairs),matches=matches,uncovered=len(true)-len(matches),false_confirmed_selected=len(selected)-len(matches)))
    label='q'+str(int(q_a)); path=DATA/'tracks'/label/f'episode_{index:03d}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    report=dict(frontend_config_sha256=hashlib.sha256((ROOT/'configs/frontend_config.json').read_bytes()).hexdigest(),frontend_revision=config.get('frontend_revision','unversioned'),association_mode=config.get('association_mode','hungarian'),episode_index=index,split=ep['split'],q_a=q_a,origin_scores=score,full_frame_identity=dict(frames=frame_count,true_object_frames=frame_true,matched_object_frames=frame_matched,confirmed_track_frames=frame_confirmed,false_confirmed_track_frames=frame_false,identity_switches=frame_switches,IDSW_per_1000_reference_object_frames=1000*frame_switches/frame_true if frame_true else None,scope='All 100ms frames after 1s burn-in, all extant confirmed tracks, 5m current-position matching, successive matches retain gaps'),identity_switches_at_scored_origins=truth_switches,track_key_truth_reassignments=id_switches,identity_metric_definition='For each true source, count changes of assigned system key between successive matched scored origins; gaps retained. Separately count one key assigned to different sources.',innovation_nis=dict(count=len(nis),effective_weight=sum(w for v,w in nis),mean=float(np.average([v for v,w in nis],weights=[w for v,w in nis])) if nis else None,fraction_over_95pct=float(np.average([v>7.815 for v,w in nis],weights=[w for v,w in nis])) if nis else None,qualification='Gated association branches weighted by posterior association probability; not an unbiased R validation sample'))
    report['jpda_components']=json.loads(json.dumps(jpda_components,default=lambda x:x.item() if isinstance(x,np.generic) else x))
    dump(OUT/'evaluation'/label/f'episode_{index:03d}.json',report)
    return report

def summarize(reports):
    origins=[s for r in reports for s in r['origin_scores']];matches=[m for s in origins for m in s['matches']]
    truth=sum(s['true_count'] for s in origins);selected=sum(s['selected_count'] for s in origins)
    vel=[m['velocity_sq_error'] for m in matches if m['velocity_sq_error'] is not None]
    result=dict(episodes=len(reports),origins=len(origins),true_object_origins=truth,selected_track_origins=selected,matched_object_origins=len(matches),coverage=len(matches)/truth if truth else 0.,position_rmse_m=float(np.sqrt(np.mean([m['position_sq_error'] for m in matches]))) if matches else None,velocity_rmse_mps=float(np.sqrt(np.mean(vel))) if vel else None,false_confirmed_fraction=sum(s['false_confirmed_pool'] for s in origins)/sum(s['confirmed_pool_count'] for s in origins) if sum(s['confirmed_pool_count'] for s in origins) else None,false_selected_fraction=(selected-len(matches))/selected if selected else None,pool_coverage=sum(s['pool_matched'] for s in origins)/truth if truth else 0.,identity_switches_at_scored_origins=sum(r['identity_switches_at_scored_origins'] for r in reports),origins_with_pool_over_8=sum(s['confirmed_pool_count']>8 for s in origins))
    total_frames=sum(r['full_frame_identity']['true_object_frames'] for r in reports)
    result['IDSW_per_1000_reference_object_frames']=1000*sum(r['full_frame_identity']['identity_switches'] for r in reports)/total_frames if total_frames else None
    result['full_frame_true_object_count']=total_frames
    stationary_total=sum(s['stationary_true_count'] for s in origins)
    result['stationary_reference_threshold_mps']=.5
    result['stationary_true_object_origins']=stationary_total
    result['stationary_coverage']=sum(s['stationary_matched_count'] for s in origins)/stationary_total if stationary_total else None
    result['passed']=bool(result['coverage']>=.9 and result['position_rmse_m'] is not None and result['position_rmse_m']<=3 and result['velocity_rmse_mps'] is not None and result['velocity_rmse_mps']<=3 and result['false_confirmed_fraction'] is not None and result['false_confirmed_fraction']<=.05)
    return result

def track(workers,q_a,indices=None):
    manifest=load(OUT/'development_manifest.json');indices=indices or manifest['train_tracking_audit_episodes']+manifest['development_episodes']
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(track_episode,indices,[q_a]*len(indices)))
    summary={split:summarize([r for r in results if r['split']==split]) for split in ('train','V_select') if any(r['split']==split for r in results)}
    summary['q_a']=q_a;dump(OUT/f'quality_q{int(q_a)}.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['calibrate','generate','track']);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--q-a',type=float,default=4.);parser.add_argument('--episodes',type=int,nargs='*');args=parser.parse_args()
    if args.action=='calibrate':calibrate(args.workers)
    elif args.action=='generate':generate(args.workers,args.episodes)
    else:track(args.workers,args.q_a,args.episodes)
