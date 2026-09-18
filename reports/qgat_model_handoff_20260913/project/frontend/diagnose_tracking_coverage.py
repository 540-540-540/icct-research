"""Evaluation-side decomposition of detector availability and track coverage."""
import json
from collections import deque
import numpy as np
from .run_frontend import load,dump,bind,wrap,OUT,DATA
from .echo_source import SourceEpisodes
from .station_geometry import visibility
from .tracker import measurement

def run():
    source=SourceEpisodes();manifest=load(OUT/'development_manifest.json');totals=dict(true=0,covered=0,detected_now=0,detected_last_5=0,stationary_true=0,stationary_detected_now=0,stationary_detected_last_5=0);details=[]
    for index in manifest['train_tracking_audit_episodes']:
        ep=source.episodes[index];evals=load(OUT/'evaluation/q4'/f'episode_{index:03d}.json');origin={r['time_ns']:r for r in evals['origin_scores']}
        cache=[json.loads(line) for line in (DATA/'detections'/f'episode_{index:03d}.jsonl').read_text().splitlines()]
        recent=deque(maxlen=5);hits=set()
        for row in cache:
            b=row['station_id'];states,slots,_=source.at_time(ep,row['measurement_time_ns']);mask=visibility(states[:,:2],source.stations,source.bores)[:,b]
            active=np.flatnonzero(mask);true=np.array([measurement(states[j],source.stations[b],source.bores[b])[0] for j in active]).reshape(-1,3)
            z=np.array([d['z'] for d in row['detections']]).reshape(-1,3);error=z[:,None]-true[None];error[:,:,1]=wrap(error[:,:,1])
            cost=np.sum((error/np.array([1.5,np.deg2rad(5),1]))**2,axis=2)
            for i,j in bind(cost,11.345):hits.add(int(slots[active[j]]))
            if b!=2:continue
            recent.append(hits);past=set().union(*recent);time_ns=row['deadline_ns']
            if time_ns in origin:
                from .run_frontend import current_truth
                states,slots,_=current_truth(source,ep,time_ns//1_000_000)
                stationary={slots[j] for j,s in enumerate(states) if np.linalg.norm(s[2:])<=.5};present=set(slots);covered={m['source_slot'] for m in origin[time_ns]['matches']}
                r=dict(episode_index=index,time_ns=time_ns,true=len(present),covered=len(covered),detected_now=len(present&hits),detected_last_5=len(present&past),stationary_true=len(stationary),stationary_detected_now=len(stationary&hits),stationary_detected_last_5=len(stationary&past),uncovered_despite_recent_detection=sorted((present-covered)&past),uncovered_without_recent_detection=sorted((present-covered)-past))
                details.append(r)
                for key in totals:totals[key]+=r[key]
            hits=set()
    result=dict(totals=totals,details=details,scope='training audit only; signal-to-truth association is diagnostic and never changes tracking',detection_binding='conservative initial-R 3D gate 11.345; union across stations, last-five refers to 100ms cycles')
    dump(OUT/'coverage_decomposition.json',result);print(json.dumps(totals,indent=2))

if __name__=='__main__':run()
