"""Rebuild a detector/tracker prefix with all later source rows removed."""
import json,time,hashlib
import numpy as np
from .echo_source import SourceEpisodes,ROOT
from .detector import detect
from .tracker import CentralTracker
from .run_frontend import load,dump,quality_bin

def run():
    source=SourceEpisodes();config=load(ROOT/'configs/frontend_config.json')
    config['detector_config']['device']='cuda:0'
    results=[]
    # Preselected early/sparse and mid-record episodes; no quality-based selection.
    for index in (0,95):
        ep=source.episodes[index];cutoff=ep['start_ms']+5000
        tracks=dict(source.tracks)
        for key in ep['source_keys']:tracks[key]=source.tracks[key][source.tracks[key]['time_ms']<=cutoff]
        cached=[json.loads(line) for line in (ROOT/'data/f01_frontend/detections'/f'episode_{index:03d}.jsonl').read_text().splitlines()]
        reference=CentralTracker(source.stations,source.bores,config)
        regenerated=CentralTracker(source.stations,source.bores,config)
        cycles=cubes=0
        for row in cached:
            if row['deadline_ns']>cutoff*1_000_000:break
            obs,_=source.observation(index,row['deadline_ns']//1_000_000,row['station_id'],tracks=tracks)
            dets,diagnostics=detect(obs,config['detector_config'])
            for d in dets:d['R']=config['R_quality_bins'][quality_bin(d)]['R']
            assert dets==row['detections'],('detector prefix changed',index,cubes)
            reference.update_station(row['detections'],row['station_id'],row['measurement_time_ns'])
            regenerated.update_station(dets,row['station_id'],int(obs['measurement_time_ns']))
            cubes+=1
            if row['station_id']==2:
                old=reference.finish_cycle(row['deadline_ns']);new=regenerated.finish_cycle(row['deadline_ns'])
                assert new==old,('tracker prefix changed',index,cycles)
                assert regenerated.select()==reference.select()
                cycles+=1
        results.append(dict(episode_index=index,cutoff_ms=cutoff,compared_cubes=cubes,compared_cycles=cycles,detections_exact=True,tracks_exact=True,selected_keys_exact=True))
    report=dict(frontend_revision=config.get('frontend_revision','unversioned'),frontend_config_sha256=hashlib.sha256((ROOT/'configs/frontend_config.json').read_bytes()).hexdigest(),passed=True,tests=results,frozen_parameters=f"F01-A geometry/episodes, F01-B seeds/waveform, trained angular mixture/R, CFAR threshold, q_a={config['q_a']}, max_missed_cycles={config.get('max_missed_cycles',5)}",mutation='Delete all rows after cutoff for every cohort source; do not refit train parameters',scope='Complete <=t detector and tracker prefix. F01-D model-input serialization is a separate gate.',maximum_absolute_difference=0.)
    dump(ROOT/'reports/f01c/causal_checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':run()
