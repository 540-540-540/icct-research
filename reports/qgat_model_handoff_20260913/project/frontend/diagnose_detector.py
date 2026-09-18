"""Training-only truth-assisted detector diagnosis; never production search."""
import json
from pathlib import Path
import numpy as np
from .echo_source import SourceEpisodes,ROOT
from .run_frontend import current_truth,wrap,bind
from .detector import _grid,candidate_map,_box_sum,load_config
from .ofdm_echo import DEFAULT_WAVEFORM as W,noise_parameters


def main():
 source=SourceEpisodes();cfg=load_config(json.loads((ROOT/'reports/f01c/baseline_before_guard_revision/configs/detector.json').read_text()));indices=[0,14,29,47,62,77,95,110,125,143,158,176]
 missing=[];summary=[]
 for ei in indices:
  ep=source.episodes[ei]
  report=json.loads((ROOT/f'reports/f01c/baseline_before_guard_revision/evaluation/q4/episode_{ei:03d}.json').read_text())
  obs=[json.loads(x) for x in (ROOT/f'data/f01_frontend_baseline/detections/episode_{ei:03d}.jsonl').read_text().splitlines()]
  obs_by={(o['deadline_ns'],o['station_id']):o for o in obs}
  for row in report['origin_scores']:
   states,slots,_=current_truth(source,ep,row['time_ns']//1000000)
   hit={m['source_slot'] for m in row['matches']}
   for state,slot in zip(states,slots):
    if np.linalg.norm(state[2:])>.5:continue
    ranges=np.sqrt(np.sum((state[:2]-source.stations)**2,axis=1)+25.)
    rec=dict(episode=ei,time_ns=row['time_ns'],slot=slot,covered=slot in hit,min_range=float(ranges.min()),best_isolated_snr_db=float(20+40*np.log10(100/ranges.min())))
    det_hits=0
    for b in range(3):
     states_b,slots_b,_=source.at_time(ep,obs_by[row['time_ns'],b]['measurement_time_ns'])
     where=np.flatnonzero(slots_b==slot)
     if not len(where):continue
     st=states_b[where[0]];delta=st[:2]-source.stations[b];r=np.sqrt(delta@delta+25);theta=wrap(np.arctan2(delta[1],delta[0])-source.bores[b]);vr=delta@st[2:]/r
     zs=np.asarray([d['z'] for d in obs_by[row['time_ns'],b]['detections']]).reshape(-1,3)
     if len(zs):
      error=zs-[r,theta,vr];error[:,1]=wrap(error[:,1]);det_hits+=int(np.any(np.sum((error/[1.5,np.deg2rad(5),1])**2,axis=1)<=11.345))
    rec['stations_with_nearby_cached_detection']=det_hits
    summary.append(rec)
    if slot not in hit:missing.append(rec)
 # Fixed deterministic examples: one first missed stationary object per episode,
 # covering up to ten distinct training episodes rather than selecting outcomes.
 chosen=[]
 for ei in indices:
  options=[r for r in missing if r['episode']==ei]
  if options:chosen.append(options[0])
 chosen=chosen[:10]
 details=[];idx,rgrid,ugrid,vgrid,valid,count,alpha,pfa=_grid(2,1)
 known_noise=noise_parameters()['matched_output_noise_variance']
 for example in chosen:
  ei=example['episode'];ep=source.episodes[ei]
  station_details=[]
  for b in range(3):
   observation,audit=source.observation(ei,example['time_ns']//1000000,b)
   loc=np.flatnonzero(np.array(audit['source_slots'])==example['slot'])
   if not len(loc):station_details.append(dict(station=b,active=False));continue
   i=int(loc[0]);r=audit['range_m'][i];vr=audit['radial_velocity_mps'][i];u=audit['spatial_direction_cosine'][i]
   rec=dict(station=b,active=True,visible=audit['visible'][i],range_m=r,radial_velocity_mps=vr,u=u,isolated_snr_db=20+40*np.log10(100/r))
   if not rec['visible']:station_details.append(rec);continue
   coords,scores,power,mean=candidate_map(observation,config=cfg)
   centre=np.array([np.argmin(abs(rgrid-r)),np.argmin(abs(vgrid-vr)),np.argmin(abs(ugrid-u))])
   lo=np.maximum(centre-[1,1,2],0);hi=np.minimum(centre+[2,2,3],power.shape)
   region=power[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
   point=tuple((np.array(np.unravel_index(np.argmax(region),region.shape))+lo).tolist())
   score=float(power[point]/(mean[point]*alpha[point]))
   is_nms=bool(np.any(np.all(coords==point,axis=1)))
   near=[]
   for j in range(len(audit['range_m'])):
    if j==i or not audit['visible'][j]:continue
    diff=np.abs([(audit['range_m'][j]-r)/(W.c/(2*W.B)),(audit['radial_velocity_mps'][j]-vr)/(W.wavelength/(2*W.CPI)),(audit['spatial_direction_cosine'][j]-u)*32])
    near.append(dict(source_slot=audit['source_slots'][j],separation_bins=diff.tolist(),within_mainlobe_box=bool(np.all(diff<=[2,2,4]))))
   masked=np.where(valid,power,0.).astype(float)
   # Diagnostic alternative, same physical FFT map and search cells. This is
   # NOT a calibrated detector and does not produce or save detections.
   alt_count=_box_sum(valid.astype(float),(9,9,17))-_box_sum(valid.astype(float),(3,3,9))
   alt_total=_box_sum(masked,(9,9,17))-_box_sum(masked,(3,3,9))
   n=float(alt_count[point]);alt_mean=float(alt_total[point]/n);alt_alpha=n*np.expm1(-np.log(pfa)/n)
   alt_score=float(power[point]/(alt_mean*alt_alpha))
   rec.update(local_peak_index=list(point),truth_nearest_index=centre.tolist(),local_peak_is_global_nms_candidate=is_nms,
      formal_cfar_score=score,formal_threshold=cfg['threshold_multiplier'],formal_passes_threshold=score>=cfg['threshold_multiplier'],
      formal_noise_to_calibrated_noise=float(mean[point]/known_noise),
      alternative_guard_4_outer_8_score=alt_score,alternative_noise_to_calibrated_noise=alt_mean/known_noise,
      alternative_passes_old_multiplier_for_diagnosis_only=alt_score>=cfg['threshold_multiplier'],
      nearby_targets=near,nearby_unresolved_targets=sum(n['within_mainlobe_box'] for n in near))
   station_details.append(rec)
  details.append(dict(example=example,stations=station_details))
  print('diagnosed episode',ei,flush=True)
 def aggregate(rows):
  return dict(count=len(rows),min_range_quantiles=np.quantile([r['min_range'] for r in rows],[0,.25,.5,.75,1]).tolist() if rows else [],
              nearest_snr_quantiles=np.quantile([r['best_isolated_snr_db'] for r in rows],[0,.25,.5,.75,1]).tolist() if rows else [],
              detection_station_histogram={str(k):sum(r['stations_with_nearby_cached_detection']==k for r in rows) for k in range(4)})
 result=dict(scope='training only; truth-assisted local checks NEVER used for production detector search',
             stationary_covered=aggregate([r for r in summary if r['covered']]),stationary_uncovered=aggregate(missing),
             deterministic_selection='first missed stationary object of each training audit episode, capped at ten episodes/30 cubes',
             cases=details,formal_detector_unchanged=True,alternative_requires_independent_noise_recalibration=True)
 visible_examples=[s for c in details for s in c['stations'] if s.get('visible')]
 nearest=[min([s for s in c['stations'] if s.get('visible')],key=lambda s:s['range_m']) for c in details]
 result['local_check_summary']=dict(cubes_generated=3*len(details),visible_station_examples=len(visible_examples),original_threshold_passes=sum(s['formal_passes_threshold'] for s in visible_examples),diagnostic_alternative_threshold_passes=sum(s['alternative_passes_old_multiplier_for_diagnosis_only'] for s in visible_examples),nearest_station_original_threshold_passes=sum(s['formal_passes_threshold'] for s in nearest),nearest_station_global_nms_candidates=sum(s['local_peak_is_global_nms_candidate'] for s in nearest),nearest_station_examples_with_mainlobe_neighbour=sum(s['nearby_unresolved_targets']>0 for s in nearest),nearest_station_noise_inflation_range=[min(s['formal_noise_to_calibrated_noise'] for s in nearest),max(s['formal_noise_to_calibrated_noise'] for s in nearest)],nearest_station_alternative_noise_inflation_range=[min(s['alternative_noise_to_calibrated_noise'] for s in nearest),max(s['alternative_noise_to_calibrated_noise'] for s in nearest)])
 result['interpretation']=['All ten nearest-station local peaks already exceed the original CFAR threshold; six survive global NMS. Four may be non-local-maxima from physical blending or suppressed by the resolution-box NMS. This diagnostic did not distinguish those two causes.', 'Nine of ten nearest-station examples have another stationary source within the 2x2x4-bin mainlobe box. These are examples, not a random estimate of population frequency.', 'The 1-bin angular guard is smaller than the first-zero angular mainlobe at four scan bins. Guard4/outer8 lowers self-contamination, but only one of twenty visible examples newly passes the old nominal threshold.', 'The alternative is diagnostic, uncalibrated, and does not prove improved detection or tracking coverage. No new formal observations or tracks were written.', 'Nearby cached detection counts are NOT one-to-one matches; multiple close true sources can share one detection. Use the main coverage_decomposition.json for unique-assignment recall.']
 result['minimal_recommendation']={'change': 'Guard angular half-width 4 and outer training angular half-width 8, leaving range/Doppler and waveform/geometry fixed', 'reason': 'Match CFAR exclusion to the 16-element array physical mainlobe on the 64-point scan', 'required_checks': 'Independent noise-only recalibration/holdout before a training-only full replay; preserve baseline outputs', 'expectation': 'Corrects demonstrated self-contamination but is unlikely alone to eliminate physical unresolved-target losses; do not promise >=90% coverage', 'if_still_failed': 'Keep the engineering gate failed and separate detector resolution/NMS from association before any larger architectural change'}
 (ROOT/'reports/f01c/detector_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


def nms_followup():
 from scipy.ndimage import maximum_filter
 source=SourceEpisodes();path=ROOT/'reports/f01c/detector_diagnosis.json';report=json.loads(path.read_text())
 cfg=json.loads((ROOT/'reports/f01c/baseline_before_guard_revision/configs/detector.json').read_text())
 idx,r,u,v,valid,count,alpha,pfa=_grid(2,1);records=[]
 for case in report['cases']:
  station=min([x for x in case['stations'] if x.get('visible')],key=lambda x:x['range_m'])
  if station['local_peak_is_global_nms_candidate']:continue
  example=case['example'];obs,_=source.observation(example['episode'],example['time_ns']//1000000,station['station'])
  coords,scores,power,mean=candidate_map(obs,config=cfg)
  local=(power==maximum_filter(np.where(valid,power,-np.inf),size=(3,3,3),mode='constant',cval=-np.inf))&valid
  points=np.argwhere(local&(power/(mean*alpha)>=cfg['threshold_multiplier']))
  truth=np.array(station['truth_nearest_index']);near=points[np.all(abs(points-truth)<=[2,2,4],axis=1)];entries=[]
  for point in near:
   pp=tuple(point);rp=r[point[0]];angle=np.arcsin(u[point[2]]/np.sqrt(1-25/rp**2));vr=v[point[1]]
   entries.append(dict(point=point.tolist(),delta_bins=(point-truth).tolist(),score=float(power[pp]/(mean[pp]*alpha[pp])),suppressed_by_large_nms=not bool(np.any(np.all(coords==point,axis=1))),z=[float(rp),float(angle),float(vr)]))
  records.append(dict(episode=example['episode'],slot=example['slot'],station=station['station'],true_range=station['range_m'],true_u=station['u'],local3_peaks_in_mainlobe_box=entries))
 report['nms_followup_same_four_cubes']=records
 report['minimal_recommendation']={'change': 'C01: angular guard4/outer8 plus local NMS1/1/1, retaining the existing ULA sidelobe check; waveform and station geometry unchanged', 'reason': 'The guard was inside the physical mainlobe; the follow-up also demonstrated four above-threshold local peaks removed by the oversized NMS box', 'required_checks': 'Independent noise-only recalibration/holdout before a training-only full replay; preserve baseline outputs', 'expectation': 'Corrects observed self-contamination and deletion of locally resolved peaks; only a fresh noise calibration and training traffic replay can establish resulting coverage', 'if_still_failed': 'Keep the engineering gate failed and separate detector resolution/NMS from association before any larger architectural change'}
 report['nms_followup_conclusion']='All four revisited observations contain above-threshold 3x3x3 local maxima suppressed by the larger NMS. The ep47 and ep62 peaks are near the missing stationary targets; ep77 still has angle bias. Evidence supports local1/1/1 NMS with retained array-sidelobe filtering, followed by independent noise recalibration.'
 path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
 main()
 nms_followup()

