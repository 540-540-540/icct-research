"""Controlled known-target symbol-level sensing adapter; no blind detector/tracker."""
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .echo_source import SourceEpisodes,ROOT
from .run_frontend import load,dump

OUT=ROOT/'reports/symbol_level';DATA=ROOT/'data/symbol_frontend';CONFIG=ROOT/'configs/symbol_frontend.json'

def configure():
    source=SourceEpisodes();position=[]
    for ep in source.episodes:
        if ep['split']!='train':continue
        for key in ep['source_keys']:
            tr=source.tracks[key];a=tr[(tr['time_ms']>=ep['start_ms'])&(tr['time_ms']<ep['end_exclusive_ms'])]
            position.append(np.column_stack([a['x'],a['y']]))
    xy=np.concatenate(position);lo=xy.min(0)-5;hi=xy.max(0)+5
    corners=np.array([[x,y] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]]]);maxr=float(np.linalg.norm(corners[:,None]-source.stations[None],axis=2).max())
    c=dict(revision='A03-symbol-v2-positive-snr',source='Wei et al IEEE TVT2024 DOI10.1109/TVT.2023.3304856',scope='Known separated targets, independently sensed then grouped into up to8; no blind detection or association claims',waveform=dict(fc=24e9,B=93.1e6,K=256,N=256,T=12.375e-6,c=299792458.),paper_reference_K=128,stations=source.stations.tolist(),xy_bounds=np.column_stack([lo,hi]).tolist(),v_bounds=[[-40.,40.],[-40.,40.]],bounds_source='Retained training position extrema plus5m each axis; velocity domain predeclared +/-40m/s each axis',max_corner_range_m=maxr,snr_db=[5,10,15,20],nominal_snr_db=20,snr_definition='Per complex received OFDM symbol entry before processing: E|a X|^2/E|noise|^2; unit QPSK; received amplitude10^(SNR/20), complex noise variance1; each BS same controlled received SNR',height_model='2D common plane as paper, no old ULA AoA channel',time_model='Snapshot at deadline, constant target parameters within symbol burst ending by deadline; no future source rows',target_access='Ideal target separation and known current correspondence assumed by user; not a modeled multi-target discovery mechanism',power_resource_model='Independent target-conditioned sensing experiments; not a full shared-bandwidth multiuser link budget',seed_recipe='SeedSequence [2026,episode_index,frame_index,source_slot,101]; no true_id enters estimator or predictor',search=dict(oversample=2,position_half_width=3.,position_step=.25,position_fine_step=.025,velocity_half_width=3.,velocity_step=.25,velocity_fine_step=.025))
    assert maxr<299792458*256/(2*93.1e6)
    dump(CONFIG,c)
    old=load(ROOT/'reports/f01c/development_manifest.json')
    dump(OUT/'audit_manifest.json',dict(train_episodes=old['train_tracking_audit_episodes'],development_episodes=old['development_episodes'],elapsed_ms=[2000,10000,18000],snr_db=c['snr_db'],scope='Fixed snapshots for current upstream model validation; not F01-D complete temporal predictor cache'))
    print(json.dumps({k:c[k] for k in ['revision','xy_bounds','max_corner_range_m','waveform','snr_db']},indent=2))

def estimate_frame(source,index,deadline_ms,snr_db,config,tracks=None):
    from .symbol_level import PaperWaveform,synthesize,divide_symbols,estimate
    ep=source.episodes[index];states,slots,used=source.at_time(ep,int(deadline_ms)*1000000,tracks=tracks)
    w=PaperWaveform(**config['waveform']);known=[];truth=[]
    for state,slot,sample in zip(states,slots,used):
        rng=np.random.default_rng(np.random.SeedSequence([2026,index,(deadline_ms-ep['start_ms'])//100,int(slot),101]))
        observation=synthesize(state[:2],state[2:],source.stations,waveform=w,snr_db=float(snr_db),rng=rng)
        b=divide_symbols(observation['Y'],observation['X'])
        result=estimate(b,source.stations,config['xy_bounds'],config['v_bounds'],waveform=w,search=config['search'])
        value=np.asarray(result['state_hat'],float);assert value.shape==(4,) and np.isfinite(value).all()
        known.append(dict(key=int(slot),state_hat=value.tolist(),exists=True,detected=True,measurement_time_ns=int(deadline_ms)*1000000,diagnostics=result['diagnostics']))
        truth.append(dict(key=int(slot),source_key=int(ep['source_keys'][slot]),state=state.tolist(),source_sample_ms=sample,position_sq_error=float(np.sum((value[:2]-state[:2])**2)),velocity_sq_error=float(np.sum((value[2:]-state[2:])**2))))
    return dict(time_ns=int(deadline_ms)*1000000,targets=known),dict(time_ns=int(deadline_ms)*1000000,targets=truth)

def audit_episode(index):
    source=SourceEpisodes();config=load(CONFIG);ep=source.episodes[index];manifest=load(OUT/'audit_manifest.json');reports=[];t=time.perf_counter()
    for snr in config['snr_db']:
        inputs=[];audit=[]
        for elapsed in manifest['elapsed_ms']:
            a,b=estimate_frame(source,index,ep['start_ms']+elapsed,snr,config);inputs.append(a);audit.append(b)
        label='snr_'+str(snr).replace('-','m')
        dump(DATA/'observations'/label/f'episode_{index:03d}.json',dict(episode_index=index,frames=inputs))
        dump(DATA/'truth_audit'/label/f'episode_{index:03d}.json',dict(episode_index=index,frames=audit))
        objs=[x for row in audit for x in row['targets']]
        reports.append(dict(snr_db=snr,targets=len(objs),position_sse=sum(x['position_sq_error'] for x in objs),velocity_sse=sum(x['velocity_sq_error'] for x in objs),estimate_outputs=sum(len(x['targets']) for x in inputs)))
    result=dict(episode_index=index,split=ep['split'],conditions=reports,elapsed_seconds=time.perf_counter()-t,revision=config['revision'],config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    dump(OUT/'episodes'/f'episode_{index:03d}.json',result);print(json.dumps(dict(index=index,seconds=result['elapsed_seconds'])),flush=True);return result

def run(workers,indices=None):
    m=load(OUT/'audit_manifest.json');indices=indices if indices is not None else m['train_episodes']+m['development_episodes']
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(audit_episode,indices))
    values=[]
    for split in ['train','V_select']:
        group=[r for r in rows if r['split']==split]
        if not group:continue
        for snr in load(CONFIG)['snr_db']:
            a=[x for r in group for x in r['conditions'] if x['snr_db']==snr];n=sum(x['targets'] for x in a)
            values.append(dict(split=split,snr_db=snr,episodes=len(group),targets=n,outputs=sum(x['estimate_outputs'] for x in a),position_rmse_m=float(np.sqrt(sum(x['position_sse'] for x in a)/n)),velocity_rmse_mps=float(np.sqrt(sum(x['velocity_sse'] for x in a)/n))))
    report=dict(revision=load(CONFIG)['revision'],config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),results=values,scope=m['scope'],qualification='Output availability follows assumed known separated targets; not detection recall. Errors include every produced estimate, no5m censoring',not_run=['V_confirm','test','F01-D full predictor packing','prediction training']);dump(OUT/'real_scene_quality.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['configure','audit']);p.add_argument('--workers',type=int,default=4);p.add_argument('--episodes',type=int,nargs='*');a=p.parse_args()
    if a.action=='configure':configure()
    else:run(a.workers,a.episodes)
