#!/usr/bin/env python3
"""Fast final gate for the frozen SinD migration.

Heavy numerical checks live in two independent validators:
- sind_dataset_validation.json: GT/window/split/selection/leakage
- sind_isac_validation.json: 3-BS cache/SNR/loader contract
This gate cross-checks those reports plus raw audit, geometry, interaction
evidence, build marker and reproducibility paths.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'reports/sind'
D=ROOT/'data/sind'
checks=[]

def load(name):
    return json.loads((R/name).read_text(encoding='utf-8'))

def ck(name,ok,detail=None):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})
    if not ok:
        print('FAIL',name,detail,flush=True)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()
gt=load('sind_dataset_validation.json')
isac=load('sind_isac_validation.json')
raw=load('sind_raw_audit.json')
inter=load('sind_interaction_audit.json')
geom=load('sind_geometry_search.json')
build=load('sind_dataset_build.json')
cfg=json.loads((ROOT/'configs/sind_controlled_isac.json').read_text(encoding='utf-8'))

ck('gt_validation_pass',gt.get('status')=='PASS',
   {'checks':len(gt.get('checks',[])),
    'passed':sum(x.get('pass',False) for x in gt.get('checks',[]))})
ck('isac_validation_pass',isac.get('status')=='PASS',
   {'checks':len(isac.get('checks',[])),
    'passed':sum(x.get('pass',False) for x in isac.get('checks',[]))})

for city,scene in raw['scenes'].items():
    formal=scene['formal']
    ck(f'raw_{city}_missing',scene['missing_required']==0,scene['missing_required'])
    ck(f'raw_{city}_duplicate',scene['duplicate_track_frame']==0,scene['duplicate_track_frame'])
    ck(f'raw_{city}_noncontiguous',formal['noncontiguous_tracks']==0,formal['noncontiguous_tracks'])
    ck(f'raw_{city}_nonfinite',formal['nonfinite_state_count']==0,formal['nonfinite_state_count'])
    ck(f'raw_{city}_jump',formal['step_speed_gt50_count']==0,formal['step_speed_gt50_count'])
    tb=scene['timebase']
    ck(f'raw_{city}_nominal_10hz',
       abs(tb['median_dt_s']-0.1)<=0.005 and tb['resampled'] is False,tb)

ck('geometry_train_only',geom.get('scope')=='train_only',geom.get('scope'))
ck('config_test_not_tuned',cfg['dataset']['test_scope'].startswith('never used'),
   cfg['dataset']['test_scope'])
cfg_by_scene={int(x['scene_id']):x for x in cfg['scenes']}
for sid_str,g in geom['scenes'].items():
    sid=int(sid_str); best=g['best']; c=cfg_by_scene[sid]
    ck(f'geometry_scene{sid}_stations_frozen',
       c['stations_xy_m']==best['stations'],
       {'config':c['stations_xy_m'],'search':best['stations']})
    ck(f'geometry_scene{sid}_boresights_frozen',
       all(abs(a-b)<1e-12 for a,b in zip(c['boresights_deg'],best['boresights_deg'])),
       {'config':c['boresights_deg'],'search':best['boresights_deg']})
    ck(f'geometry_scene{sid}_coverage_ge2',best['coverage_ge2']==1.0,best['coverage_ge2'])

supply=inter['source_supply']
mf=inter['model_facing']
sind_supply=supply['sind_public_changchun_xian']
auto_supply=supply['automatum_formal_split_trajectories']
ck('interaction_supply_gain',
   sind_supply['high_interaction_fraction']>auto_supply['high_interaction_fraction'],
   {'sind':sind_supply['high_interaction_fraction'],
    'automatum':auto_supply['high_interaction_fraction']})
ck('interaction_active3_gain',
   mf['sind']['active_neighbor']['3']['target_fraction']>
   mf['automatum']['active_neighbor']['3']['target_fraction'],
   {'sind':mf['sind']['active_neighbor']['3']['target_fraction'],
    'automatum':mf['automatum']['active_neighbor']['3']['target_fraction']})
ck('interaction_nge4_gain',
   mf['sind']['n_ge_fraction']['4']>mf['automatum']['n_ge_fraction']['4'],
   {'sind':mf['sind']['n_ge_fraction']['4'],
    'automatum':mf['automatum']['n_ge_fraction']['4']})
marker=D/'.BUILD_COMPLETE'
ck('build_marker_exists',marker.exists(),'data/sind/.BUILD_COMPLETE')
if marker.exists():
    m=json.loads(marker.read_text(encoding='utf-8'))
    report=ROOT/m['report']
    ck('build_marker_report_hash',sha256(report)==m['report_sha256'],
       {'actual':sha256(report),'marker':m['report_sha256']})

expected_counts={'train':15802,'val':1880,'test':2086}
for split,n in expected_counts.items():
    got=build['splits'][split]['samples']
    ck(f'{split}_sample_count',got==n,got)
    for rel in [
        f'data/sind/splits/{split}/samples.npz',
        f'data/sind/splits/{split}/trajectories.csv',
        f'data/sind/isac/{split}/sensing_cache.npz',
        f'data/sind/isac/{split}/sensing_manifest.json',
    ]:
        ck(f'{split}_file_{Path(rel).name}',(ROOT/rel).exists(),rel)

bad_paths=[]
for p in list((ROOT/'tools/data_preprocessing').glob('*sind*.py')) + [
    ROOT/'frontend/sind_prediction_dataset.py',
    ROOT/'frontend/controlled_isac/sind_frontend.py',
]:
    if p.resolve()==Path(__file__).resolve():
        continue
    text=p.read_text(encoding='utf-8')
    forbidden='ICCT_' + 'sind_final_'
    if forbidden in text:
        bad_paths.append(str(p.relative_to(ROOT)))
ck('no_temp_worktree_absolute_paths',not bad_paths,bad_paths)
status=all(x['pass'] for x in checks)
payload={
    'status':'PASS' if status else 'FAIL',
    'checks_total':len(checks),
    'checks_passed':sum(x['pass'] for x in checks),
    'heavy_validation':{
        'gt':{'status':gt.get('status'),'checks':len(gt.get('checks',[]))},
        'isac':{'status':isac.get('status'),'checks':len(isac.get('checks',[]))},
    },
    'sample_counts':{s:build['splits'][s]['samples'] for s in ['train','val','test']},
    'interaction_key':{
        'source_high_interaction_fraction_sind':sind_supply['high_interaction_fraction'],
        'source_high_interaction_fraction_automatum':auto_supply['high_interaction_fraction'],
        'model_active_neighbors_ge3_sind':mf['sind']['active_neighbor']['3']['target_fraction'],
        'model_active_neighbors_ge3_automatum':mf['automatum']['active_neighbor']['3']['target_fraction'],
    },
    'checks':checks,
}
(R/'sind_migration_validation.json').write_text(
    json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:payload[k] for k in
                  ['status','checks_total','checks_passed','sample_counts','interaction_key']},
                 indent=2,ensure_ascii=False),flush=True)
if not status:
    raise SystemExit(2)
