"""Read-only F01-A output review and exploratory training trajectory plots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from station_geometry import visibility
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports/f01a'
a=np.load(ROOT/'data/f01_source/source_states.npy',mmap_mode='r',allow_pickle=False)
episodes=[json.loads(l) for l in (R/'episodes.jsonl').read_text().splitlines()]
geom=json.loads((R/'geometry.json').read_text()); bs=np.array(geom['stations_xy_m'])
keys,first,counts=np.unique(a['source_key'],return_index=True,return_counts=True)
tracks={int(k):a[i:i+n] for k,i,n in zip(keys,first,counts)}
def source_rows(e):
    chunks=[]
    for k in e['source_keys']:
        t=tracks[k]
        chunks.append(t[(t['time_ms']>=e['start_ms'])&(t['time_ms']<e['end_exclusive_ms'])])
    return np.concatenate(chunks)
def visible(rows):
    return visibility(np.column_stack([rows['x'],rows['y']]),bs,boresights=np.array(geom['boresight_rad']))
stats={}
for split in ['train','V_select','V_confirm','test']:
    es=[e for e in episodes if e['split']==split]
    points=np.concatenate([source_rows(e) for e in es]) if es else np.empty(0,dtype=a.dtype)
    active=points[points['past_frames']>=3]
    vis=visible(active)
    stats[split]={'episodes':len(es),'source_time_blocks':len(set(e['source_block'] for e in es)),'source_intervals':len(set(e['start_ms'] for e in es)),'prediction_grid_count':sum(len(e['prediction_grid_ms']) for e in es),'active_vehicle_time_rows_with_anchor_repeats':len(active),'any_station_geometry_fraction':float(vis.any(1).mean()) if len(active) else None,'two_station_geometry_fraction':float((vis.sum(1)>=2).mean()) if len(active) else None,'all_station_geometry_fraction':float(vis.all(1).mean()) if len(active) else None,'cohort_size_range':[min(map(lambda e:len(e['source_keys']),es),default=0),max(map(lambda e:len(e['source_keys']),es),default=0)]}
(R/'scene_quality.json').write_text(json.dumps(stats,indent=2))
train=[e for e in episodes if e['split']=='train']
chosen=[train[i] for i in np.linspace(0,len(train)-1,4,dtype=int)]
plt.rcParams.update({'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(11,9))
for ax,e in zip(axes.flat,chosen):
    for slot,k in enumerate(e['source_keys']):
        t=tracks[k]; t=t[(t['time_ms']>=e['start_ms'])&(t['time_ms']<e['end_exclusive_ms'])]
        line,=ax.plot(t['x'],t['y'],label=f'Vehicle {slot+1}',lw=1.5)
        ax.scatter(t['x'][0],t['y'][0],s=22,color=line.get_color())
        ax.scatter(t['x'][-1],t['y'][-1],s=25,marker='x',color=line.get_color())
    ax.set_title(f"Training episode at +{(e['start_ms']-1118935680200)/1000:.0f}s | {len(e['source_keys'])} vehicles")
    ax.set(xlabel='Local x (m)',ylabel='Local y (m)'); ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2)
    ax.legend(fontsize=7,ncol=2)
fig.suptitle('Actual source trajectories: four fixed training samples\nDot = first observed position; cross = last observed position',fontsize=13)
fig.tight_layout(); fig.savefig(R/'training_trajectories.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,9))
# Training only: no test trajectory visualization.
manifest=json.loads((R/'split_manifest.json').read_text()); lo,hi=manifest['guarded_intervals_ms']['train']
sub=a[(a['time_ms']>=lo)&(a['time_ms']<hi)&a['owned']][::100]
v=visible(sub).any(1)
ax.scatter(sub['x'][~v],sub['y'][~v],s=2,c='#c25a45',label='Outside all station sectors')
ax.scatter(sub['x'][v],sub['y'][v],s=2,c='#457b9d',label='Inside at least one sector')
ax.scatter(bs[:,0],bs[:,1],s=110,marker='^',c='black',label='Base stations')
for i,b in enumerate(bs): ax.annotate(f'BS{i+1}',b,xytext=(6,6),textcoords='offset points')
ax.set(xlabel='Local x (m)',ylabel='Local y (m)',title='Frozen station geometry over training source positions\nGeometry only: not detection accuracy')
ax.set_aspect('equal'); ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(R/'training_geometry.png',dpi=150); plt.close(fig)
e=chosen[0]; preview=source_rows(e)
preview=preview[preview['time_ms']<e['start_ms']+1000]
pd.DataFrame.from_records(preview).to_csv(R/'sample_source_first_second.csv',index=False)
print(json.dumps(stats,indent=2))


# Compare revisions on exactly the same unique active training source rows.
fit=np.zeros(len(a),dtype=bool)
for e in train:
    fit |= np.isin(a['source_key'],e['source_keys']) & (a['time_ms']>=e['start_ms']) & (a['time_ms']<e['end_exclusive_ms'])
rows=a[fit & (a['past_frames']>=3)]
old=json.loads((ROOT/'reports/amendment_A01_before/f01a/geometry.json').read_text())
xy=np.column_stack([rows['x'],rows['y']])
old_xy=xy+np.array(geom['source_origin_m'])-np.array(old['source_origin_m'])
old_bs=np.array(old['stations_xy_m'])
comparison={}
for name,coords,stations,bore in [('old',old_xy,old_bs,np.arctan2(-old_bs[:,1],-old_bs[:,0])),
                                  ('A01',xy,bs,np.array(geom['boresight_rad']))]:
    v=visibility(coords,stations,bore)
    comparison[name]={'rows_same_unique_retained_training_source':len(rows),'one':float(v.any(1).mean()),'two':float((v.sum(1)>=2).mean()),'three':float(v.all(1).mean())}
(R/'geometry_same_training_comparison.json').write_text(json.dumps(comparison,indent=2))
assert comparison['A01']['one']>=.99 and comparison['A01']['two']>=.95
