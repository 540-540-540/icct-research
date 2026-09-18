#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/'configs/sind_controlled_isac.json').read_text())
out=ROOT/'reports/sind/plots'
out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ROOT/'data/sind/splits/train/trajectories.csv')

for scene in cfg['scenes']:
    sid=int(scene['scene_id'])
    g=d[d.scene_id==sid]
    fig,ax=plt.subplots(figsize=(8,8))
    for _,t in g.groupby('vehicle_id'):
        ax.plot(t.x,t.y,linewidth=.25,alpha=.12)
    stations=scene['stations_xy_m']
    ax.scatter([x[0] for x in stations],[x[1] for x in stations],marker='^',s=90,label='ISAC BS')
    for i,(x,y) in enumerate(stations):
        ax.text(x,y,f' BS{i+1}',fontsize=9)
    cx,cy=scene['junction_center_xy_m']
    ax.scatter([cx],[cy],marker='x',s=70,label='geometry center')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(scene['label']+' — frozen 3-BS geometry (train trajectories)')
    ax.axis('equal')
    ax.grid(True,alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/f'sind_scene{sid}_3bs_geometry.png',dpi=180)
    plt.close(fig)
    print(out/f'sind_scene{sid}_3bs_geometry.png')
