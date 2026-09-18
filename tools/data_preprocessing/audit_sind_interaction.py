#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RADII=[10,15,20,25,30,45]
CLOSE=[0.0,0.5,1.0]
CPA_D=[2,5,10]
HORIZON=4.0
TOTAL=40; HIST=20

def graph_stats(states):
    n=len(states)
    dp=states[None,:,:2]-states[:,None,:2]
    dv=states[None,:,2:]-states[:,None,2:]
    dist=np.linalg.norm(dp,axis=2)
    dot=np.sum(dp*dv,axis=2)
    vv=np.sum(dv*dv,axis=2)
    closing=np.maximum(0.0,-dot/np.maximum(dist,1e-9))
    tcpa=np.where(vv>1e-9,-dot/np.maximum(vv,1e-9),np.inf)
    valid=(tcpa>0)&(tcpa<=HORIZON)
    safe=np.where(valid,tcpa,0.0)
    dcpa=np.where(valid,np.linalg.norm(dp+dv*safe[:,:,None],axis=2),np.inf)
    tri=np.triu(np.ones((n,n),bool),1)
    active=(dist<=30)&((closing>.5)|(valid&(dcpa<=10)))&tri
    spatial30=(dist<=30)&tri
    return dist,closing,tcpa,dcpa,active,spatial30,tri

def component_max(n,edge_mask):
    adj=[[] for _ in range(n)]
    ii,jj=np.nonzero(edge_mask)
    for i,j in zip(ii.tolist(),jj.tolist()):
        adj[i].append(j);adj[j].append(i)
    seen=set();best=1
    for s in range(n):
        if s in seen: continue
        q=[s];seen.add(s);k=0
        while q:
            u=q.pop();k+=1
            for v in adj[u]:
                if v not in seen:seen.add(v);q.append(v)
        best=max(best,k)
    return best
class Acc:
    def __init__(self):
        self.windows=0;self.targets=0;self.pairs=0
        self.n={k:0 for k in range(2,9)}
        self.r={x:[0,0] for x in RADII}
        self.cl={x:[0,0] for x in CLOSE}
        self.cpa={x:[0,0] for x in CPA_D}
        self.neigh={k:[0,0] for k in [2,3,4]}
        self.active={k:[0,0] for k in [2,3,4]}
        self.comp={k:0 for k in [3,4,5]}
        self.tcpa=[];self.dcpa=[]
        self.original_gt8=0
    def add(self,states,original_n=None):
        n=len(states);self.windows+=1;self.targets+=n;self.pairs+=n*(n-1)//2
        if 2<=n<=8:self.n[n]+=1
        if original_n is not None:self.original_gt8+=int(original_n>8)
        dist,closing,tcpa,dcpa,ae,se,tri=graph_stats(states)
        for x in RADII:
            q=int(((dist<=x)&tri).sum());self.r[x][0]+=q;self.r[x][1]+=int(q>0)
        for x in CLOSE:
            q=int(((closing>x)&tri).sum());self.cl[x][0]+=q;self.cl[x][1]+=int(q>0)
        for x in CPA_D:
            q=int(((dcpa<=x)&tri).sum());self.cpa[x][0]+=q;self.cpa[x][1]+=int(q>0)
        deg=(se|se.T).sum(1);adeg=(ae|ae.T).sum(1)
        for k in [2,3,4]:
            self.neigh[k][0]+=int((deg>=k).sum());self.neigh[k][1]+=int((deg>=k).any())
            self.active[k][0]+=int((adeg>=k).sum());self.active[k][1]+=int((adeg>=k).any())
        cm=component_max(n,ae)
        for k in [3,4,5]:self.comp[k]+=int(cm>=k)
        pair_valid=tri&np.isfinite(dcpa)
        self.tcpa.extend(tcpa[pair_valid].tolist());self.dcpa.extend(dcpa[pair_valid].tolist())
    def report(self):
        W=max(self.windows,1); T=max(self.targets,1); P=max(self.pairs,1)
        def qs(a):
            return [float(np.quantile(a,x)) for x in [.1,.5,.9,.95]] if a else []
        return {
            'windows':self.windows,'targets':self.targets,'pairs':self.pairs,
            'n_distribution':{str(k):self.n[k] for k in range(2,9)},
            'n_ge_fraction':{str(k):sum(v for n,v in self.n.items() if n>=k)/W for k in [4,5,6]},
            'original_n_gt8_fraction':self.original_gt8/W,
            'radius':{str(x):{'pairs':self.r[x][0],'pairs_per_window':self.r[x][0]/W,
                              'pair_fraction':self.r[x][0]/P,'window_fraction_any':self.r[x][1]/W}
                      for x in RADII},
            'closing':{str(x):{'pairs':self.cl[x][0],'pairs_per_window':self.cl[x][0]/W,
                               'pair_fraction':self.cl[x][0]/P,'window_fraction_any':self.cl[x][1]/W}
                       for x in CLOSE},
            'cpa':{str(x):{'pairs':self.cpa[x][0],'pairs_per_window':self.cpa[x][0]/W,
                           'pair_fraction':self.cpa[x][0]/P,'window_fraction_any':self.cpa[x][1]/W}
                   for x in CPA_D},
            'spatial_neighbor':{str(k):{'target_fraction':self.neigh[k][0]/T,
                                        'window_fraction_any':self.neigh[k][1]/W} for k in [2,3,4]},
            'active_neighbor':{str(k):{'target_fraction':self.active[k][0]/T,
                                       'window_fraction_any':self.active[k][1]/W} for k in [2,3,4]},
            'active_component_window_fraction':{str(k):self.comp[k]/W for k in [3,4,5]},
            'tcpa_s_quantiles_p10_p50_p90_p95':qs(self.tcpa),
            'dcpa_m_quantiles_p10_p50_p90_p95':qs(self.dcpa),
        }
def audit_npz(paths):
    acc=Acc()
    for p in paths:
        z=np.load(p,allow_pickle=False)
        orig=z['original_num_vehicles'] if 'original_num_vehicles' in z.files else np.full(len(z['history']),-1)
        for h,n,o in zip(z['history'],z['num_vehicles'],orig):
            acc.add(h[-1,:int(n)].astype(float),None if int(o)<0 else int(o))
    return acc.report()

def audit_supply_csv(paths):
    acc=Acc(); total_candidates=0; qualified=0
    for p in paths:
        d=pd.read_csv(p)
        if 'frame_id' not in d.columns:
            d['frame_id']=np.rint(d.timestamp.to_numpy(float)*29.97).astype(int)//3
        groups=d.groupby('scene_id') if 'scene_id' in d.columns else [(0,d)]
        for _,s in groups:
            tracks={int(t):g.set_index('frame_id')[['x','y','vx','vy']] for t,g in s.groupby('vehicle_id')}
            spans={t:(int(g.index.min()),int(g.index.max())) for t,g in tracks.items()}
            f0=int(s.frame_id.min()); f1=int(s.frame_id.max())
            for f in range(f0,f1-TOTAL+2):
                ids=[t for t,(lo,hi) in spans.items() if lo<=f and hi>=f+TOTAL-1]
                if len(ids)<2:
                    continue
                total_candidates+=1
                S=np.array([tracks[t].loc[f+HIST-1].to_numpy(float) for t in ids])
                _,_,_,_,ae,_,_=graph_stats(S)
                deg=(ae|ae.T).sum(1)
                if (deg>=2).any():
                    qualified+=1
                    acc.add(S,len(ids))
    rep=acc.report()
    rep['candidate_windows_ge2']=total_candidates
    rep['high_interaction_qualified']=qualified
    rep['high_interaction_fraction']=qualified/max(total_candidates,1)
    return rep
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--automatum-root',default=str(Path(__file__).resolve().parents[2]))
    args=ap.parse_args()
    root=Path(args.root); auto=Path(args.automatum_root)

    sind_npz=[root/f'data/sind/splits/{s}/samples.npz' for s in ['train','val','test']]
    auto_npz=[auto/f'data/automatum_t_crossing/splits/{s}/samples.npz' for s in ['train','val','test']]

    supply=[]
    raw_sources=[
        (0,root/'data/sind/raw/changchun/Veh_smoothed_tracks.csv'),
        (1,root/'data/sind/raw/xian/Veh_smoothed_tracks.csv')
    ]
    for sid,p in raw_sources:
        d=pd.read_csv(p,usecols=['track_id','frame_id','agent_type','x','y','vx','vy'])
        d=d[d.agent_type.isin(['car','bus','truck'])].copy()
        mapping={v:i+1 for i,v in enumerate(sorted(d.track_id.unique()))}
        d['vehicle_id']=d.track_id.map(mapping)
        d['scene_id']=sid
        supply.append(d[['scene_id','vehicle_id','frame_id','x','y','vx','vy']])
    tmp=root/'data/sind/.audit_supply.csv'
    pd.concat(supply,ignore_index=True).to_csv(tmp,index=False)

    auto_paths=[auto/f'data/automatum_t_crossing/splits/{s}/trajectories.csv'
                for s in ['train','val','test']]
    out={
        'definitions':{
            'snapshot':'last history frame',
            'active_edge':'distance<=30m AND (closing>0.5m/s OR 0<TCPA<=4s AND DCPA<=10m)',
            'closing':'max(0,-dot(delta_p,delta_v)/distance), all unordered pairs',
            'cpa':'constant-relative-velocity CPA over the next 4 seconds',
            'higher_order':'largest connected component of active-edge graph',
        },
        'source_supply':{
            'sind_public_changchun_xian':audit_supply_csv([tmp]),
            'automatum_formal_split_trajectories':audit_supply_csv(auto_paths),
        },
        'model_facing':{
            'sind':audit_npz(sind_npz),
            'automatum':audit_npz(auto_npz),
        },
    }
    tmp.unlink(missing_ok=True)
    outp=root/'reports/sind/sind_interaction_audit.json'
    outp.parent.mkdir(parents=True,exist_ok=True)
    outp.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    s=out['source_supply']; m=out['model_facing']
    print('SUPPLY Sind',s['sind_public_changchun_xian']['high_interaction_fraction'],
          'Auto',s['automatum_formal_split_trajectories']['high_interaction_fraction'])
    print('MODEL active>=3 Sind',m['sind']['active_neighbor']['3']['target_fraction'],
          'Auto',m['automatum']['active_neighbor']['3']['target_fraction'])
    print('WROTE',outp)

if __name__=='__main__':
    main()
