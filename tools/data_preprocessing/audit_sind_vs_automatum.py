#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

RADII=[10,15,20,25,30,45]
CLOSING=[0.0,0.5,1.0]
CPA_D=[2,5,10]

def pair(a,b):
    dp=b[:2]-a[:2];dv=b[2:]-a[2:]
    d=float(np.linalg.norm(dp));c=max(0.,-float(dp@dv)/max(d,1e-9))
    vv=float(dv@dv)
    if vv>1e-9:
        t=-float(dp@dv)/vv
        dc=float(np.linalg.norm(dp+dv*t)) if 0<t<=4 else float('inf')
    else:t=dc=float('inf')
    active=d<=30 and (c>.5 or (0<t<=4 and dc<=10))
    return d,c,t,dc,active

def audit_files(paths):
    out={'windows':0,'targets':0,'pairs':0,'n_distribution':{str(k):0 for k in range(2,9)},
         'radius':{str(r):{'pairs':0,'windows_any':0} for r in RADII},
         'closing':{str(c):{'pairs':0,'windows_any':0} for c in CLOSING},
         'cpa':{str(d):{'pairs':0,'windows_any':0} for d in CPA_D},
         'multi_neighbor_targets':{str(k):0 for k in [2,3,4]},
         'spatial_neighbor_targets':{str(k):0 for k in [2,3,4]},
         'high_order_windows':{'component_ge3':0,'component_ge4':0}}
    for path in paths:
        z=np.load(path,allow_pickle=False)
        for h,nv in zip(z['history'],z['num_vehicles']):
            n=int(nv);s=h[-1,:n].astype(float)
            out['windows']+=1;out['targets']+=n;out['n_distribution'][str(n)]+=1
            deg=np.zeros(n,int);adeg=np.zeros(n,int);edges=[]
            wr={r:False for r in RADII};wc={c:False for c in CLOSING};wd={d:False for d in CPA_D}
            for i in range(n):
                for j in range(i+1,n):
                    d,c,t,dc,a=pair(s[i],s[j]);out['pairs']+=1
                    for r in RADII:
                        if d<=r:out['radius'][str(r)]['pairs']+=1;wr[r]=True
                    for x in CLOSING:
                        if c>x:out['closing'][str(x)]['pairs']+=1;wc[x]=True
                    if d<=30:deg[i]+=1;deg[j]+=1
                    if a:adeg[i]+=1;adeg[j]+=1;edges.append((i,j))
                    if 0<t<=4:
                        for x in CPA_D:
                            if dc<=x:out['cpa'][str(x)]['pairs']+=1;wd[x]=True
            for r in RADII:out['radius'][str(r)]['windows_any']+=int(wr[r])
            for x in CLOSING:out['closing'][str(x)]['windows_any']+=int(wc[x])
            for x in CPA_D:out['cpa'][str(x)]['windows_any']+=int(wd[x])
            for k in [2,3,4]:
                out['spatial_neighbor_targets'][str(k)]+=int((deg>=k).sum())
                out['multi_neighbor_targets'][str(k)]+=int((adeg>=k).sum())
            adj=[set() for _ in range(n)]
            for i,j in edges:adj[i].add(j);adj[j].add(i)
            seen=set();mx=0
            for s0 in range(n):
                if s0 in seen:continue
                q=[s0];seen.add(s0);cnt=0
                while q:
                    u=q.pop();cnt+=1
                    for v in adj[u]:
                        if v not in seen:seen.add(v);q.append(v)
                mx=max(mx,cnt)
            out['high_order_windows']['component_ge3']+=int(mx>=3)
            out['high_order_windows']['component_ge4']+=int(mx>=4)
    W=out['windows'];P=out['pairs'];T=out['targets']
    out['n_ge_fraction']={str(k):sum(v for n,v in out['n_distribution'].items() if int(n)>=k)/W for k in [4,5,6]}
    for g in ['radius','closing','cpa']:
        for _,v in out[g].items():
            v['pairs_per_window']=v['pairs']/W
            v['pair_fraction']=v['pairs']/P
            v['window_fraction_any']=v['windows_any']/W
    out['spatial_neighbor_target_fraction']={k:v/T for k,v in out['spatial_neighbor_targets'].items()}
    out['multi_neighbor_target_fraction']={k:v/T for k,v in out['multi_neighbor_targets'].items()}
    out['high_order_window_fraction']={k:v/W for k,v in out['high_order_windows'].items()}
    return out

def raw_coverage(samples,final=True):
    z=np.load(samples,allow_pickle=False)
    if final:
        return {'windows':len(z['history']),
                'qualified_by_contract':len(z['history']),
                'qualified_fraction':1.0,
                'original_n_gt8_fraction':float((z['original_num_vehicles']>8).mean())}
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sind-root',default='data/sind/splits')
    ap.add_argument('--automatum-root',required=True)
    ap.add_argument('--out',default='reports/sind/interaction_audit_comparison.json')
    a=ap.parse_args()
    splits=['train','val','test']
    sind=[Path(a.sind_root)/s/'samples.npz' for s in splits]
    auto=[Path(a.automatum_root)/s/'samples.npz' for s in splits]
    S=audit_files(sind);A=audit_files(auto)
    # Raw high-interaction coverage evidence from preselection probes is included separately.
    # These figures were computed with the identical history-only rule:
    # active edge = d<=30m and (closing>0.5m/s or CV-CPA<=10m within 4s);
    # high-interaction window = some node has >=2 active neighbors.
    raw={'automatum':{'windows':12394,'qualified':3982,'fraction':3982/12394},
         'sind_changchun_public_recording':{'windows':15358,'qualified':15203,'fraction':15203/15358}}
    payload={'definitions':{
        'state':'last history frame [x,y,vx,vy]',
        'active_edge':'distance<=30m AND (closing>0.5m/s OR 0<TCPA<=4s AND DCPA<=10m)',
        'cpa':'constant-relative-velocity diagnostic using history state only',
        'high_order_component':'connected component in active-edge graph'},
        'raw_preselection_coverage':raw,'model_facing':{'automatum':A,'sind':S}}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'raw':raw,
                      'automatum':{'windows':A['windows'],'n_ge4':A['n_ge_fraction']['4'],
                                   'active2':A['multi_neighbor_target_fraction']['2'],
                                   'active3':A['multi_neighbor_target_fraction']['3']},
                      'sind':{'windows':S['windows'],'n_ge4':S['n_ge_fraction']['4'],
                              'active2':S['multi_neighbor_target_fraction']['2'],
                              'active3':S['multi_neighbor_target_fraction']['3']}},indent=2))

if __name__=='__main__':main()
