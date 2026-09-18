"""A03 causal prefix replay and uncensored search diagnostics."""
import json
import numpy as np
from .run_symbol_frontend import SourceEpisodes,CONFIG,OUT,DATA,load,dump,estimate_frame

def main():
    source=SourceEpisodes();cfg=load(CONFIG);checks=[]
    for index in [0,95]:
        ep=source.episodes[index];cutoff=ep['start_ms']+5000
        truncated={k:v[v['time_ms']<=cutoff].copy() for k,v in source.tracks.items()}
        count=0
        for deadline in range(ep['start_ms']+100,cutoff+1,100):
            original,_=estimate_frame(source,index,deadline,cfg['nominal_snr_db'],cfg)
            replay,_=estimate_frame(source,index,deadline,cfg['nominal_snr_db'],cfg,tracks=truncated)
            assert original==replay,(index,deadline)
            for obj in original['targets']:
                assert 'source_key' not in obj and 'state' not in obj
            count+=len(original['targets'])
        checks.append(dict(episode_index=index,frames=50,target_states=count,exact_prefix_match=True))
        print(checks[-1],flush=True)
    counts={}
    for file in sorted((DATA/'observations').glob('snr_*/episode_*.json')):
        key=file.parent.name
        a=counts.setdefault(key,dict(outputs=0,search_failures=0,position_boundary=0,velocity_boundary=0))
        for row in load(file)['frames']:
            for obj in row['targets']:
                d=obj['diagnostics'];a['outputs']+=1;a['search_failures']+=int(d['search_failure'])
                a['position_boundary']+=int(np.any(np.isclose(np.asarray(obj['state_hat'][:2])[:,None],np.asarray(cfg['xy_bounds']),rtol=0,atol=1e-8)))
                a['velocity_boundary']+=int(np.any(np.isclose(np.asarray(obj['state_hat'][2:])[:,None],np.asarray(cfg['v_bounds']),rtol=0,atol=1e-8)))
    dump(OUT/'causal_checks.json',dict(passed=True,checks=checks,intervention='Delete all source rows after fixed 5s cutoff, replay entire 0.1..5s prefix with fixed config/cohort/seeds',scope='Upstream sensing only; F01-D predictor loader not implemented'))
    dump(OUT/'search_diagnostics.json',counts)
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
