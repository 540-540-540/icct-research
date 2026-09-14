"""Evaluation-only covariance consistency for the frozen diagnostic replay."""
import numpy as np
from .run_frontend import SourceEpisodes,OUT,DATA,load,dump,current_truth,bind

def main():
    source=SourceEpisodes(); manifest=load(OUT/'development_manifest.json'); config=load(OUT.parent.parent/'configs/frontend_config.json');revision=config.get('frontend_revision','C03');q=int(config['q_a']); ns=[];ps=[]; comps=[]; nw=nv=0.
    for index in manifest['train_tracking_audit_episodes']:
        ep=source.episodes[index];grid=set(ep['prediction_grid_ms'])
        report=load(OUT/f'evaluation/q{q}'/f'episode_{index:03d}.json');comps.extend(report['jpda_components'])
        ni=report['innovation_nis'];nw+=ni['effective_weight'];nv+=ni['effective_weight']*ni['mean']
        for line in (DATA/f'tracks/q{q}'/f'episode_{index:03d}.jsonl').read_text().splitlines():
            import json
            row=json.loads(line);tm=row['time_ns']//1000000
            if tm not in grid:continue
            tr=[t for t in row['tracks'] if t['selected'] and t['exists']];truth,slots,valid=current_truth(source,ep,tm)
            x=np.array([t['state_hat'] for t in tr]).reshape(-1,4)
            for i,j in bind(np.linalg.norm(x[:,None,:2]-truth[None,:,:2],axis=2),5.):
                e=x[i]-truth[j];p=np.array(tr[i]['P']);ps.append(float(e[:2]@np.linalg.solve(p[:2,:2],e[:2])))
                if valid[j]:ns.append(float(e@np.linalg.solve(p,e)))
    def summary(v,threshold):return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),fraction_above_chi95=float(np.mean(np.array(v)>threshold)))
    result=dict(scope=revision+f' JPDA q{q} fixed12train scored matched selected origins only; 5m binding truncates errors, not an unbiased calibration test',state_nees=summary(ns,9.488),position_nees=summary(ps,5.991),association_weighted_nis=dict(mean=nv/nw,effective_weight=nw,qualification='gated posterior branches'),jpda=dict(components=len(comps),truncated_components=sum(c['truncated'] for c in comps),max_tracks=max(c['tracks'] for c in comps),max_detections=max(c['detections'] for c in comps),min_ESS=min(c['effective_sample_size'] for c in comps),qualification='top50 probabilities conditional on retained hypotheses; omitted mass unknown'),interpretation='Diagnostic only; does not establish calibrated posterior uncertainty or replace the separate engineering gate')
    dump(OUT/f'covariance_consistency_{revision}.json',result);print(result)
if __name__=='__main__':main()
