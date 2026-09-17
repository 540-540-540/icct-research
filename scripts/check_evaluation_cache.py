"""Exact no-grad evaluation feature reuse: correctness and actual cache speed."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prediction.quantum import QuantumGraph
from prediction.evaluation_cache import evaluation_graph_features


def main():
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(20260912)
    report={'cases':[]}
    for legacy in (False,True):
        graph=QuantumGraph([40,40,15,15],legacy=legacy).cuda().eval()
        with torch.no_grad(): graph.theta.add_(torch.randn_like(graph.theta)*.2)
        x=np.random.default_rng(7).normal(size=(2,5,4,4))
        mask=np.array([[[0,0,0,0],[1,0,1,0],[1,1,1,1],[0,1,1,1],[1,0,0,0]]]*2,dtype=bool)
        x[1]=x[0]
        x[~mask]=np.nan
        with torch.no_grad():
            plain=graph.forward_history(x,mask)
            cached,stats=evaluation_graph_features(graph,x,mask)
            torch.testing.assert_close(cached,plain,atol=2e-10,rtol=2e-9)
            assert stats['unique_frames']==5 and (cached[~torch.as_tensor(mask,device='cuda')]==0).all()
            # No persistent feature survives a parameter change between calls.
            graph.theta[:,4:].add_(.07)
            changed,_=evaluation_graph_features(graph,x,mask)
            new_plain=graph.forward_history(x,mask)
            torch.testing.assert_close(changed,new_plain,atol=2e-10,rtol=2e-9)
            assert not torch.allclose(changed,cached)
        try:
            evaluation_graph_features(graph,x,mask)
            raise AssertionError('Grad-enabled call accepted')
        except RuntimeError:
            pass
        with torch.no_grad():
            graph.train()
            try:
                evaluation_graph_features(graph,x,mask)
                raise AssertionError('Training-mode call accepted')
            except RuntimeError:
                pass
        report['cases'].append(dict(legacy=legacy,max_error=float((cached-plain).abs().max()),
                                    weight_change_recomputed=True,guard_checks=True,**stats))
    norm=json.loads((ROOT/'data/f01d/normalization.json').read_text())
    graph=QuantumGraph(norm['quantum_scale']).cuda().eval()
    with torch.no_grad(): graph.theta.add_(torch.randn_like(graph.theta)*.2)
    with np.load(ROOT/'data/f01d/inputs/V_select_snr_20.npz') as f:
        x=f['state_hat'][:80];mask=f['track_exists'][:80]
    with torch.no_grad():
        graph.forward_history(x[:1],mask[:1])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        plain=torch.cat([graph.forward_history(xx[None],mm[None]) for xx,mm in zip(x,mask)])
        torch.cuda.synchronize()
        baseline_seconds=time.perf_counter()-tick
        baseline_peak=torch.cuda.max_memory_allocated()/2**20
        torch.cuda.reset_peak_memory_stats()
        cached,stats=evaluation_graph_features(graph,x,mask)
        torch.testing.assert_close(cached,plain,atol=2e-10,rtol=2e-9)
        peak=torch.cuda.max_memory_allocated()/2**20
    report['cases'].append(dict(split='V_select',snr=20,origins=80,indices='first 80 in source order',
                                baseline_seconds=baseline_seconds,baseline_peak_mib=baseline_peak,
                                cached_peak_mib=peak,max_error=float((cached-plain).abs().max()),
                                speedup=baseline_seconds/stats['seconds'],**stats))
    report['passed']=True
    folder=ROOT/'reports/evaluation_cache';folder.mkdir(exist_ok=True)
    path=folder/f'check-{time.strftime("%Y%m%d-%H%M%S")}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print(path,flush=True)


if __name__=='__main__': main()
