"""Local association posterior: exhaustive small components, Murty top-50 otherwise."""
import heapq
from itertools import count
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def enumerate_assignments(cost):
    n=cost.shape[0]
    output=[]
    def visit(row,used,assignment,total):
        if row==n:
            output.append((float(total),tuple(assignment)))
            return
        for col in np.flatnonzero(np.isfinite(cost[row])):
            if int(col) not in used:
                visit(row+1,used|{int(col)},assignment+[int(col)],total+cost[row,col])
    visit(0,set(),[],0.)
    return sorted(output,key=lambda x:(x[0],x[1]))


def murty_assignments(cost,k=50):
    """True k-best rectangular one-to-one assignments by disjoint Murty partitions."""
    cost=np.asarray(cost,float)
    n=cost.shape[0]
    if not n:return [(0.,())],False
    serial=count()
    queue=[]
    def push(fixed,forbidden):
        work=cost.copy()
        for row,col in forbidden:
            work[row,col]=np.inf
        if len(set(fixed))!=len(fixed):return
        for row,col in enumerate(fixed):
            value=work[row,col]
            work[row,:]=np.inf;work[:,col]=np.inf;work[row,col]=value
        try:
            rows,cols=linear_sum_assignment(work)
        except ValueError:
            return
        if len(rows)!=n or not np.isfinite(work[rows,cols]).all():return
        assignment=tuple(map(int,cols))
        total=float(cost[rows,cols].sum())
        heapq.heappush(queue,(total,next(serial),fixed,forbidden,assignment))
    push((),frozenset())
    output=[]
    # Fetch one additional complete hypothesis to identify actual truncation.
    while queue and len(output)<k+1:
        total,_,fixed,forbidden,assignment=heapq.heappop(queue)
        output.append((total,assignment))
        for row in range(len(fixed),n):
            push(assignment[:row],forbidden|{(row,assignment[row])})
    return output[:k],len(output)>k


def association_marginals(cost,detection_count):
    """Return detection beta, private-miss beta, and component evidence."""
    cost=np.asarray(cost,float)
    n=cost.shape[0];m=int(detection_count)
    if cost.shape!=(n,m+n):raise ValueError('Expected detection columns plus private miss columns')
    beta=np.zeros((n,m));miss=np.ones(n)
    remaining=set(range(n));components=[]
    gate=np.isfinite(cost[:,:m])
    while remaining:
        rows={min(remaining)};columns=set()
        while True:
            newcols=set(np.flatnonzero(gate[list(rows)].any(axis=0)))
            newrows=set(np.flatnonzero(gate[:,list(newcols)].any(axis=1))) if newcols else set()
            if newcols==columns and newrows.issubset(rows):break
            rows|=newrows;columns=newcols
        remaining-=rows
        rr=sorted(rows);cc=sorted(columns);nr=len(rr);nd=len(cc)
        local=np.full((nr,nd+nr),np.inf)
        if nd:local[:,:nd]=cost[np.ix_(rr,cc)]
        for i,row in enumerate(rr):local[i,nd+i]=cost[row,m+row]
        if max(nr,nd)<=4:
            hypotheses=enumerate_assignments(local);truncated=False;method='exact'
        else:
            hypotheses,truncated=murty_assignments(local,50);method='murty_top50'
        if not hypotheses:raise ValueError('No feasible association hypotheses')
        totals=np.array([h[0] for h in hypotheses]);weights=np.exp(-(totals-totals.min()));weights/=weights.sum()
        miss[rr]=0.
        for weight,(_,assignment) in zip(weights,hypotheses):
            for i,j in enumerate(assignment):
                if j<nd:beta[rr[i],cc[j]]+=weight
                else:miss[rr[i]]+=weight
        components.append({'track_indices':rr,'detection_indices':cc,'tracks':nr,'detections':nd,
                           'hypothesis_count':len(hypotheses),'truncated':truncated,'method':method,
                           'effective_sample_size':float(1./(weights@weights)),
                           'best_cost':float(totals.min()),'last_retained_cost':float(totals.max()),
                           'posterior_normalization':'all hypotheses' if not truncated else 'conditional on retained top50; omitted mass unknown'})
    return beta,miss,components


def mixture_moments(means,covariances,weights):
    weights=np.asarray(weights,float);means=np.asarray(means,float)
    if not np.isclose(weights.sum(),1.) or np.any(weights<0):raise ValueError('Invalid mixture weights')
    mean=weights@means
    delta=means-mean
    covariance=sum(w*(P+np.outer(d,d)) for w,P,d in zip(weights,covariances,delta))
    return mean,(covariance+covariance.T)/2


def selfcheck():
    # Two tracks contest one detection: three legal assignments, not four.
    c=np.array([[0.,0.,np.inf],[0.,np.inf,0.]])
    b,miss,diag=association_marginals(c,1)
    assert np.allclose(b,1/3) and np.allclose(miss,2/3)
    assert np.allclose(b.sum(1)+miss,1) and np.all(b.sum(0)<=1+1e-12)
    assert diag[0]['hypothesis_count']==3 and not diag[0]['truncated']
    rng=np.random.default_rng(2026)
    checked=0
    for n,m in [(1,2),(2,2),(3,3),(4,3),(5,4)]:
        cost=np.full((n,m+n),np.inf);cost[:,:m]=rng.normal(size=(n,m))
        cost[:,:m][rng.random((n,m))<.15]=np.inf
        for i in range(n):cost[i,m+i]=.5+i*.1
        exact=enumerate_assignments(cost)
        result,truncated=murty_assignments(cost,50)
        assert np.allclose([a[0] for a in result],[a[0] for a in exact[:50]])
        assert len({a[1] for a in result})==len(result)
        assert truncated==(len(exact)>50)
        for _,assignment in result:assert len(set(assignment))==len(assignment)
        checked+=1
    mean,P=mixture_moments([[-1.,0.],[1.,0.]],[np.eye(2),np.eye(2)],[.5,.5])
    assert np.allclose(mean,0) and np.allclose(P,np.diag([2.,1.])) and np.linalg.eigvalsh(P).min()>0
    b,miss,d=association_marginals(np.diag([2.,2.])+np.array([[0.,np.inf],[np.inf,0.]]),0)
    assert b.shape==(2,0) and np.allclose(miss,1)
    from .tracker import CentralTracker,measurement
    stations=np.array([[-65.,-110.],[115.,0.],[-65.,110.]])
    bores=np.array([0.,np.pi,0.]);tr=CentralTracker(stations,bores,{'association_mode':'jpda'})
    state=np.array([0.,0.,0.,0.]);z,_=measurement(state,stations[0],bores[0])
    for cycle in range(3):
        ns=cycle*100_000_000+10_000_000
        out=tr.update_station([{'z':z}],0,ns)
        tr.finish_cycle((cycle+1)*100_000_000)
    assert len(tr.select())==1 and len(tr.tracks)==1
    for t in tr.tracks.values():assert np.linalg.eigvalsh(t['P']).min()>0
    empty=tr.update_station([],0,310_000_000)
    assert empty['jpda_diagnostics']['detection_unassigned_probability']==[]
    assert not empty['birth_keys']
    # Exercise tracker branch mixing, not only the generic mixture formula.
    toy=CentralTracker(stations,bores,{'association_mode':'jpda'})
    toy._birth(z,toy.default_R,0)
    toy.tracks[0]['x']=np.zeros(4);toy.tracks[0]['P']=np.eye(4)
    H=np.eye(3,4);S=2*np.eye(3);R=np.eye(3)
    cache={(0,0):(np.array([-2.,0.,0.]),H,S,R),
           (0,1):(np.array([2.,0.,0.]),H,S,R)}
    toy._update_jpda([0],[(z,R),(z,R)],0,np.zeros((1,3)),cache,0)
    assert np.isclose(toy.tracks[0]['P'][0,0],4/3)
    assert np.linalg.eigvalsh(toy.tracks[0]['P']).min()>0
    assert len(toy.tracks)==3  # each detection has unassigned marginal 2/3
    # Exact half probability is neither a hard hit nor a birth.
    half=CentralTracker(stations,bores,{'association_mode':'jpda'})
    half._birth(z,half.default_R,0);half.cycle_hits.clear()
    half._update_jpda([0],[(z,R)],0,np.zeros((1,2)),{(0,0):(np.zeros(3),H,S,R)},0)
    assert not half.cycle_hits and len(half.tracks)==1
    from .tracker import selfcheck as legacy_check
    from unittest.mock import patch
    with patch.object(Path,'write_text',return_value=0):
        legacy_check()
    result={'exact_probability_normalization':True,'unique_detection_marginals':True,
            'murty_vs_exhaustive_cases':checked,'murty_hypotheses_unique':True,
            'mixture_contains_between_hypothesis_covariance':True,'mixture_psd':True,'tracker_mixture_spread_verified':True,
            'half_probability_does_not_hit_or_birth':True,'legacy_hungarian_regression':True,
            'no_observations':True,'jpda_tracker_three_cycle_integration':True,
            'hit_and_birth_probability_threshold':'strictly greater than 0.5',
            'large_component_rule':'actual Murty top50; posterior conditional on retained set',
            'small_component_rule':'exhaustive if max(track_count,detection_count)<=4'}
    path=Path(__file__).resolve().parents[1]/'reports/f01c/jpda_checks.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':selfcheck()
