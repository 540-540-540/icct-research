"""Zero-mean measurement-noise mixtures; gate components without renormalizing priors."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp


def validate_bins(records,edges):
    bins={}
    if len(edges)!=2 or not np.isfinite(edges).all() or not np.all(np.diff(edges)>0):
        raise ValueError('Expected two increasing finite quality edges')
    for row in records:
        index=int(row['bin']);components=[]
        if index in bins or index not in (0,1,2):raise ValueError('Duplicate or invalid quality bin')
        for item in row['components']:
            weight=float(item['weight']);R=np.asarray(item['R'],float)
            if weight<0 or not np.isfinite(weight) or R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R,R.T):
                raise ValueError('Invalid mixture component')
            np.linalg.cholesky(R)
            components.append((weight,R))
        if not components or not np.isclose(sum(w for w,R in components),1.,rtol=1e-10,atol=1e-12):
            raise ValueError('Mixture prior weights must sum to one')
        bins[index]=components
    if set(bins)!={0,1,2}:raise ValueError('All three quality bins are required')
    return bins


def gated_likelihood(nu,H,P,components):
    branches=[];logs=[]
    for index,(weight,R) in enumerate(components):
        if weight==0:continue
        S=H@P@H.T+R
        distance=float(nu@np.linalg.solve(S,nu))
        if distance>11.345:continue
        sign,ldet=np.linalg.slogdet(S)
        if sign<=0:raise ValueError('Nonpositive innovation covariance')
        logs.append(np.log(weight)-.5*(distance+ldet+3*np.log(2*np.pi)))
        branches.append(dict(component_index=index,nu=nu,H=H,S=S,R=R))
    if not branches:return -np.inf,[]
    likelihood=float(logsumexp(logs))
    for branch,logweight in zip(branches,logs):
        branch['probability']=float(np.exp(logweight-likelihood))
    return likelihood,branches


def selfcheck():
    from .tracker import CentralTracker,measurement
    from .jpda import selfcheck as jpda_checks
    from unittest.mock import patch
    R=np.diag([.35**2,np.deg2rad(1.)**2,.2**2]);wide=R.copy();wide[1,1]=np.deg2rad(5.)**2
    H=np.eye(3,4);P=np.eye(4)*1e-7
    weights=[(.8,R),(.2,wide)]
    near,bnear=gated_likelihood(np.zeros(3),H,P,weights)
    far,bfar=gated_likelihood(np.array([0.,np.deg2rad(5.),0.]),H,P,weights)
    assert np.isclose(sum(b['probability'] for b in bnear),1.)
    assert bfar[0]['component_index']==1 and bfar[0]['probability']==1.
    fullwide,_=gated_likelihood(np.array([0.,np.deg2rad(5.),0.]),H,P,[(1.,wide)])
    assert np.isclose(far,fullwide+np.log(.2))
    assert bfar[0]['probability']>next(b['probability'] for b in bnear if b['component_index']==1)
    stations=np.array([[-65.,-110.],[115.,0.],[-65.,110.]])
    bores=np.array([0.,np.pi,0.]);state=np.array([1.,2.,0.,0.]);z,_=measurement(state,stations[0],bores[0])
    one=[{'bin':i,'components':[{'weight':1.,'R':R.tolist()}]} for i in range(3)]
    two=[{'bin':i,'components':[{'weight':w,'R':r.tolist()} for w,r in weights]} for i in range(3)]
    for mode in ('hungarian','jpda'):
        old=CentralTracker(stations,bores,{'association_mode':mode})
        new=CentralTracker(stations,bores,{'association_mode':mode,'measurement_noise_mixture_bins':one})
        for cycle in range(4):
            ns=cycle*100_000_000+10_000_000
            d={'z':z,'R':R,'quality':{'peak_to_noise_db':15.}}
            old.update_station([d],0,ns);new.update_station([d],0,ns)
            a=old.finish_cycle((cycle+1)*100_000_000);b=new.finish_cycle((cycle+1)*100_000_000)
            assert len(a)==len(b)
            for x,y in zip(a,b):
                assert np.allclose(x['state_hat'],y['state_hat'],atol=1e-10)
                assert np.allclose(x['P'],y['P'],atol=1e-10)
                assert x['confirmed']==y['confirmed'] and x['selected']==y['selected']
    tracker=CentralTracker(stations,bores,{'association_mode':'jpda','measurement_noise_mixture_bins':two})
    prior=.8*R+.2*wide
    for cycle in range(4):
        ns=cycle*100_000_000+10_000_000
        trial=z.copy();trial[1]+=np.deg2rad(1.5)*(-1)**cycle
        update=tracker.update_station([{'z':trial,'R':prior,'quality':{'peak_to_noise_db':25.}}],0,ns)
        records=tracker.finish_cycle((cycle+1)*100_000_000)
        for row in records:assert np.linalg.eigvalsh(row['P']).min()>0
        for match in update['matches']:
            assert np.isclose(match['association_probability'],match['detection_association_probability']*match['component_probability'])
    empty=tracker.update_station([],0,410_000_000);assert not empty['matches'] and not empty['birth_keys']
    with patch.object(Path,'write_text',return_value=0):jpda_checks()
    result={'single_component_hungarian_and_jpda_regression':True,'component_posterior_normalization':True,
            'outlier_increases_wide_posterior':True,'gated_prior_weight_not_renormalized':True,
            'joint_branch_probability_beta_times_component':True,'full_mixture_covariance_psd':True,
            'no_detections':True,'legacy_jpda_and_hungarian_checks':True,
            'birth_R':'caller supplied zero-mean prior covariance moment',
            'NIS_weight':'association_probability equals beta_ij times conditional component probability'}
    path=Path(__file__).resolve().parents[1]/'reports/f01c/measurement_mixture_checks.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':selfcheck()
