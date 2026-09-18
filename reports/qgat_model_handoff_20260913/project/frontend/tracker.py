"""Shared anonymous central EKF/Hungarian tracker; no source identity input."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def measurement(state, station, boresight, height=5.):
    d = state[:2] - station
    rho2 = float(d @ d)
    if rho2 <= 1e-12:
        raise ValueError("Horizontal bearing is undefined at station position")
    r = np.sqrt(rho2 + height**2)
    dot = float(d @ state[2:])
    z = np.array([r, wrap(np.arctan2(d[1], d[0])-boresight), dot/r])
    H = np.zeros((3,4))
    H[0,:2] = d/r
    H[1,:2] = [-d[1]/rho2, d[0]/rho2]
    H[2,:2] = state[2:]/r - dot*d/r**3
    H[2,2:] = d/r
    return z, H


class CentralTracker:
    def __init__(self, stations, boresights, config=None):
        self.stations = np.asarray(stations, dtype=float)
        self.boresights = np.asarray(boresights, dtype=float)
        if self.stations.shape != (3,2) or self.boresights.shape != (3,) or not np.isfinite(self.stations).all() or not np.isfinite(self.boresights).all():
            raise ValueError("Expected three finite stations and boresights")
        c = dict(config or {})
        self.association_mode = c.get('association_mode','hungarian')
        if self.association_mode not in ('hungarian','jpda'):
            raise ValueError('association_mode must be hungarian or jpda')
        self.measurement_mixture=None
        self.quality_edges=np.asarray(c.get('quality_bin_edges_db',[12.,20.]),float)
        if 'measurement_noise_mixture_bins' in c:
            from .measurement_mixture import validate_bins
            self.measurement_mixture=validate_bins(c['measurement_noise_mixture_bins'],self.quality_edges)
        self.max_missed_cycles=c.get('max_missed_cycles',5)
        if isinstance(self.max_missed_cycles,bool) or not isinstance(self.max_missed_cycles,int) or self.max_missed_cycles<1:raise ValueError('max_missed_cycles must be a positive integer')
        self.q_a = float(c.get('q_a',4.))
        self.pD = float(c.get('p_D',.9))
        # Startup uniform density, to be replaced by noise-only calibration.
        self.clutter = float(c.get('clutter_intensity',.1/(290*np.deg2rad(140)*2*43.57)))
        self.height = float(c.get('height_difference_m',5.))
        if self.q_a not in (1.,4.,9.) or not 0 < self.pD < 1 or not np.isfinite([self.clutter,self.height]).all() or self.clutter <= 0 or self.height < 0:
            raise ValueError("Invalid process, detection or clutter configuration")
        self.tracks = {}
        self.history = {}
        self.next_key = 0
        self.time_ns = None
        self.cycle_stations = set()
        self.cycle_hits = set()
        self.default_R = np.diag([1.5**2,np.deg2rad(5)**2,1.])
        self.last_records = []

    def _predict(self, time_ns):
        time_ns = int(time_ns)
        if self.time_ns is not None and time_ns < self.time_ns:
            raise ValueError("Measurements must be processed in timestamp order")
        if self.time_ns is not None:
            dt = (time_ns-self.time_ns)*1e-9
            I = np.eye(2)
            F = np.block([[I,dt*I],[np.zeros((2,2)),I]])
            Q = self.q_a*np.block([[dt**3/3*I,dt**2/2*I],[dt**2/2*I,dt*I]])
            for t in self.tracks.values():
                t['x'] = F @ t['x']
                t['P'] = F @ t['P'] @ F.T + Q
                t['P'] = (t['P']+t['P'].T)/2
        self.time_ns = time_ns

    def _birth(self, z, R, sid):
        r,theta,vr = z
        if r <= self.height:
            return None
        rho = np.sqrt(r*r-self.height**2)
        a = self.boresights[sid]+theta
        e = np.array([np.cos(a),np.sin(a)])
        tangent = np.array([-e[1],e[0]])
        u = rho/r*e
        x = np.r_[self.stations[sid]+rho*e, u*vr/(u@u)]
        J = np.column_stack([r/rho*e,rho*tangent])
        P = np.zeros((4,4))
        P[:2,:2] = J @ R[:2,:2] @ J.T
        # Explicit approximation: initial position/velocity cross covariance is zero.
        radial_var = max(float(R[2,2]/(u@u)),1e-6)
        P[2:,2:] = radial_var*np.outer(e,e)+100*np.outer(tangent,tangent)
        key = self.next_key
        self.next_key += 1
        self.tracks[key] = {'x':x,'P':P,'hits':deque(maxlen=3),'confirmed':False,'misses':0,'age':0}
        self.history[key] = []
        self.cycle_hits.add(key)
        return key

    def update_station(self, detections, station_id, time_ns):
        sid = int(station_id)
        if sid not in (0,1,2) or sid in self.cycle_stations:
            raise ValueError("Station must be 0/1/2 and processed once per cycle")
        time_ns = int(time_ns)
        valid = []
        mixture_components=[]
        invalid = 0
        for detection in detections:
            if int(detection.get('station_id',sid)) != sid or int(detection.get('measurement_time_ns',time_ns)) != time_ns:
                raise ValueError("Detection station/timestamp mismatch")
            z = np.asarray(detection['z'],dtype=float)
            R = np.asarray(detection.get('R',self.default_R),dtype=float)
            if z.shape != (3,) or R.shape != (3,3) or not np.isfinite(z).all() or not np.isfinite(R).all() or not np.allclose(R,R.T):
                raise ValueError("Invalid detection or covariance")
            np.linalg.cholesky(R)
            if z[0] <= self.height:
                invalid += 1
                continue
            valid.append((z,R))
            if self.measurement_mixture is not None:
                quality=detection.get('quality',{})
                db=float(detection.get('peak_to_noise_db',quality.get('peak_to_noise_db',0.)))
                if not np.isfinite(db):raise ValueError('Nonfinite detection quality')
                quality_bin=int(np.searchsorted(self.quality_edges,db,side='right'))
                mixture_components.append(self.measurement_mixture[quality_bin])
        self._predict(time_ns)
        self.cycle_stations.add(sid)
        keys = list(self.tracks)
        n,m = len(keys),len(valid)
        costs = np.full((n,m+n),np.inf)
        cache = {}
        for i,key in enumerate(keys):
            t = self.tracks[key]
            predicted,H = measurement(t['x'],self.stations[sid],self.boresights[sid],self.height)
            costs[i,m+i] = -np.log1p(-self.pD)
            for j,(z,R) in enumerate(valid):
                nu = z-predicted
                nu[1] = wrap(nu[1])
                if self.measurement_mixture is not None:
                    from .measurement_mixture import gated_likelihood
                    loglik,branches=gated_likelihood(nu,H,t['P'],mixture_components[j])
                    if branches:
                        costs[i,j]=-np.log(self.pD)-loglik+np.log(self.clutter)
                        cache[i,j]=branches
                    continue
                S = H @ t['P'] @ H.T+R
                distance = float(nu @ np.linalg.solve(S,nu))
                if distance <= 11.345:
                    sign,ldet = np.linalg.slogdet(S)
                    if sign <= 0:
                        raise ValueError("Nonpositive innovation covariance")
                    costs[i,j] = -np.log(self.pD)+.5*(distance+ldet+3*np.log(2*np.pi))+np.log(self.clutter)
                    cache[i,j] = (nu,H,S,R)
        if self.association_mode == 'jpda':
            return self._update_jpda(keys,valid,sid,costs,cache,invalid)
        if self.measurement_mixture is not None:
            beta=np.zeros((n,m));miss=np.ones(n)
            if n:
                ri,ci=linear_sum_assignment(costs)
                for i,j in zip(ri,ci):
                    if j<m:beta[i,j]=1.;miss[i]=0.
            return self._update_jpda(keys,valid,sid,costs,cache,invalid,(beta,miss,[]))
        assigned = set()
        matches = []
        if n:
            ri,ci = linear_sum_assignment(costs)
            for i,j in zip(ri,ci):
                if j >= m:
                    continue
                key = keys[i]
                t = self.tracks[key]
                nu,H,S,R = cache[i,j]
                K = np.linalg.solve(S,H @ t['P']).T
                t['x'] = t['x']+K@nu
                A = np.eye(4)-K@H
                t['P'] = A@t['P']@A.T+K@R@K.T
                t['P'] = (t['P']+t['P'].T)/2
                assigned.add(j)
                self.cycle_hits.add(key)
                matches.append({'key':key,'detection_index':int(j),'cost':float(costs[i,j]),
                                'innovation':nu.tolist(),'S':S.tolist(),
                                'nis':float(nu@np.linalg.solve(S,nu))})
        births = [self._birth(z,R,sid) for j,(z,R) in enumerate(valid) if j not in assigned]
        return {'matches':matches,'birth_keys':births,'invalid_detections':invalid}

    def _update_jpda(self,keys,valid,sid,costs,cache,invalid,posterior=None):
        from .jpda import association_marginals,mixture_moments
        beta,miss,components=association_marginals(costs,len(valid)) if posterior is None else posterior
        matches=[]
        for i,key in enumerate(keys):
            t=self.tracks[key]
            prior_x=t['x'];prior_P=t['P']
            means=[prior_x];covariances=[prior_P];weights=[float(miss[i])]
            for j in np.flatnonzero(beta[i]>0):
                detection_probability=float(beta[i,j])
                item=cache[i,int(j)]
                if isinstance(item,tuple):
                    nu,H,S,R=item
                    branches=[dict(probability=1.,component_index=0,nu=nu,H=H,S=S,R=R)]
                else:branches=item
                for branch in branches:
                    probability=detection_probability*branch['probability']
                    nu,H,S,R=(branch[name] for name in ('nu','H','S','R'))
                    K=np.linalg.solve(S,H@prior_P).T
                    A=np.eye(4)-K@H
                    means.append(prior_x+K@nu)
                    covariances.append(A@prior_P@A.T+K@R@K.T)
                    weights.append(probability)
                    matches.append({'key':key,'detection_index':int(j),'cost':float(costs[i,j]),
                                    'association_probability':probability,
                                    'detection_association_probability':detection_probability,
                                    'component_probability':branch['probability'],
                                    'component_index':branch['component_index'],
                                    'innovation':nu.tolist(),'S':S.tolist(),
                                    'nis':float(nu@np.linalg.solve(S,nu))})
            t['x'],t['P']=mixture_moments(means,covariances,weights)
            # Fixed declaration, not a calibrated confirmation threshold.
            if float(beta[i].sum())>.5:
                self.cycle_hits.add(key)
        unassigned=np.clip(1-beta.sum(axis=0),0.,1.)
        births=[self._birth(z,R,sid) for j,(z,R) in enumerate(valid) if unassigned[j]>.5]
        for component in components:
            component['track_keys']=[keys[i] for i in component['track_indices']]
        return {'matches':matches,'birth_keys':births,'invalid_detections':invalid,
                'jpda_diagnostics':{'components':components,
                    'track_assigned_probability':beta.sum(axis=1).tolist(),
                    'detection_unassigned_probability':unassigned.tolist(),
                    'hit_threshold_strict':.5,'birth_threshold_strict':.5,
                    'nis_interpretation':'hypothesis branches; weight by association_probability'}}

    def finish_cycle(self, deadline_ns):
        self._predict(deadline_ns)
        records = []
        deleted = []
        for key,t in self.tracks.items():
            hit = key in self.cycle_hits
            t['hits'].append(hit)
            t['age'] += 1
            t['misses'] = 0 if hit else t['misses']+1
            t['confirmed'] = t['confirmed'] or sum(t['hits']) >= 2
            exists = t['misses'] < self.max_missed_cycles
            record = {'key':key,'time_ns':int(deadline_ns),'state_hat':t['x'].tolist(),'P':t['P'].tolist(),
                      'exists':exists,'detected':hit,'confirmed':bool(t['confirmed']),
                      'misses':t['misses'],'age':t['age'],'selected':False}
            records.append(record)
            self.history[key].append(record)
            if not exists:
                deleted.append(key)
        for key in deleted:
            del self.tracks[key]
        # Age counts completed 100ms existence cycles; filter history before the top-eight cap.
        eligible = [r for r in records if r['exists'] and r['confirmed'] and r['age'] >= 3]
        eligible.sort(key=lambda r:(r['misses'],np.trace(np.asarray(r['P'])[:2,:2]),-r['age'],r['key']))
        for r in eligible[:8]:
            r['selected'] = True
        self.last_records = records
        self.cycle_hits.clear()
        self.cycle_stations.clear()
        return records

    def select(self):
        rows = [r for r in self.last_records if r['selected']]
        return sorted(rows,key=lambda r:(r['misses'],np.trace(np.asarray(r['P'])[:2,:2]),-r['age'],r['key']))


def selfcheck():
    stations = np.array([[-65.,-110.],[115.,0.],[-65.,110.]])
    bores = np.array([0.,np.pi,0.])
    state = np.array([2.,3.,4.,-2.])
    _,H = measurement(state,stations[0],bores[0])
    numerical = np.zeros((3,4))
    for k in range(4):
        delta = np.zeros(4); delta[k] = 1e-5
        a,_ = measurement(state+delta,stations[0],bores[0])
        b,_ = measurement(state-delta,stations[0],bores[0])
        difference = a-b; difference[1] = wrap(difference[1])
        numerical[:,k] = difference/(2e-5)
    assert np.allclose(H,numerical,atol=1e-8)
    state = np.array([2.,3.,0.,0.])
    def det(sid,ns,x=state):
        z,_ = measurement(x,stations[sid],bores[sid])
        return {'station_id':sid,'measurement_time_ns':ns,'z':z}
    tr = CentralTracker(stations,bores)
    for sid in range(3):
        tr.update_station([det(sid,10_000_000*(sid+1))],sid,10_000_000*(sid+1))
    rows = tr.finish_cycle(100_000_000)
    assert len(rows)==1 and not rows[0]['confirmed'] and rows[0]['detected']
    tr.update_station([det(0,110_000_000)],0,110_000_000)
    rows = tr.finish_cycle(200_000_000)
    assert rows[0]['confirmed'] and not rows[0]['selected'] and np.linalg.norm(rows[0]['state_hat'][2:])<1e-8
    tr.update_station([det(0,210_000_000)],0,210_000_000)
    rows = tr.finish_cycle(300_000_000)
    assert rows[0]['selected'] and rows[0]['age']==3
    for cycle in range(4,9):
        for sid in range(3):
            tr.update_station([],sid,(cycle-1)*100_000_000+(sid+1)*10_000_000)
        rows = tr.finish_cycle(cycle*100_000_000)
        assert rows[0]['exists'] == (cycle<8)
        assert np.linalg.eigvalsh(rows[0]['P']).min()>0
    assert not tr.tracks and len(tr.history[0])==8
    # Even an in-gate detection may be cheaper to leave unassigned.
    tr = CentralTracker(stations,bores,{'clutter_intensity':1e10})
    tr.update_station([det(0,10_000_000)],0,10_000_000)
    tr.finish_cycle(100_000_000)
    out = tr.update_station([det(0,110_000_000)],0,110_000_000)
    assert not out['matches'] and len(tr.tracks)==2
    gated = CentralTracker(stations,bores)
    gated.update_station([det(0,10_000_000)],0,10_000_000)
    gated.finish_cycle(100_000_000)
    out = gated.update_station([det(0,110_000_000,np.array([900.,900.,0.,0.]))],0,110_000_000)
    assert not out['matches'] and out['birth_keys']==[1]
    tentative = CentralTracker(stations,bores)
    tentative.update_station([det(0,10_000_000)],0,10_000_000)
    tentative.finish_cycle(100_000_000)
    for cycle in range(2,7):
        tentative.finish_cycle(cycle*100_000_000)
    assert not tentative.tracks and tentative.history[0][0]['detected']
    pool = CentralTracker(stations,bores)
    pool.update_station([det(0,10_000_000,np.array([float(i)*20,0.,0.,0.])) for i in range(10)],0,10_000_000)
    assert len(pool.tracks)==10
    for key,t in pool.tracks.items():
        t['confirmed']=True
        t['age']=2 if key else 1
        t['P']=np.eye(4)*(key+1)
    pool.finish_cycle(100_000_000)
    assert [r['key'] for r in pool.select()]==list(range(1,9))
    result = {'jacobian_finite_difference':True,'no_forced_match_inside_gate':True,
              'same_cycle_three_stations_not_confirmation':True,'two_cycles_confirmation':True,
              'five_missed_cycles_deletion':True,'covariance_positive_definite':True,
              'stationary_track_retained':True,'deleted_history_preserved':True,
              'three_cycle_history_before_selection':True,
              'outside_gate_no_match':True,'tentative_expiration':True,
              'uncapped_pool_top8_after_history_filter':True,
              'initial_cross_covariance_approximation':'zero; innovation calibration remains required',
              'clutter_density_status':'startup uniform; noise-only calibration remains required'}
    outpath = Path(__file__).resolve().parents[1]/'reports/f01c/tracker_checks.json'
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    selfcheck()
