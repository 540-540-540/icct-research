"""Wei2024 III symbol-level three-BS sensing for a pre-separated known target.

Simulation and estimation are separate. Positive radial velocity approaches a BS.
Distances are planar as in the paper. This module does not solve target separation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class PaperWaveform:
    fc: float = 24e9
    B: float = 93.1e6
    K: int = 128
    N: int = 256
    # Independent full-symbol interval; df=B/K, not 1/T. No time-domain CP simulation.
    T: float = 12.375e-6
    c: float = 299792458.

    def __post_init__(self):
        if self.K<2 or self.N<2 or int(self.K)!=self.K or int(self.N)!=self.N or not np.isfinite([self.fc,self.B,self.T,self.c]).all() or min(self.fc,self.B,self.T,self.c)<=0:
            raise ValueError('Invalid waveform')

    @property
    def df(self):return self.B/self.K
    @property
    def unambiguous_range(self):return self.c/(2*self.df)
    @property
    def unambiguous_velocity(self):return self.c/(4*self.fc*self.T)
    @property
    def range_resolution(self):return self.c/(2*self.B)
    @property
    def velocity_resolution(self):return self.c/(2*self.fc*self.N*self.T)


Waveform=PaperWaveform


def _stations(stations):
    stations=np.asarray(stations,float)
    if stations.shape!=(3,2) or not np.isfinite(stations).all():raise ValueError('Exactly three finite planar stations are required')
    return stations


def synthesize(position,velocity,stations,waveform=None,snr_db=20.,rng=None,phases=None,noise=True):
    """Simulation only: Eq4, fixed received-symbol SNR, independent unit CN noise."""
    w=waveform or PaperWaveform();stations=_stations(stations)
    p=np.asarray(position,float);v=np.asarray(velocity,float)
    if p.shape!=(2,) or v.shape!=(2,) or not np.isfinite(np.r_[p,v,snr_db]).all():raise ValueError('Invalid simulation state/SNR')
    if not isinstance(rng,np.random.Generator):raise ValueError('An explicit numpy Generator is required')
    delta=stations-p;radius=np.linalg.norm(delta,axis=1)
    if np.any(radius<=0):raise ValueError('Target coincides with station')
    radial=delta@v/radius
    X=np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0,4,(3,w.K,w.N))))
    phases=rng.uniform(-np.pi,np.pi,3) if phases is None else np.asarray(phases,float)
    if phases.shape!=(3,) or not np.isfinite(phases).all():raise ValueError('Expected three finite scattering phases')
    range_phase=np.exp(-1j*4*np.pi*w.df/w.c*radius[:,None]*np.arange(w.K))
    velocity_phase=np.exp(1j*4*np.pi*w.fc*w.T/w.c*radial[:,None]*np.arange(w.N))
    clean=10**(float(snr_db)/20)*np.exp(1j*phases)[:,None,None]*range_phase[:,:,None]*velocity_phase[:,None,:]*X
    additive=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2) if noise else 0.
    return {'Y':clean+additive,'X':X}


received_echo=synthesize


def divide_symbols(Y,X):
    Y=np.asarray(Y,complex);X=np.asarray(X,complex)
    if Y.shape!=X.shape or not np.isfinite(Y).all() or not np.isfinite(X).all() or np.any(np.abs(X)==0):
        raise ValueError('Finite aligned symbols and nonzero known transmit symbols are required')
    return Y/X


def lag_correlation(vector):
    """Eq22/32: mean v[a] conj(v[a+k]), k=1,...,length-1."""
    vector=np.asarray(vector,complex)
    if vector.ndim!=1 or len(vector)<2 or not np.isfinite(vector).all():raise ValueError('Expected a finite feature vector')
    n=len(vector)
    return np.correlate(vector,vector,mode='full')[n-2::-1]/np.arange(n-1,0,-1)


def _bounds(value,name):
    value=np.asarray(value,float)
    if value.shape!=(2,2) or not np.isfinite(value).all() or np.any(value[:,0]>=value[:,1]):raise ValueError(name+' must be [[min,max],[min,max]]')
    return value


def _coarse(B,w,r_bounds,v_bounds,oversample):
    # IFFT over range cancels the negative delay phase; FFT cancels positive Doppler.
    nr=w.K*oversample;nv=w.N*oversample
    spectrum=np.fft.fft(np.fft.ifft(B,n=nr,axis=0),n=nv,axis=1)
    ranges=np.arange(nr)*w.unambiguous_range/nr
    velocities=np.fft.fftfreq(nv,d=w.T)*w.c/(2*w.fc)
    ri=np.flatnonzero((ranges>=r_bounds[0])&(ranges<=r_bounds[1]))
    vi=np.flatnonzero((velocities>=v_bounds[0])&(velocities<=v_bounds[1]))
    if not len(ri) or not len(vi):raise ValueError('Configured bounds contain no coarse FFT cells')
    ir,iv=np.unravel_index(np.abs(spectrum[np.ix_(ri,vi)]).argmax(),(len(ri),len(vi)))
    r0=ranges[ri[ir]];v0=velocities[vi[iv]]
    dr=w.range_resolution/oversample;dv=w.velocity_resolution/oversample
    rg=np.linspace(max(r_bounds[0],r0-dr),min(r_bounds[1],r0+dr),9)
    vg=np.linspace(max(v_bounds[0],v0-dv),min(v_bounds[1],v0+dv),9)
    A=np.exp(1j*4*np.pi*w.df/w.c*rg[:,None]*np.arange(w.K))
    C=np.exp(-1j*4*np.pi*w.fc*w.T/w.c*np.arange(w.N)[:,None]*vg)
    score=np.abs(A@B@C)
    i,j=np.unravel_index(score.argmax(),score.shape)
    return float(rg[i]),float(vg[j]),{'fft_cells':int(len(ri)*len(vi)),'fft_transform_shape':[nr,nv],'local_steering_cells':81,
        'range_local_edge':bool(i in (0,8)),'velocity_local_edge':bool(j in (0,8))}


def _local_grid(center,bounds,width,step):
    lower=np.maximum(bounds[:,0],center-width);upper=np.minimum(bounds[:,1],center+width)
    axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/step))+1)) for a,b in zip(lower,upper)]
    xx,yy=np.meshgrid(*axes,indexing='ij')
    return np.column_stack([xx.ravel(),yy.ravel()]),tuple(map(len,axes)),np.column_stack([lower,upper])


def _grid_search(center,bounds,width,step,fine_step,score):
    candidates,shape,local_bounds=_local_grid(center,bounds,width,step)
    scores=score(candidates);index=int(np.argmax(scores));ij=np.unravel_index(index,shape)
    initial_edge=any(i in (0,n-1) for i,n in zip(ij,shape))
    chosen=candidates[index]
    refined,rshape,rbounds=_local_grid(chosen,bounds,step,fine_step)
    rscores=score(refined);j=int(np.argmax(rscores));rij=np.unravel_index(j,rshape)
    edge=any(i in (0,n-1) for i,n in zip(rij,rshape))
    return refined[j],{'coarse_candidates':len(candidates),'fine_candidates':len(refined),
        'coarse_local_edge':bool(initial_edge),'fine_local_edge':bool(edge),
        'coarse_bounds':local_bounds.tolist(),'fine_bounds':rbounds.tolist(),'score':float(rscores[j])}


def _fused_score(parameters,features,phase_factor):
    """Eq26/37: sum over lag of product over stations of compensated real parts."""
    features=np.asarray(features,complex)
    # Candidate-independent positive scaling preserves the exact argmax.
    scale=np.maximum(np.max(np.abs(features),axis=1),np.finfo(float).tiny)
    features=features/scale[:,None]
    lags=np.arange(1,features.shape[1]+1)
    score=np.empty(len(parameters))
    for start in range(0,len(parameters),256):
        values=parameters[start:start+256]
        product=np.ones((len(values),len(lags)))
        for station in range(3):
            product*=np.real(features[station][None,:]*np.exp(1j*phase_factor*values[:,station,None]*lags))
        score[start:start+len(values)]=product.sum(axis=1)
    return score


def estimate(B,stations,xy_bounds,v_bounds,waveform=None,search=None):
    """Estimate one pre-separated target from B and declared bounds, with no truth input."""
    w=waveform or PaperWaveform();stations=_stations(stations)
    B=np.asarray(B,complex);xy_bounds=_bounds(xy_bounds,'xy_bounds');v_bounds=_bounds(v_bounds,'v_bounds')
    if B.shape!=(3,w.K,w.N) or not np.isfinite(B).all():raise ValueError('B must be finite (3,K,N) symbols')
    defaults=dict(oversample=2,position_half_width=3.,position_step=.25,position_fine_step=.025,
                  velocity_half_width=3.,velocity_step=.25,velocity_fine_step=.025)
    if search:
        unknown=set(search)-set(defaults)
        if unknown:raise ValueError('Unknown search settings: '+str(unknown))
        defaults.update(search)
    cfg=defaults
    if int(cfg['oversample'])!=cfg['oversample'] or cfg['oversample']<1 or any(not np.isfinite(v) or v<=0 for v in cfg.values()):raise ValueError('Invalid search configuration')
    corners=np.array([[x,y] for x in xy_bounds[0] for y in xy_bounds[1]])
    max_ranges=np.linalg.norm(corners[:,None,:]-stations,axis=2).max(axis=0)
    nearest=np.clip(stations,xy_bounds[:,0],xy_bounds[:,1])
    min_ranges=np.linalg.norm(nearest-stations,axis=1)
    max_speed=np.linalg.norm(np.max(np.abs(v_bounds),axis=1))
    if np.any(max_ranges>=w.unambiguous_range) or max_speed>=w.unambiguous_velocity:
        raise ValueError('Configured geometry/velocity region exceeds waveform ambiguity domain')
    coarse=[];coarse_diag=[];E=[];F=[]
    for station in range(3):
        r,v,d=_coarse(B[station],w,(min_ranges[station],max_ranges[station]),(-max_speed,max_speed),int(cfg['oversample']))
        coarse.append([r,v]);coarse_diag.append(d)
        C=np.exp(-1j*4*np.pi*w.fc*w.T/w.c*np.arange(w.N)*v)
        A=np.exp(1j*4*np.pi*w.df/w.c*np.arange(w.K)*r)
        E.append(B[station]@C);F.append(A@B[station])
    coarse=np.asarray(coarse);E=np.asarray(E);F=np.asarray(F)
    G=np.array([lag_correlation(e) for e in E]);I=np.array([lag_correlation(f) for f in F])
    A=2*(stations[1:]-stations[0]);rhs=coarse[0,0]**2-coarse[1:,0]**2+np.sum(stations[1:]**2,axis=1)-stations[0]@stations[0]
    rank=int(np.linalg.matrix_rank(A));condition=float(np.linalg.cond(A))
    if rank<2:raise ValueError('Three stations must be noncollinear for rough range localization')
    initial=np.clip(np.linalg.lstsq(A,rhs,rcond=None)[0],xy_bounds[:,0],xy_bounds[:,1])
    rough=least_squares(lambda p:np.linalg.norm(p-stations,axis=1)-coarse[:,0],initial,
                        bounds=(xy_bounds[:,0],xy_bounds[:,1]),max_nfev=40)
    position,pdiag=_grid_search(rough.x,xy_bounds,cfg['position_half_width'],cfg['position_step'],cfg['position_fine_step'],
        lambda p:_fused_score(np.linalg.norm(p[:,None,:]-stations,axis=2),G,-4*np.pi*w.df/w.c))
    radii=np.linalg.norm(stations-position,axis=1)
    if np.any(radii<=1e-9):raise ValueError('Estimated position coincides with station')
    U=(stations-position)/radii[:,None]
    velocity_initial=np.clip(np.linalg.lstsq(U,coarse[:,1],rcond=None)[0],v_bounds[:,0],v_bounds[:,1])
    velocity,vdiag=_grid_search(velocity_initial,v_bounds,cfg['velocity_half_width'],cfg['velocity_step'],cfg['velocity_fine_step'],
        lambda v:_fused_score(v@U.T,I,4*np.pi*w.fc*w.T/w.c))
    failure=not rough.success or pdiag['coarse_local_edge'] or pdiag['fine_local_edge'] or vdiag['coarse_local_edge'] or vdiag['fine_local_edge']
    return {'state_hat':np.r_[position,velocity],'coarse':{'range_radial_velocity':coarse,'position':rough.x,'velocity':velocity_initial},
        'symbol_features':{'E':E,'F':F,'G':G,'I':I},
        'diagnostics':{'search_failure':bool(failure),'localization_optimizer_success':bool(rough.success),
            'position_search':pdiag,'velocity_search':vdiag,'single_station':coarse_diag,
            'rough_geometry_condition':condition,'velocity_geometry_condition':float(np.linalg.cond(U)),
            'unambiguous_range_m':w.unambiguous_range,'unambiguous_velocity_mps':w.unambiguous_velocity,
            'configured_xy_bounds':xy_bounds.tolist(),'configured_velocity_bounds':v_bounds.tolist(),
            'radial_velocity_convention':'positive toward station; U=(station-estimated_position)/estimated_range',
            'scope':'Three BSs; pre-separated target and known cross-station association; no detection/tracking claim'}}


def selfcheck():
    w=PaperWaveform();stations=np.array([[0.,0.],[80.,0.],[0.,80.]])
    position=np.array([30.,35.]);velocity=np.array([3.,-2.])
    echo=synthesize(position,velocity,stations,w,rng=np.random.default_rng(2026),noise=False)
    result=estimate(divide_symbols(**echo),stations,[[10.,60.],[10.,60.]],[[-10.,10.],[-10.,10.]],w)
    assert np.linalg.norm(result['state_hat'][:2]-position)<.06
    assert np.linalg.norm(result['state_hat'][2:]-velocity)<.06
    assert not result['diagnostics']['search_failure']
    print({'noiseless_state_hat':result['state_hat'].tolist(),'search_failure':False})

if __name__=='__main__':selfcheck()
