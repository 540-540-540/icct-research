"""Independent CUDA validation. Never invokes the CPU symbol estimator."""
import argparse
import json
import time
from pathlib import Path
import torch
from .echo_source import ROOT,SourceEpisodes


def scalar(x):return float(x.detach().item())
def relative(a,b):return scalar((a-b).abs().max()/b.abs().max().clamp_min(1e-30))
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def physical_check(gpu,w,device):
    stations=torch.tensor([[-90.,-70.],[95.,-65.],[0.,110.]],device=device,dtype=torch.float64)
    position=torch.tensor([[2.371,-3.149],[0.,0.],[-7.125,5.225],[8.85,-6.625]],device=device,dtype=torch.float64)
    velocity=torch.tensor([[3.217,-1.463],[0.,0.],[-4.2,2.9],[6.12,-5.43]],device=device,dtype=torch.float64)
    phases=torch.tensor([[.2,-1.1,2.4],[-.5,.8,1.3],[1.7,.4,-2.2],[2.1,-.3,.9]],device=device,dtype=torch.float64)
    seeds=[640010+i for i in range(len(position))]
    echo=gpu.synthesize_batch(position,velocity,stations,waveform=w,snr_db=10,seeds=seeds,device=device,phases=phases,noise=False)
    B=gpu.divide_symbols(echo['Y'],echo['X'])
    delta=stations[None]-position[:,None];ranges=torch.linalg.vector_norm(delta,dim=-1)
    radial=(delta*velocity[:,None]).sum(-1)/ranges
    k=torch.arange(w.K,device=device,dtype=torch.float64);n=torch.arange(w.N,device=device,dtype=torch.float64)
    reference=10**.5*torch.exp(1j*phases)[:,:,None,None]*torch.exp(-1j*4*torch.pi*w.df/w.c*ranges[:,:,None]*k)[:,:,:,None]*torch.exp(1j*4*torch.pi*w.fc*w.T/w.c*radial[:,:,None]*n)[:,:,None,:]
    eq6=relative(B,reference);assert eq6<1e-10
    assert scalar((echo['X'].abs()-1).abs().max())<1e-12
    result=gpu.estimate_batch(B,stations,[[-15.,15.],[-12.,12.]],[[-10.,10.],[-10.,10.]],waveform=w)
    state=result['state_hat'];errors=state-torch.cat([position,velocity],dim=-1)
    assert torch.isfinite(state).all() and scalar(torch.linalg.vector_norm(errors[:,:2],dim=-1).max())<.06
    assert scalar(torch.linalg.vector_norm(errors[:,2:],dim=-1).max())<.06
    coarse=result['coarse']['range_radial_velocity']
    C=torch.exp(-1j*4*torch.pi*w.fc*w.T/w.c*coarse[:,:,1,None]*n)
    A=torch.exp(1j*4*torch.pi*w.df/w.c*coarse[:,:,0,None]*k)
    E=torch.einsum('bskt,bst->bsk',B,C);F=torch.einsum('bsk,bskt->bst',A,B)
    features=result['symbol_features']
    feature_errors={'E':relative(features['E'],E),'F':relative(features['F'],F)}
    for name,vector in [('G',E),('I',F)]:
        for lag in (1,2,vector.shape[-1]//2,vector.shape[-1]-1):
            direct=(vector[...,:-lag]*vector[...,lag:].conj()).mean(-1)
            feature_errors[name+'_lag'+str(lag)]=relative(features[name][...,lag-1],direct)
    assert max(feature_errors.values())<1e-9
    for name,parameter,factor in [('G',ranges,4*torch.pi*w.df/w.c),('I',radial,-4*torch.pi*w.fc*w.T/w.c)]:
        vector=features[name];lag=torch.arange(1,vector.shape[-1]+1,device=device,dtype=torch.float64)
        theoretical=vector[...,0].abs()[...,None]*torch.exp(1j*factor*parameter[...,None]*lag)
        assert relative(vector,theoretical)<1e-8
    # Independent Eq26/37 at the actual chosen lattice points, all arithmetic on CUDA.
    score_errors={}
    estimated_ranges=torch.linalg.vector_norm(stations[None]-state[:,None,:2],dim=-1)
    U=(stations[None]-state[:,None,:2])/estimated_ranges[:,:,None]
    estimated_radial=(U*state[:,None,2:]).sum(-1)
    for feature,param,factor,field in [('G',estimated_ranges,-4*torch.pi*w.df/w.c,'position_search'),('I',estimated_radial,4*torch.pi*w.fc*w.T/w.c,'velocity_search')]:
        f=features[feature]/features[feature].abs().amax(-1,keepdim=True).clamp_min(1e-30)
        lag=torch.arange(1,f.shape[-1]+1,device=device,dtype=torch.float64)
        scores=(f*torch.exp(1j*factor*param[...,None]*lag)).real.prod(1).sum(-1)
        stored=torch.tensor([d[field]['score'] for d in result['diagnostics']],device=device,dtype=torch.float64)
        score_errors[field]=relative(stored,scores)
    assert max(score_errors.values())<1e-8
    phase_rotation=torch.exp(1j*torch.tensor([.6,-2.1,1.4],device=device,dtype=torch.float64))[None,:,None,None]
    rotated=gpu.estimate_batch(B*phase_rotation,stations,[[-15.,15.],[-12.,12.]],[[-10.,10.],[-10.,10.]],waveform=w)
    assert scalar((rotated['state_hat']-state).abs().max())<1e-8
    old=json.loads((ROOT/'reports/symbol_level/validation_K256_selfchecks.json').read_text())
    old_state=torch.tensor(old['checks']['noiseless_offgrid_position_velocity']['state_hat'],device=device,dtype=torch.float64)
    old_difference=state[0]-old_state
    assert scalar(torch.linalg.vector_norm(old_difference[:2]))<=.025 and scalar(torch.linalg.vector_norm(old_difference[2:]))<=.025
    return dict(equation6_relative_error=eq6,feature_reference_errors=feature_errors,
                equation26_37_score_relative_errors=score_errors,noiseless_state_hat=state.tolist(),
                noiseless_errors=errors.tolist(),constant_phase_invariance=True,
                saved_CPU_reference_difference=old_difference.tolist(),
                reference_qualification='Previously saved CPU result only; CPU estimator not called'),(stations,position,velocity)


def noise_and_grouping(gpu,w,device,fixture):
    stations,position,velocity=fixture
    position=position.repeat(2,1);velocity=velocity.repeat(2,1)
    seeds=[740010+i for i in range(len(position))]
    rows=[];test_B=None
    for snr in (5,10,15,20):
        noisy=gpu.synthesize_batch(position,velocity,stations,waveform=w,snr_db=snr,seeds=seeds,device=device)
        clean=gpu.synthesize_batch(position,velocity,stations,waveform=w,snr_db=snr,seeds=seeds,device=device,noise=False)
        noise=noisy['Y']-clean['Y']
        measured=10*torch.log10(clean['Y'].abs().square().mean((-2,-1))/noise.abs().square().mean((-2,-1)))
        assert scalar((measured-snr).abs().max())<.15
        repeated=gpu.synthesize_batch(position,velocity,stations,waveform=w,snr_db=snr,seeds=seeds,device=device)
        assert torch.equal(noisy['Y'],repeated['Y']) and torch.equal(noisy['X'],repeated['X'])
        portions=[gpu.synthesize_batch(position[j:j+3],velocity[j:j+3],stations,waveform=w,snr_db=snr,seeds=seeds[j:j+3],device=device) for j in range(0,len(position),3)]
        assert torch.equal(noisy['Y'],torch.cat([p['Y'] for p in portions]))
        assert torch.equal(noisy['X'],torch.cat([p['X'] for p in portions]))
        B=gpu.divide_symbols(noisy['Y'],noisy['X'])
        result=gpu.estimate_batch(B,stations,[[-15.,15.],[-12.,12.]],[[-10.,10.],[-10.,10.]],waveform=w)
        assert torch.isfinite(result['state_hat']).all()
        grouped=[gpu.estimate_batch(B[j:j+3],stations,[[-15.,15.],[-12.,12.]],[[-10.,10.],[-10.,10.]],waveform=w)['state_hat'] for j in range(0,len(B),3)]
        difference=scalar((result['state_hat']-torch.cat(grouped)).abs().max())
        assert difference<1e-8
        error=result['state_hat']-torch.cat([position,velocity],dim=-1)
        rows.append(dict(requested_snr_db=snr,measured_snr_db=measured.tolist(),
                         position_rmse_m=scalar(torch.sqrt(error[:,:2].square().sum(-1).mean())),
                         velocity_rmse_mps=scalar(torch.sqrt(error[:,2:].square().sum(-1).mean())),
                         batch_grouping_max_state_difference=difference,
                         search_failures=sum(bool(d['search_failure']) for d in result['diagnostics'])))
        test_B=B
    return dict(per_SNR=rows,seed_reproducibility_exact=True,grouping_symbol_samples_exact=True,grouping_estimates_tolerance_1e8=True),test_B


def real_states(gpu,w,device):
    # Only file lookup and indexing on the host. Signal/estimator/error arithmetic is CUDA.
    source=SourceEpisodes();config=json.loads((ROOT/'configs/symbol_frontend.json').read_text())
    positions=[];velocities=[];routes=[]
    for index in (0,95):
        ep=source.episodes[index];wanted=ep['start_ms']+2000
        for slot,key in enumerate(ep['source_keys']):
            track=source.tracks[key]
            for row in track:
                if int(row['time_ms'])==wanted and int(row['past_frames'])>=3:
                    positions.append([float(row['x']),float(row['y'])]);velocities.append([float(row['vx']),float(row['vy'])]);routes.append([index,slot,wanted]);break
    assert positions and {r[0] for r in routes}=={0,95}
    p=torch.tensor(positions,device=device,dtype=torch.float64);v=torch.tensor(velocities,device=device,dtype=torch.float64)
    stations=torch.tensor(config['stations'],device=device,dtype=torch.float64)
    echo=gpu.synthesize_batch(p,v,stations,waveform=w,snr_db=10,seeds=[840010+i for i in range(len(p))],device=device)
    B=gpu.divide_symbols(echo['Y'],echo['X'])
    result=gpu.estimate_batch(B,stations,config['xy_bounds'],config['v_bounds'],waveform=w,search=config['search'])
    assert torch.isfinite(result['state_hat']).all()
    error=result['state_hat']-torch.cat([p,v],dim=-1)
    return dict(episodes=[0,95],targets=len(p),routes=routes,position_rmse_m=scalar(torch.sqrt(error[:,:2].square().sum(-1).mean())),
                velocity_rmse_mps=scalar(torch.sqrt(error[:,2:].square().sum(-1).mean())),
                search_failures=sum(bool(d['search_failure']) for d in result['diagnostics'])),(B,stations,config)


def precision_and_timing(gpu,w,device,real_fixture):
    B,stations,config=real_fixture
    kwargs=dict(stations=stations,xy_bounds=config['xy_bounds'],v_bounds=config['v_bounds'],waveform=w,search=config['search'])
    baseline=gpu.estimate_batch(B,**kwargs)
    lower=gpu.estimate_batch(B.to(torch.complex64),**kwargs)
    difference=lower['state_hat'].to(torch.float64)-baseline['state_hat']
    pmax=scalar(torch.linalg.vector_norm(difference[:,:2],dim=-1).max());vmax=scalar(torch.linalg.vector_norm(difference[:,2:],dim=-1).max())
    flag_signature=lambda rows:[(d['search_failure'],d['position_search']['coarse_local_edge'],d['position_search']['fine_local_edge'],d['velocity_search']['coarse_local_edge'],d['velocity_search']['fine_local_edge']) for d in rows]
    flags_equal=flag_signature(lower['diagnostics'])==flag_signature(baseline['diagnostics'])
    fp32_pass=bool(pmax<=.025 and vmax<=.025 and flags_equal)
    timings=[]
    pool=B.repeat((max(1,(16+len(B)-1)//len(B)),1,1,1))[:16]
    for size in (1,4,8,16):
        batch=pool[:size]
        gpu.estimate_batch(batch,**kwargs);torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device);durations=[]
        for _ in range(3):
            torch.cuda.synchronize(device);start=time.perf_counter()
            result=gpu.estimate_batch(batch,**kwargs)
            torch.cuda.synchronize(device);durations.append(time.perf_counter()-start)
        median=sorted(durations)[1]
        timings.append(dict(batch_size=size,elapsed_seconds=durations,median_batch_seconds=median,
                            median_seconds_per_target=median/size,targets_per_second=size/median,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    return dict(float32_vs_float64=dict(position_max_l2_m=pmax,velocity_max_l2_mps=vmax,search_failure_flags_equal=flags_equal,all_search_edge_flags_equal=flags_equal,
                predeclared_limits=dict(position_l2_m=.025,velocity_l2_mps=.025),passed=fp32_pass,
                qualification='Same B explicitly cast to complex64; finite sampled real states, not universal precision guarantee'),
                recommended_precision='float64 until main accepts this bounded numerical evidence',
                synchronized_estimator_only_timings=timings,
                old_CPU_reference_seconds_per_target=.0764,
                timing_qualification='CPU value was measured earlier on another workload; no new CPU run and no strict same-batch speedup claim; CUDA timings include synchronization and host orchestration, exclude synthesis/source IO')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
    torch.backends.cuda.matmul.allow_tf32=False
    from . import symbol_level_gpu as gpu
    from .symbol_level import PaperWaveform
    w=PaperWaveform(K=256)
    with torch.inference_mode():
        physics,fixture=physical_check(gpu,w,args.device)
        noise,_=noise_and_grouping(gpu,w,args.device,fixture)
        real,real_fixture=real_states(gpu,w,args.device)
        performance=precision_and_timing(gpu,w,args.device,real_fixture)
    result=dict(device=args.device,device_name=torch.cuda.get_device_name(args.device),waveform=dict(K=w.K,N=w.N,B=w.B,fc=w.fc,T=w.T),
                all_required_float64_checks_passed=True,physics=physics,noise_and_reproducibility=noise,
                real_source_check=real,precision_and_performance=performance,
                scope='GPU-only heavy numerical operations; ideal separated target symbol model; no prediction training or real sensor performance claim')
    save(ROOT/'reports/symbol_gpu/validation.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
