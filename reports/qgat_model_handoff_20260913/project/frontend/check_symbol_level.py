"""Independent Wei2024 symbol-level algebra/numerical checks; exactly three BSs.

Fixed geometry/search bounds are declared before any random target is drawn.
Truth is used only by the simulator and error calculations, never by estimate().
"""
from dataclasses import asdict
from pathlib import Path
import inspect,json,time
import numpy as np
from . import symbol_level as core

ROOT=Path(__file__).resolve().parents[1]
STATIONS=np.array([[-90.,-70.],[95.,-65.],[0.,110.]])
XY_BOUNDS=np.array([[-15.,15.],[-12.,12.]])
VELOCITY_BOUNDS=np.array([[-10.,10.],[-10.,10.]])
SEARCH=dict(oversample=2,position_half_width=3.,position_step=.25,position_fine_step=.025,
            velocity_half_width=3.,velocity_step=.25,velocity_fine_step=.025)
SNRS=(-20.,-15.,-10.,-5.)
TRIALS=64


def estimate(symbols,w):
    return core.estimate(symbols,STATIONS,XY_BOUNDS,VELOCITY_BOUNDS,w,search=SEARCH)


def relative(a,b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(float(np.max(np.abs(b))),1e-30))


def direct_lags(vector):
    return np.array([np.mean(vector[:-k]*np.conj(vector[k:])) for k in range(1,len(vector))])


def waveform_info(w):
    return dict(asdict(w),df=w.B/w.K,useful_symbol_duration_from_df_s=w.K/w.B,
                unambiguous_range_m=w.c*w.K/(2*w.B),range_resolution_m=w.c/(2*w.B),
                unambiguous_approaching_velocity_mps=w.c/(4*w.fc*w.T),
                velocity_resolution_mps=w.c/(2*w.fc*w.N*w.T),
                observation_duration_s=w.N*w.T,
                df_equals_inverse_T=bool(np.isclose(w.B/w.K,1/w.T)))


def selfchecks(w):
    checks={};p=np.array([2.371,-3.149]);v=np.array([3.217,-1.463]);phases=np.array([.31,-1.27,2.19])
    delta=STATIONS-p;r=np.linalg.norm(delta,axis=1);vr=delta@v/r
    sample=core.synthesize(p,v,STATIONS,w,snr_db=-5,rng=np.random.default_rng(41001),phases=phases,noise=False)
    B=core.divide_symbols(**sample)
    expected=10**(-5/20)*np.exp(1j*phases)[:,None,None]*np.exp(-1j*4*np.pi*w.df/w.c*r[:,None]*np.arange(w.K))[:,:,None]*np.exp(1j*4*np.pi*w.fc*w.T/w.c*vr[:,None]*np.arange(w.N))[:,None,:]
    error=relative(B,expected)
    checks['equation6_demodulated_symbols']=dict(relative_error=error,passed=error<1e-12)
    extraX=np.exp(1j*(np.pi/4+np.pi/2*np.random.default_rng(41002).integers(0,4,B.shape)))
    separated=core.divide_symbols(B*extraX,extraX)
    checks['known_payload_separation']=dict(relative_error=relative(separated,B),passed=relative(separated,B)<1e-12)
    try:core.divide_symbols(sample['Y'],np.zeros_like(sample['X']))
    except ValueError:zero_rejected=True
    else:zero_rejected=False
    checks['zero_reference_rejected']=dict(passed=zero_rejected)
    result=estimate(B,w);state=np.asarray(result['state_hat']);e=state-np.r_[p,v]
    checks['noiseless_offgrid_position_velocity']=dict(state_hat=state.tolist(),error=e.tolist(),
        search_failure=result['diagnostics']['search_failure'],passed=bool(np.linalg.norm(e[:2])<.06 and np.linalg.norm(e[2:])<.06 and not result['diagnostics']['search_failure']))
    parked=core.synthesize(p,[0.,0.],STATIONS,w,snr_db=-5,rng=np.random.default_rng(41003),phases=phases,noise=False)
    zero=estimate(core.divide_symbols(**parked),w)
    checks['stationary_target_retained']=dict(state_hat=zero['state_hat'].tolist(),passed=bool(np.linalg.norm(zero['state_hat'][2:])<.06))
    # Direct phase slopes resolve the paper's approaching-positive convention.
    observed_v=np.angle(B[:,0,1]*np.conj(B[:,0,0]))*w.c/(4*np.pi*w.fc*w.T)
    observed_r_phase=np.angle(B[:,1,0]*np.conj(B[:,0,0]))
    expected_r_phase=np.angle(np.exp(-1j*4*np.pi*w.df/w.c*r))
    checks['range_and_approaching_velocity_sign']=dict(radial_truth=vr.tolist(),radial_phase_estimate=observed_v.tolist(),
        passed=bool(np.allclose(observed_v,vr,atol=1e-10) and np.allclose(observed_r_phase,expected_r_phase,atol=1e-12) and np.any(vr>0) and np.any(vr<0)))
    features=result['symbol_features'];coarse=result['coarse']['range_radial_velocity']
    refE=np.array([B[b]@np.exp(-1j*4*np.pi*w.fc*w.T/w.c*np.arange(w.N)*coarse[b,1]) for b in range(3)])
    refF=np.array([np.exp(1j*4*np.pi*w.df/w.c*np.arange(w.K)*coarse[b,0])@B[b] for b in range(3)])
    errors={key:relative(features[key],value) for key,value in [('E',refE),('F',refF),('G',np.array([direct_lags(x) for x in refE])),('I',np.array([direct_lags(x) for x in refF]))]}
    checks['equations13_15_22_32_direct_reference']=dict(relative_errors=errors,passed=max(errors.values())<1e-11)
    analyticG=np.abs(refE[:,0])[:,None]**2*np.exp(1j*4*np.pi*w.df/w.c*r[:,None]*np.arange(1,w.K))
    analyticI=np.abs(refF[:,0])[:,None]**2*np.exp(-1j*4*np.pi*w.fc*w.T/w.c*vr[:,None]*np.arange(1,w.N))
    checks['lag_range_positive_and_velocity_negative_phase']=dict(G_relative_error=relative(features['G'],analyticG),I_relative_error=relative(features['I'],analyticI),
        passed=bool(relative(features['G'],analyticG)<1e-10 and relative(features['I'],analyticI)<1e-10))
    # Evaluate the published sum-of-products directly; no Gaussian-product premise.
    fixed_positions=np.array([[-5.,4.],[3.,-2.],[0.,0.],[10.,8.]])
    candidate_ranges=np.linalg.norm(fixed_positions[:,None,:]-STATIONS,axis=2)
    fixed_velocities=np.array([[0.,0.],[2.,-3.],[-4.,1.],[6.,5.]])
    U=(STATIONS-state[:2])/np.linalg.norm(STATIONS-state[:2],axis=1)[:,None]
    score_errors={}
    for name,parameters,vector,factor in [('H',candidate_ranges,features['G'],-4*np.pi*w.df/w.c),('J',fixed_velocities@U.T,features['I'],4*np.pi*w.fc*w.T/w.c)]:
        scale=np.max(np.abs(vector),axis=1)
        direct=np.array([sum(np.prod([np.real(vector[b,k-1]*np.exp(1j*factor*values[b]*k))/scale[b] for b in range(3)]) for k in range(1,vector.shape[1]+1)) for values in parameters])
        computed=core._fused_score(parameters,vector,factor)
        score_errors[name]=relative(computed,direct)
    checks['equations26_37_fused_score_direct_reference']=dict(relative_errors=score_errors,passed=max(score_errors.values())<1e-10)
    noisy=core.synthesize(p,v,STATIONS,w,snr_db=-10,rng=np.random.default_rng(41004),phases=phases,noise=True)
    noisyB=core.divide_symbols(**noisy);baseline=estimate(noisyB,w)
    rotated=noisyB*np.exp(1j*np.array([1.731,-2.413,.987]))[:,None,None]
    rotation=estimate(rotated,w)
    phase_errors={key:relative(rotation['symbol_features'][key],baseline['symbol_features'][key]) for key in ('G','I')}
    checks['independent_station_constant_phase_invariance']=dict(lag_relative_errors=phase_errors,
        state_difference=(rotation['state_hat']-baseline['state_hat']).tolist(),
        passed=bool(max(phase_errors.values())<1e-10 and np.allclose(rotation['state_hat'],baseline['state_hat'],atol=1e-10)),
        interpretation='Each entire received station matrix is rotated; circular-noise distribution is unchanged. Independent new noise draws are not claimed samplewise identical.')
    # Constant coefficient phase cancellation does not remove delay/frequency bias.
    delay_bias_m=.4;doppler_bias_mps=.7
    e_bias=refE[0]*np.exp(-1j*4*np.pi*w.df/w.c*delay_bias_m*np.arange(w.K))
    f_bias=refF[0]*np.exp(1j*4*np.pi*w.fc*w.T/w.c*doppler_bias_mps*np.arange(w.N))
    rg=core.lag_correlation(e_bias)/features['G'][0];ri=core.lag_correlation(f_bias)/features['I'][0]
    checks['constant_phase_cancellation_does_not_cancel_bias']=dict(
        passed=bool(np.allclose(rg,np.exp(1j*4*np.pi*w.df/w.c*delay_bias_m*np.arange(1,w.K)),atol=1e-10) and np.allclose(ri,np.exp(-1j*4*np.pi*w.fc*w.T/w.c*doppler_bias_mps*np.arange(1,w.N)),atol=1e-10)),
        delay_bias_m=delay_bias_m,radial_velocity_bias_mps=doppler_bias_mps)
    first=core.synthesize(p,v,STATIONS,w,snr_db=-15,rng=np.random.default_rng(41005))
    second=core.synthesize(p,v,STATIONS,w,snr_db=-15,rng=np.random.default_rng(41005))
    checks['deterministic_seed']=dict(passed=bool(np.array_equal(first['Y'],second['Y']) and np.array_equal(first['X'],second['X'])))
    measurements=[]
    for snr in SNRS:
        clean=core.synthesize(p,v,STATIONS,w,snr_db=snr,rng=np.random.default_rng(41006),phases=phases,noise=False)
        received=core.synthesize(p,v,STATIONS,w,snr_db=snr,rng=np.random.default_rng(41006),phases=phases,noise=True)
        noise=received['Y']-clean['Y'];signal_power=np.mean(abs(clean['Y'])**2,axis=(1,2));noise_power=np.mean(abs(noise)**2,axis=(1,2))
        measured=10*np.log10(signal_power/noise_power)
        measurements.append(dict(requested_snr_db=snr,measured_snr_db=measured.tolist(),signal_power=signal_power.tolist(),noise_variance=noise_power.tolist(),real_noise_variance=float(noise.real.var()),imag_noise_variance=float(noise.imag.var())))
    checks['raw_received_symbol_snr']=dict(measurements=measurements,passed=bool(all(max(abs(np.array(r['measured_snr_db'])-r['requested_snr_db']))<.15 for r in measurements)),
        definition='Per-station received demodulated complex symbol: E|signal|^2/E|noise|^2; unit complex-noise variance and signal amplitude10^(SNR/20). Unit-modulus QPSK division preserves noise variance.')
    parameters=list(inspect.signature(core.estimate).parameters)
    checks['estimator_interface_no_truth_parameters']=dict(parameters=parameters,passed=parameters==['B','stations','xy_bounds','v_bounds','waveform','search'],
        qualification='Signature and manual source review: coarse estimates and search centers depend on observed B, with configured global bounds; truth is only used in this validator for synthesis and error measurement.')
    checks['configured_bounds_and_geometry']=dict(passed=bool(np.linalg.matrix_rank(STATIONS[1:]-STATIONS[0])==2 and result['diagnostics']['velocity_geometry_condition']<3 and np.array_equal(result['diagnostics']['configured_xy_bounds'],XY_BOUNDS)),
        station_count=3,velocity_geometry_condition=result['diagnostics']['velocity_geometry_condition'])
    return checks


def monte_carlo(w,profile_index):
    # Bounds and all search settings already fixed above, before sampling truths.
    truth_rng=np.random.default_rng(2026091501)
    positions=truth_rng.uniform([-10.,-8.],[10.,8.],size=(TRIALS,2))
    velocities=truth_rng.uniform([-8.,-8.],[8.,8.],size=(TRIALS,2))
    rows=[];summaries=[]
    for snr in SNRS:
        local=[]
        for i,(position,velocity) in enumerate(zip(positions,velocities)):
            seed=2026091502+profile_index*10000+i
            obs=core.synthesize(position,velocity,STATIONS,w,snr_db=snr,rng=np.random.default_rng(seed))
            result=estimate(core.divide_symbols(**obs),w);error=result['state_hat']-np.r_[position,velocity]
            record=dict(snr_db=snr,trial=i,seed=seed,truth=np.r_[position,velocity].tolist(),estimate=result['state_hat'].tolist(),
                position_sq_error=float(error[:2]@error[:2]),velocity_sq_error=float(error[2:]@error[2:]),
                search_failure=bool(result['diagnostics']['search_failure']))
            rows.append(record);local.append(record)
        summaries.append(dict(snr_db=snr,trials=TRIALS,position_rmse_m=float(np.sqrt(np.mean([r['position_sq_error'] for r in local]))),
            velocity_rmse_mps=float(np.sqrt(np.mean([r['velocity_sq_error'] for r in local]))),search_failures=sum(r['search_failure'] for r in local)))
        print('K',w.K,'SNR',snr,'summary',summaries[-1],flush=True)
    low=[r for r in rows if r['snr_db']==SNRS[0]];high=[r for r in rows if r['snr_db']==SNRS[-1]]
    bootstrap_rng=np.random.default_rng(2026091503);sample_indices=bootstrap_rng.integers(0,TRIALS,(2000,TRIALS));statistics={}
    for label,key in [('position','position_sq_error'),('velocity','velocity_sq_error')]:
        difference=np.array([a[key]-b[key] for a,b in zip(low,high)])
        interval=np.quantile(difference[sample_indices].mean(axis=1),[.025,.975])
        statistics[label]=dict(mean_low_minus_high_squared_error=float(difference.mean()),paired_bootstrap_95ci=interval.tolist(),
            lower_snr_worse_fraction=float(np.mean(difference>0)),passed=bool(interval[0]>0))
    nonzero=all(s['position_rmse_m']>1e-5 and s['velocity_rmse_mps']>1e-5 for s in summaries)
    return dict(per_snr=summaries,paired_extreme_snr_comparison=statistics,noise_error_nonzero=nonzero,
        passed=bool(nonzero and all(s['passed'] for s in statistics.values()) and not any(r['search_failure'] for r in rows)),
        notes='Same truths and noise seed paired across SNR within profile; assess average squared error, not monotonicity of every sample. No station-count comparison.',trials=rows)


def main():
    start=time.perf_counter();out=ROOT/'reports/symbol_level';out.mkdir(parents=True,exist_ok=True)
    profiles=[]
    for profile_index,K in enumerate((128,256)):
        w=core.PaperWaveform(K=K);checks=selfchecks(w)
        print('K',K,'selfchecks', {k:v['passed'] for k,v in checks.items()},flush=True)
        result=dict(profile='paper_parameter_reference' if K==128 else 'project_carrier_extension',waveform=waveform_info(w),checks=checks,
                    selfchecks_passed=bool(all(c['passed'] for c in checks.values())))
        # Persist numerical checks even if a later Monte Carlo check fails.
        (out/f'validation_K{K}_selfchecks.json').write_text(json.dumps(result,indent=2)+'\n')
        result['monte_carlo']=monte_carlo(w,profile_index)
        result['passed']=bool(result['selfchecks_passed'] and result['monte_carlo']['passed'])
        (out/f'validation_K{K}.json').write_text(json.dumps(result,indent=2)+'\n');profiles.append(result)
    report=dict(stage='symbol-level single-target estimator validation',passed=bool(all(p['passed'] for p in profiles)),
        station_count=3,stations=STATIONS.tolist(),configured_xy_bounds=XY_BOUNDS.tolist(),configured_velocity_bounds=VELOCITY_BOUNDS.tolist(),search=SEARCH,
        monte_carlo_trials_per_snr_per_profile=TRIALS,snr_db=list(SNRS),profiles=[{k:v for k,v in p.items() if k!='monte_carlo'}|{'monte_carlo':{k:v for k,v in p['monte_carlo'].items() if k!='trials'}} for p in profiles],
        elapsed_seconds=time.perf_counter()-start,assumptions=[
            'Targets are already separated and cross-station correspondence is known; this does not test blind detection, association, or tracking.',
            'Exactly three fixed noncollinear planar BSs; global search bounds fixed before random truths; local refinement is observation-derived.',
            'Ideal demodulated OFDM symbols, known unit-modulus QPSK, independent circular Gaussian noise, unknown constant per-station phase, constant CPI range/radial velocity.',
            'Received-symbol SNR is prescribed equally at BSs; channel gain is normalized and path loss is not simulated.',
            'No ranging-clock bias, residual carrier-frequency bias, extended-target multipath or unresolved-target leakage is modeled.',
            'df=B/K and T are independent in this frequency-domain implementation; TableII numbers do not satisfy the textual df=1/T relation.',
            'K256 extends ambiguity range at fixed bandwidth; it is a project modification, not an exact reproduction of published figures.'],
        paper_audit=[
            'Eq20 printed lim|cos(x)| as x tends to0 is incorrect; the limit is1. Correctness checks do not rely on it.',
            'Eq51 discussion does not establish a generally Gaussian product or universally reduced variance. Products of independent Gaussian random variables are not generally Gaussian.',
            'The implemented lag conjugation and real-part product score are checked directly against Eq6,13,15,22,26,32,37.',
            'Original sites/refinement grid steps are insufficiently specified for exact graph reproduction; this64-trial-per-SNR test is a numerical smoke study, not the paper1000-trial replication.'])
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 符号级三基站估计器：独立验证','',f"综合数值检查：{'通过' if report['passed'] else '存在未通过项'}。所有试验固定使用三个基站，不比较基站数量。",'',
        '## 配置与证据范围','',
        '先固定三站坐标、位置/速度搜索边界和细化步长，再独立随机抽取真值。估计器只接收符号矩阵B与预设参数；真值仅用于生成信号和统计误差。逐车分离及跨站对应已知属于输入假设，不代表完成盲检测或关联。','',
        '参考配置K=128与项目扩展K=256均采用24GHz、93.1MHz、256个符号、12.375微秒符号间隔。实现取Δf=B/K，并把T作为独立完整符号间隔；论文表中数值与正文Δf=1/T存在不一致，因此这里是理想频域符号模型，不声称完成严格等价的原始时域链路或原论文图复现。','',
        '## 四档接收符号SNR的蒙特卡洛结果','',
        '| K | SNR/dB | 次数 | 位置RMSE/m | 速度RMSE/(m/s) | 搜索边界/优化异常 |','|---:|---:|---:|---:|---:|---:|']
    for profile in profiles:
        for row in profile['monte_carlo']['per_snr']:
            lines.append(f"| {profile['waveform']['K']} | {row['snr_db']:g} | {TRIALS} | {row['position_rmse_m']:.6f} | {row['velocity_rmse_mps']:.6f} | {row['search_failures']} |")
    lines+=['','使用配对噪声种子比较-20dB与-5dB的平均平方误差及bootstrap区间，不要求每个随机样本都随SNR单调改善。非零噪声下保留实际非零误差，不作100%精度结论；无噪声检查也受有限搜索网格影响。','',
        '## 物理与代数核对','',
        '核对已知QPSK除法、接近基站为正的径向速度、距离负相位及慢时间正相位、E/F压缩、G/I滞后共轭相关的相位、跨三站逐lag实部相乘求和，以及固定种子的可复现性。常数站相位在lag相关后抵消；延迟或频率偏差形成的相位斜率不会因此消失。','',
        'SNR定义为每站解调后复符号的信号功率与复噪声方差之比；接收信号幅度为10^(SNR/20)，噪声复方差为1。没有距离路径损耗，不把该定义混同此前加窗相干输出SNR或ADC采样SNR。','',
        '论文Eq20的余弦极限表述有误，Eq51附近关于高斯随机量乘积仍为高斯、方差必然降低的结论不具有一般性。实现验收依赖代数逐式检查及实际数值结果，不沿用上述论证作为性能保证。','',
        '详细逐项检查与全部随机试验保存在validation_K128.json、validation_K256.json，汇总为validation.json。64次/档是小规模独立验证，论文的1000次蒙特卡洛曲线不在本次复现声明内。真实NGSIM适配与因果前缀检查由主流程另行完成。']
    (out/'validation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=report['passed'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    if not report['passed']:raise SystemExit('Independent symbol-level validation has failed checks; inspect artifacts')

if __name__=='__main__':main()
