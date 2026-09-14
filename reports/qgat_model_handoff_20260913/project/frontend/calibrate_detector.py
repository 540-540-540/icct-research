"""Design-only noise threshold calibration, independent holdout and smoke checks."""
from pathlib import Path
import json,time
import numpy as np
from scipy.stats import chi2
from .ofdm_echo import DEFAULT_WAVEFORM as W,echo_cube
from .detector import DEFAULTS,detect,candidate_map,candidate_map_cpu,_grid

ROOT=Path(__file__).resolve().parents[1]


def noise_observation(seed):
    rng=np.random.default_rng(seed)
    Y=(rng.standard_normal((W.A,W.K,W.M),dtype=np.float32)+1j*rng.standard_normal((W.A,W.K,W.M),dtype=np.float32))/np.sqrt(2.)
    return dict(Y=Y.astype(np.complex64),X=np.ones((W.K,W.M),np.complex64),station_id=0,measurement_time_ns=123456)


def main():
    started=time.perf_counter()
    output=ROOT/'reports/f01c';output.mkdir(parents=True,exist_ok=True)
    config=dict(DEFAULTS)
    design_n=1024;holdout_n=1024;design_seed=6202600;holdout_seed=7202600
    scores=[];design_counts=[];censor_bounds=[]
    broad=dict(config,threshold_multiplier=.5,max_candidates=32)
    for i in range(design_n):
        detections,design_diag=detect(noise_observation(design_seed+i),broad)
        if design_diag['overflow']:
            censor_bounds.append(min(d['quality']['cfar_score'] for d in detections))
        scores.extend(d['quality']['cfar_score'] for d in detections)
        design_counts.append(len(detections))
        if i%128==127: print('noise design',i+1,flush=True)
    scores=np.sort(scores)[::-1]
    target_count=round(.1*design_n)
    if len(scores)<=target_count: raise RuntimeError('Calibration score floor too high')
    threshold=float((scores[target_count-1]+scores[target_count])/2)
    if censor_bounds and threshold<=max(censor_bounds):
        raise RuntimeError('Design candidate cap could affect selected threshold')
    config.update(threshold_multiplier=threshold,revision='C01',revision_reason='Array mainlobe guard4/outer8; local NMS1/1/1 avoids deleting measured resolved peaks; ULA sidelobe suppression and waveform unchanged',calibration_design_cpis=design_n,calibration_holdout_cpis=holdout_n,
                  calibration_design_seed_start=design_seed,calibration_holdout_seed_start=holdout_seed,
                  design_false_alarms=target_count,design_false_alarms_per_cpi=target_count/design_n,
                  p_D=.9,p_D_status='initial value; training detection calibration pending',
                  window_description='periodic Hann range/slow, rectangular 16-element array, scan FFT64',
                  backend='torch CUDA with scipy CPU reference/fallback',
                  noise_model='unit-variance circular CN frequency noise; CFAR scale invariant',
                  calibration_note='NMS and interpolation geometry filtering included; no source traffic or target truth used')
    counts=[];overflow=0
    for i in range(holdout_n):
        ds,diag=detect(noise_observation(holdout_seed+i),config)
        counts.append(len(ds));overflow+=int(diag['overflow'])
        if i%128==127: print('noise holdout',i+1,flush=True)
    total=sum(counts);rate=total/holdout_n
    poisson_ci=[float(.5*chi2.ppf(.025,2*total)/holdout_n) if total else 0.,float(.5*chi2.ppf(.975,2*(total+1))/holdout_n)]
    volume=(300-10)*(2*np.deg2rad(70))*(W.wavelength/(2*W.Tr))
    config.update(lambda_c=max((target_count/design_n)/volume,1e-12),
                  lambda_c_units='detections/(m rad m/s)',lambda_c_source='design mean divided by continuous valid measurement-domain volume',
                  measurement_domain_volume=float(volume))
    # Holdout remains assessment only; no threshold/lambda retuning after seeing it.
    calibration=dict(design=dict(cpis=design_n,seed_start=design_seed,total_false_alarms=target_count,
                                    mean=target_count/design_n,threshold_multiplier=threshold),
                     holdout=dict(cpis=holdout_n,seed_start=holdout_seed,counts=counts,total_false_alarms=total,
                                  mean=rate,poisson_95ci=poisson_ci,
                                  variance=float(np.var(counts,ddof=1)),overflow_cpis=overflow,
                                  interpretation='Poisson interval for independent-CPI mean; cell correlations are not assumed independent'),
                     target_mean=.1,calibration_frozen_before_holdout=True,design_cap_check_passed=True,design_capped_cpis=len(censor_bounds),holdout_target_in_poisson95ci=bool(poisson_ci[0]<=.1<=poisson_ci[1]),holdout_assessment='Interpret the independently measured mean and interval; do not retune on this holdout.')
    tests={}
    cases=dict(stationary=[[100,0,0,0]],moving=[[80,25,8,-2]],
               two_resolvable=[[70,0,0,0],[85,35,3,4]],
               same_range_two_angles=[[80,40,0,0],[80,-40,0,0]],
               out_of_fov=[[-100,0,0,0]])
    for name,states in cases.items():
        y,x,audit=echo_cube(states,[0,0],0,rng=20260912,snr_db=25)
        obs=dict(Y=y,X=x,station_id=1,measurement_time_ns=654321)
        ds,diag=detect(obs,config)
        truth=np.column_stack([audit['range_m'],audit['theta_rad'],audit['radial_velocity_mps']])
        errors=[]
        for z in truth:
            if ds:
                nearest=min(ds,key=lambda d:np.linalg.norm((np.array(d['z'])-z)/[1.5,.1,1]))
                errors.append((np.array(nearest['z'])-z).tolist())
        passed=(len(ds)>=len(states) and all(abs(e[0])<1.5 and abs(e[1])<np.deg2rad(5) and abs(e[2])<.5 for e in errors)) if name!='out_of_fov' else len(ds)==0
        tests[name]=dict(passed=passed,detections=ds,diagnostics=diag,truth_used_only_for_validation=truth.tolist(),errors=errors)
    obs=noise_observation(8202600)
    cpu=candidate_map_cpu(obs,config);gpu=candidate_map(obs,config=config)
    tests['cpu_gpu_reference']=dict(power_max_relative=float(np.max(np.abs(cpu[2]-gpu[2]))/np.max(cpu[2])),
        cfar_noise_relative_rms=float(np.linalg.norm(cpu[3]-gpu[3])/np.linalg.norm(cpu[3])))
    tests['cpu_gpu_reference']['passed']=tests['cpu_gpu_reference']['power_max_relative']<1e-5 and tests['cpu_gpu_reference']['cfar_noise_relative_rms']<1e-5
    # Boundary training cells use actual valid cells, rather than fixed interior N.
    grid=_grid(config['training_half_width'][2],config['guard_half_width'][2]);counts_grid=grid[5][grid[4]]
    tests['boundary_training_counts']=dict(min=int(counts_grid.min()),max=int(counts_grid.max()),passed=bool(counts_grid.min()>0 and counts_grid.min()<counts_grid.max()))
    # Same observation with arbitrary forbidden audit keys cannot affect output.
    d1,q1=detect(obs,config); d2,q2=detect(dict(obs,source_key=[999],target_count=800,audit={'truth':'unused'}),config)
    tests['truth_metadata_ignored']=dict(passed=d1==d2 and q1==q2)
    times=[]
    for _ in range(5):
        start=time.perf_counter();detect(obs,config);times.append(time.perf_counter()-start)
    report=dict(revision=config.get('revision'),calibration=calibration,synthetic_checks=tests,all_synthetic_checks_passed=bool(all(v['passed'] for v in tests.values())),
                detector_seconds_median=float(np.median(times)),elapsed_seconds=time.perf_counter()-started,
                limitations=['Default R remains a conservative initial covariance until training residual calibration.',
                             'Uniform clutter density approximates the nonuniform angle-grid search.',
                             'Noise holdout is finite; 0.1 false alarms/CPI is a target, not a deterministic guarantee.',
                             'Rectangular-array sidelobe suppression can merge weaker co-range/co-Doppler sources under a strong sidelobe.'])
    (ROOT/'configs/detector.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'detector_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_checks_passed=report['all_synthetic_checks_passed'],threshold_multiplier=threshold,holdout_mean=rate,holdout_poisson_95ci=poisson_ci,median_seconds=report['detector_seconds_median'])),flush=True)
    if not report['all_synthetic_checks_passed']: raise SystemExit('Synthetic validation failed; inspect report')

if __name__=='__main__': main()
