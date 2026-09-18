"""F01-B bounded echo artifact audit; no detection, tracking, or learning."""
from pathlib import Path
import json
import time
import numpy as np
from .echo_source import SourceEpisodes, cpi_schedule, ROOT, W
from .ofdm_echo import processing_windows

OUT = ROOT/'reports/f01b'
DATA = ROOT/'data/f01_echo_audit'


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def distribution(a):
    a = np.asarray(a)
    return dict(count=int(a.size), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), p99=float(np.quantile(a, .99)),
                max=float(a.max()))


def training_bounds(source):
    """All retained training station-centre source states; no echoes allocated.

    Deduplicate repeated anchor cohorts by source key/time/station. This is a
    physical-parameter scan, not extra independent traffic or training data.
    """
    seen = set()
    values = [[] for _ in range(3)]
    eligible_count = 0
    for episode in source.episodes:
        if episode['split'] != 'train':
            continue
        for deadline in range(episode['start_ms']+100, episode['end_exclusive_ms'], 100):
            for station in range(3):
                schedule = cpi_schedule(deadline, station)
                states, slots, used = source.at_time(episode, schedule['measurement_time_ns'])
                for state, slot, row_ms in zip(states, slots, used):
                    key = (int(episode['source_keys'][slot]), row_ms, station)
                    if key in seen:
                        continue
                    seen.add(key)
                    assert row_ms*1_000_000 <= schedule['measurement_time_ns']
                    delta = state[:2]-source.stations[station]
                    r = np.sqrt(np.dot(delta, delta)+25.)
                    theta = (np.arctan2(delta[1], delta[0])-source.bores[station]+np.pi)%(2*np.pi)-np.pi
                    vr = np.dot(delta, state[2:])/r
                    speed = np.linalg.norm(state[2:])
                    visible = 10. <= r <= 300. and abs(theta) <= np.deg2rad(70.)
                    values[station].append([r, vr, abs(2*vr/W.wavelength)/W.df,
                                            speed*W.CPI, abs(vr)*W.CPI, visible])
                    eligible_count += 1
    per_station = []
    visible_all = []
    for station, rows in enumerate(values):
        a = np.asarray(rows)
        v = a[a[:, 5].astype(bool)]
        visible_all.append(v)
        per_station.append(dict(station_id=station, eligible_state_count=len(a),
                                visible_state_count=len(v), visibility_fraction=float(len(v)/len(a)),
                                visible_range_m=distribution(v[:, 0]),
                                visible_radial_velocity_mps=distribution(v[:, 1]),
                                visible_ici_ratio=distribution(v[:, 2]),
                                visible_cpi_planar_displacement_m=distribution(v[:, 3])))
    allv = np.concatenate(visible_all)
    return dict(scope='all retained training episodes, all 199 deadlines, actual station measurement centres; repeated source-key/time/station deduplicated',
                eligible_station_state_count=eligible_count, visible_station_state_count=len(allv),
                per_station=per_station, visible_ici_ratio=distribution(allv[:, 2]),
                visible_cpi_planar_displacement_m=distribution(allv[:, 3]),
                visible_cpi_radial_displacement_m=distribution(allv[:, 4]),
                visible_range_m=distribution(allv[:, 0]),
                visible_radial_velocity_mps=distribution(allv[:, 1]),
                fraction_abs_fd_over_df_gt_001=float(np.mean(allv[:, 2]>.01)),
                fraction_abs_fd_over_df_gt_004=float(np.mean(allv[:, 2]>.04)),
                fraction_radial_displacement_gt_range_bin=float(np.mean(allv[:, 4]>W.c/(2*W.B))),
                fraction_outside_cp_delay=float(np.mean(2*allv[:, 0]/W.c >= W.cp)),
                fraction_doppler_aliased=float(np.mean(np.abs(allv[:, 1]) >= W.wavelength/(4*W.Tr))),
                interpretation='ICI ratios 0.01 and 0.04 are diagnostic levels, not newly tuned acceptance gates. Frozen geometry remains an approximation; values are not clipped.')


def plot_range_doppler(saved):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True, constrained_layout=True)
    _, wk, wm = processing_windows(W)
    ranges = np.arange(W.K)*W.c/(2*W.B)
    speeds = -np.fft.fftshift(np.fft.fftfreq(W.M, W.Tr))*W.wavelength/2
    order = np.argsort(speeds)
    selected = [r for r in saved if r['elapsed_ms']==5000]
    episode_ids = list(dict.fromkeys(r['episode_id'] for r in selected))
    for rec in selected:
        with np.load(DATA/rec['file'], allow_pickle=False) as z:
            rx = z['Y']/z['X'][None, :, :]
        rd = np.fft.fftshift(np.fft.fft(np.fft.ifft(rx*wk[None,:,None]*wm[None,None,:],axis=1),axis=2),axes=2)
        power = np.mean(np.abs(rd)**2, axis=0)
        db = 10*np.log10(np.maximum(power,1e-30)/power.max())
        ax = axes[rec['station_id'], episode_ids.index(rec['episode_id'])]
        im=ax.pcolormesh(ranges, speeds[order], db[:, order].T, shading='auto', cmap='viridis', vmin=-35, vmax=0, rasterized=True)
        ax.set_title(f"Episode {episode_ids.index(rec['episode_id'])+1}, BS{rec['station_id']+1}, +5s")
        ax.set_xlabel('Slant range (m), full unambiguous FFT axis')
        ax.set_ylabel('Radial velocity (m/s), positive=receding')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Power relative to each panel maximum (dB)')
    fig.suptitle('Three-station range-Doppler diagnostics; mean antenna power, no detections')
    path=OUT/'range_doppler.png'
    fig.savefig(path,dpi=160)
    plt.close(fig)
    return path.name


def main():
    started=time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    source=SourceEpisodes()
    assert np.all(source.rows['source_key'][1:] >= source.rows['source_key'][:-1])
    assert all(np.all(np.diff(t['time_ms'])>0) for t in source.tracks.values())
    # Earliest two distinct training time intervals, not selected for performance.
    selected=[]; starts=set()
    for i,e in enumerate(source.episodes):
        if e['split']=='train' and e['start_ms'] not in starts:
            selected.append(i); starts.add(e['start_ms'])
            if len(selected)==2: break
    assert len(selected)==2
    config=dict(stage='F01-B small-scale physical echo audit only', seed=2026,
                waveform=W.as_dict(), snr_ref_db=20., episode_indices=selected,
                selection='earliest two distinct retained training episode start times',
                elapsed_deadlines_ms=[1000,5000], stations=[0,1,2], expected_saved_cubes=12,
                observation_keys=['Y','X','station_id','measurement_time_ns','deadline_ns'],
                data_directory='data/f01_echo_audit', truth_audit_directory='reports/f01b',
                pressure='separate deterministic audit at 20dB; no saved pressure production cache')
    write_json(ROOT/'configs/f01b.json',config)
    records=[]; truth=[]; causal=0; reproducible=0; schedules=[]
    for episode_index in selected:
        e=source.episodes[episode_index]
        assert np.array_equal(source.pressure_states(episode_index),source.pressure_states(episode_index))
        for elapsed in (1000,5000):
            deadline=e['start_ms']+elapsed
            triplet=[cpi_schedule(deadline,b) for b in range(3)]
            assert triplet[0]['slot_end_ns']==triplet[1]['slot_start_ns']
            assert triplet[1]['slot_end_ns']==triplet[2]['slot_start_ns']
            assert triplet[2]['slot_end_ns']==deadline*1_000_000
            # Delete every post-deadline row: target departure and future labels
            # cannot influence source activation or emitted observations.
            prefix={k:source.tracks[k][source.tracks[k]['time_ms']<=deadline]
                    for k in e['source_keys']}
            for station in range(3):
                obs,audit=source.observation(episode_index,deadline,station)
                repeated,_=source.observation(episode_index,deadline,station)
                truncated,_=source.observation(episode_index,deadline,station,tracks=prefix)
                assert all(np.array_equal(obs[k],repeated[k]) for k in obs)
                reproducible+=1
                assert all(np.array_equal(obs[k],truncated[k]) for k in obs)
                causal+=1
                schedule=audit['schedule']
                assert schedule['reference_last_sample_end_ns']<=schedule['slot_end_ns']<=schedule['deadline_ns']
                assert all(t*1_000_000<=schedule['measurement_time_ns'] for t in audit['source_sample_ms'])
                assert set(obs)==set(config['observation_keys'])
                assert obs['Y'].shape==(W.A,W.K,W.M) and obs['Y'].dtype==np.complex64
                assert np.isfinite(obs['Y']).all() and np.isfinite(obs['X']).all()
                filename=f"episode{episode_index:03d}_t{elapsed:05d}_bs{station}.npz"
                path=DATA/filename
                np.savez(path,**obs)
                with np.load(path,allow_pickle=False) as stored:
                    assert set(stored.files)==set(obs)
                    assert all(np.array_equal(stored[k],obs[k]) for k in obs)
                records.append(dict(file=filename,episode_id=e['episode_id'],episode_index=episode_index,
                                    elapsed_ms=elapsed,station_id=station,bytes=path.stat().st_size))
                truth.append(dict(file=filename,**audit))
                schedules.append(schedule)
    pressure_checks=[]
    for episode_index in selected:
        e=source.episodes[episode_index]; deadline=e['start_ms']+5000
        for station in range(3):
            a,aa=source.observation(episode_index,deadline,station,pressure=True)
            b,bb=source.observation(episode_index,deadline,station,pressure=True)
            assert all(np.array_equal(a[k],b[k]) for k in a)
            pressure_checks.append(dict(episode_index=episode_index,station_id=station,
                                        deterministic=True,amplitude_scales=aa['per_target_amplitude_scale']))
    bounds=training_bounds(source)
    cp=bounds['fraction_outside_cp_delay']; alias=bounds['fraction_doppler_aliased']
    assert cp==0 and alias==0, (cp,alias)
    write_json(OUT/'training_physical_bounds.json',bounds)
    plot=plot_range_doppler(records)
    (OUT/'echo_truth_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in truth))
    checks=dict(passed=True,saved_cube_count=len(records),saved_bytes=sum(r['bytes'] for r in records),
                same_seed_exact_reproductions=reproducible,future_prefix_deletion_exact_matches=causal,
                pressure_exact_reproductions=len(pressure_checks),pressure_details=pressure_checks,
                scheduling_nonoverlap=True,all_reference_samples_before_deadline=True,
                saved_keys_exclude_truth=True,npz_readback_exact=True,source_track_sort_checked=True,
                elapsed_seconds=time.perf_counter()-started)
    write_json(OUT/'source_checks.json',checks)
    write_json(OUT/'observation_manifest.json',dict(scope=config['stage'],files=records,schedules=schedules,
        runtime_executable=__import__('sys').executable,
        seed_recipe='SeedSequence([2026,episode_index,frame_index,station_id,stream]); frame_index=elapsed_ms/100-1',
        streams={'QPSK':0,'phase':1,'noise':2,'pressure':44},
        generator_sha256={name:__import__('hashlib').sha256((ROOT/'frontend'/name).read_bytes()).hexdigest() for name in ('ofdm_echo.py','echo_source.py','run_echo_audit.py')}))
    report=f'''# F01-B small-scale three-station echo audit

Status: source adapter and 12 observation cubes passed the listed checks. Independent physical tests are recorded in validation.json / validation.md. This is not a full production cache, detector validation, or prediction result.

## Delivered observations

Two earliest distinct retained training time intervals, episode indices {selected}, are evaluated at +1s and +5s, three stations each. The saved 12 NPZ files total {checks['saved_bytes']/1024**2:.2f} MiB. Each Y is complex64 with shape 16 x 512 x 256; X is the known 512 x 256 QPSK reference array. Only Y, X, station_id, measurement_time_ns, and deadline_ns are saved. True target state/slot/phase/visibility information stays separately in reports/f01b/echo_truth_audit.jsonl, and must not be passed downstream.

The +1s examples are burn-in observations, not valid 20-step prediction histories. Episodes sharing traffic are not claimed as independent statistical samples.

## Executed checks

- 12 exact repeated observations under common deterministic seeds.
- 12 exact observations after deleting all post-deadline source rows, holding the sealed F01-A source/episode configuration fixed. This tests the source adapter and echoes; it does not newly rerun F01-A preprocessing or test a future tracker.
- 6 exact pressure-mode repeated observations; pressure fields are deterministic Markov attenuation, not real mapped building occlusion.
- Three TDMA slots are adjacent and nonoverlapping; every reference sample ends before or at the deadline. Measurement time is the average reference useful-symbol centre, not the slot midpoint.
- All files were reopened and compared exactly; finite values, shapes, dtypes and absence of true target metadata in saved observation files were checked.

## Full-training physical parameter scan

Without constructing full echoes, every 100ms update in every retained training episode was checked at the actual three station measurement centres. Repeated source-key/time/station tuples were deduplicated. Eligible station-source states: {bounds['eligible_station_state_count']}; visible: {bounds['visible_station_state_count']}.

- Maximum visible |fd|/df: {bounds['visible_ici_ratio']['max']:.6f}; 95th percentile: {bounds['visible_ici_ratio']['p95']:.6f}.
- Fraction |fd|/df > 0.01: {bounds['fraction_abs_fd_over_df_gt_001']:.6%}; > 0.04: {bounds['fraction_abs_fd_over_df_gt_004']:.6%}.
- Maximum visible planar displacement per CPI: {bounds['visible_cpi_planar_displacement_m']['max']:.6f} m.
- Maximum visible radial displacement per CPI: {bounds['visible_cpi_radial_displacement_m']['max']:.6f} m; fraction exceeding one range bin: {bounds['fraction_radial_displacement_gt_range_bin']:.6%}.
- Fraction visible round-trip delay beyond CP: {cp:.6%}; Doppler alias fraction: {alias:.6%}.

No fast vehicles were clipped. Small ICI and frozen within-CPI geometry remain approximations; the independent time-domain check quantifies waveform mismatch rather than claiming strict equivalence. Per-station and percentile details are in training_physical_bounds.json.

## Diagnostic figure and next boundary

range_doppler.png shows all unambiguous range/velocity bins at +5s for each selected episode and all three stations, using antenna-averaged power. It contains no truth-guided search or detections. Noise-floor appearance is not detection performance.

Next work is shared unknown-identity detection and tracking. No CFAR, tracker, model training, station-count ablation or full-scale production cache was run in F01-B.

Reproduce with the active project environment from project root: `/home/dell/YrM/envs/ICCT/bin/python -m frontend.run_echo_audit`. Configuration: configs/f01b.json. Source checks: source_checks.json. Runtime for this audit: {checks['elapsed_seconds']:.1f}s.
'''
    (OUT/'F01B_REPORT.md').write_text(report)
    print(json.dumps(checks,indent=2))
    print(json.dumps({k:v for k,v in bounds.items() if k!='per_station'},indent=2))


if __name__=='__main__':
    main()
