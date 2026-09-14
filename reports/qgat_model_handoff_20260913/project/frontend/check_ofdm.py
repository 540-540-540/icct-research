"""Independent OFDM physical checks; never used by the batch echo generator."""
import json
import numpy as np


def time_domain_reference(X, delay_samples, fd_hz, *, bandwidth_hz=1e8,
                          cp_samples=256, reference_period_s=61.44e-6):
    """IFFT, actual CP, delayed channel, per-sample Doppler, CP removal, FFT.

    Fractional delay evaluates the periodic bandlimited transmitted waveform
    using unwrapped nonnegative OFDM carrier indices, as in the protocol.
    Integer delays use direct samples of the CP-bearing transmitted buffer.
    Delay must fit inside CP; no previous-symbol contamination is modelled.
    """
    X = np.asarray(X, dtype=np.complex128)
    K, M = X.shape
    if not 0 <= delay_samples < cp_samples or not 0 < cp_samples <= K:
        raise ValueError("Delay must fit inside a nonempty CP")
    useful = np.fft.ifft(X, axis=0)
    tx = np.concatenate([useful[-cp_samples:], useful], axis=0)
    # Leading receive-CP samples may belong to the prior symbol; they are
    # discarded. A zero-filled preceding symbol is sufficient for this check.
    receive_index = np.arange(cp_samples + K)
    source_index = receive_index - delay_samples
    valid = source_index >= 0
    delayed = np.zeros_like(tx)
    if float(delay_samples).is_integer():
        delayed[valid] = tx[source_index[valid].astype(int)]
    else:
        # Fourier interpolation from actual time samples of the transmitted CP
        # waveform, not from the analytic channel transfer function.
        spectrum_from_tx = np.fft.fft(tx[cp_samples:], axis=0)
        interpolation = np.exp(2j*np.pi*np.outer(source_index[valid]-cp_samples,
                                                np.arange(K))/K) / K
        delayed[valid] = interpolation @ spectrum_from_tx
    fast_time = (receive_index-cp_samples-K/2) / bandwidth_hz
    slow_time = (np.arange(M)-(M-1)/2)*reference_period_s
    received = delayed*np.exp(2j*np.pi*fd_hz*(fast_time[:, None]+slow_time))
    received_without_cp = received[cp_samples:]
    return np.fft.fft(received_without_cp, axis=0)


def check_time_reference():
    rng = np.random.default_rng(1749)
    K, M, B, Tr = 512, 8, 1e8, 61.44e-6
    X = np.exp(.5j*np.pi*rng.integers(0, 4, size=(K, M)))
    rows = []
    for delay in (67., 66.71281904):
        for ratio in (0., 1e-5, 1e-4, 1e-3, .01, .04):
            fd = -ratio * B/K
            reference = time_domain_reference(X, delay, fd)
            analytic = X*np.exp(-2j*np.pi*np.arange(K)[:, None]*delay/K)
            analytic *= np.exp(2j*np.pi*fd*(np.arange(M)-(M-1)/2)*Tr)
            err = np.linalg.norm(reference-analytic)/np.linalg.norm(analytic)
            rows.append(dict(delay_samples=delay, fd_hz=fd,
                             abs_fd_over_subcarrier_spacing=ratio,
                             relative_complex_l2_error=float(err)))
        errors = [r['relative_complex_l2_error'] for r in rows[-6:]]
        assert errors[0] < 1e-11, errors
        assert all(a < b for a, b in zip(errors, errors[1:])), errors
        assert errors[-2] < .02 and errors[-1] < .08, errors
    return dict(passed=True, K=K, M=M, cp_samples=256,
                fast_time_reference='(n-K/2)/B, useful-symbol centre',
                fractional_delay_convention='periodic waveform; carriers k=0..K-1',
                results=rows,
                interpretation='Exact numerical agreement only at zero Doppler. '
                'Nonzero Doppler causes intra-symbol phase and ICI omitted by the '
                'batch model; relative error converges to zero as |fd|/df tends to zero.')




def run_checks(core=None):
    """Run protocol sign, normalization, superposition, and noise checks."""
    if core is None:
        from frontend import ofdm_echo as core
    from dataclasses import replace
    result = {'time_domain_reference': check_time_reference()}
    w = core.DEFAULT_WAVEFORM
    # Match the actual core to the independent time-domain channel too.
    small = replace(w, M=8)
    rng = np.random.default_rng(1750)
    Xsmall = np.exp(.5j*np.pi*rng.integers(0, 4, (small.K, small.M)))
    cross = []
    for delay in (67., 66.71281904):
        r = delay*w.c/(2*w.B)
        x = np.sqrt(r*r-25.)
        for vr in (0., 10., -10.):
            state = [[x, 0., vr*r/x, 0.]]
            Y, _, _ = core.clean_echo(state, [0., 0.], 0., waveform=small,
                                      X=Xsmall, phases=[0.])
            td = time_domain_reference(Xsmall, delay, -2*vr/w.wavelength,
                                       bandwidth_hz=w.B,
                                       cp_samples=round(w.cp*w.B),
                                       reference_period_s=w.Tr)*(100/r)**2
            error = float(np.linalg.norm(Y[0]-td)/np.linalg.norm(td))
            assert error < (1e-11 if vr == 0 else .02), error
            cross.append(dict(delay_samples=delay, radial_velocity_mps=vr,
                              relative_complex_l2_error=error))
    result['core_vs_time_domain'] = dict(passed=True, cases=cross)

    # Choose independent physical states exactly on FFT bins.
    rbin, dbin, ubin = 67, -8, 2
    r = rbin*w.c/(2*w.B)
    u = 2*ubin/w.A
    y = u*r
    x = np.sqrt(r*r-25.-y*y)
    fd = dbin/(w.M*w.Tr)
    vr = -fd*w.wavelength/2
    state = np.array([[x, y, vr*r/x, 0.]])
    X = np.ones((w.K, w.M), complex)
    Y, _, audit = core.clean_echo(state, [0., 0.], 0., X=X, phases=[0.])
    observed = dict(range_bin=int(np.argmax(np.abs(np.fft.ifft(Y[0, :, 0])))),
                    doppler_bin=int(np.argmax(np.abs(np.fft.fft(Y[0, 0, :])))),
                    spatial_bin=int(np.argmax(np.abs(np.fft.fft(Y[:, 0, 0])))))
    assert observed == dict(range_bin=rbin, doppler_bin=dbin % w.M, spatial_bin=ubin), observed
    assert audit['radial_velocity_mps'][0] > 0 and audit['doppler_hz'][0] < 0
    # Zero-speed vehicles must have a nonzero DC Doppler peak.
    stopped = state.copy()
    stopped[:, 2:] = 0
    Y0, _, _ = core.clean_echo(stopped, [0., 0.], 0., X=X, phases=[0.])
    zero_bin = int(np.argmax(np.abs(np.fft.fft(Y0[0, 0, :]))))
    assert zero_bin == 0 and np.linalg.norm(Y0) > 0
    result['physical_signs'] = dict(passed=True, observed_fft_bins=observed,
                                    receding_velocity_mps=vr, expected_fd_hz=fd,
                                    expected_positive_u=u, zero_velocity_bin=zero_bin)

    second = np.array([[90., -20., -3., 2.]])
    states = np.concatenate([state, second])
    YA, _, _ = core.clean_echo(state, [0., 0.], 0., X=X, phases=[.3])
    YB, _, _ = core.clean_echo(second, [0., 0.], 0., X=X, phases=[-.7])
    both, _, _ = core.clean_echo(states, [0., 0.], 0., X=X, phases=[.3, -.7])
    superposition_error = float(np.max(np.abs(both-YA-YB)))
    assert superposition_error < 1e-12
    noisy, _, _ = core.echo_cube(states, [0., 0.], 0., X=X, phases=[.3, -.7], rng=77, dtype=np.complex128)
    noiseonly, _, _ = core.echo_cube(np.empty((0, 4)), [0., 0.], 0., X=X, phases=[], rng=77, dtype=np.complex128)
    noise_error = float(np.max(np.abs((noisy-both)-noiseonly)))
    assert noise_error < 1e-12
    cancel, _, _ = core.clean_echo(np.repeat(state, 2, axis=0), [0., 0.], 0., X=X, phases=[0., np.pi])
    assert np.linalg.norm(cancel)/np.linalg.norm(Y) < 1e-12
    result['complex_superposition_and_noise_after_sum'] = dict(passed=True,
                max_linearity_abs_error=superposition_error,
                max_common_noise_residual_error=noise_error,
                opposite_phase_targets_cancel=True)

    wa, wk, wm = core.processing_windows(w)
    weights = wa[:, None, None]*wk[None, :, None]*wm[None, None, :]
    p = core.noise_parameters(20., w)
    # Verify actual FFT amplitudes, not just a duplicated gain formula.
    transformed = np.fft.fft(np.fft.fft(np.fft.ifft(Y*weights, axis=1), axis=2), axis=0)
    measured_power = float(abs(transformed[ubin, rbin, dbin % w.M])**2)
    amplitude = (100/r)**2
    predicted_power = amplitude**2*p['matched_reference_output_power']
    assert abs(measured_power/predicted_power-1) < 1e-11
    del transformed

    # Direct full-dimensional CN fields sample the implemented noise generator.
    # Their matched projections have exponential power with wide finite-N error.
    # Independent 100000 low-dimensional draws characterize that distribution;
    # these are labelled distribution draws, not additional full-cube trials.
    calibration = []
    projection_results = []
    reference100, _, _ = core.clean_echo([[np.sqrt(100.**2-25.), 0., 0., 0.]],
                                         [0., 0.], 0., X=X, phases=[0.])
    refsignal = weights*np.conj(reference100)  # exact matched 100m steering
    matched100_power = float(abs(np.sum(reference100*refsignal)/w.K)**2)
    assert abs(matched100_power/p['matched_reference_output_power']-1) < 1e-11
    norm = w.K
    rng = np.random.default_rng(73411)
    for snr in (10., 15., 20., 25.):
        params = core.noise_parameters(snr, w)
        powers, raw_powers, means = [], [], []
        trials = 32
        for _ in range(trials):
            N, _, _ = core.echo_cube(np.empty((0, 4)), [0., 0.], 0., X=X,
                                     phases=[], rng=rng, snr_db=snr, dtype=np.complex128)
            powers.append(float(abs(np.sum(N*refsignal)/norm)**2))
            raw_powers.append(float(np.mean(np.abs(N)**2)))
            means.append(complex(np.mean(N)))
        variance_ratio = np.mean(raw_powers)/params['sigma2']
        projected_ratio = np.mean(powers)/params['matched_output_noise_variance']
        assert abs(variance_ratio-1) < .005, variance_ratio
        assert abs(np.mean(means))/np.sqrt(params['sigma2']) < .003
        assert .25 < projected_ratio < 2.5, projected_ratio
        measured_snr = 10*np.log10(params['matched_reference_output_power']/np.mean(powers))
        calibration.append(dict(snr_ref_db=snr, full_cube_trials=trials,
            full_cube_shape=[w.A, w.K, w.M], measured_raw_variance_ratio=float(variance_ratio),
            measured_matched_noise_variance_ratio=float(projected_ratio),
            full_cube_empirical_snr_db=float(measured_snr),
            matched_noise_power_relative_standard_error=1/np.sqrt(trials)))
        z=(rng.standard_normal(100000)+1j*rng.standard_normal(100000))*np.sqrt(params['matched_output_noise_variance']/2)
        projected_mc_snr = 10*np.log10(params['matched_reference_output_power']/np.mean(abs(z)**2))
        assert abs(projected_mc_snr-snr) < .07
        projection_results.append(dict(snr_ref_db=snr, projected_distribution_draws=100000,
            empirical_snr_db=float(projected_mc_snr)))
    result['normalization_and_snr'] = dict(passed=True,
        actual_fft_peak_power=measured_power, predicted_fft_peak_power=predicted_power,
        G_w=p['G_w'], reference_range_m=100.,
        reference100_actual_matched_signal_power=matched100_power,
        reference_qualification='100m is generally off the FFT grid; direct exact matched steering is used for SNR calibration. Separate on-grid FFT normalization is tested above.',
        full_cube_monte_carlo=calibration,
        projected_distribution_monte_carlo=projection_results,
        qualification='Full-cube trials test the actual generator and FFT/matched normalization. '
        '100000 scalar CN projections test the equivalent output distribution, '
        'not 100000 independent cube simulations. Small full-cube trial counts '
        'have the reported exponential-power sampling uncertainty.')
    result['passed'] = all(v.get('passed', False) for v in result.values())
    return result


if __name__ == '__main__':
    from pathlib import Path
    import sys, hashlib
    from .ofdm_echo import DEFAULT_WAVEFORM, noise_parameters
    root=Path(__file__).resolve().parents[1]
    out=root/'reports/f01b'
    out.mkdir(parents=True,exist_ok=True)
    result=run_checks()
    result['runtime']={'executable':sys.executable,'python':sys.version,'numpy':np.__version__}
    result['generator_sha256']={name:hashlib.sha256((root/'frontend'/name).read_bytes()).hexdigest() for name in ('ofdm_echo.py','echo_source.py','check_ofdm.py')}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    waveform={k:v for k,v in result.items() if k!='normalization_and_snr'}
    waveform['waveform']=DEFAULT_WAVEFORM.as_dict()
    (out/'waveform_checks.json').write_text(json.dumps(waveform,indent=2)+'\n')
    snr=dict(result['normalization_and_snr'],runtime=result['runtime'],parameters=[noise_parameters(s) for s in (10,15,20,25)])
    (out/'snr_calibration.json').write_text(json.dumps(snr,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'executable':sys.executable,'reports':['validation.json','waveform_checks.json','snr_calibration.json']}))
