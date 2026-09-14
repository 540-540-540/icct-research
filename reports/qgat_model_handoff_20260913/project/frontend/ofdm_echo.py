"""Frequency-domain, post-CP/FFT monostatic OFDM echo reference.

No detector or target identity is present here. Truth geometry is only used to
synthesize the received waveform and diagnostic metadata.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Waveform:
    fc: float = 28e9
    B: float = 100e6
    K: int = 512
    M: int = 256
    A: int = 16
    cp: float = 2.56e-6
    stride: int = 8
    dt: float = 0.1
    c: float = 299792458.

    def __post_init__(self):
        if not all(isinstance(n, (int, np.integer)) and not isinstance(n, bool)
                   for n in (self.K, self.M, self.A, self.stride)):
            raise ValueError('K/M/A/stride must be integers')
        if min(self.K, self.M, self.A) < 2 or self.stride < 1:
            raise ValueError('K/M/A >= 2 and stride >= 1 required')
        if not np.isfinite([self.fc, self.B, self.cp, self.dt, self.c]).all() or min(self.fc, self.B, self.cp, self.dt, self.c) <= 0:
            raise ValueError('Waveform physical parameters must be finite positive')
        if 3*self.CPI > self.dt:
            raise ValueError('Three non-overlapping CPIs must fit in one update')

    @property
    def df(self): return self.B/self.K
    @property
    def Tu(self): return 1/self.df
    @property
    def Ts(self): return self.Tu+self.cp
    @property
    def Tr(self): return self.stride*self.Ts
    @property
    def CPI(self): return self.M*self.Tr
    @property
    def wavelength(self): return self.c/self.fc

    def as_dict(self):
        return dict(asdict(self), df=self.df, Tu=self.Tu, Ts=self.Ts,
                    Tr=self.Tr, CPI=self.CPI, wavelength=self.wavelength,
                    range_resolution_m=self.c/(2*self.B),
                    range_alias_m=self.c/(2*self.df), range_cp_m=self.c*self.cp/2,
                    radial_velocity_resolution_mps=self.wavelength/(2*self.CPI),
                    radial_velocity_alias_mps=self.wavelength/(4*self.Tr),
                    cp_fraction=self.cp/self.Ts, reference_fraction=1/self.stride,
                    three_station_cpi_fraction=3*self.CPI/self.dt)


DEFAULT_WAVEFORM = Waveform()


def processing_windows(waveform=DEFAULT_WAVEFORM):
    """Periodic Hann for range/slow time, rectangular for physical ULA."""
    w = waveform
    return (np.ones(w.A), 0.5-0.5*np.cos(2*np.pi*np.arange(w.K)/w.K),
            0.5-0.5*np.cos(2*np.pi*np.arange(w.M)/w.M))


def noise_parameters(snr_db=20., waveform=DEFAULT_WAVEFORM):
    """CN variance in the post-FFT received-frequency cube (E|W|^2).

    G is invariant under an overall transform normalization. A detector using
    NumPy ifft(range), fft(slow), fft(spatial) has matched amplitude sum(w)/K,
    and output noise variance sigma2*sum(w^2)/K^2. Spatial zero-padding changes
    neither physical sample count nor G. Unit-modulus X removal preserves W.
    This is not an ADC-domain variance; a time-domain implementation must
    account for its FFT scaling (unnormalized K-FFT multiplies variance by K).
    """
    if not np.isfinite(snr_db):
        raise ValueError('snr_db must be finite')
    windows = processing_windows(waveform)
    weight_sum = float(np.prod([x.sum() for x in windows]))
    weight_energy = float(np.prod([np.dot(x, x) for x in windows]))
    gain = weight_sum**2/weight_energy
    sigma2 = gain/10.**(float(snr_db)/10.)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError('snr_db produces an unrepresentable noise variance')
    return dict(snr_ref_db=float(snr_db), alpha_ref=1., reference_range_m=100.,
                G_w=gain, weight_sum=weight_sum, weight_energy=weight_energy,
                sigma2=sigma2, variance_domain='post-CP/FFT frequency cube, E|W|^2',
                windows=dict(spatial='rectangular', range='periodic Hann', slow_time='periodic Hann'),
                transform_normalization='numpy ifft range (1/K), fft slow/spatial (unscaled)',
                matched_output_noise_variance=sigma2*weight_energy/waveform.K**2,
                matched_reference_output_power=(weight_sum/waveform.K)**2)


def _generator(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def clean_echo(states_xyvxvy, station_xy, boresight_rad, *,
               waveform=DEFAULT_WAVEFORM, X=None, phases=None, rng=None,
               per_target_amplitude_scale=None, dtype=np.complex128):
    """Return (clean Y[A,K,M], known unit-modulus X[K,M], truth audit).

    States are evaluated at the station CPI centre. phases are optional real
    radians, fixed within this CPI only; by default each call draws fresh
    phases. No persistent per-car scattering signature is assigned.
    """
    s, b = np.asarray(states_xyvxvy, float), np.asarray(station_xy, float)
    if s.ndim != 2 or s.shape[1] != 4 or len(s) > 8 or not np.isfinite(s).all():
        raise ValueError('Expected finite N x 4 states with 0 <= N <= 8')
    if b.shape != (2,) or not np.isfinite(b).all() or not np.isfinite(boresight_rad):
        raise ValueError('Expected finite station xy and boresight radians')
    if np.dtype(dtype) not in (np.dtype('complex64'), np.dtype('complex128')):
        raise ValueError('dtype must be complex64 or complex128')
    w, gen, n = waveform, _generator(rng), len(s)
    if X is None:
        X = np.exp(1j*(np.pi/4+np.pi/2*gen.integers(0, 4, (w.K, w.M))))
    X = np.asarray(X, np.complex128)
    if X.shape != (w.K, w.M) or not np.isfinite(X).all() or not np.allclose(np.abs(X), 1., rtol=1e-6, atol=1e-7):
        raise ValueError('X must be a finite unit-modulus K x M reference array')
    phase = gen.uniform(-np.pi, np.pi, n) if phases is None else np.asarray(phases, float)
    scale = np.ones(n) if per_target_amplitude_scale is None else np.asarray(per_target_amplitude_scale, float)
    if phase.shape != (n,) or not np.isfinite(phase).all():
        raise ValueError('phases must contain one finite radian value per target')
    if scale.shape != (n,) or not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError('amplitude scales must be finite nonnegative per-target values')
    d = s[:, :2]-b
    r = np.sqrt(np.sum(d*d, axis=1)+25.)
    theta = (np.arctan2(d[:, 1], d[:, 0])-boresight_rad+np.pi)%(2*np.pi)-np.pi
    vr = np.sum(d*s[:, 2:], axis=1)/r
    tau, fd = 2*r/w.c, -2*vr/w.wavelength
    u = np.sqrt(1-25./r**2)*np.sin(theta)
    visible = (r >= 10.) & (r <= 300.) & (np.abs(theta) <= np.deg2rad(70.))
    amplitude = visible*scale*(100./r)**2
    cube = np.zeros((w.A, w.K, w.M), np.complex128)
    a, k = np.arange(w.A), np.arange(w.K)
    slow_t = (np.arange(w.M)-(w.M-1)/2)*w.Tr
    for i in np.flatnonzero(amplitude):
        spatial = np.exp(1j*np.pi*a*u[i])
        delay = np.exp(-2j*np.pi*k*w.df*tau[i])
        doppler = np.exp(2j*np.pi*fd[i]*slow_t)
        cube += (amplitude[i]*np.exp(1j*phase[i])*spatial[:, None, None]
                 *delay[None, :, None]*doppler[None, None, :])
    cube *= X[None, :, :]
    audit = dict(waveform=w.as_dict(), station_xy=b.tolist(), boresight_rad=float(boresight_rad),
                 height_difference_m=5., half_angle_deg=70., range_limits_m=[10., 300.],
                 gain_model='flat unit gain within sector; equal baseline RCS',
                 target_count=n, visible_count=int(visible.sum()),
                 range_m=r.tolist(), radial_velocity_mps=vr.tolist(), theta_rad=theta.tolist(),
                 spatial_direction_cosine=u.tolist(), delay_s=tau.tolist(), doppler_hz=fd.tolist(),
                 visible=visible.tolist(), amplitude=amplitude.tolist(),
                 scattering_phase_rad=phase.tolist(), per_target_amplitude_scale=scale.tolist(),
                 max_ici_ratio=float(np.max(np.abs(fd)/w.df)) if n else 0.,
                 max_cpi_planar_displacement_m=float(np.max(np.linalg.norm(s[:, 2:], axis=1)*w.CPI)) if n else 0.,
                 approximation='frozen CPI-centre geometry, constant Doppler, narrowband ULA, no ICI',
                 output_dtype=np.dtype(dtype).name,
                 audit_use='simulation diagnostics only; never observation or predictor input')
    return cube.astype(dtype, copy=False), X.astype(dtype, copy=False), audit


def echo_cube(states_xyvxvy, station_xy, boresight_rad, *, snr_db=20.,
              waveform=DEFAULT_WAVEFORM, rng=None, X=None, phases=None,
              per_target_amplitude_scale=None, dtype=np.complex64):
    """Sum all complex target echoes, then add one shared circular CN field."""
    if np.dtype(dtype) not in (np.dtype('complex64'), np.dtype('complex128')):
        raise ValueError('dtype must be complex64 or complex128')
    gen = _generator(rng)
    cube, X, audit = clean_echo(states_xyvxvy, station_xy, boresight_rad,
                               waveform=waveform, rng=gen, X=X, phases=phases,
                               per_target_amplitude_scale=per_target_amplitude_scale)
    noise = noise_parameters(snr_db, waveform)
    standard_deviation = np.sqrt(noise['sigma2']/2)
    cube += standard_deviation*(gen.standard_normal(cube.shape)+1j*gen.standard_normal(cube.shape))
    audit.update(noise=noise, output_dtype=np.dtype(dtype).name)
    return cube.astype(dtype, copy=False), X.astype(dtype, copy=False), audit
