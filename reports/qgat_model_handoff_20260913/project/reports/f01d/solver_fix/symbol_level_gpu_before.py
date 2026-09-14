"""CUDA-only batched Wei2024 symbol sensing; known separated targets.

Torch CUDA RNG is explicitly seeded per target and differs from NumPy RNG.
No CPU numerical fallback, FFT, least-squares solver, or optimization is used.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class PaperWaveform:
    fc: float = 24e9
    B: float = 93.1e6
    K: int = 256
    N: int = 256
    T: float = 12.375e-6
    c: float = 299792458.

    def __post_init__(self):
        if self.K < 2 or self.N < 2 or int(self.K) != self.K or int(self.N) != self.N or any(not math.isfinite(v) or v <= 0 for v in (self.fc, self.B, self.T, self.c)):
            raise ValueError("Invalid waveform")

    @property
    def df(self): return self.B / self.K
    @property
    def unambiguous_range(self): return self.c / (2 * self.df)
    @property
    def unambiguous_velocity(self): return self.c / (4 * self.fc * self.T)
    @property
    def range_resolution(self): return self.c / (2 * self.B)
    @property
    def velocity_resolution(self): return self.c / (2 * self.fc * self.N * self.T)


def _waveform(w):
    return PaperWaveform(**w) if isinstance(w, dict) else (w or PaperWaveform())


def _real(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def _cuda(tensor):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise ValueError("A CUDA tensor is required; CPU numerical fallback is disabled")


@torch.no_grad()
def synthesize_batch(position, velocity, stations, waveform=None, snr_db=20., seeds=None,
                     device="cuda:0", phases=None, noise=True, dtype=torch.float64):
    """Eq4 on CUDA. Explicit per-target seeds make batch partitioning irrelevant."""
    w = _waveform(waveform)
    if dtype not in (torch.float64, torch.float32):
        raise ValueError("dtype must be float64 or float32")
    p = _real(position, device, dtype)
    _cuda(p)
    v = _real(velocity, p.device, dtype)
    stations = _real(stations, p.device, dtype)
    if p.ndim != 2 or p.shape[1] != 2 or v.shape != p.shape or stations.shape != (3, 2):
        raise ValueError("Expected position/velocity [batch,2] and stations [3,2]")
    if not torch.isfinite(p).all() or not torch.isfinite(v).all() or not torch.isfinite(stations).all() or not math.isfinite(float(snr_db)):
        raise ValueError("Nonfinite simulation input")
    batch = len(p)
    if not batch or seeds is None or len(seeds) != batch:
        raise ValueError("One explicit integer seed per nonempty target batch is required")
    supplied_phases = None if phases is None else _real(phases, p.device, dtype)
    if supplied_phases is not None and (supplied_phases.shape != (batch, 3) or not torch.isfinite(supplied_phases).all()):
        raise ValueError("Scattering phases must be finite [batch,3]")
    delta = stations[None] - p[:, None]
    radii = torch.linalg.vector_norm(delta, dim=-1)
    if torch.any(radii <= 0):
        raise ValueError("Target coincides with station")
    radial = (delta * v[:, None]).sum(-1) / radii
    xs, phase_rows, noises = [], [], []
    for i, seed in enumerate(seeds):
        generator = torch.Generator(device=p.device).manual_seed(int(seed))
        q = torch.randint(0, 4, (3, w.K, w.N), generator=generator, device=p.device)
        xs.append(torch.polar(torch.ones_like(q, dtype=dtype), math.pi/4 + math.pi/2*q.to(dtype)))
        phase_rows.append((torch.rand(3, device=p.device, dtype=dtype, generator=generator)*2-1)*math.pi
                          if supplied_phases is None else supplied_phases[i])
        if noise:
            real = torch.randn((3, w.K, w.N), device=p.device, dtype=dtype, generator=generator)
            imag = torch.randn((3, w.K, w.N), device=p.device, dtype=dtype, generator=generator)
            noises.append(torch.complex(real, imag) / math.sqrt(2))
    X = torch.stack(xs)
    phases_t = torch.stack(phase_rows)
    k = torch.arange(w.K, device=p.device, dtype=dtype)
    n = torch.arange(w.N, device=p.device, dtype=dtype)
    delay = -4*math.pi*w.df/w.c*radii[:, :, None]*k
    doppler = 4*math.pi*w.fc*w.T/w.c*radial[:, :, None]*n
    channel = torch.polar(torch.ones_like(delay), delay)[:, :, :, None] * torch.polar(torch.ones_like(doppler), doppler)[:, :, None, :]
    clean = 10**(float(snr_db)/20) * torch.polar(torch.ones_like(phases_t), phases_t)[:, :, None, None] * channel * X
    return dict(Y=clean + torch.stack(noises) if noise else clean, X=X)


@torch.no_grad()
def divide_symbols(Y, X):
    _cuda(Y)
    _cuda(X)
    if Y.shape != X.shape or Y.device != X.device or not torch.isfinite(Y).all() or not torch.isfinite(X).all() or torch.any(X.abs() == 0):
        raise ValueError("Finite aligned CUDA symbols and nonzero X are required")
    return Y / X


divide = divide_symbols


@torch.no_grad()
def lag_correlation(vector):
    """Eq22/32: mean v[a] conj(v[a+lag]); arbitrary CUDA batch dimensions."""
    _cuda(vector)
    length = vector.shape[-1]
    if length < 2 or not torch.isfinite(vector).all():
        raise ValueError("Expected finite vectors of length >=2")
    nfft = 1 << (2*length-2).bit_length()
    spectrum = torch.fft.fft(vector, n=nfft, dim=-1)
    correlation = torch.fft.ifft(spectrum.abs().square(), dim=-1)[..., 1:length].conj()
    return correlation / torch.arange(length-1, 0, -1, device=vector.device, dtype=vector.real.dtype)


def _bounds(value, B, name):
    value = _real(value, B.device, B.real.dtype)
    if value.shape != (2, 2) or not torch.isfinite(value).all() or torch.any(value[:, 0] >= value[:, 1]):
        raise ValueError(name + " must be finite [[min,max],[min,max]]")
    return value


def _coarse(B, w, min_ranges, max_ranges, max_speed, oversample):
    batch = B.shape[0]
    device, dtype = B.device, B.real.dtype
    nr, nv = w.K*oversample, w.N*oversample
    spectrum = torch.fft.fft(torch.fft.ifft(B, n=nr, dim=-2), n=nv, dim=-1)
    ranges = torch.arange(nr, device=device, dtype=dtype)*w.unambiguous_range/nr
    velocities = torch.fft.fftfreq(nv, d=w.T, device=device, dtype=dtype)*w.c/(2*w.fc)
    range_ok = (ranges[None] >= min_ranges[:, None]) & (ranges[None] <= max_ranges[:, None])
    velocity_ok = (velocities >= -max_speed) & (velocities <= max_speed)
    if not range_ok.any(-1).all() or not velocity_ok.any():
        raise ValueError("Configured bounds contain no coarse FFT cells")
    allowed = range_ok[:, :, None] & velocity_ok[None, None]
    index = spectrum.abs().masked_fill(~allowed[None], -torch.inf).flatten(-2).argmax(-1)
    r0, v0 = ranges[index//nv], velocities[index%nv]
    dr, dv = w.range_resolution/oversample, w.velocity_resolution/oversample
    rl, ru = torch.maximum(min_ranges[None], r0-dr), torch.minimum(max_ranges[None], r0+dr)
    vl, vu = torch.maximum(-max_speed, v0-dv), torch.minimum(max_speed, v0+dv)
    fraction = torch.linspace(0, 1, 9, device=device, dtype=dtype)
    rg, vg = rl[..., None] + (ru-rl)[..., None]*fraction, vl[..., None] + (vu-vl)[..., None]*fraction
    k = torch.arange(w.K, device=device, dtype=dtype)
    n = torch.arange(w.N, device=device, dtype=dtype)
    A = torch.exp(1j*4*math.pi*w.df/w.c*rg[..., :, None]*k)
    C = torch.exp(-1j*4*math.pi*w.fc*w.T/w.c*n[:, None]*vg[..., None, :])
    scores = (A @ B @ C).abs()
    chosen = scores.flatten(-2).argmax(-1)
    ri, vi = chosen//9, chosen%9
    r = rg.gather(-1, ri[..., None]).squeeze(-1)
    v = vg.gather(-1, vi[..., None]).squeeze(-1)
    diag = dict(fft_cells=range_ok.sum(-1)*velocity_ok.sum(), range_local_edge=(ri == 0) | (ri == 8), velocity_local_edge=(vi == 0) | (vi == 8), fft_transform_shape=[nr, nv])
    return torch.stack([r, v], -1), diag


@torch.no_grad()
def bounded_range_least_squares(ranges, stations, initial, bounds, max_iterations=40):
    """Projected damped Gauss-Newton for the same bounded sum of range residual squares.

    Per-target damping and accepted-step convergence; active-bound projected gradient
    is the stationarity test. Every numerical operation stays on CUDA.
    """
    _cuda(ranges)
    p = initial.clone()
    batch = len(p)
    dtype, device = p.dtype, p.device
    eye = torch.eye(2, device=device, dtype=dtype)[None]
    damping = torch.full((batch,), 1e-4, device=device, dtype=dtype)
    converged = torch.zeros(batch, device=device, dtype=torch.bool)
    iterations = torch.zeros(batch, device=device, dtype=torch.int64)
    tolerance = 1e-10 if dtype == torch.float64 else 2e-5
    for iteration in range(max_iterations):
        delta = p[:, None] - stations[None]
        radius = torch.linalg.vector_norm(delta, dim=-1).clamp_min(torch.finfo(dtype).eps)
        residual = radius - ranges
        J = delta / radius[..., None]
        gradient = (J * residual[..., None]).sum(1)
        projected = p - torch.clamp(p-gradient, min=bounds[:, 0], max=bounds[:, 1])
        converged |= projected.abs().amax(-1) <= tolerance
        active = ~converged
        if not active.any():
            break
        iterations += active
        hessian = J.transpose(-1, -2) @ J
        step = torch.linalg.solve(hessian + damping[:, None, None]*eye, -gradient[..., None]).squeeze(-1)
        candidate = torch.clamp(p+step, min=bounds[:, 0], max=bounds[:, 1])
        cost = residual.square().sum(-1)
        newcost = (torch.linalg.vector_norm(candidate[:, None]-stations[None], dim=-1)-ranges).square().sum(-1)
        accepted = (newcost < cost) & active
        change = (candidate-p).abs().amax(-1)
        # Tiny accepted progress is a convergence criterion, never a failed trial.
        converged |= accepted & ((change <= tolerance*(1+p.abs().amax(-1))) | ((cost-newcost).abs() <= tolerance*tolerance*(1+cost)))
        p = torch.where(accepted[:, None], candidate, p)
        damping = torch.where(accepted, (damping/3).clamp_min(1e-12), (damping*10).clamp_max(1e12))
    delta = p[:, None]-stations[None]
    radius = torch.linalg.vector_norm(delta, dim=-1).clamp_min(torch.finfo(dtype).eps)
    residual = radius-ranges
    gradient = ((delta/radius[..., None])*residual[..., None]).sum(1)
    projected = p-torch.clamp(p-gradient, min=bounds[:, 0], max=bounds[:, 1])
    converged |= projected.abs().amax(-1) <= tolerance*10
    return p, dict(success=converged, iterations=iterations, objective=residual.square().sum(-1), projected_gradient_inf=projected.abs().amax(-1))


def _local_grid(center, bounds, width, step):
    lower = torch.maximum(bounds[:, 0], center-width)
    upper = torch.minimum(bounds[:, 1], center+width)
    counts = torch.ceil((upper-lower)/step).to(torch.int64).add(1).clamp_min(2)
    nx, ny = int(counts[:, 0].max().item()), int(counts[:, 1].max().item())
    ix = torch.arange(nx, device=center.device)
    iy = torch.arange(ny, device=center.device)
    x = lower[:, 0, None] + (upper-lower)[:, 0, None]*ix/(counts[:, 0, None]-1)
    y = lower[:, 1, None] + (upper-lower)[:, 1, None]*iy/(counts[:, 1, None]-1)
    candidates = torch.stack([x[:, :, None].expand(-1, -1, ny), y[:, None, :].expand(-1, nx, -1)], -1).flatten(1, 2)
    valid = ((ix[None, :, None] < counts[:, 0, None, None]) & (iy[None, None, :] < counts[:, 1, None, None])).flatten(1)
    return candidates, valid, counts, torch.stack([lower, upper], -1), ny


@torch.no_grad()
def fused_score(parameters, features, phase_factor, chunk_size=128):
    """Eq26/37: parameters [B,C,3], features [B,3,L], result [B,C]."""
    _cuda(parameters)
    _cuda(features)
    scale = features.abs().amax(-1).clamp_min(torch.finfo(features.real.dtype).tiny)
    features = features / scale[..., None]
    lags = torch.arange(1, features.shape[-1]+1, device=features.device, dtype=features.real.dtype)
    scores = []
    for start in range(0, parameters.shape[1], chunk_size):
        values = parameters[:, start:start+chunk_size]
        product = torch.ones((*values.shape[:2], len(lags)), device=features.device, dtype=features.real.dtype)
        for station in range(3):
            phase = phase_factor*values[..., station, None]*lags
            product *= features[:, station, None].real*torch.cos(phase) - features[:, station, None].imag*torch.sin(phase)
        scores.append(product.sum(-1))
    return torch.cat(scores, dim=-1)


def _grid_search(center, bounds, width, step, fine_step, score):
    def choose(c, width_, step_):
        candidates, valid, counts, local_bounds, ny = _local_grid(c, bounds, width_, step_)
        scores = score(candidates).masked_fill(~valid, -torch.inf)
        chosen = scores.argmax(-1)
        selected = candidates[torch.arange(len(c), device=c.device), chosen]
        ix, iy = chosen//ny, chosen%ny
        edge = (ix == 0) | (ix == counts[:, 0]-1) | (iy == 0) | (iy == counts[:, 1]-1)
        return selected, counts.prod(-1), local_bounds, edge, scores.gather(1, chosen[:, None]).squeeze(1)
    chosen, count, local_bounds, edge, _ = choose(center, width, step)
    result, finecount, finebounds, fineedge, scorevalue = choose(chosen, step, fine_step)
    return result, dict(coarse_candidates=count, fine_candidates=finecount, coarse_local_edge=edge,
                        fine_local_edge=fineedge, coarse_bounds=local_bounds, fine_bounds=finebounds, score=scorevalue)


@torch.no_grad()
def estimate_batch(B, stations, xy_bounds, v_bounds, waveform=None, search=None):
    """Estimate CUDA [B,3,K,N] symbols without source state or target identity."""
    _cuda(B)
    w = _waveform(waveform)
    if B.dtype not in (torch.complex64, torch.complex128) or B.ndim != 4 or B.shape[1:] != (3, w.K, w.N) or not len(B) or not torch.isfinite(B).all():
        raise ValueError("Expected finite nonempty complex CUDA [batch,3,K,N]")
    stations = _real(stations, B.device, B.real.dtype)
    if stations.shape != (3, 2) or not torch.isfinite(stations).all():
        raise ValueError("Exactly three finite planar stations required")
    xy_bounds, v_bounds = _bounds(xy_bounds, B, "xy_bounds"), _bounds(v_bounds, B, "v_bounds")
    cfg = dict(oversample=2, position_half_width=3., position_step=.25, position_fine_step=.025,
               velocity_half_width=3., velocity_step=.25, velocity_fine_step=.025)
    if search:
        if set(search)-set(cfg):
            raise ValueError("Unknown search setting")
        cfg.update(search)
    if any(not math.isfinite(float(v)) or v <= 0 for v in cfg.values()) or int(cfg["oversample"]) != cfg["oversample"]:
        raise ValueError("Invalid search configuration")
    corners = torch.cartesian_prod(xy_bounds[0], xy_bounds[1])
    max_ranges = torch.linalg.vector_norm(corners[:, None]-stations[None], dim=-1).amax(0)
    nearest = torch.clamp(stations, min=xy_bounds[:, 0], max=xy_bounds[:, 1])
    min_ranges = torch.linalg.vector_norm(nearest-stations, dim=-1)
    max_speed = torch.linalg.vector_norm(v_bounds.abs().amax(-1))
    if torch.any(max_ranges >= w.unambiguous_range) or max_speed >= w.unambiguous_velocity:
        raise ValueError("Configured geometry/velocity exceeds waveform ambiguity domain")
    coarse, cdiag = _coarse(B, w, min_ranges, max_ranges, max_speed, int(cfg["oversample"]))
    k = torch.arange(w.K, device=B.device, dtype=B.real.dtype)
    n = torch.arange(w.N, device=B.device, dtype=B.real.dtype)
    C = torch.exp(-1j*4*math.pi*w.fc*w.T/w.c*coarse[..., 1, None]*n)
    A = torch.exp(1j*4*math.pi*w.df/w.c*coarse[..., 0, None]*k)
    E = (B @ C[..., None]).squeeze(-1)
    F = (A[..., None, :] @ B).squeeze(-2)
    G, I = lag_correlation(E), lag_correlation(F)
    matrix = 2*(stations[1:]-stations[0])
    if torch.linalg.matrix_rank(matrix) < 2:
        raise ValueError("Three noncollinear stations required")
    rhs = coarse[:, :1, 0].square()-coarse[:, 1:, 0].square()+stations[1:].square().sum(-1)-stations[0].square().sum()
    initial = torch.linalg.solve(matrix, rhs.T).T.clamp(min=xy_bounds[:, 0], max=xy_bounds[:, 1])
    rough, optimizer = bounded_range_least_squares(coarse[..., 0], stations, initial, xy_bounds)
    position, pdiag = _grid_search(rough, xy_bounds, cfg["position_half_width"], cfg["position_step"], cfg["position_fine_step"],
        lambda p: fused_score(torch.linalg.vector_norm(p[:, :, None]-stations[None, None], dim=-1), G, -4*math.pi*w.df/w.c))
    delta = stations[None]-position[:, None]
    radii = torch.linalg.vector_norm(delta, dim=-1)
    if torch.any(radii <= 1e-9):
        raise ValueError("Estimated position coincides with station")
    U = delta/radii[..., None]
    velocity_initial = torch.linalg.lstsq(U, coarse[..., 1, None]).solution.squeeze(-1).clamp(min=v_bounds[:, 0], max=v_bounds[:, 1])
    velocity, vdiag = _grid_search(velocity_initial, v_bounds, cfg["velocity_half_width"], cfg["velocity_step"], cfg["velocity_fine_step"],
        lambda v: fused_score(v @ U.transpose(-1, -2), I, 4*math.pi*w.fc*w.T/w.c))
    failure = ~optimizer["success"] | pdiag["coarse_local_edge"] | pdiag["fine_local_edge"] | vdiag["coarse_local_edge"] | vdiag["fine_local_edge"]
    # Only final diagnostic serialization transfers values to the CPU.
    op = {key: value.tolist() for key, value in optimizer.items()}
    pd = {key: value.tolist() for key, value in pdiag.items()}
    vd = {key: value.tolist() for key, value in vdiag.items()}
    cd = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in cdiag.items()}
    failures = failure.tolist()
    condition = float(torch.linalg.cond(matrix).item())
    vcondition = torch.linalg.cond(U).tolist()
    diagnostics = []
    xy_list, v_list = xy_bounds.tolist(), v_bounds.tolist()
    for i in range(len(B)):
        single = [dict(fft_cells=cd["fft_cells"][j], fft_transform_shape=cd["fft_transform_shape"], local_steering_cells=81,
                       range_local_edge=cd["range_local_edge"][i][j], velocity_local_edge=cd["velocity_local_edge"][i][j]) for j in range(3)]
        diagnostics.append(dict(search_failure=failures[i], localization_optimizer_success=op["success"][i],
            localization_optimizer={key: value[i] for key, value in op.items()},
            position_search={key: value[i] for key, value in pd.items()}, velocity_search={key: value[i] for key, value in vd.items()},
            single_station=single, rough_geometry_condition=condition, velocity_geometry_condition=vcondition[i],
            unambiguous_range_m=w.unambiguous_range, unambiguous_velocity_mps=w.unambiguous_velocity,
            configured_xy_bounds=xy_list, configured_velocity_bounds=v_list,
            radial_velocity_convention="positive toward station; U=(station-estimated_position)/estimated_range",
            backend="torch CUDA", dtype=str(B.dtype), optimizer="projected damped Gauss-Newton; same bounded range residual objective",
            scope="Three BSs; pre-separated target and known cross-station association; no detection/tracking claim"))
    return dict(state_hat=torch.cat([position, velocity], -1),
                coarse=dict(range_radial_velocity=coarse, position=rough, velocity=velocity_initial),
                symbol_features=dict(E=E, F=F, G=G, I=I), diagnostics=diagnostics)


if __name__ == "__main__":
    stations = [[0., 0.], [80., 0.], [0., 80.]]
    echo = synthesize_batch([[30., 35.]], [[3., -2.]], stations, seeds=[2026], noise=False)
    result = estimate_batch(divide_symbols(**echo), stations, [[10., 60.], [10., 60.]], [[-10., 10.], [-10., 10.]])
    expected = torch.tensor([[30., 35., 3., -2.]], device="cuda", dtype=torch.float64)
    assert torch.linalg.vector_norm(result["state_hat"][:, :2]-expected[:, :2]) < .06
    assert torch.linalg.vector_norm(result["state_hat"][:, 2:]-expected[:, 2:]) < .06
    print(dict(state_hat=result["state_hat"].tolist(), diagnostics=result["diagnostics"]))
