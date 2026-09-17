"""2D Range-Doppler CA-CFAR, NMS and peak-wise AoA detector (B domain).

Frozen design: SYSTEM_MODEL.md sections 4-5, DECISIONS.md D07/D08/D23.
This module must never receive target lists, identities or truth states.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as TF

from . import coords
from .waveform import ArrayConfig, PaperWaveform, SensingResource, periodic_hann

DETECTION_FIELDS = (
    "station_id", "time_ns", "r_m", "vr_mps", "u", "bearing_rad", "x_m", "y_m", "C_xy",
    "peak_power", "noise_floor", "peak_to_noise_db", "cfar_score", "grid_index",
)


def _integral(x: torch.Tensor) -> torch.Tensor:
    c = torch.cumsum(torch.cumsum(x, dim=0), dim=1)
    return TF.pad(c, (1, 0, 1, 0))


def box_sum(x: torch.Tensor, half_h: int, half_w: int) -> torch.Tensor:
    """Edge-adaptive rectangular sum via an integral image (clipped window)."""
    height, width = x.shape
    c = _integral(x)
    i = torch.arange(height, device=x.device)
    j = torch.arange(width, device=x.device)
    i0 = (i - half_h).clamp_min(0)
    i1 = (i + half_h).clamp_max(height - 1)
    j0 = (j - half_w).clamp_min(0)
    j1 = (j + half_w).clamp_max(width - 1)
    return c[i1 + 1][:, j1 + 1] - c[i0][:, j1 + 1] - c[i1 + 1][:, j0] + c[i0][:, j0]


@torch.no_grad()
def rd_spectrum(Y_b: torch.Tensor, X_b: torch.Tensor, waveform: PaperWaveform, array: ArrayConfig,
                resource: SensingResource | None = None) -> dict:
    """2D Range-Doppler spectrum over the active sensing resource (B domain, no CFAR).

    ``resource=None`` keeps the full-grid oracle path (all N symbols). A ``contiguous_burst``
    resource integrates only the active consecutive symbols; the range grid stays at nr=2K and
    the velocity grid becomes nv=2*active_symbols with the slow-time spacing still T, so the
    unambiguous velocity is unchanged.
    """
    if Y_b.ndim != 3 or Y_b.shape != (array.elements, waveform.K, waveform.N):
        raise ValueError("Y_b must be [A,K,N]")
    if X_b.shape != (waveform.K, waveform.N):
        raise ValueError("X_b must be [K,N]")
    if not torch.isfinite(Y_b).all() or not torch.isfinite(X_b).all() or torch.any(X_b.abs() == 0):
        raise ValueError("Finite aligned Y/X symbols are required")
    resource = resource if resource is not None else SensingResource()
    if resource.total_symbols != waveform.N:
        raise ValueError("sensing resource total_symbols must match the waveform block N")
    samples = waveform.N if resource.mode == "full" else resource.active_symbols
    device, dtype = Y_b.device, Y_b.real.dtype
    nr, nv = 2 * waveform.K, 2 * samples
    range_window = periodic_hann(waveform.K, device, dtype)[None, :, None]
    doppler_window = periodic_hann(samples, device, dtype)[None, None, :]
    z = Y_b / X_b[None]
    if resource.mode != "full":
        z = z[:, :, resource.start_symbol:resource.start_symbol + resource.active_symbols]
    spectrum = torch.fft.ifft(z * range_window, n=nr, dim=1)
    spectrum = torch.fft.fftshift(torch.fft.fft(spectrum * doppler_window, n=nv, dim=2), dim=2)
    power = spectrum.abs().square().sum(dim=0)
    ranges = torch.arange(nr, device=device, dtype=dtype) * (waveform.unambiguous_range / nr)
    velocities = torch.fft.fftshift(torch.fft.fftfreq(nv, d=waveform.T, device=device, dtype=dtype))
    velocities = velocities * (waveform.c / (2 * waveform.fc))
    return {"P_RD": power, "spectrum": spectrum, "ranges": ranges, "velocities": velocities,
            "nr": nr, "nv": nv, "samples": samples}


@torch.no_grad()
def compute_maps(Y_b: torch.Tensor, X_b: torch.Tensor, waveform: PaperWaveform, array: ArrayConfig,
                 detector: dict, resource: SensingResource | None = None) -> dict:
    """Legacy CFAR path: RD spectrum plus the sliding-window noise/alpha fields."""
    maps = rd_spectrum(Y_b, X_b, waveform, array, resource=resource)

    train_r, train_v = (int(v) for v in detector["cfar_train"])
    guard_r, guard_v = (int(v) for v in detector["cfar_guard"])
    if not (0 <= guard_r < train_r and 0 <= guard_v < train_v):
        raise ValueError("CFAR guard must be strictly inside the training window")
    power = maps["P_RD"]
    ring_sum = box_sum(power, train_r, train_v) - box_sum(power, guard_r, guard_v)
    one = torch.ones_like(power)
    ring_count = box_sum(one, train_r, train_v) - box_sum(one, guard_r, guard_v)
    ring_count = ring_count.clamp_min(1.0)
    noise = ring_sum / ring_count
    target_false_alarms = float(detector["target_false_alarms_per_bs_frame"])
    if not 0 < target_false_alarms < maps["nr"] * maps["nv"]:
        raise ValueError("Invalid target false alarm budget")
    pfa_cell = target_false_alarms / (maps["nr"] * maps["nv"])
    alpha = ring_count * (pfa_cell ** (-1.0 / ring_count) - 1.0)

    maps.update(noise=noise, alpha=alpha, ring_count=ring_count)
    return maps


@torch.no_grad()
def cfar_mask(maps: dict, multiplier: float) -> torch.Tensor:
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("CFAR multiplier must be finite and positive")
    return maps["P_RD"] > maps["noise"] * maps["alpha"] * multiplier


@torch.no_grad()
def local_maxima(mask: torch.Tensor, power: torch.Tensor, radius: tuple[int, int]) -> torch.Tensor:
    kernel = (2 * int(radius[0]) + 1, 2 * int(radius[1]) + 1)
    pooled = TF.max_pool2d(power[None, None], kernel_size=kernel, stride=1,
                           padding=(int(radius[0]), int(radius[1])))[0, 0]
    return mask & (power >= pooled)


def _parabolic(log_power: torch.Tensor, index: tuple[int, int], axis: int) -> float:
    i = index[axis]
    if i <= 0 or i >= log_power.shape[axis] - 1:
        return 0.0
    left, right = list(index), list(index)
    left[axis] -= 1
    right[axis] += 1
    lm = float(log_power[tuple(left)])
    lc = float(log_power[tuple(index)])
    lp = float(log_power[tuple(right)])
    denom = lm - 2 * lc + lp
    if denom >= 0:
        return 0.0
    return float(min(max(0.5 * (lm - lp) / denom, -0.5), 0.5))


def _nms_1d(values: torch.Tensor, radius: int) -> list[int]:
    length = values.numel()
    if length == 0:
        return []
    pooled = TF.max_pool1d(values.reshape(1, 1, -1), kernel_size=2 * radius + 1, stride=1,
                           padding=radius)[0, 0]
    peaks = [int(i) for i in torch.nonzero(values >= pooled).flatten().tolist()]
    return peaks or [int(torch.argmax(values).item())]


@torch.no_grad()
def aoa_estimate(snapshot: torch.Tensor, array: ArrayConfig) -> tuple[float, int]:
    spectrum = torch.fft.fftshift(torch.fft.fft(snapshot, n=array.fft_size))
    power = spectrum.abs().square()
    candidates = _nms_1d(power, array.aoa_nms_radius)
    best = max(candidates, key=lambda i: float(power[i]))
    log_power = torch.log(power.clamp_min(torch.finfo(power.dtype).tiny))
    left = log_power[(best - 1) % array.fft_size] if best > 0 else log_power[best]
    centre = log_power[best]
    right = log_power[(best + 1) % array.fft_size] if best < array.fft_size - 1 else log_power[best]
    denom = float(left - 2 * centre + right)
    fraction = 0.0
    if denom < 0:
        fraction = float(min(max(0.5 * float(left - right) / denom, -0.5), 0.5))
    step = 2.0 / array.fft_size
    u_grid = 2.0 * torch.fft.fftshift(torch.fft.fftfreq(array.fft_size))
    u_hat = float(u_grid[best]) + fraction * step
    return float(min(max(u_hat, -1.0), 1.0)), best


@torch.no_grad()
def detect_from_maps(maps: dict, station_id: int, time_ns: int, station, boresight: float,
                     config: dict, array: ArrayConfig, multiplier: float | None = None,
                     covariance_lut: dict | None = None, height: float = 5.0) -> tuple[list[dict], dict, list]:
    detector = config["detector"]
    if multiplier is None:
        multiplier = detector.get("cfar_threshold_multiplier") or 1.0
    mask = cfar_mask(maps, float(multiplier))
    power = maps["P_RD"]
    radius = (int(detector["nms_radius"][0]), int(detector["nms_radius"][1]))
    candidates = local_maxima(mask, power, radius)
    indices = torch.nonzero(candidates).tolist()
    log_power = torch.log(power.clamp_min(torch.finfo(power.dtype).tiny))
    order = sorted(range(len(indices)), key=lambda n: (-float(power[tuple(indices[n])]), indices[n]))
    suppressed = torch.zeros_like(mask)
    d_r = float(maps["ranges"][1] - maps["ranges"][0])
    d_v = float(maps["velocities"][1] - maps["velocities"][0])
    max_candidates = int(detector["max_candidates"])
    hh, hw = radius
    detections, snapshots = [], []
    rejected_interpolation = 0
    suppressed_count = 0
    candidates_before_cap = 0
    overflow_count = 0
    for position in order:
        index = indices[position]
        if suppressed[index[0], index[1]]:
            suppressed_count += 1
            continue
        r0, r1 = max(0, index[0] - hh), min(maps["nr"] - 1, index[0] + hh)
        v0, v1 = max(0, index[1] - hw), min(maps["nv"] - 1, index[1] + hw)
        suppressed[r0:r1 + 1, v0:v1 + 1] = True
        fraction_r = _parabolic(log_power, (index[0], index[1]), 0)
        fraction_v = _parabolic(log_power, (index[0], index[1]), 1)
        r_hat = float(maps["ranges"][index[0]]) + fraction_r * d_r
        v_hat = float(maps["velocities"][index[1]]) + fraction_v * d_v
        snapshot = maps["spectrum"][:, index[0], index[1]]
        u_hat, u_bin = aoa_estimate(snapshot, array)
        bearing = math.asin(u_hat)
        if r_hat <= height or not (10.0 <= r_hat <= 300.0) or abs(bearing) > math.radians(70.0):
            rejected_interpolation += 1
            continue
        candidates_before_cap += 1
        if len(detections) >= max_candidates:
            overflow_count += 1
            continue
        x_m, y_m = coords.polar_to_cartesian(r_hat, u_hat, station, boresight, height)
        peak_power = float(power[index[0], index[1]])
        noise_floor = float(maps["noise"][index[0], index[1]])
        peak_to_noise_db = 10 * math.log10(peak_power / max(noise_floor, 1e-300))
        covariance = None
        if covariance_lut is not None:
            covariance = coords.covariance_from_lut(r_hat, u_hat, peak_to_noise_db, boresight,
                                                    covariance_lut, detector.get("covariance_floor_m2"))
        detection = {
            "station_id": int(station_id),
            "time_ns": int(time_ns),
            "r_m": r_hat,
            "vr_mps": v_hat,
            "u": u_hat,
            "bearing_rad": bearing,
            "x_m": float(x_m),
            "y_m": float(y_m),
            "C_xy": covariance.tolist() if covariance is not None else None,
            "peak_power": peak_power,
            "noise_floor": noise_floor,
            "peak_to_noise_db": float(peak_to_noise_db),
            "cfar_score": float(peak_power / max(noise_floor * float(maps["alpha"][index[0], index[1]]), 1e-300)),
            "grid_index": [int(index[0]), int(index[1]), int(u_bin)],
        }
        detections.append(detection)
        snapshots.append(snapshot)
    counters = {
        "candidates_before_cap": candidates_before_cap,
        "local_maxima": len(indices),
        "nms_suppressed": suppressed_count,
        "interpolation_geometry_rejected": rejected_interpolation,
        "overflow": overflow_count,
        "returned_count": len(detections),
        "multiplier": float(multiplier),
    }
    return detections, counters, snapshots


@torch.no_grad()
def process_baseline(Y_b: torch.Tensor, X_b: torch.Tensor, station_id: int, waveform: PaperWaveform,
                     array: ArrayConfig, config: dict, time_ns: int = 0, multiplier: float | None = None,
                     covariance_lut: dict | None = None, station=None, boresight: float = 0.0,
                     height: float = 5.0, resource: SensingResource | None = None) -> dict:
    resource = resource if resource is not None else SensingResource.from_config(config)
    maps = compute_maps(Y_b, X_b, waveform, array, config["detector"], resource=resource)
    detections, counters, snapshots = detect_from_maps(maps, station_id, time_ns, station, boresight,
                                                       config, array, multiplier=multiplier,
                                                       covariance_lut=covariance_lut, height=height)
    return {"power_rd": maps["P_RD"], "detections": detections,
            "snapshots": snapshots, "counters": counters}


def _angle_parabolic(log_power: torch.Tensor, index: int) -> float:
    size = log_power.numel()
    left = log_power[(index - 1) % size] if index > 0 else log_power[index]
    centre = log_power[index]
    right = log_power[(index + 1) % size] if index < size - 1 else log_power[index]
    denom = float(left - 2 * centre + right)
    if denom >= 0:
        return 0.0
    return float(min(max(0.5 * float(left - right) / denom, -0.5), 0.5))


@torch.no_grad()
def aoa_angle_peaks(snapshot: torch.Tensor, array: ArrayConfig, max_peaks: int, rel_threshold_db: float,
                    nms_radius: int, noise_reference: float | None = None,
                    floor_factor_db: float | None = None) -> list[dict]:
    """All resolvable angle peaks of one RD-bin snapshot (no CFAR, no target count).

    Two internal spectrum-candidate filters keep the extraction honest without any
    false-alarm calibration: a peak must stay within ``rel_threshold_db`` of the strongest
    angle peak (rejects the -13.3 dB rectangular-array sidelobes) and, when a noise
    reference is supplied, must beat it by ``floor_factor_db`` (rejects noise angle peaks).
    """
    if snapshot.ndim != 1 or snapshot.numel() != array.elements:
        raise ValueError("snapshot must contain one complex sample per array element")
    if int(max_peaks) < 1 or not math.isfinite(rel_threshold_db):
        raise ValueError("max_peaks >= 1 and a finite relative threshold are required")
    absolute_floor = 0.0
    if noise_reference is not None:
        if not math.isfinite(float(noise_reference)) or float(noise_reference) <= 0:
            raise ValueError("noise_reference must be finite and positive")
        if floor_factor_db is None or not math.isfinite(float(floor_factor_db)):
            raise ValueError("floor_factor_db is required together with noise_reference")
        absolute_floor = float(noise_reference) * 10 ** (float(floor_factor_db) / 10)
    spectrum = torch.fft.fftshift(torch.fft.fft(snapshot, n=array.fft_size))
    power = spectrum.abs().square()
    log_power = torch.log(power.clamp_min(torch.finfo(power.dtype).tiny))
    candidates = sorted(_nms_1d(power, int(nms_radius)),
                        key=lambda i: (-float(power[i]), int(i)))
    suppressed = torch.zeros(array.fft_size, dtype=torch.bool, device=power.device)
    kept: list[int] = []
    for index in candidates:
        if suppressed[index]:
            continue
        low = max(0, index - int(nms_radius))
        high = min(array.fft_size, index + int(nms_radius) + 1)
        suppressed[low:high] = True
        kept.append(int(index))
        if len(kept) >= int(max_peaks):
            break
    strongest = float(power[kept[0]])
    step = 2.0 / array.fft_size
    u_grid = 2.0 * torch.fft.fftshift(torch.fft.fftfreq(array.fft_size))
    peaks = []
    for index in kept:
        if float(power[index]) < absolute_floor:
            break
        relative_db = 10 * math.log10(float(power[index]) / strongest)
        if relative_db < -float(rel_threshold_db):
            continue
        fraction = _angle_parabolic(log_power, index)
        u_hat = float(u_grid[index]) + fraction * step
        peaks.append({"u": float(min(max(u_hat, -1.0), 1.0)), "bin": int(index),
                      "relative_db": float(relative_db), "power": float(power[index])})
    return peaks


@torch.no_grad()
def extract_rd_peaks(maps: dict, params: dict, range_roi_m: tuple[float, float]) -> tuple[list[dict], dict]:
    """Local spectral RD peaks above a global median floor (no CFAR false-alarm logic)."""
    power = maps["P_RD"]
    if not torch.isfinite(power).all():
        raise ValueError("RD power map must be finite")
    floor = float(power.median())
    floor_db = 10 * math.log10(max(floor, 1e-300))
    threshold = floor * 10 ** (float(params["rd_floor_factor_db"]) / 10)
    in_roi = (maps["ranges"] >= float(range_roi_m[0])) & (maps["ranges"] <= float(range_roi_m[1]))
    mask = (power > threshold) & in_roi[:, None]
    radius = (int(params["rd_nms_radius"][0]), int(params["rd_nms_radius"][1]))
    candidates = local_maxima(mask, power, radius)
    indices = torch.nonzero(candidates).tolist()
    log_power = torch.log(power.clamp_min(torch.finfo(power.dtype).tiny))
    order = sorted(range(len(indices)), key=lambda n: (-float(power[tuple(indices[n])]), indices[n]))
    suppressed = torch.zeros_like(mask)
    d_r = float(maps["ranges"][1] - maps["ranges"][0])
    d_v = float(maps["velocities"][1] - maps["velocities"][0])
    hh, hw = radius
    peaks, nms_suppressed, roi_rejected = [], 0, 0
    for position in order:
        index = indices[position]
        if suppressed[index[0], index[1]]:
            nms_suppressed += 1
            continue
        suppressed[max(0, index[0] - hh):min(maps["nr"], index[0] + hh + 1),
                   max(0, index[1] - hw):min(maps["nv"], index[1] + hw + 1)] = True
        fraction_r = _parabolic(log_power, (index[0], index[1]), 0)
        fraction_v = _parabolic(log_power, (index[0], index[1]), 1)
        r_hat = float(maps["ranges"][index[0]]) + fraction_r * d_r
        v_hat = float(maps["velocities"][index[1]]) + fraction_v * d_v
        if not (float(range_roi_m[0]) <= r_hat <= float(range_roi_m[1])):
            roi_rejected += 1
            continue
        peaks.append({"range_m": r_hat, "radial_velocity_mps": v_hat,
                      "peak_power": float(power[index[0], index[1]]),
                      "grid_index": [int(index[0]), int(index[1])]})
        if len(peaks) >= int(params["rd_max_peaks"]):
            break
    counters = {"local_maxima": len(indices), "nms_suppressed": nms_suppressed,
                "roi_rejected": roi_rejected, "returned_count": len(peaks),
                "median_floor": float(floor), "median_floor_db": float(floor_db),
                "threshold_db": float(floor_db + float(params["rd_floor_factor_db"]))}
    return peaks, counters


@torch.no_grad()
def detect_anonymous_measurements(Y_b: torch.Tensor, X_b: torch.Tensor, station_id: int, station,
                                  boresight: float, waveform: PaperWaveform, array: ArrayConfig,
                                  params: dict, visibility: dict,
                                  resource: SensingResource | None = None) -> dict:
    """Non-CFAR single-BS detector: shared echo in, anonymous measurements out.

    The callable sees only the received cube, the known communication symbols, its own
    station geometry and the waveform/visibility model parameters. No target list, no
    identity, no ground-truth count enters this function.
    """
    maps = rd_spectrum(Y_b, X_b, waveform, array, resource=resource)
    roi = (float(visibility["range_min_m"]), float(visibility["range_max_m"]))
    height = float(visibility["height_difference_m"])
    fov_half = math.radians(float(visibility["fov_half_angle_deg"]))
    peaks, rd_counters = extract_rd_peaks(maps, params, roi)
    measurements, fov_rejected = [], 0
    for peak in peaks:
        i, j = peak["grid_index"]
        snapshot = maps["spectrum"][:, i, j]
        angles = aoa_angle_peaks(snapshot, array, int(params["aoa_max_peaks"]),
                                 float(params["aoa_rel_threshold_db"]), int(params["aoa_nms_radius"]),
                                 noise_reference=rd_counters["median_floor"],
                                 floor_factor_db=float(params.get("aoa_floor_factor_db", 10.0)))
        for rank, angle in enumerate(angles[:int(params["max_measurements_per_rd_peak"])]):
            bearing = math.asin(angle["u"])
            if abs(bearing) > fov_half:
                fov_rejected += 1
                continue
            x_m, y_m = coords.polar_to_cartesian(peak["range_m"], angle["u"], station, boresight, height)
            measurements.append({
                "station_id": int(station_id),
                "range_hat": float(peak["range_m"]),
                "radial_velocity_hat": float(peak["radial_velocity_mps"]),
                "u_hat": float(angle["u"]),
                "bearing_hat": float(bearing),
                "x_hat_bs": float(x_m),
                "y_hat_bs": float(y_m),
                "rd_peak_power": peak["peak_power"],
                "angle_peak_power": angle["power"],
                "angle_peak_relative_db": angle["relative_db"],
                "angle_rank": int(rank),
                "grid_index": [int(i), int(j), int(angle["bin"])],
            })
    counters = dict(rd_counters)
    counters.update(fov_rejected=fov_rejected, measurements=len(measurements),
                    rd_peaks=len(peaks))
    return {"measurements": measurements, "rd_peaks": peaks, "counters": counters,
            "power_rd": maps["P_RD"], "spectrum": maps["spectrum"],
            "ranges": maps["ranges"], "velocities": maps["velocities"]}