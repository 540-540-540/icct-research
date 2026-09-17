"""Shared multi-target echo simulator (A domain only).

Implements the frozen V1 model in reports/sensing_design/sens_design_02/SYSTEM_MODEL.md
sections 2-3: one shared observation per BS and frame, all target echoes summed before a
single receiver noise realization. Truth states never leave this module.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch

from .waveform import ArrayConfig, PaperWaveform


def derive_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(seed: int, device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _phases(episode: int, frame: int, bs: int, source_keys: Sequence[int], device, dtype) -> torch.Tensor:
    values = []
    for key in source_keys:
        seed = derive_seed(f"2026:{episode}:{frame}:{bs}:{int(key)}:phase")
        generator = _generator(seed, device)
        values.append((torch.rand(1, device=device, dtype=dtype, generator=generator) * 2 - 1) * math.pi)
    if not values:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.stack(values).reshape(-1)


@torch.no_grad()
def synthesize_shared(
    positions,
    velocities,
    source_keys: Sequence[int],
    stations,
    boresights,
    waveform: PaperWaveform,
    array: ArrayConfig,
    rcs_m2,
    snr_ref_db: float,
    episode: int,
    frame: int,
    height_m: float | None = None,
    noise: bool = True,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float64,
) -> dict:
    if dtype not in (torch.float64, torch.float32):
        raise ValueError("dtype must be float64 or float32")
    if height_m is None or not math.isfinite(float(height_m)) or float(height_m) < 0:
        raise ValueError("height_m must be the finite A01/config height difference (5 m)")
    p = torch.as_tensor(positions, device=device, dtype=dtype)
    v = torch.as_tensor(velocities, device=device, dtype=dtype)
    if p.numel() == 0:
        p = torch.zeros((0, 2), device=device, dtype=dtype)
    if v.numel() == 0:
        v = torch.zeros((0, 2), device=device, dtype=dtype)
    s = torch.as_tensor(stations, device=device, dtype=dtype)
    b = torch.as_tensor(boresights, device=device, dtype=dtype)
    sigma = torch.as_tensor(rcs_m2, device=device, dtype=dtype).reshape(-1, 1)
    if p.ndim != 2 or p.shape[1] != 2 or v.shape != p.shape or s.shape != (3, 2) or b.shape != (3,):
        raise ValueError("Expected positions/velocities [Nt,2], stations [3,2], boresights [3]")
    keys = [int(k) for k in source_keys]
    if len(keys) != len(p):
        raise ValueError("source_keys must provide one stable key per target")
    if sigma.numel() != len(p):
        raise ValueError("rcs_m2 must provide one value per target")
    if not math.isfinite(float(snr_ref_db)):
        raise ValueError("snr_ref_db must be finite")

    delta = s[None] - p[:, None]
    to_target = p[:, None] - s[None]
    rho = torch.linalg.vector_norm(delta, dim=-1)
    if len(p) and torch.any(rho <= 0):
        raise ValueError("Target coincides with station")
    height = rho.new_tensor(float(height_m))
    radii = torch.sqrt(rho ** 2 + height ** 2)
    radial = (delta * v[:, None]).sum(-1) / radii
    bearing = (torch.remainder(torch.atan2(to_target[..., 1], to_target[..., 0]) - b[None] + math.pi,
                               2 * math.pi) - math.pi)
    visibility = ((radii >= 10.0) & (radii <= 300.0) & (bearing.abs() <= math.radians(70.0))) if len(p) else \
        torch.zeros((0, 3), device=device, dtype=torch.bool)
    u = torch.sin(bearing)
    amplitude = 10 ** (float(snr_ref_db) / 20) * (100.0 / radii.clamp_min(1e-9)) ** 2 * torch.sqrt(sigma / 10.0)
    amplitude = torch.where(visibility, amplitude, torch.zeros_like(amplitude))

    xs, ys, noise_cubes = [], [], []
    k = torch.arange(waveform.K, device=device, dtype=dtype)
    n = torch.arange(waveform.N, device=device, dtype=dtype)
    a = torch.arange(array.elements, device=device, dtype=dtype)
    for bs in range(3):
        seed_waveform = derive_seed(f"2026:{episode}:{frame}:{bs}:waveform")
        generator = _generator(seed_waveform, device)
        q = torch.randint(0, 4, (waveform.K, waveform.N), generator=generator, device=device)
        x = torch.polar(torch.ones_like(q, dtype=dtype), math.pi / 4 + math.pi / 2 * q.to(dtype))
        phases = _phases(episode, frame, bs, keys, device, dtype)
        if len(p):
            delay = torch.exp(-1j * 4 * math.pi * waveform.df / waveform.c * radii[:, bs, None] * k)
            doppler = torch.exp(1j * 4 * math.pi * waveform.fc * waveform.T / waveform.c * radial[:, bs, None] * n)
            element = torch.exp(1j * math.pi * a[None] * u[:, bs, None])
            scalar = amplitude[:, bs] * torch.polar(torch.ones_like(phases), phases)
            contribution = (scalar[:, None, None, None] * element[:, :, None, None]
                            * delay[:, None, :, None] * doppler[:, None, None, :]).sum(0)
        else:
            contribution = torch.zeros((array.elements, waveform.K, waveform.N), device=device, dtype=torch.complex128)
            contribution = contribution.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
        clean = contribution * x[None, :, :]
        seed_noise = derive_seed(f"2026:{episode}:{frame}:{bs}:noise")
        noise_generator = _generator(seed_noise, device)
        real = torch.randn((array.elements, waveform.K, waveform.N), device=device, dtype=dtype,
                           generator=noise_generator)
        imag = torch.randn((array.elements, waveform.K, waveform.N), device=device, dtype=dtype,
                           generator=noise_generator)
        noise_cube = torch.complex(real, imag) / math.sqrt(2)
        xs.append(x)
        ys.append(clean + noise_cube if noise else clean)
        noise_cubes.append(noise_cube)

    output_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return {
        "Y": torch.stack(ys).to(output_dtype),
        "X": torch.stack(xs).to(output_dtype),
        "W": torch.stack(noise_cubes).to(output_dtype),
        "alpha": amplitude.T.contiguous(),
        "visible": visibility.T.contiguous(),
    }


@torch.no_grad()
def scale_shared_noise(S, W0, snr_db: float | None = None, noise_variance: float = 1.0) -> dict:
    """Scale the shared base noise realisation to the frame-level SNR definition.

    ``SNR_b = 10*log10(mean|S_b|^2 / mean|W_b|^2)`` with ``W_b = scale_b * W0_b`` and ``W0_b``
    the unit-variance base realisation. ``snr_db=None`` keeps ``W = sqrt(noise_variance)*W0``.
    """
    if S.shape != W0.shape or S.ndim != 4:
        raise ValueError("S/W0 must share the [BS,A,K,N] shape")
    signal_power = S.abs().square().mean(dim=(1, 2, 3))
    base_power = W0.abs().square().mean(dim=(1, 2, 3))
    if not torch.isfinite(signal_power).all() or torch.any(base_power <= 0):
        raise ValueError("Signal power must be finite and the base noise non-degenerate")
    if snr_db is None:
        variance = float(noise_variance)
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
        target_noise_power = base_power * variance
    else:
        if not math.isfinite(float(snr_db)):
            raise ValueError("snr_db must be finite")
        target_noise_power = signal_power / 10 ** (float(snr_db) / 10)
        target_noise_power = torch.where(signal_power > 0, target_noise_power, base_power)
    scale = torch.sqrt(target_noise_power / base_power)
    noise = W0 * scale[:, None, None, None]
    noise_power = noise.abs().square().mean(dim=(1, 2, 3))
    achieved = torch.where(signal_power > 0,
                           10 * torch.log10(signal_power / noise_power.clamp_min(1e-300)),
                           torch.full_like(signal_power, float("nan")))
    return {"W": noise, "scale": scale, "signal_power": signal_power,
            "noise_power": noise_power, "achieved_snr_db": achieved}


@torch.no_grad()
def synthesize_shared_frame_snr(
    positions,
    velocities,
    source_keys: Sequence[int],
    stations,
    boresights,
    waveform: PaperWaveform,
    array: ArrayConfig,
    rcs_m2,
    visibility: dict,
    scene_id: int,
    frame: int,
    snr_db: float | None = None,
    noise_variance: float | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Shared multi-target echo with the frame-level SNR definition (A domain only).

    Per BS: ``S_b`` sums every geometric-visible target echo, ``W0_b`` is one unit-variance
    complex Gaussian realisation whose seed depends only on ``(scene_id, frame, bs)``, and
    ``W_b = scale_b * W0_b`` with ``scale_b`` chosen so that
    ``SNR_b = 10*log10(mean|S_b|^2 / mean|W_b|^2)`` equals ``snr_db``. The SNR value never
    enters the noise seed, so all SNR levels of one (scene, frame, BS) share the same base
    realisation. Amplitude follows ``sqrt(rcs_m2) / r^2``; no 100 m reference SNR exists here.
    """
    if dtype not in (torch.float64, torch.float32):
        raise ValueError("dtype must be float64 or float32")
    height_m = float(visibility["height_difference_m"])
    if not math.isfinite(height_m) or height_m < 0:
        raise ValueError("visibility.height_difference_m must be finite and non-negative")
    range_min = float(visibility["range_min_m"])
    range_max = float(visibility["range_max_m"])
    fov_half = float(visibility["fov_half_angle_deg"])
    if not (0 <= range_min < range_max and 0 < fov_half < 90):
        raise ValueError("Invalid visibility range/FOV")
    if snr_db is not None and not math.isfinite(float(snr_db)):
        raise ValueError("snr_db must be finite when given")
    if snr_db is None:
        variance = 1.0 if noise_variance is None else float(noise_variance)
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
    else:
        variance = 0.0

    p = torch.as_tensor(positions, device=device, dtype=dtype)
    v = torch.as_tensor(velocities, device=device, dtype=dtype)
    if p.numel() == 0:
        p = torch.zeros((0, 2), device=device, dtype=dtype)
    if v.numel() == 0:
        v = torch.zeros((0, 2), device=device, dtype=dtype)
    s = torch.as_tensor(stations, device=device, dtype=dtype)
    b = torch.as_tensor(boresights, device=device, dtype=dtype)
    sigma = torch.as_tensor(rcs_m2, device=device, dtype=dtype).reshape(-1, 1)
    if p.ndim != 2 or p.shape[1] != 2 or v.shape != p.shape or s.shape != (3, 2) or b.shape != (3,):
        raise ValueError("Expected positions/velocities [Nt,2], stations [3,2], boresights [3]")
    keys = [int(k) for k in source_keys]
    if len(keys) != len(p):
        raise ValueError("source_keys must provide one stable key per target")
    if sigma.numel() != len(p) or (len(p) and torch.any(sigma <= 0)):
        raise ValueError("rcs_m2 must provide one positive value per target")

    delta = s[None] - p[:, None]
    to_target = p[:, None] - s[None]
    rho = torch.linalg.vector_norm(delta, dim=-1)
    if len(p) and torch.any(rho <= 0):
        raise ValueError("Target coincides with station")
    height = rho.new_tensor(height_m)
    radii = torch.sqrt(rho ** 2 + height ** 2)
    radial = (delta * v[:, None]).sum(-1) / radii if len(p) else torch.zeros_like(rho)
    bearing = (torch.remainder(torch.atan2(to_target[..., 1], to_target[..., 0]) - b[None] + math.pi,
                               2 * math.pi) - math.pi) if len(p) else torch.zeros_like(rho)
    visible = ((radii >= range_min) & (radii <= range_max) & (bearing.abs() <= math.radians(fov_half))) \
        if len(p) else torch.zeros((0, 3), device=device, dtype=torch.bool)
    u = torch.sin(bearing)
    amplitude = torch.sqrt(sigma) / radii.clamp_min(1e-9) ** 2
    amplitude = torch.where(visible, amplitude, torch.zeros_like(amplitude))

    k = torch.arange(waveform.K, device=device, dtype=dtype)
    n = torch.arange(waveform.N, device=device, dtype=dtype)
    a = torch.arange(array.elements, device=device, dtype=dtype)
    x_list, s_list, w0_list = [], [], []
    for bs in range(3):
        generator = _generator(derive_seed(f"automatum:{scene_id}:{frame}:{bs}:waveform"), device)
        q = torch.randint(0, 4, (waveform.K, waveform.N), generator=generator, device=device)
        x = torch.polar(torch.ones_like(q, dtype=dtype), math.pi / 4 + math.pi / 2 * q.to(dtype))
        if len(p):
            phases = _phases(scene_id, frame, bs, keys, device, dtype)
            delay = torch.exp(-1j * 4 * math.pi * waveform.df / waveform.c * radii[:, bs, None] * k)
            doppler = torch.exp(1j * 4 * math.pi * waveform.fc * waveform.T / waveform.c * radial[:, bs, None] * n)
            element = torch.exp(1j * math.pi * a[None] * u[:, bs, None])
            scalar = amplitude[:, bs] * torch.polar(torch.ones_like(phases), phases)
            contribution = (scalar[:, None, None, None] * element[:, :, None, None]
                            * delay[:, None, :, None] * doppler[:, None, None, :]).sum(0)
        else:
            contribution = torch.zeros((array.elements, waveform.K, waveform.N),
                                       device=device, dtype=torch.complex64 if dtype == torch.float32
                                       else torch.complex128)
        clean = contribution * x[None, :, :]
        noise_generator = _generator(derive_seed(f"automatum:{scene_id}:{frame}:{bs}:noise"), device)
        real = torch.randn((array.elements, waveform.K, waveform.N), device=device, dtype=dtype,
                           generator=noise_generator)
        imag = torch.randn((array.elements, waveform.K, waveform.N), device=device, dtype=dtype,
                           generator=noise_generator)
        w0 = torch.complex(real, imag) / math.sqrt(2)
        x_list.append(x)
        s_list.append(clean)
        w0_list.append(w0)

    clean_stack = torch.stack(s_list).to(torch.complex64 if dtype == torch.float32 else torch.complex128)
    base_stack = torch.stack(w0_list).to(clean_stack.dtype)
    noise = scale_shared_noise(clean_stack, base_stack, snr_db=snr_db, noise_variance=variance)
    output_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return {
        "Y": (clean_stack + noise["W"]).to(output_dtype),
        "X": torch.stack(x_list).to(output_dtype),
        "S": clean_stack,
        "W0": base_stack,
        "W": noise["W"],
        "alpha": amplitude.T.contiguous(),
        "visible": visible.T.contiguous(),
        "r_m": radii.T.contiguous(),
        "radial_velocity_mps": radial.T.contiguous(),
        "bearing_rad": bearing.T.contiguous(),
        "u": u.T.contiguous(),
        "signal_power": noise["signal_power"],
        "noise_power": noise["noise_power"],
        "scale": noise["scale"],
        "target_snr_db": None if snr_db is None else float(snr_db),
        "achieved_snr_db": noise["achieved_snr_db"],
    }


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("synthesize_shared requires CUDA per frozen design")
    w = PaperWaveform()
    array = ArrayConfig()
    stations = [[-64.23372566, -106.90678644], [114.23372566, 0.0], [-64.23372566, 106.90678644]]
    boresights = [0.0, math.pi, 0.0]
    out = synthesize_shared([[0.0, 0.0]], [[5.0, -3.0]], [7], stations, boresights, w, array, [10.0], 20.0,
                            episode=0, frame=0, height_m=5.0, device=device)
    assert out["Y"].shape == (3, array.elements, w.K, w.N)
    assert out["X"].shape == (3, w.K, w.N)
    assert out["W"].shape == out["Y"].shape
    assert out["Y"].dtype == torch.complex128 and out["alpha"].shape == (3, 1)
    print({"Y": list(out["Y"].shape), "alpha": out["alpha"].tolist(), "visible": out["visible"].tolist()})