"""Polar/Cartesian geometry and measurement covariance propagation (B domain).

Frozen formulas: SYSTEM_MODEL.md section 5.1, DECISIONS.md D23.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

H_M = 5.0


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def cartesian_to_polar(xy, station, boresight, height: float = H_M):
    """Truth-side conversion used only by C-domain calibration/evaluation."""
    x = torch.as_tensor(xy, dtype=torch.float64)
    s = torch.as_tensor(station, dtype=torch.float64)
    delta = x - s
    rho = torch.linalg.vector_norm(delta, dim=-1)
    r = torch.sqrt(rho ** 2 + height ** 2)
    bearing = wrap_angle(torch.atan2(delta[..., 1], delta[..., 0]) - float(boresight))
    u = torch.sin(bearing)
    return rho, r, bearing, u


def polar_to_cartesian(r, u, station, boresight, height: float = H_M):
    r = torch.as_tensor(r, dtype=torch.float64)
    u = torch.as_tensor(u, dtype=torch.float64)
    s = torch.as_tensor(station, dtype=torch.float64)
    rho = torch.sqrt(r ** 2 - height ** 2)
    phi = float(boresight) + torch.asin(u.clamp(-1.0, 1.0))
    x = s[0] + rho * torch.cos(phi)
    y = s[1] + rho * torch.sin(phi)
    return x, y


def jacobian_ru_to_xy(r, u, boresight, height: float = H_M):
    r = torch.as_tensor(r, dtype=torch.float64)
    u = torch.as_tensor(u, dtype=torch.float64)
    rho = torch.sqrt(r ** 2 - height ** 2)
    phi = float(boresight) + torch.asin(u.clamp(-1.0, 1.0))
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    d_dr = torch.stack([(r / rho) * cos_phi, (r / rho) * sin_phi])
    scale = rho / torch.sqrt((1 - u ** 2).clamp_min(1e-12))
    d_du = torch.stack([scale * (-sin_phi), scale * cos_phi])
    return torch.stack([d_dr, d_du], dim=1)


def lut_sigmas(q_db: float, lut: Mapping) -> tuple[float, float]:
    """Conservative LUT lookup keyed only by the observed peak-to-noise value."""
    bins: Sequence[Mapping] = lut["bins"]
    if not bins:
        raise ValueError("Covariance LUT has no bins")
    for entry in bins:
        limit = entry.get("q_max_db")
        if limit is None or float(q_db) <= float(limit):
            return float(entry["sigma_r_m"]), float(entry["sigma_u"])
    last = bins[-1]
    return float(last["sigma_r_m"]), float(last["sigma_u"])


def covariance_from_lut(r, u, q_db: float, boresight, lut: Mapping, floor_m2: float | None = None):
    sigma_r, sigma_u = lut_sigmas(q_db, lut)
    jacobian = jacobian_ru_to_xy(r, u, boresight, float(lut.get("height_difference_m", H_M)))
    r_ru = torch.diag(torch.tensor([sigma_r ** 2, sigma_u ** 2], dtype=torch.float64))
    floor = float(lut.get("floor_m2", 0.01) if floor_m2 is None else floor_m2)
    covariance = jacobian @ r_ru @ jacobian.T + floor * torch.eye(2, dtype=torch.float64)
    return covariance


def selfcheck() -> dict:
    station = torch.tensor([-64.23372566, -106.90678644], dtype=torch.float64)
    boresight = 0.0
    height = H_M
    r = 137.4
    u = math.sin(math.radians(25.0))
    x, y = polar_to_cartesian(r, u, station, boresight, height)
    rho, r_back, _, u_back = cartesian_to_polar(torch.stack([x, y]), station, boresight, height)
    assert abs(float(r_back) - r) < 1e-9 and abs(float(u_back) - u) < 1e-12
    repeat = polar_to_cartesian(r, u, station, boresight, height)
    assert torch.equal(x, repeat[0]) and torch.equal(y, repeat[1])

    def reference(r_value: float, u_value: float):
        rho_value = math.sqrt(r_value ** 2 - height ** 2)
        phi_value = float(boresight) + math.asin(max(min(u_value, 1.0), -1.0))
        return float(station[0]) + rho_value * math.cos(phi_value), float(station[1]) + rho_value * math.sin(phi_value)

    reference_x, reference_y = reference(r, u)
    assert abs(reference_x - float(x)) < 1e-9 and abs(reference_y - float(y)) < 1e-9
    numerical = torch.zeros(2, 2, dtype=torch.float64)
    eps = 1e-5
    numerical[:, 0] = torch.tensor([(reference(r + eps, u)[0] - reference(r - eps, u)[0]) / (2 * eps),
                                    (reference(r + eps, u)[1] - reference(r - eps, u)[1]) / (2 * eps)])
    numerical[:, 1] = torch.tensor([(reference(r, u + eps)[0] - reference(r, u - eps)[0]) / (2 * eps),
                                    (reference(r, u + eps)[1] - reference(r, u - eps)[1]) / (2 * eps)])
    analytic = jacobian_ru_to_xy(r, u, boresight)
    assert torch.allclose(analytic, numerical, atol=1e-4), (analytic, numerical)
    lut = {"bins": [{"q_max_db": 6.0, "sigma_r_m": 2.0, "sigma_u": 0.03},
                    {"q_max_db": 12.0, "sigma_r_m": 1.0, "sigma_u": 0.02},
                    {"q_max_db": None, "sigma_r_m": 0.5, "sigma_u": 0.01}],
           "floor_m2": 0.01, "height_difference_m": 5.0}
    covariance = covariance_from_lut(r, u, 7.5, boresight, lut)
    assert covariance.shape == (2, 2) and float(torch.linalg.eigvalsh(covariance).min()) > 0
    assert lut_sigmas(3.0, lut) == (2.0, 0.03) and lut_sigmas(99.0, lut) == (0.5, 0.01)
    return {"forward_backward": True, "jacobian_finite_difference": True,
            "lut_lookup": True, "covariance_positive_definite": True}


if __name__ == "__main__":
    print(selfcheck())