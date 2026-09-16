"""Current-frame 2D velocity recovery from multiple BS radial velocities.

For a target at fused position p and BS b with direction ``u_b = (p - station_b)/|p - station_b|``,
the calibrated radial measurement is ``vr_b ~ -(u_b . v)`` (toward-station positive). The current
frame's 2D velocity is a MAP estimate over all associated BS members with a weakly informative
speed prior sigma_prior = 30 m/s (NGSIM freeway speed scale). The prior only matters when the BS
geometry is nearly collinear and the along-line component is unobservable; for well-conditioned
two/three-BS geometry its weight is negligible. No history, no smoothing, no Kalman filter.
"""
from __future__ import annotations

import numpy as np

VELOCITY_PRIOR_SIGMA_MPS = 30.0


def recover_velocity(fused_position, members: list, sigma_radial: float,
                     velocity_prior_sigma: float = VELOCITY_PRIOR_SIGMA_MPS) -> dict:
    position = np.asarray(fused_position, dtype=float)
    normal = np.zeros((2, 2), dtype=float)
    rhs = np.zeros(2, dtype=float)
    directions = []
    for member in members:
        delta = position - np.asarray(member["station"], dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            continue
        direction = delta / norm
        weight = 1.0 / max(float(sigma_radial) ** 2, 1e-12)
        normal += weight * np.outer(direction, direction)
        rhs += -weight * direction * float(member["radial_velocity"])
        directions.append(direction)
    eigenvalues = np.linalg.eigvalsh(normal)
    condition_ratio = (float(eigenvalues[0] / eigenvalues[1])
                       if eigenvalues[1] > 0 else 0.0)
    prior_information = 1.0 / max(float(velocity_prior_sigma) ** 2, 1e-12)
    solution = np.linalg.solve(normal + prior_information * np.eye(2), rhs)
    rank = int(np.linalg.matrix_rank(normal, tol=1e-9))
    return {"vx": float(solution[0]), "vy": float(solution[1]), "rank": rank,
            "condition_ratio": condition_ratio, "n_radial": len(directions),
            "prior_sigma_mps": float(velocity_prior_sigma)}