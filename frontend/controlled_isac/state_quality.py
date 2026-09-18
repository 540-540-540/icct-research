"""LEGACY descriptive score; not used for calibration or evaluation.

Final Audit (``reports/isac_final_audit/quality_audit.md``) showed that this percentage is a
self-declared score whose 1 m / 1 m/s reference scales are arbitrary relative to the Automatum
motion scales; it must never be reported as an accuracy. The frozen Route-B evaluation uses
physical ``position RMSE [m]`` / ``velocity RMSE [m/s]`` with the Good / Medium /
Poor-but-usable SNR semantics instead.

The function below is kept for backward compatibility only:

    s       = sqrt( (dp / p_ref)^2 + (dv / v_ref)^2 )
    quality = 100 * exp(-s)
"""
from __future__ import annotations

import math

DEFAULT_P_REF_M = 1.0
DEFAULT_V_REF_MPS = 1.0


def quality_from_errors(position_error_m: float, velocity_error_mps: float,
                        p_ref_m: float = DEFAULT_P_REF_M,
                        v_ref_mps: float = DEFAULT_V_REF_MPS) -> float:
    if p_ref_m <= 0 or v_ref_mps <= 0:
        raise ValueError("quality reference scales must be positive")
    score = math.sqrt((float(position_error_m) / float(p_ref_m)) ** 2
                      + (float(velocity_error_mps) / float(v_ref_mps)) ** 2)
    return 100.0 * math.exp(-score)


def quality_from_state(state: dict, position_true, velocity_true,
                       p_ref_m: float = DEFAULT_P_REF_M,
                       v_ref_mps: float = DEFAULT_V_REF_MPS) -> float:
    dp = math.hypot(state["x_hat"] - float(position_true[0]),
                    state["y_hat"] - float(position_true[1]))
    dv = math.hypot(state["vx_hat"] - float(velocity_true[0]),
                    state["vy_hat"] - float(velocity_true[1]))
    return quality_from_errors(dp, dv, p_ref_m, v_ref_mps)


def mapping_table(p_ref_m: float = DEFAULT_P_REF_M, v_ref_mps: float = DEFAULT_V_REF_MPS,
                  fractions: tuple = (0.95, 0.9, 0.8, 0.6, 0.5, 0.4, 0.25, 0.1)) -> list[dict]:
    """Quality values for representative error combinations (report/plot aid)."""
    rows = []
    for fraction in fractions:
        s = -math.log(fraction)
        for share in (0.0, 0.5, 1.0):
            dp = s * p_ref_m * math.sqrt(1 - share ** 2)
            dv = s * v_ref_mps * share
            rows.append({"quality": 100.0 * fraction, "score": s,
                         "position_error_m": dp, "velocity_error_mps": dv})
    return rows


if __name__ == "__main__":
    assert abs(quality_from_errors(0.0, 0.0) - 100.0) < 1e-12
    assert quality_from_errors(0.1, 0.0) > quality_from_errors(1.0, 0.0)
    assert 0 < quality_from_errors(10.0, 10.0) < 1
    print({"q(0.1m,0.1m/s)": quality_from_errors(0.1, 0.1),
           "q(0.5m,0.5m/s)": quality_from_errors(0.5, 0.5),
           "q(1m,1m/s)": quality_from_errors(1.0, 1.0)})