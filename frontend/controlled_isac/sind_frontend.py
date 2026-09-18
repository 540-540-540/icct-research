"""SinD adapter for the frozen controlled Route-B sensing core.

The sensing mathematics are intentionally unchanged from Automatum V2:
three fixed BS polar measurements, covariance-weighted position fusion,
and multi-BS radial-velocity recovery. Only dataset geometry/configuration
and state keys differ.
"""
from __future__ import annotations

from .automatum_frontend import STATE_FIELDS
from .automatum_frontend import sense_vehicle as _sense_vehicle

def sense_vehicle(scene_id, frame, vehicle_id, position, velocity,
                  snr_db, calibration, setup):
    return _sense_vehicle(scene_id, frame, vehicle_id, position, velocity,
                          snr_db, calibration, setup)
