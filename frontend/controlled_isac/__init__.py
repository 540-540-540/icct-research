"""Controlled ISAC frontend (ISAC-REDESIGN-PROBE-01): measurement model.

New prototype chain that keeps multi-BS association, fusion and temporal linking while exposing
raw per-frame measurement error to the downstream model (no Kalman smoothing, no coast, no
track birth/death). This package is additive: the frozen production frontend is unchanged.
"""