# Round4 Raj Paper-Math Semantics-Preserving Vectorization — 2026-09-19

The first paper-math prototype was correct but too slow at batch32 (~4.21 s/optimizer step) because node-register and joint-register Givens operations were executed inside per-scene Python loops.

The implementation was vectorized across the batch without changing V/A/W/M/readout semantics. The frozen slow path remains available as `_forward_reference` for equivalence testing.

Equivalence checks in float64:
- joint-register output max abs error: 0
- joint-register gradient max abs error: 8.67e-19
- adjacency output max abs error: 5.59e-17
- adjacency gradient max abs error: 9.72e-17
- joint fidelity verification rerun: PASS
- full paper-math fidelity verification rerun: PASS

Batch32 engineering profile after warmup:
- before: ~4.207 s/step
- after: ~0.265 s/step
- speedup: ~15.85x
- peak allocated memory: ~3.62 GiB

This is an engineering optimization only. No architecture, objective, data permission, or model-selection criterion changed.
