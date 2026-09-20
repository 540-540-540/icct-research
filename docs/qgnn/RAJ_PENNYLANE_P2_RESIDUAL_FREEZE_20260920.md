# Raj PennyLane P2 residual interface freeze (2026-09-20)

## Decision

P2 freezes one deliberately small interface between the already accepted
PennyLane Raj core and the trajectory LLM:

```text
own history -> Self LLM ---------------------------> self residual
multi-vehicle history -> PennyLane Raj -> [N,64] -> interaction residual + gate
CV + self residual + gate * interaction residual -> 20-step trajectory
```

The QGNN and LLM remain the two primary model components. The interface is an
auxiliary connector, not a third forecasting backbone.

## Frozen tensor contract

- Raj input: `history [B,20,N,4]`, `vehicle_mask [B,N]`.
- Raj formal P1.1 output used by P2: fused `readout [B,N,64]` and boolean
  `interaction_mask [B,N]`.
- The diagnostic `j=2/j=3` branch tokens are not passed to the LLM.
- The LLM produces 20 future hidden states for every active vehicle.
- Every future state receives the same per-vehicle Raj readout plus a learned
  future-step embedding. A 32-dimensional fusion then produces a bounded
  interaction residual and a scalar gate per future step.
- Final prediction is
  `Y_CV + delta_self + gate * delta_interaction`.

This replaces the pre-P2 prototype that expected two branch tokens and therefore
did not match the accepted P1.1 core.

## Training phases

1. `self`: Raj and the interaction interface are frozen, and Raj is not executed.
   The LLM self-motion path learns without graph information.
2. `interaction`: the self LLM is frozen; Raj and the small interaction interface
   learn the interaction correction.
3. `joint`: Raj, the interaction interface, and GPT-2 LoRA parameters are
   trainable. The self residual head remains frozen.

The phase switch is a parameter and compute contract. This freeze does not yet
authorize long training or select checkpoints.

## Acceptance boundary

`scripts/check_raj_pennylane_p2.py` must establish:

- the interface has at most 50,000 parameters;
- phase-specific trainability matches the three-phase contract;
- `self` performs zero Raj calls and is graph-exclusive;
- one actual SinD train-history forward/backward pass is finite;
- gradients reach both Raj and the interface in `interaction` phase;
- frozen LLM/self parameters receive no gradients;
- validation labels, test data, ADE, and FDE remain unopened.

Passing this gate means only that the P2 interface is internally consistent and
cheap enough to proceed. It is not evidence of trajectory improvement, quantum
advantage, convergence, or superiority over a baseline.
