# QGNN Selection Candidate A — Raj-Style Weighted Multi-j Freeze — 2026-09-19

## Status

This file freezes the strongest Raj-style candidate while the project formally returns to the QGNN architecture-selection stage.

This is **not** a declaration that Raj is the final QGNN.

Candidate A is retained because it currently has the strongest project-level evidence among tested quantum routes.

## Candidate A

Model kind:

`raj_weighted_multij_quantum`

Core:

`RajWeightedMultiJQGNNCore`

Main mechanism:
- both j=2 and j=3 subset branches run quantum dynamics for every scene;
- pair/triplet subset identity is preserved;
- SinD history-only continuous physical relations directly condition weighted Johnson-subset quantum mixing;
- fixed-particle-number embedding evolution is retained;
- layerwise feature re-upload is retained;
- per-agent incidence readout produces 64D continuous interaction features;
- j=2 and j=3 are fused per agent;
- same Motion-Token GPT-2 downstream is used.

This is a **Raj-style ICCT adaptation**, not a verbatim reproduction of Raj et al. 2026.

## Strongest evidence

### 4096-window paired diagnostics

Seed 2026:
- Quantum ADE 0.516457940
- Quantum FDE 1.102781152
- Johnson ADE 0.525952385
- Johnson FDE 1.129781810
- relative gain: ADE +1.805%, FDE +2.390%

Seed 2027:
- Quantum ADE 0.530895042
- Quantum FDE 1.148233937
- Johnson ADE 0.550422603
- Johnson FDE 1.189891210
- relative gain: ADE +3.548%, FDE +3.501%

Two-seed mean relative gain:
- ADE +2.696%
- FDE +2.960%
- J +2.833%

This is the strongest repeated positive signal currently observed among project QGNN candidates.

### Full-train 15,802-window confirmation, seed 2026

Quantum:
- ADE 0.481258665
- FDE 1.051172996
- J 1.006845163

Johnson:
- ADE 0.482314337
- FDE 1.049728396
- J 1.007178535

Relative Quantum gain:
- ADE +0.219%
- FDE -0.138%
- J +0.033%

Conclusion: the full-data result is effectively a near tie and does **not** establish a stable quantum advantage.

## Interpretation

Candidate A remains the strongest candidate because:
1. it is the only currently tested route with repeat positive 4096-window results across two seeds;
2. its absolute prediction quality is strong, around ADE 0.48 / FDE 1.05 under full training;
3. the mechanism directly uses higher-order pair/triplet subset identity, which is aligned with the SinD interaction evidence;
4. it satisfies the project requirement that graph structure directly control quantum computation.

But Candidate A has **not passed final selection** because:
- full-data advantage was not retained;
- a same-family strong classical higher-order baseline can catch up;
- the source of any small-data advantage is not yet fully isolated;
- the paper's j-WL guarantees do not directly imply ADE/FDE superiority on continuous trajectory prediction.

## Paper-fidelity branch status

The later single-j full paper-math reconstruction is a **mechanism/fidelity research branch**, not a replacement for Candidate A.

Its poor ADE/FDE values must not be interpreted as degradation of Candidate A.

The paper-fidelity investigation has already established:
- official Raj WL-climb behaviour is reproducible;
- official JohnsonGIN reproduces the same higher-order WL climb;
- the public repository is not a complete executable transcription of every paper diagram block;
- a SinD full paper-math adaptation performs much worse than Candidate A;
- several fidelity factors were isolated without rescuing the model;
- adjacency operator composition is a material factor, but this branch remains far below Candidate A.

No further paper-fidelity micro-ablation is part of the main QGNN selection loop unless explicitly reopened.

## Future Raj work

Raj-style research is now frozen for a dedicated future GPT-6 Pro thread.

That future task should **not** ask for a more literal Raj reproduction.

It should ask:

> Design the strongest scientifically defensible Raj-style QGNN specifically for SinD multi-vehicle trajectory prediction, using all current evidence, while preserving a fair strong classical higher-order baseline and explaining why the chosen quantum mechanism could retain an advantage at full-data scale.

The future Raj thread may revisit:
- sample-efficiency versus full-data behaviour;
- multi-j fusion;
- capacity matching;
- higher-order interaction selection;
- graph-conditioned quantum dynamics;
- task-specific readout;
- stronger matched classical controls.

It must not erase or reinterpret the frozen negative fidelity evidence.

## Selection-stage status after this freeze

- Candidate A: **Raj-style Weighted Multi-j QGNN — retained / strongest tested candidate / not selected**
- Candidate B: **SQM-GNN-style quantum message passing — next route to explore**
- RC-HQGNN / Scene-Adaptive RC-HQGNN: not promoted after failed overall gates
- Edge-Local QGCN: comparator / reference route
- Skolik-style EQC: symmetry / historical quantum reference

The project is formally back in **QGNN architecture selection**.

