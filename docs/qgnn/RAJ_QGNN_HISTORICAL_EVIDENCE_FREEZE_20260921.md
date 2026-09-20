# RAJ QGNN Historical Evidence Freeze — 2026-09-21

## Status

This document freezes the evidence package imported from the externally completed RAJ experiment campaign into the formal ICCT repository.

Evidence package path inside the formal project:

`reports/qgnn/raj_qgnn_full_evidence_20260921/`

The package is preserved as **historical experimental evidence**. Its source snapshot is not the active implementation used by the current PennyLane / task-redesign mainline.

## Frozen headline result

Frozen test split, 0 dB, 2086 scenes, two seeds (2026/2027), 4096 training samples per seed:

- all scenes: ADE gain 3.006%, FDE gain 3.307%, J gain **3.161%**;
- validation-defined high-risk top 10%: ADE gain 4.418%, FDE gain 4.144%, J gain **4.276%**.

Per-seed overall J gains:

- seed 2026: +3.1537%;
- seed 2027: +3.1685%.

The high-risk threshold is frozen from validation history inputs before test predictions are evaluated.

## Matched classical control

The exact-match comparison uses:

- RAJ weighted Multi-J complex-state graph core: 126,535 trainable graph parameters;
- matched Multi-J Johnson classical core: 125,697 trainable graph parameters;
- graph-core capacity gap: about 0.66%.

Within each seed, Q/C runs share the same:

- training subset;
- GPT-2 / LoRA interface;
- optimizer protocol;
- history input;
- initialization hash for shared LLM/interface parameters.

## Critical interpretation boundaries

### 1. The 3.161% result is a 4096-sample result

The headline frozen-test models are trained on 4096 training windows, not all 15,802 training windows.

The package also contains a seed-2026 full-data experiment. In that run the quantum model is compared against a larger 179,201-parameter Johnson classical baseline and is nearly tied in J. Therefore the current evidence does **not** justify the claim that a 3.161% gain holds on full-data exact-match training.

A future full-data exact-match comparison must keep the matched 125,697-parameter classical control fixed if the project wants to isolate data-scale effects.

### 2. This is not the current PennyLane-native core

The historical winner uses differentiable complex state-vector / Hamiltonian simulation, including operations such as

[
U = e^{iH}
]

on lifted subset states, fixed-particle-number embedding evolution, and complex reduced-density-matrix style readout.

It is quantum-structured / quantum-state-simulation based, but it is not the current formal PennyLane-native implementation.

Therefore this evidence must not be described as:

- physical quantum hardware acceleration;
- hardware quantum advantage;
- a PennyLane-native predictive result.

### 3. Keep the successful simulator architecture

The successful RAJ Weighted Multi-J structure is now the primary QGNN architecture reference. The project should not invent a new QGNN family unless later evidence invalidates it.

The PennyLane branch remains useful for quantum-realizability validation, but it should not automatically replace this performance champion if doing so changes the model and degrades predictive performance.

### 4. Task redesign remains necessary

The historical result proves QGNN > matched classical graph under the frozen legacy setting. It does not prove Graph > Self.

The current project has separately observed that strong target-only TCN/LSTM/Transformer baselines can reach approximately ADE 0.37–0.39 on the legacy 2s→2s task. Therefore task redesign remains active:

[
	ext{Task Definition} > 	ext{GPT-2} > 	ext{QGNN}
]

The intended future confirmation is:

1. demonstrate that the redesigned interaction-critical task genuinely needs neighbor information;
2. repair/fairly evaluate the GPT-2 self-motion branch;
3. re-evaluate the frozen RAJ architecture against matched classical graph baselines.

## Artifact policy

The formal project keeps the full evidence-package directory structure.

Eight `best.pt` checkpoints are retained on both the local and server formal project copies for reproducibility.

Checkpoints are intentionally excluded from Git/GitHub by ignore rules.

GitHub contains:

- final aggregate JSONs;
- per-seed configs / summaries / training histories / validation rows / logs;
- source snapshot;
- evaluation / aggregation scripts;
- SHA256SUMS;
- this freeze note.

The package SHA256 manifest must be preserved unchanged so the retained checkpoints can be verified independently.
