# Round4 Raj Row-Local Reupload Isolation Preregistration — 2026-09-19

## Purpose

Isolate whether the failed full paper-math SinD transfer is caused primarily by the chosen global-state construction / controlled Householder re-upload interpretation.

Only the V(x) state-construction and re-upload semantics change.

## Frozen reference

Full paper-math Quantum:
`raj_paper_math_quantum`

Frozen 4096 / seed2026 / 20e result:
- ADE 0.810536438
- FDE 1.707538617
- J 1.664305747
- best epoch 1
- graph params 56,763.

## New diagnostic kind

`raj_paper_math_rowlocal_quantum`

Keep identical to the frozen full paper-math core:
- single j=3
- D=6, k=3
- rounds=3
- same `SubsetFeatureBuilder(32)`
- same graph-conditioned `EquivariantSubsetAdjacency`
- same compound/Cayley W(theta)
- same final `JointRegisterMixer`
- same conditional projection
- same probability-weighted full-complex 1-RDM incidence readout
- same 64D graph output
- no local-history residual.

Change only loader/re-upload semantics.

### Initial load

Use the same feature MLP and non-negative feature amplitudes as `ControlledFeatureLoader`.

Instead of allowing subset-row norms to encode non-uniform global subset probability:
- normalize every valid subset row independently;
- assign equal norm to every valid subset row;
- divide by sqrt(number of valid subsets), producing one globally normalized factored state.

### Re-upload

Instead of controlled Householder evolution:
- compute the same row-normalized conditional target;
- add the target to the current subset row;
- normalize each valid subset row independently;
- assign equal row norm 1/sqrt(number of valid subsets).

This mirrors the successful row-local/additive semantics of the intermediate Raj route while retaining the rest of the full paper-math pipeline.

This is a **mechanism ablation**, not a claim of greater paper fidelity.

## Protocol

- SinD, 0 dB
- seed2026
- deterministic 4096 train windows
- all 1880 validation windows
- 20 epochs
- batch32
- LR 3e-4
- depth3
- correction cap16
- same optimizer / scheduler
- same fixed GPT-2 / Tokenizer / trajectory+token loss
- test closed
- fresh run, no checkpoint reuse.

## Interpretation gates

Primary rescue:
- ADE lower than full paper-math reference;
- FDE lower than full paper-math reference;
- J improvement >= 3%.

Strong rescue:
- both metrics lower;
- J improvement >= 5%.

Near-no-effect:
- absolute J change <= 1%.

If primary rescue passes:
- the chosen global-state probability allocation / controlled-Householder re-upload package is materially responsible for the failed transfer.

If strong rescue passes and the result approaches the intermediate Quantum J=1.525528:
- the loader/re-upload interpretation becomes the dominant identified failure mechanism.

If near-no-effect passes:
- loader/re-upload is not the main explanation and the remaining joint/readout/interface interpretation must be revisited despite the small inference sensitivity.

No tuning is allowed from the outcome before it is frozen.

## Exact command

`env CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_paper_math_rowlocal_quantum --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --train-limit 4096 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_paper_math_rowlocal_quantum_4096_seed2026 --max-seconds 7200 --save-steps 100`

No full-train, five-SNR or test-set experiment is authorised.
