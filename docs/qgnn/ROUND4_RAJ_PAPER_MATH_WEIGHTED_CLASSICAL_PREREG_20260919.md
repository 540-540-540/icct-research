# Round4 Raj Paper-Math Weighted-Classical Diagnostic Preregistration — 2026-09-19

## Purpose

Disentangle two major confounds in the failed paper-math 4096 transfer:
1. the existing `raj_johnson` comparator contains an ICCT local-history residual;
2. the paper-math Quantum uses graph-conditioned weighted subset dynamics.

The frozen paper-math Quantum result at commit `2e418f9` is reused unchanged. Only a new weighted/no-local classical reference is trained.

## Frozen Quantum reference

`raj_paper_math_quantum`, seed2026, 0 dB, 4096 train / 1880 val / 20 epochs:
- ADE 0.810536438
- FDE 1.707538617
- J 1.664305747
- best epoch 1

## New comparator

Kind: `raj_paper_johnson`

Properties:
- single j=3
- graph-conditioned weighted Johnson dynamics
- no `local(own)` residual in the graph readout
- continuous subset features
- same 64D graph output contract
- same fixed GPT-2 / Tokenizer / trajectory+token objective

Limitation frozen before result:
`raj_paper_johnson` uses `ContinuousSubsetLoader`, while the new Quantum prototype uses `SubsetFeatureBuilder` plus the controlled feature loader. Therefore this is a **first causal diagnostic**, not a strict substrate-only control.

## Protocol

- SinD, 0 dB
- seed2026
- deterministic 4096-window subset
- all 1880 validation windows
- 20 epochs, batch32
- LR 3e-4
- depth3
- correction cap16
- same downstream initialization seed
- test closed
- fresh comparator initialization

## Interpretation

If weighted/no-local Johnson remains far better than the frozen Quantum:
- the old Johnson local residual is not sufficient to explain the failure;
- attention shifts toward the Quantum V/W/M/readout transfer and/or its per-agent task interface.

If weighted/no-local Johnson collapses toward the Quantum:
- a large fraction of the previous gap is attributable to the local/interface support rather than the quantum substrate.

No architecture tuning is authorised from this diagnostic.

## Exact command

`env CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_paper_johnson --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --train-limit 4096 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_paper_math_weighted_johnson_4096_seed2026 --max-seconds 7200 --save-steps 100`

