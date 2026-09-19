# Round4 Raj Intermediate Weighted-Matched Diagnostic Preregistration — 2026-09-19

## Purpose

Test whether the simpler Raj intermediate quantum implementation remains competitive with the frozen weighted/no-local classical comparator under a fresh, current-source 20-epoch run.

This diagnostic isolates an important transition:

- intermediate Quantum: weighted Johnson quantum mixing + compound embedding evolution + additive data re-upload + conditional 1-RDM;
- full paper-math Quantum: globally normalised factored state + controlled orthogonal re-upload + final joint-register M + conditional projection + 1-RDM.

The weighted/no-local classical result is already frozen at commit `a50ca44`.

## Frozen comparator

Kind: `raj_paper_johnson`

Frozen seed2026 / 4096 / 20-epoch result:
- ADE 0.768920247
- FDE 1.568731510
- J 1.553286002
- best epoch 19
- graph params 267,354

This comparator is not retrained in this diagnostic.

## New Quantum run

Kind: `raj_paper_quantum`

Current implementation:
- single j=3;
- `ContinuousSubsetLoader`;
- graph-conditioned `WeightedJohnsonMix`;
- compound/Cayley embedding evolution;
- additive layerwise data re-upload;
- conditional full-complex 1-RDM incidence readout;
- no final joint-register `M(theta_M)`;
- no ICCT local-history residual inside the graph core.

This is an intermediate Raj-inspired implementation, not the full paper-math prototype.

## Protocol

- Dataset: SinD
- SNR: 0 dB
- Seed: 2026
- deterministic 4096 train windows
- all 1880 validation windows
- 20 epochs
- batch size 32
- LR 3e-4
- depth 3
- correction cap 16
- same fixed GPT-2
- same Tokenizer
- same trajectory + token loss
- same optimizer / scheduler
- test closed
- fresh run directory
- no checkpoint reuse

## Interpretation gates

Primary mechanism gate — near-match:
- absolute relative J difference between intermediate Quantum and weighted/no-local Johnson <= 1%.

Supporting near-match:
- absolute relative ADE difference <= 2%, AND
- absolute relative FDE difference <= 2%.

Quantum-positive gate:
- Quantum ADE and FDE both lower than weighted/no-local Johnson, AND
- J gain >= 1%.

Clear-negative gate:
- Quantum J worse by >3%, OR
- both ADE and FDE worse by >2%.

Interpretation:
- If near-match passes, while the full paper-math Quantum remains ~7% worse, the evidence points to the newly added paper-math transfer blocks/interface as the main source of degradation rather than weighted graph conditioning alone.
- If the intermediate Quantum is also clearly worse, the historical 12-epoch near-tie is not reproduced under current source / full 20-epoch exposure and the quantum update itself remains suspect.
- No architecture change is authorised before this result is frozen.

## Exact command

`env CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_paper_quantum --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --train-limit 4096 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_intermediate_quantum_4096_seed2026 --max-seconds 7200 --save-steps 100`

No full-train, five-SNR or test-set experiment is authorised by this preregistration.

