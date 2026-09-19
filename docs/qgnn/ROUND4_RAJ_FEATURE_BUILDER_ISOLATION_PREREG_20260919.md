# Round4 Raj Feature-Builder Isolation Preregistration — 2026-09-19

## Purpose

Test whether the performance gap between the successful intermediate Raj route and the failed full paper-math route is materially explained by the subset feature-definition difference.

Only one factor is changed relative to the frozen weighted/no-local classical comparator.

## Frozen reference

Comparator: `raj_paper_johnson`

Frozen result:
- ADE 0.768920247
- FDE 1.568731510
- J 1.553286002
- graph params 267,354
- best epoch 19

Its loader uses `ContinuousSubsetLoader`.

## New diagnostic model

Kind: `raj_subsetbuilder_weighted_johnson`

Keep identical to `RajPaperJohnsonGINCore`:
- single j=3
- depth3
- hidden128
- graph-conditioned `WeightedJohnsonMix`
- same classical update equations
- same jumping/concatenated depth representation
- same incidence pooling
- same 64D no-local readout
- no local-history residual
- same downstream GPT-2 / Tokenizer / objective

Change only the subset feature source:
- replace `ContinuousSubsetLoader` with `SubsetFeatureBuilder(32)`
- obtain the same frozen physical edge tensor through `physical_graph`.

Feature difference under test:
- both include own-node mean/std, physical-edge mean/std and risk mean/std;
- `ContinuousSubsetLoader` additionally retains risk max/min and mean distance/closing statistics.

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
- same optimizer/scheduler
- same fixed GPT-2 / Tokenizer / loss
- test closed
- fresh run; no checkpoint reuse

## Interpretation

Feature-equivalent gate:
- absolute relative J difference versus frozen `raj_paper_johnson` <= 1%.

Strong feature-equivalence:
- absolute relative ADE <= 2%
- absolute relative FDE <= 2%
- absolute relative J <= 1%.

Feature-material gate:
- new model J worse by >3%.

If feature-equivalent passes, the feature-builder difference is not a credible primary explanation for the ~8.34% intermediate-vs-full-paper Quantum gap.

If feature-material passes, the next loader/re-upload diagnostic must retain `ContinuousSubsetLoader` features before attributing the failure to quantum-state construction.

No architecture tuning is allowed from the outcome before it is frozen.

## Exact command

`env CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_subsetbuilder_weighted_johnson --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --train-limit 4096 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_subsetbuilder_weighted_johnson_4096_seed2026 --max-seconds 7200 --save-steps 100`

No full-train, five-SNR or test-set experiment is authorised.

