# Senior R0 reproduction report (2026-09-21)

## Verdict

The original senior multi-target Graph + Motion-Token GPT-2 experiment was reproduced successfully on the server. The regenerated scene cache preserves the exact scene membership, masks, target IDs, and start indices. The two primary gates passed again:

- Target Interaction GNN outperforms Independent GRU on both ADE and FDE.
- Graph + Motion-Token GPT-2 outperforms the reproduced Target Interaction GNN on both ADE and FDE.

This is a historical reproduction result only. It uses seed 2026 and the original exposed test split; it is not new multi-seed evidence and does not include Raj QGNN.

## Git and runtime

- Branch: `codex/senior-r0-reproduction`
- Base commit: `7e14be2cf6645187a0c9e36f2017beec9eee5f66`
- Worktree: `/home/dell/YrM/ICCT-senior-r0-reproduction`
- Python: 3.11.15
- PyTorch: 2.13.0+cu130
- Transformers: 4.57.6
- NumPy: 2.4.6
- pandas: 3.0.5
- GPU: NVIDIA GeForce RTX 4090
- Seed: 2026

The raw CSV and local GPT-2 weights remain server-only and are not committed to Git.

## Data-cache reproduction

Configuration: history 20, prediction 20, maximum 8 targets, minimum 2 targets, 45 m radius, stride 3, two scenes per time, seed 2026, temporal split 70/15/15.

| Split | Shape | Count | Active targets per scene |
|---|---:|---:|---:|
| Train | `[6000,40,8,4]` | 6000 | 2-8 |
| Validation | `[1200,40,8,4]` | 1200 | 2-8 |
| Test | `[1200,40,8,4]` | 1200 | 2-8 |

Compared with the archived historical cache:

- masks: exact equality;
- target IDs: exact equality;
- start indices: exact equality;
- state arrays: maximum absolute difference below `6.11e-5`;
- the only metadata difference is the absolute source-CSV path.

The tiny state difference is consistent with floating-point/library drift and does not change scene construction.

## Main test results

| Model | Historical ADE | Reproduced ADE | Historical FDE | Reproduced FDE |
|---|---:|---:|---:|---:|
| Constant Velocity | 1.056293 | 1.056293 | 2.038661 | 2.038661 |
| Independent GRU | 0.700099 | 0.700366 | 1.468509 | 1.470346 |
| LSTM | 0.626552 | 0.622822 | 1.297952 | 1.290216 |
| TCN | 0.647183 | 0.646647 | 1.337266 | 1.339345 |
| Transformer | 0.683634 | 0.650979 | 1.401837 | 1.350592 |
| Target Interaction GNN | 0.620168 | 0.619633 | 1.252882 | 1.252824 |
| Graph + Motion-Token GPT-2 | 0.592450 | 0.591153 | 1.195643 | 1.195073 |

The reproduced Target Interaction GNN improves over Independent GRU by 11.53% ADE and 14.79% FDE. The reproduced Graph + Motion-Token GPT-2 improves over the reproduced GNN by 4.60% ADE and 4.61% FDE.

The Transformer result is materially better than the historical snapshot while the parameter count, data, seed, and schedule remain the same. The current scripts do not enforce deterministic CUDA algorithms, and the historical environment manifest is incomplete, so this row is recorded as environment/non-determinism drift. It does not change the model ordering or either primary gate.

## Parameter counts

| Model | Parameters |
|---|---:|
| Independent GRU | 98,210 trainable |
| Target Interaction GNN | 400,558 trainable |
| LSTM | 903,266 trainable |
| TCN | 531,362 trainable |
| Transformer | 3,272,546 trainable |
| Graph + Motion-Token GPT-2 | 2,505,958 trainable / 70,643,348 total |

## Multi-SNR reproduction

The original empirically calibrated position/velocity noise map and common test-noise seed 4026 were reused.

| SNR | GNN ADE/FDE | Graph+GPT-2 ADE/FDE | ADE gain | FDE gain |
|---:|---:|---:|---:|---:|
| 5 dB | 0.869001 / 1.589413 | 0.833860 / 1.522751 | 4.04% | 4.19% |
| 10 dB | 0.708602 / 1.371344 | 0.668321 / 1.290355 | 5.68% | 5.91% |
| 15 dB | 0.692196 / 1.349952 | 0.652636 / 1.270188 | 5.72% | 5.91% |
| 20 dB | 0.623497 / 1.263810 | 0.589473 / 1.194138 | 5.46% | 5.51% |

All four SNR conditions preserve the historical conclusion that Graph + Motion-Token GPT-2 is better than Target Interaction GNN on both ADE and FDE.

## Component ablation at 20 dB

| Variant | ADE | FDE |
|---|---:|---:|
| Full retrained model | 0.589473 | 1.194138 |
| Without uncertainty | 0.591331 | 1.196953 |
| Without soft token | 0.601504 | 1.221056 |
| Without graph context | 0.617851 | 1.250860 |
| Without token loss | 0.603115 | 1.224831 |
| Without LoRA | 0.591254 | 1.196282 |

The historical ablation ordering is retained. Removing graph context causes the largest degradation among the listed variants.

## Server artifacts

All regenerated artifacts are under:

`/home/dell/YrM/ICCT-senior-r0-reproduction/reports/senior_r0_reproduction/`

- `phase1/`: Independent GRU and Target Interaction GNN checkpoints, logs, results.
- `phase2/`: Graph + Motion-Token GPT-2 checkpoint, logs, results.
- `comparisons/`: LSTM, TCN, Transformer checkpoints, logs, comparison results.
- `ablation/`: six ablation checkpoints, logs, and 20 dB results.
- `snr/`: main and ablation multi-SNR results plus regenerated trajectory examples.
- `figures/`: four figures in PDF/SVG/PNG plus visual-QA metadata.

The four regenerated figures passed the built-in checks and manual preview inspection for clipping, missing glyphs, and legend/layout defects.

## R0 boundary

R0 is complete for the original data cache, classical baselines, interaction GNN, Motion-Token GPT-2/LoRA, multi-SNR evaluation, core component ablations, checkpoints, logs, parameter counts, and key figures. No Raj QGNN code or experiment was introduced in this reproduction.
