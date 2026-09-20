# ICCT QGNN Finalists Implementation Handoff — 2026-09-20

## Status

Two theory-selection finalists are implemented and ready for user-started formal train/validation experiments on the `qgnn` branch. No formal 15,802-window run was started by ChatGPT.

- Finalist A: **TRC-QGNN** (`kind=trc`) with matched **RTCN** (`kind=rtcn`).
- Finalist B: **TO-JQGNN** (`kind=toj`) with matched **TR-TGN** (`kind=trtgn`).
- SinD, 3-BS sensing history, 20→20 task, 41×41 mixed Motion Token, GPT-2 first 4 blocks and QKV LoRA r8/alpha16 are retained.
- Prediction test split is never constructed by the training harness.

## Implementation

- `prediction/qgnn_finalists/common.py`: four-block single-agent temporal encoder, physical relations, routing, statevector Pauli utilities.
- `prediction/qgnn_finalists/trc.py`: TRC-QGNN and RTCN.
- `prediction/qgnn_finalists/toj.py`: TO-JQGNN and TR-TGN.
- `prediction/qgnn_finalists/llm.py`: 48-token TRC reader and 44-token TO reader using the frozen Motion Token/GPT-2/LoRA base.
- `prediction/qgnn_finalists/model.py`: end-to-end wrappers and build entry point.
- `scripts/train_qgnn_finalists.py`: resumable train/validation-only runner.

TRC uses current formal N≤8 input and processes rooted pair and ordered rooted-triplet configuration branches in a reduced configuration basis. The eight target roots are vectorized into the batch dimension; this changes execution only, not the TRC equations.

TO uses per-target sensed-history routing to a bounded six-vehicle joint register (root + up to five neighbors), two qubits per selected vehicle, four temporal stages and four encoding/dynamics banks per stage. Commuting edge-Hamiltonian families are fused using basis changes plus diagonal phases rather than launched as thousands of separate gates.

## Fairness

The same seed produces an identical common GPT-2/Token initialization SHA across all four smoke runs:

`018e5c740fa3a6586213097e2cbe45749468bce9a17ad29bb51db3c08747bc3b`

The same two-sample training smoke uses identical train-index SHA:

`e5c6d2f5acfd81d931206b9470d0d722514cebf8a9e89165b9f675bb9b667a2d`

Recommended matched baseline for TRC formal selection is RTCN width 128 (the stronger preregistered capacity); width 64 remains available as a sensitivity control. TR-TGN uses the theory-selected width 32 / tensor rank 16 design.

## Correctness checks

Synthetic core checks on RTX 4090:

| Core | output | permutation max abs | padding effect on active nodes |
|---|---|---:|---:|
| TRC-QGNN | `[B,8,2,4,35]` | ~3.6e-7 | 0 |
| RTCN | `[B,8,2,4,35]` | ~3.0e-7 | 0 |
| TO-JQGNN | `[B,8,4,31]` | 0 | 0 |
| TR-TGN | `[B,8,4,31]` | 0 | 0 |

All end-to-end B=1 backward checks produced finite losses and nonzero finite gradients on every trainable core tensor.

Two-sample, one-epoch train/validation smoke completed for TRC, RTCN-64, RTCN-128, TO and TR-TGN, including best checkpoint and summary generation.

TO resume smoke: a forced `PAUSED_BUDGET` after global step 1 resumed with `--resume`, finished global step 2 and completed validation.

## Parameters

| Model | core trainable | total trainable |
|---|---:|---:|
| TRC-QGNN | 7,052 | 4,252,073 |
| RTCN-64 | 68,036 | 4,313,057 |
| RTCN-128 | 203,140 | 4,448,161 |
| TO-JQGNN | 6,722 | 4,143,959 |
| TR-TGN | 30,167 | 4,167,404 |

## RTX 4090 profile

Synthetic B32 end-to-end forward+backward (not a formal training result):

| Model | B32 step | peak allocated | peak reserved |
|---|---:|---:|---:|
| TRC-QGNN | 1.175 s | 5.51 GiB | 5.98 GiB |
| TO-JQGNN | 0.966 s | 7.13 GiB | 8.03 GiB |

For 15,802 samples, B32 and 20 epochs, there are 9,880 training steps. Multiplying by the isolated profile gives about 3.2 h pure training for TRC and 2.7 h for TO before validation/checkpoint/data overhead. These are planning estimates, not measured full-run wall clocks.

## Formal first gate

Run only 0 dB / seed 2026 first. Use full 15,802 train and full 1,880 validation. Do not use `--train-limit` or `--val-limit`. Test remains closed.

Pair A should compare TRC against RTCN-128. Pair B should compare TO against TR-TGN. Run each pair concurrently on separate GPUs if both GPUs are available.

The formal run directory contains `config.json`, `heartbeat.json`, `last.pt`, `best.pt`, `training.json`, `summary.json` and `best_validation_rows.json`. `last.pt` supports `--resume` with RNG, optimizer and scheduler state.
