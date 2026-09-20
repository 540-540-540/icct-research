# ICCT QGNN Finalists Engineering Handoff — 2026-09-20

## Frozen scope

Authority is the server workspace `/home/dell/YrM/ICCT` on branch `qgnn`. This handoff finishes engineering preflight only: no 15,802-window formal run and no prediction test run was started.

The formal first gate is fixed at SinD 0 dB, seed 2026, 15,802 full-train windows, 1,880 full-validation windows, 20 epochs, and batch size 32. `J=ADE+0.5*FDE` is the validation checkpoint-selection value; ADE and FDE remain the reported prediction metrics.

| Formal pair | Quantum arm | Matched classical arm |
|---|---|---|
| A | `trc` — TRC-QGNN | `rtcn --rtcn-width 128` — RTCN-128 |
| B | `toj` — TO-JQGNN | `trtgn` — TR-TGN |

## Theory-to-code audit

### TRC-QGNN

- `TRCQuantumCore` keeps the rooted pair `Omega1` and ordered rooted-triplet `Omega2` branches, four 5-frame temporal blocks, the first-observation anchored shared GRU32, 42D role descriptions, and all six directed 12D physical relations for triplets.
- Each stage applies the real-symmetric, risk-conditioned configuration Hamiltonian before the configuration-controlled feature encoding. The legal configuration basis is represented directly rather than padded to the entire binary index register; this is an exact reduced-basis implementation of the legal subspace, not a branch-wise classical pool.
- Node encoding is 12 learned history angles + 10 direct physics/role angles + 2 identity angles; each stage has four ordered subrounds. The 35D readout is the 15 root Pauli moments, 12 root-neighbor correlators, and 8 symmetrized triplet Z moments. No real/imaginary state amplitude is read out.
- The reader emits `own + 8 interaction + 19 motion + 20 future-query = 48` tokens. Missing pair/triplet branches are attention-masked.

### TO-JQGNN

- `TOJQGNNCore` uses a root plus at most five sensed-history-selected neighbors (`M<=6`, two qubits per selected vehicle), a last-observation anchored shared GRU32, four stages, and four banks per stage.
- The selector was corrected to the report-defined score `0.5*max_stage_risk + 0.5*stage4_risk`. Its stable tie order is score, summed four-stage distance, then full sensed-history lexicographic order; exactly identical histories are intentionally indistinguishable. It never uses future state, ID, or slot as a semantic key.
- TO acceleration is now the per-5-frame velocity least-squares slope against the supplied timestamps, as specified for TO. TRC keeps its separate first-block-LS / later-block-difference acceleration definition.
- All target-neighbor and neighbor-neighbor even/odd relations enter the four noncommuting Hamiltonian families in the frozen order `V -> W -> H_Z -> H_X -> H_D1 -> H_D2`; the same joint register continues across all four stages without reset or feedback.
- The 31D stage readout is 15 root RDM Pauli moments plus unweighted and risk-weighted eight-correlator summaries. The reader emits `own + 4 interaction + 19 motion + 20 future-query = 44` tokens and has no direct graph bypass into its coordinate head.
- A preflight float64 execution exposed a complex64 initial-state mismatch in the joint register. `init_product_plus` now receives the requested complex dtype, so the float64 path is finite for TO as well as the other three arms.

### Matched controls and common protocol

- RTCN-128 shares TRC's history encoder, role/edge features, `Omega1/Omega2`, directed physical bias, risk potential, masks, four stages, 35D readout, 8-token reader, frozen GPT-2, QKV-LoRA r8/alpha16, Motion Token, loss, optimizer/scheduler, sampling order, validation, and checkpoint rule. Width 128 is the stronger preregistered capacity, not a weakened baseline.
- TR-TGN shares TO's sensed-history selector, `M<=6` context, 38D node/12D edge inputs, four stages, four rounds per stage, 31D readout, 44-token reader, GPT-2/LoRA, loss, optimizer/scheduler, sampling order, validation, and checkpoint rule. Its rank-16 tensor mediator and width 32 are the theory-selected matched control.
- All four arms construct only train and validation datasets. Prediction test remains closed.

## Training-run reliability

`scripts/train_qgnn_finalists.py` now writes `config.json`, `parameters.json`, `heartbeat.json`, `last.pt`, `best.pt`, `training.json`, `summary.json`, and `best_validation_rows.json` for a completed run. An initial `last.pt` is written before the first optimization step.

`--resume` verifies source/token hashes and immutable run settings, then restores model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, epoch/batch progress, accumulated epoch loss, best state, and smoothed throughput state. `--max-seconds` is now opt-in (`0` means no automatic pause), so a normal formal command cannot stop after the former four-hour default.

The terminal prints the first, last, and every 25th step by default. Each training line includes model, epoch/step/global-step counters, loss, ADE, FDE, LR, current and smoothed step time, elapsed time, dynamic ETA, estimated finish time, current allocated/reserved GPU memory, and peak allocated/reserved GPU memory. ETA uses observed exponential-smoothed training throughput after warm-up and adds measured validation-batch throughput after the first validation. Validation prints ADE, FDE, J, best epoch, and best J explicitly.

## Correctness evidence

Source of record: `reports/qgnn/finalists_preflight/preflight_20260920.json`.

| Arm | core output | nonzero/finite core gradients | permutation max abs | active-node padding effect | float64 forward |
|---|---|---:|---:|---:|---|
| TRC-QGNN | `[1,8,2,4,35]` | 15/15 | `2.98e-7` | `0` | pass |
| RTCN-128 | `[1,8,2,4,35]` | 24/24 | `2.98e-7` | `0` | pass |
| TO-JQGNN | `[1,8,4,31]` | 14/14 | `0` | `0` | pass |
| TR-TGN | `[1,8,4,31]` | 36/36 | `0` | `0` | pass |

All-padding output is exactly zero for all arms. End-to-end forward/backward is finite for all four; every trainable core tensor receives a nonzero finite gradient. Compact checkpoint save/load gives maximum prediction difference `0.0` for every arm.

Two-sample, one-epoch train/validation smokes completed for all four arms with every required run artifact present. The shared GPT-2/Motion Token initialization hash and the two-sample train-index hash match across all four:

```text
shared initialization: 018e5c740fa3a6586213097e2cbe45749468bce9a17ad29bb51db3c08747bc3b
train indices:         e5c6d2f5acfd81d931206b9470d0d722514cebf8a9e89165b9f675bb9b667a2d
```

TO-JQGNN was forced to pause after global step 1, resumed with `--resume`, and was compared with an uninterrupted identical two-step reference. Final model-state maximum absolute difference, train-loss difference, and validation ADE/FDE/J differences are all `0.0`; only wall-clock timing fields differ.

## Instantiated parameters and RTX 4090 profile

Profile definition: synthetic full-model train step on one RTX 4090, including forward, ADE/FDE loss, token loss, backward, gradient clipping, and AdamW update. Numbers are means of three post-warm-up steps; VRAM is peak allocated/reserved GiB. This is resource evidence, not an ADE/FDE result.

| Arm | core trainable | total trainable | B=8: s, alloc/resv GiB | B=16: s, alloc/resv GiB | B=32: s, alloc/resv GiB |
|---|---:|---:|---|---|---|
| TRC-QGNN | 7,052 | 4,252,073 | 0.566, 1.66/1.75 | 0.651, 2.94/3.20 | 0.667, 5.54/6.28 |
| RTCN-128 | 203,140 | 4,448,161 | 0.157, 1.43/1.54 | 0.180, 2.51/2.71 | 0.257, 4.68/5.08 |
| TO-JQGNN | 6,722 | 4,143,959 | 0.378, 2.06/2.17 | 0.424, 3.75/3.94 | 0.540, 7.16/8.31 |
| TR-TGN | 30,167 | 4,167,404 | 0.102, 1.33/1.43 | 0.161, 2.28/2.45 | 0.213, 4.22/4.61 |

Actual-data B32 smokes also completed. Their single validation-batch times were 0.152 s (TRC), 0.119 s (RTCN-128), 0.218 s (TO), and 0.170 s (TR-TGN).

With `ceil(15802/32)*20 = 9,880` training steps and `ceil(1880/32)*20 = 1,180` validation batches, the profile-based base/planning estimates are:

| Arm | base estimate | base plus 20% planning margin |
|---|---:|---:|
| TRC-QGNN | 1.88 h | 2.26 h |
| RTCN-128 | 0.74 h | 0.89 h |
| TO-JQGNN | 1.55 h | 1.86 h |
| TR-TGN | 0.64 h | 0.77 h |

Data loading, checkpoint I/O, GPU contention, and later-epoch throughput can move these values. The live ETA printed by the formal runner is authoritative once it has warmed up.

## Formal run files

For a formal command below, inspect its own run directory:

- `heartbeat.json`: current step, smoothed time, ETA, finish time, and current/peak memory.
- `summary.json`: current/completed status, selected best validation result, parameter count, and data-index hashes.
- `training.json`: one row per completed epoch.
- `last.pt`: resume source; `best.pt` and `best_validation_rows.json`: best-J validation checkpoint and rows.
- `config.json` and `parameters.json`: frozen invocation and instantiated counts.

## Frozen formal commands

Run Pair A concurrently on GPU0/GPU1:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trc --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trc

cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind rtcn --rtcn-width 128 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_rtcn128
```

Run Pair B concurrently on GPU0/GPU1:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind toj --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_toj

cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trtgn --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trtgn
```

Resume with the identical command plus `--resume`:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trc --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trc --resume

cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind rtcn --rtcn-width 128 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_rtcn128 --resume

cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind toj --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_toj --resume

cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trtgn --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trtgn --resume
```
