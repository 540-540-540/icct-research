# QGNN Finalists Formal Run Commands — 2026-09-20

## Scope frozen for this first gate

- Server project: `/home/dell/YrM/ICCT`, branch `qgnn`.
- Data: SinD full train 15,802 and full validation 1,880; prediction test stays closed.
- Protocol: SNR 0 dB, seed 2026, 20 epochs, batch size 32, LR `3e-4`.
- Formal pairs: `TRC-QGNN vs RTCN-128` and `TO-JQGNN vs TR-TGN`.
- Do not add `--train-limit`, `--val-limit`, `--max-seconds`, or `--no-checkpoint` to these commands.

The runner writes an initial checkpoint before the first step. Its terminal ETA starts after a few measured training steps and continues to update from smoothed throughput rather than this document's planning estimate.

## Pair A: TRC-QGNN vs RTCN-128

Start these in two terminals at the same time.

### GPU0 — TRC-QGNN

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trc --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trc
```

Resume:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trc --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trc --resume
```

### GPU1 — RTCN-128

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind rtcn --rtcn-width 128 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_rtcn128
```

Resume:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind rtcn --rtcn-width 128 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_rtcn128 --resume
```

## Pair B: TO-JQGNN vs TR-TGN

Start these in two terminals at the same time.

### GPU0 — TO-JQGNN

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind toj --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_toj
```

Resume:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind toj --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_toj --resume
```

### GPU1 — TR-TGN

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trtgn --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trtgn
```

Resume:

```bash
cd /home/dell/YrM/ICCT && CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_finalists.py --kind trtgn --seed 2026 --snr 0 --epochs 20 --batch-size 32 --lr 3e-4 --progress-steps 25 --save-steps 25 --run-dir reports/qgnn/finalists/formal_0db_seed2026_trtgn --resume
```

## What to watch during a formal run

Every 25 steps, plus each epoch's first and last step, the terminal shows model, epoch/step/global-step progress, loss, ADE, FDE, LR, current and smoothed step time, elapsed time, ETA, estimated finish time, and GPU allocated/reserved plus peak memory.

After validation it prints validation ADE, validation FDE, J, best epoch, and best J. The runner does not use prediction test.

For a run directory such as `reports/qgnn/finalists/formal_0db_seed2026_trc/`:

| File | Use |
|---|---|
| `heartbeat.json` | Current progress, dynamic ETA, estimated finish, and GPU memory. |
| `summary.json` | Current/completed status, current best validation, parameter count, and data hashes. |
| `training.json` | Completed epoch history. |
| `last.pt` | Resume checkpoint. |
| `best.pt` / `best_validation_rows.json` | Best-J checkpoint and its validation rows. |
| `config.json` / `parameters.json` | Frozen command settings and instantiated parameter counts. |

## Planning-only B32 estimates

These are not formal performance results. They use the completed RTX 4090 profile, 9,880 full training steps, 1,180 full validation batches, and a 20% planning margin:

| Arm | Base estimate | Planning estimate |
|---|---:|---:|
| TRC-QGNN | 1.88 h | 2.26 h |
| RTCN-128 | 0.74 h | 0.89 h |
| TO-JQGNN | 1.55 h | 1.86 h |
| TR-TGN | 0.64 h | 0.77 h |

Use the runner's live ETA after warm-up as the operational estimate.
