# QGNN Finalists Engineering Completion Task

Date: 2026-09-20
Project: /home/dell/YrM/ICCT
Branch: qgnn

## Goal

Finish the two implemented QGNN finalists and their matched classical controls so the user can launch full SinD training manually from the terminal.

Finalist A:
- TRC-QGNN: kind=trc
- RTCN-128: kind=rtcn --rtcn-width 128

Finalist B:
- TO-JQGNN: kind=toj
- TR-TGN: kind=trtgn

Do not run the formal 15,802-window experiments in this task.

## Current state

Implementation already exists:
- prediction/qgnn_finalists/common.py
- prediction/qgnn_finalists/trc.py
- prediction/qgnn_finalists/toj.py
- prediction/qgnn_finalists/llm.py
- prediction/qgnn_finalists/model.py
- scripts/check_qgnn_finalists.py
- scripts/train_qgnn_finalists.py

Theory references:
- docs/qgnn/theory_selection/20260920/ICCT_QGNN_Architecture_Selection_Report.md
- docs/qgnn/theory_selection/20260920_complete/ICCT_QGNN_Architecture_Selection_Server_Report.md

Existing smoke checks show all four main models can forward/backward. TO-JQGNN had one tensor-shape bug during development and has already been repaired.

## Tasks

### 1. Theory-to-code audit

Verify TRC-QGNN and TO-JQGNN against their respective theory reports.

TRC:
- rooted pair + ordered rooted-triplet configuration branches
- four temporal blocks
- directed physical relations
- configuration-space quantum evolution
- measurable readout
- 48-token GPT-2 interface

TO-JQGNN:
- root + up to five selected neighbors
- two qubits per selected vehicle
- four temporal stages
- four encoding/dynamics banks per stage
- target-neighbor and neighbor-neighbor relations
- time-ordered noncommuting joint-register evolution
- 31D stage readout
- 44-token GPT-2 interface

Fix clear implementation mistakes if found and record any material deviation from the theory reports.

### 2. Matched-pair fairness

Freeze the formal pairs:
- trc vs rtcn --rtcn-width 128
- toj vs trtgn

Both sides of each pair must share dataset, seed, SNR, GPT-2, Motion Token, LoRA, loss, optimizer schedule, batch semantics, validation rule and checkpoint rule.

### 3. Correctness / smoke

Run only short checks:
- shape
- mask/padding
- permutation/equivariance
- finite forward/backward
- finite nonzero core gradients
- checkpoint save/load
- resume
- validation output

No full formal training.

### 4. Training script UX

Make scripts/train_qgnn_finalists.py suitable for a user manually watching a multi-hour run.

The terminal must continuously show useful progress, including at least:

- model kind
- epoch current/total
- step current/total for the epoch
- global step / total global steps
- current loss
- ADE
- FDE
- learning rate
- step time
- elapsed wall time
- ETA / remaining time
- estimated finish time
- GPU peak allocated/reserved memory

Print progress regularly, e.g. every 20-50 steps, plus first/last step of each epoch.

ETA should be based on measured running throughput rather than a fixed precomputed guess. Prefer a smoothed moving average so the remaining-time display stabilizes after warm-up.

Validation should clearly print:
- validation ADE
- validation FDE
- J
- best epoch / best J so far

At the end print a concise final summary and exact run directory.

### 5. Run resilience

Ensure each run writes:
- config.json
- parameters.json
- heartbeat.json
- last.pt
- best.pt
- training.json
- summary.json
- best_validation_rows.json

Resume should restore model/optimizer/scheduler/RNG/progress.

### 6. Short GPU profile

Profile all four formal arms on RTX 4090 at B=8/16/32 when feasible:
- trc
- rtcn --rtcn-width 128
- toj
- trtgn

Record:
- forward+backward step time
- peak allocated/reserved VRAM
- instantiated trainable parameter count

Use this only to estimate expected full-run duration.

### 7. Freeze user-run commands

First formal experiment:
- SinD full train 15,802
- full val 1,880
- 0 dB
- seed 2026
- 20 epochs
- batch size 32 unless profiling shows a concrete reason to change it
- no train-limit / val-limit
- test closed

Prepare exact shell commands for:
1. TRC on GPU0
2. RTCN-128 on GPU1
3. TO-JQGNN on GPU0
4. TR-TGN on GPU1

Also provide corresponding --resume commands.

The user will start these commands manually.

## Deliverables

Update/create:
- docs/qgnn/finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md
- docs/qgnn/finalists/QGNN_FINALISTS_FORMAL_RUN_COMMANDS_20260920.md
- reports/qgnn/finalists_preflight/preflight_20260920.json

The handoff should end with:
- implementation status
- correctness status
- profile table
- estimated full-run time
- exact formal commands
- exact resume commands
