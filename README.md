# ICCT: 3-BS ISAC Vehicle Tracking and Trajectory Prediction

Static navigation document. Research progress lives in Git branches and commit history; experiment
facts live in `reports/` and `results/`.

## Repository layout

| Path | Contents |
|---|---|
| `frontend/` | Frozen ISAC frontend: shared echo source adapter, B64 OFDM sensing, CA-CFAR, AoA, fusion, CV-KF tracking, F01-E dataset loader |
| `prediction/` | QGAT/QGNN runtime: classical and quantum graph cores, shared GPT-2/LoRA temporal predictor, training utilities (import path only; not run by the ISAC flow) |
| `experiments/` | QGAT/QGNN experiment entry points and their configurations |
| `scripts/` | Stage-1 training/evaluation/protocol entry scripts and environment checks |
| `code/` | Shared research code: remote dependency snapshots, PennyLane QGNN core, quantum association, diagnostics, figure tools |
| `configs/` | Frozen frontend configuration (`shared_frontend.json`) and experiment configurations |
| `docs/` | Static references: conference template (`docs/ICCT2026官方模板/`) and reference material |
| `reports/` | Machine-readable experiment evidence (JSON/CSV/figures) and integrity manifests |
| `results/` | Small machine-readable development evidence; large training outputs stay on the server |

## Environment

- Dedicated environment on the server: `/home/dell/YrM/envs/ICCT` (`source .../bin/activate`).
- Requirements and environment notes: `ENVIRONMENT.md`, `requirements.txt`, `requirements-lock.txt`.
- All production runs execute on the server at `/home/dell/YrM/ICCT`.

## Large assets (not in Git)

Raw datasets (`data/`), model weights (`models/`), training checkpoints (`checkpoints/`) and large
generated arrays/weights (`*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`) are intentionally
excluded from Git and stay on the server (or the local mirror). Integrity evidence for the frozen
F01-E dataset is kept under `reports/f01e/final_freeze_11/`.

## History

| Tag | Meaning |
|---|---|
| `stage0-senior-original` | Senior-delivered project snapshot before handover on 2026-09-11 17:00 +08:00 |
| `stage1-meeting-freeze-20260915` | First stable project version before the 2026-09-15 group meeting |

Historical development branches are fully contained in the current development history and are
preserved until branch cleanup is approved.

## Current development module

ISAC (shared frontend) development continues on the `isac-redesign-probe-01` line; the formal
dataset contract and loader entry are `data/f01e/` and
`frontend.f01e_dataset.F01EDataset`. QGNN/LLM modules follow after the ISAC freeze.