# ICCT QGNN Finalists Formal Results — 2026-09-20

## Frozen experiment boundary

Authority is `/home/dell/YrM/ICCT` on branch `qgnn`. The four finalists completed the first formal SinD gate under one fixed development protocol: full train 15,802 windows, full validation 1,880 windows, SNR 0 dB, seed 2026, 20 epochs, batch size 32, and learning rate `3e-4`. No `train-limit` or `val-limit` was used, and prediction test remains closed.

All four run configurations record the same source hash set, shared initialization SHA-256 `018e5c740fa3a6586213097e2cbe45749468bce9a17ad29bb51db3c08747bc3b`, train-index SHA-256 `18eeffab9a311866f3cb67c352e9882d86e7068fb5527d82cd2d15d100ef73ca`, and validation-index SHA-256 `8f8e21c50a9ab126400a3c7225cce0c9b3aef47597d3dce67563202b64b34903`. The runs started from Git commit `bdd4aa0a21889c9a0e987ece96db620a19b0a323`.

`J = ADE + 0.5 * FDE` selected the checkpoint. ADE and FDE are shown separately and are the prediction metrics to report.

## Completion and best-validation results

| Pair | Arm | Status | Best epoch | Best ADE | Best FDE | Best J | Elapsed | Trainable parameters |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | TRC-QGNN (`trc`) | COMPLETED, 20/20 | 16 | 0.576768 | 1.180421 | 1.166978 | 02:01:15 | 4,252,073 |
| A | RTCN-128 (`rtcn --rtcn-width 128`) | COMPLETED, 20/20 | 20 | 0.498817 | 1.035420 | 1.016527 | 00:48:28 | 4,448,161 |
| B | TO-JQGNN (`toj`) | COMPLETED, 20/20 | 19 | 0.613655 | 1.260698 | 1.244004 | 01:43:13 | 4,143,959 |
| B | TR-TGN (`trtgn`) | COMPLETED, 20/20 | 18 | 0.547933 | 1.167918 | 1.131892 | 00:54:04 | 4,167,404 |

Each run completed exactly 9,880/9,880 training steps. The classical matched controls have lower best full-validation ADE, FDE, and J in both pairs for this fixed seed. This is completed single-seed validation evidence only: it is not a prediction-test result, multi-seed stability claim, or quantum-advantage claim.

The actual device allocation was TRC on GPU0, RTCN-128 on GPU1, TO-JQGNN on GPU1, and TR-TGN on GPU0. This pair-B device swap only changes process placement; all four runs retain the same frozen data, source, initialization, and selection protocol.

## Versioned run artifacts

For each directory below, Git contains `config.json`, `parameters.json`, `heartbeat.json`, `training.json`, `summary.json`, and `best_validation_rows.json`:

- `reports/qgnn/finalists/formal_0db_seed2026_trc/`
- `reports/qgnn/finalists/formal_0db_seed2026_rtcn128/`
- `reports/qgnn/finalists/formal_0db_seed2026_toj/`
- `reports/qgnn/finalists/formal_0db_seed2026_trtgn/`

`training.json` preserves all completed epochs. `best_validation_rows.json` is the full 1,880-window validation record corresponding to the selected `best.pt`; `summary.json` records the status, selected result, data hashes, and parameter counts.

The binary `best.pt` and `last.pt` remain on the authoritative server under their run directories. They are intentionally excluded by the repository-wide `*.pt` rule rather than force-added as ordinary Git blobs. Their size, SHA-256, and exact paths are frozen in `reports/qgnn/finalists/QGNN_FINALISTS_FORMAL_CHECKPOINT_MANIFEST_20260920.json`.

## Engineering evidence retained with this freeze

- `docs/qgnn/finalists/QGNN_FINALISTS_IMPLEMENTATION_HANDOFF_20260920.md`: theory/code audit, matched-control audit, resume and terminal reliability implementation.
- `docs/qgnn/finalists/QGNN_FINALISTS_FORMAL_RUN_COMMANDS_20260920.md`: exact completed invocation provenance and resume form.
- `reports/qgnn/finalists_preflight/`: synthetic core checks, short actual-data smoke/profile, ETA behavior, and resume-equivalence evidence.
- `reports/qgnn/finalists_preflight/preflight_20260920.json`: compact preflight source of record.

No prediction-test artifact exists in this freeze.
