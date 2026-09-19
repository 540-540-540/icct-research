# Round4 Raj full-train 0 dB paired confirmation preregistration

Date: 2026-09-19

## Purpose

Confirm whether the repeated 4096-window paper-native positive signal for the accepted Raj-inspired Weighted multi-j QGNN persists when training exposure is expanded to the full SinD training split.

This is an evidence-confirmation run, not an architecture-search or hyperparameter-tuning run.

## Frozen protocol

- Dataset: SinD
- SNR: 0 dB
- Seed: 2026 for both arms
- Train: full 15,802 training windows
- Validation: all 1,880 validation windows
- Test: closed / never constructed
- Epochs: 20
- Batch size: 32
- Base LR: 3e-4
- LoRA LR: 0.25 × base LR, as already implemented
- Optimizer: existing AdamW configuration
- Scheduler: existing cosine annealing configuration
- Depth / rounds: 3
- Correction cap: 16 m
- Same GPT-2 backbone
- Same Tokenizer / token config
- Same trajectory + token loss
- Same data permissions and history-only sensing input

## Frozen paired models

Quantum:
- kind: `raj_weighted_multij_quantum`
- accepted Weighted multi-j Raj-style QGNN
- j=2 and j=3 quantum subset branches both active in every scene

Classical:
- kind: `raj_multij_johnson`
- paper-native matched multi-j JohnsonGIN baseline

No architecture, subset definition, LR, loss, Tokenizer, GPT-2, optimizer, scheduler, or training-objective changes are allowed after this preregistration and before reading the paired full-train result.

## Execution

- GPU0: Quantum
- GPU1: JohnsonGIN
- Fresh run directories; no reuse of 4096-window checkpoints
- Omit `--train-limit` so the complete training split is used
- Independent background processes
- stdout/stderr -> each run's `run.log`
- `job.json` records exact command, PID, GPU, Git HEAD, and launch time
- trainer heartbeat approximately every 25 steps
- checkpoint approximately every 100 global steps
- resume supported by the existing trainer
- `--max-seconds 7200`

Exact training commands are frozen as:

`env CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_weighted_multij_quantum --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_fulltrain_quantum_0db_seed2026 --max-seconds 7200 --save-steps 100`

`env CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_multij_johnson --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_fulltrain_johnson_0db_seed2026 --max-seconds 7200 --save-steps 100`

## Acceptance gates

Primary gate:
- Quantum validation ADE < JohnsonGIN validation ADE, AND
- Quantum validation FDE < JohnsonGIN validation FDE.

Strong gate:
- Primary gate passes, AND
- relative gain in J = ADE + 0.5 × FDE is at least 1%.

If either ADE or FDE fails the primary gate, freeze the result before any tuning and first analyze why the 4096-window two-seed positive signal did not extend to the full-training regime.

No five-SNR run, final test use, baseline weakening, or SQM-GNN switch is authorized by this preregistration alone.
