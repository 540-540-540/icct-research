# Round4 Raj Paper-Math 4096 Paired Diagnostic Preregistration — 2026-09-19

## Purpose

Test whether restoring the major Raj paper-math blocks missing from the accepted ICCT adaptation changes the 4096-window SinD result against the single-j paper-native JohnsonGIN family.

This is a fidelity/transfer diagnostic. It is not a claim that the paper predicts quantum superiority over JohnsonGIN.

## Frozen protocol

- Dataset: SinD
- SNR: 0 dB
- Seed: 2026 for both arms
- Train: deterministic 4096-window seed2026 subset
- Validation: all 1,880 windows
- Test: closed / never constructed
- Epochs: 20
- Batch size: 32
- Base LR: 3e-4
- LoRA LR: 0.25 × base LR
- Optimizer: existing AdamW
- Scheduler: existing cosine schedule
- Correction cap: 16 m
- Same GPT-2 / Tokenizer / trajectory+token loss
- Same train indices / validation exposure
- Fresh initialization; no reuse of prior checkpoints

## Quantum arm

Kind: `raj_paper_math_quantum`

Frozen graph core:
- single j=3
- D=6, k=3
- L=3
- controlled embedding re-upload
- graph-conditioned node-register Givens message passing
- compound/Cayley embedding evolution
- one final three-block joint-register M(theta_M)
- conditional projection and full complex embedding 1-RDM
- incidence-weighted subset-to-agent readout
- no multi-j fusion
- no local graph residual

## Classical arm

Kind: `raj_johnson`

- single j=3
- same ICCT subset feature builder
- classical Johnson-graph message passing
- same downstream GPT-2 / Tokenizer / loss / data

This is the closest existing ICCT adaptation of the official JohnsonGIN family, but it is **not a strict substrate-only control** because the quantum arm uses graph-conditioned weighted node mixing while this JohnsonGIN uses fixed binary Johnson adjacency. That limitation is frozen before results.

## Interpretation gates

Primary transfer gate:
- Quantum ADE < JohnsonGIN ADE, AND
- Quantum FDE < JohnsonGIN FDE.

Strong transfer signal:
- primary gate passes, AND
- J relative gain >= 1%.

If the primary gate fails:
- do not tune V/A/W/M/readout from this result;
- freeze the result;
- analyze which paper mechanism fails to translate to continuous SinD;
- run a separately preregistered weighted-classical causal diagnostic before attributing failure to the quantum substrate.

If the primary gate passes:
- it means the more faithful Raj adaptation recovers a positive paper-native signal under the 4096 regime;
- it still does not prove a quantum-substrate advantage until the weighted-classical causal diagnostic is performed.

## Exact commands

GPU0 Quantum:

`env CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_paper_math_quantum --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --train-limit 4096 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_paper_math_quantum_4096_seed2026 --max-seconds 7200 --save-steps 100`

GPU1 JohnsonGIN:

`env CUDA_VISIBLE_DEVICES=1 /home/dell/YrM/envs/ICCT/bin/python -u scripts/train_qgnn_paper_native.py --kind raj_johnson --j 3 --seed 2026 --snr 0 --epochs 20 --batch-size 32 --train-limit 4096 --depth 3 --correction-cap 16 --lr 3e-4 --run-dir reports/qgnn/round4_raj_paper_math_johnson_j3_4096_seed2026 --max-seconds 7200 --save-steps 100`

Both jobs must use background processes, job.json, heartbeat.json, run.log, checkpoint/resume support and short polling.

