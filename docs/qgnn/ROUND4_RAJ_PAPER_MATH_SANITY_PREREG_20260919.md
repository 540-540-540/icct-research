# Round4 Raj Paper-Math Fixed-Set Sanity Preregistration — 2026-09-19

## Purpose

Verify that the isolated `raj_paper_math_quantum` stack can learn through V/A/W/M/1-RDM + the fixed GPT-2 interface without numerical failure.

This is an overfit/trainability sanity check, not a baseline comparison and not evidence of predictive superiority.

## Frozen protocol

- Dataset: SinD train only
- SNR: 0 dB
- Seed: 2026
- Fixed subset: first 32 indices from `np.random.default_rng(2026).permutation(len(train))`
- Batch size: 4
- Optimizer steps: 32
- Model: `raj_paper_math_quantum`
- j=3, D=6, k=3, rounds=3
- Same GPT-2 / Tokenizer / trajectory+token loss as the existing training stack
- LR: 3e-4 for non-LoRA trainables
- LoRA LR: 0.25 × base LR
- AdamW weight decay: 2e-4
- No validation selection
- Test split closed / never constructed
- No baseline run

## Acceptance gate

PASS only if:
1. every loss and gradient norm is finite;
2. no OOM / exception occurs;
3. graph-module gradients are non-zero;
4. mean loss of steps 25-32 is at least 10% lower than mean loss of steps 1-8.

ADE/FDE are not acceptance metrics for this diagnostic.

If the gate fails, freeze the failure before modifying the prototype.
