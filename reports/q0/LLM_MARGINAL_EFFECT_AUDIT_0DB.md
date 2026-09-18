# LLM Marginal-Effect Audit — Automatum 0 dB validation

Single-seed diagnostic, validation only; test remains closed.

## Matched 20-epoch exposure

| Downstream | NoGraph ADE/FDE | MPNN ADE/FDE | Graph gain ADE/FDE |
|---|---:|---:|---:|
| Simple-GRU | 0.434671 / 0.725519 | 0.466372 / 0.843173 | -7.29% / -16.22% |
| GPT-2+LoRA | 0.543987 / 0.950401 | 0.529046 / 0.953208 | +2.75% / -0.30% |

- GPT-2 changes Graph marginal gain by +10.04 percentage points in ADE and +15.92 points in FDE relative to Simple-GRU.
- Simple-GRU vs GPT-2 absolute gain at 20 epochs: NoGraph ADE +20.10%, FDE +23.66%; MPNN ADE +11.85%, FDE +11.54%.

## Simple-GRU 40-epoch convergence extension

- NoGraph best: ADE 0.379483, FDE 0.643283, epoch 39.
- MPNN best: ADE 0.451491, FDE 0.837012, epoch 31.
- Graph gain remains negative: ADE -18.98%, FDE -30.12%.

## Interpretation

- The current evidence does **not** support “GPT-2 is already so strong that it creates diminishing returns for Graph.”
- In the matched 20-epoch diagnostic, Graph is much less harmful under GPT-2 than under Simple-GRU; GPT-2 therefore does not appear to be suppressing Graph marginal utility.
- More importantly, the Simple-GRU downstream is substantially better in absolute ADE/FDE than the current GPT-2+LoRA pipeline. The LLM module itself is therefore not yet demonstrated to provide an accuracy benefit on Automatum.
- This points to a possible LLM/interface/optimization bottleneck rather than a saturation effect. The project can still retain LLM as a core module, but its role and training recipe need to be separated from claims about Graph/QGNN gains.
- All conclusions are 0 dB, one seed, validation-only diagnostics.
