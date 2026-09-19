# Round4 Raj Paper-Math Fixed-Set Sanity Findings — 2026-09-19

The preregistered 32-step train-only sanity diagnostic **PASSed**.

Frozen prereg commit: `d971635`.
Harness commit at launch: `0bb23f6`.

Protocol:
- SinD train only, 0 dB, seed2026
- deterministic 32-window subset
- batch4, 32 optimizer steps
- `raj_paper_math_quantum`, j=3, D=6, k=3, rounds=3
- same GPT-2 / tokenizer / trajectory+token objective
- test split never constructed

Acceptance result:
- first 8-step mean loss: 1.645754829
- final 8-step mean loss: 1.388392270
- relative loss decrease: **15.637965%**
- prereg threshold: >=10%
- minimum graph gradient norm: 0.05047536
- maximum graph gradient norm: 0.83652554
- all losses/gradients finite
- 32/32 steps completed
- elapsed: 14.145 s
- peak allocated GPU memory: 0.757 GiB

This establishes numerical trainability only. Per-batch ADE/FDE values are not performance evidence and are not used for acceptance.

A separate batch32 engineering profile measured ~4.21 s/optimizer step and ~3.55 GiB peak allocation. Therefore a 4096-window x 20-epoch run is not launched from this unoptimised implementation. The next operation is semantics-preserving vectorisation with old/new equivalence tests.
