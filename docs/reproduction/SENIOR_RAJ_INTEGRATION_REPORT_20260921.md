# Senior R0 + Raj QGNN integration report (2026-09-21)

## Verdict

The direct graph-backbone replacement did not improve the reproduced senior
Graph + Motion-Token GPT-2 system. Across seeds 2026, 2027, and 2028, the Raj
variant was stable but consistently worse than the reproduced senior model.

The result does show that the unchanged Motion-Token GPT-2 stage improves the
Raj backbone by about 3.95% on both ADE and FDE. The failure is therefore in the
direct Raj backbone replacement, not in a broken LLM connection.

## Protocol

- Branch: `codex/senior-r0-reproduction`
- Integration commits: `a6bf22c`, `0b44e90`, `b5e7108`, `a94c3bc`
- Dataset/cache: `data/multitarget_lankershim_v1.npz`
- Fixed task: 20 history steps to 20 future steps, at most eight vehicles
- Fixed noise: position sigma 0.35 m, velocity sigma 0.20 m/s
- Seeds: 2026, 2027, 2028
- Raj core: `RajWeightedMultiJQGNNCore`, j=2+j=3, three rounds
- Adapter: 64-dimensional Raj per-vehicle readout projected to the original
  128-dimensional graph-feature contract
- Raj training: 30 epochs, batch 24, learning rate 4e-4
- LLM training: the reproduced four-layer GPT-2, LoRA rank 8, Motion Tokenizer,
  loss, checkpoint selection, and 16-epoch budget were left unchanged
- Test was evaluated once after validation checkpoint selection for each seed

## Three-seed results

| Model | seed 2026 ADE/FDE | seed 2027 ADE/FDE | seed 2028 ADE/FDE | mean ± sample std ADE | mean ± sample std FDE |
|---|---:|---:|---:|---:|---:|
| Raj QGNN | 0.675425 / 1.364710 | 0.669890 / 1.360972 | 0.668875 / 1.351868 | 0.671396 ± 0.003525 | 1.359183 ± 0.006605 |
| Raj QGNN + Motion-Token GPT-2 | 0.648511 / 1.309086 | 0.641657 / 1.302350 | 0.644504 / 1.305675 | 0.644891 ± 0.003443 | 1.305704 ± 0.003368 |

Reproduced senior seed-2026 references:

| Model | ADE | FDE |
|---|---:|---:|
| Target Interaction GNN | 0.619633 | 1.252824 |
| Graph + Motion-Token GPT-2 | 0.591153 | 1.195073 |

## Comparisons

- Adding the unchanged GPT-2 stage to Raj improves the Raj mean by 3.95% ADE
  and 3.93% FDE.
- Raj QGNN alone is worse than the reproduced Target Interaction GNN by 8.35%
  ADE and 8.49% FDE.
- Raj + GPT-2 is worse than the reproduced Target Interaction GNN by 4.08% ADE
  and 4.22% FDE.
- Raj + GPT-2 is worse than the reproduced senior Graph + GPT-2 model by 9.09%
  ADE and 9.26% FDE.

## Interpretation and boundary

This experiment rejects the simplest hypothesis that replacing the senior
interaction GNN with the existing Raj multi-j readout will improve the senior
stack. The three Raj seeds are tightly clustered, so the negative result is not
explained by one unlucky initialization.

The reference senior system is still a single seed, so this is a screening
comparison rather than a paired three-seed significance claim. The margin is
large and all three Raj runs agree; a multi-SNR expansion is not justified for
this direct-replacement variant.

The next technically distinct hypothesis, if pursued, should keep the proven
senior GNN forecast and use Raj only as a gated interaction residual. That is a
new experiment and is not part of this completed direct-replacement run.

## Artifacts

Server-only checkpoints, logs, per-seed JSON, and the aggregate JSON are under:

`/home/dell/YrM/ICCT-senior-r0-reproduction/results/senior_raj_stack/`

Aggregate metrics:

`results/senior_raj_stack/three_seed_summary.json`
