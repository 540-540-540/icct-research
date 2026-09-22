# Raj QGNN+LLM metric-aligned quantum refinement freeze

## Status

This is the strongest validation-only, single-model, three-seed result obtained after the previous freeze. It uses no distillation, teacher model, ensemble, routing, or model soup.

The test split was not accessed during optimization. After commit `3cd274e` froze the protocol and checkpoints, the user explicitly authorized one final test evaluation. The selection was not changed afterward. Because the test has now been opened for this checkpoint, any later optimization or formal confirmation must use a new untouched holdout or an external test set.

## Fixed training protocol

1. Start from each seed's previously frozen Raj QGNN graph checkpoint.
2. Update only `core.*` and `raj_projection.*` for up to 60 epochs with learning rate `5e-5` and the metric-aligned loss `ADE + 0.35 * FDE`.
3. Train the existing Motion-Token GPT-2 head for 32 epochs with head learning rate `3e-4`, quantum-core learning rate `5e-5`, metric-aligned loss, quantum coordinate features, and token weight `0.035`. The shared classical graph components and GPT-2 base remain frozen.
4. Run the same optional 16-epoch refinement for every seed with head learning rate `3e-5` and quantum-core learning rate `5e-6`. Early stopping is allowed to retain the incoming stage-3 checkpoint when refinement does not improve validation.
5. Select checkpoints only on validation using the existing `ADE + 0.35 * FDE` rule.

## Frozen validation result

| Seed | ADE | FDE | Selected stage |
| --- | ---: | ---: | --- |
| 2026 | 0.5203611211 | 1.0465148230 | refinement |
| 2027 | 0.5225408967 | 1.0510690980 | stage 3 retained by early stopping |
| 2028 | 0.5197052294 | 1.0485403395 | refinement |
| Mean | 0.5208690824 | 1.0487080868 | — |
| Sample std | 0.0014845103 | 0.0022817667 | — |

Compared with the matched classical GNN+LLM mean (`0.5478482412 / 1.1034988306`), the gains are `4.9246% / 4.9652%`. Compared with the prior frozen Raj QGNN+LLM (`0.5235982595 / 1.0551554282`), this protocol improves another `0.5212% / 0.6110%`.

The three-seed quantum-off mean is `0.6000849906 / 1.2459992137`; enabling the quantum core improves those values by `13.2008% / 15.8340%`. This confirms that the retained gain is quantum-core dependent.

On validation, the requested 5% margin is not claimed: the measured margins remain 0.0754 and 0.0348 percentage points short for ADE and FDE, respectively.

## One-shot frozen test

The post-freeze three-seed test result is `0.5604533705 ± 0.0016014879` ADE and `1.1352680597 ± 0.0016808355` FDE. Against the senior original Graph+GPT-2 test result (`0.5911531277 / 1.1950730480`), this is a `5.1932% / 5.0043%` improvement, so the 5% target is reached on both test metrics.

The test quantum-off mean is `0.6360783858 / 1.3211961639`; enabling the quantum core improves it by `11.8893% / 14.0727%`. No parameter, checkpoint, or selection rule was changed after seeing the test.
