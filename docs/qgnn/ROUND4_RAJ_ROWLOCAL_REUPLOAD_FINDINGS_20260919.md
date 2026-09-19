# Round4 Raj Row-Local Reupload Isolation Findings — 2026-09-19

## 1. Decision

The preregistered row-local/additive re-upload isolation shows **near no effect**.

Replacing the full paper-math prototype's global subset-probability allocation and controlled-Householder re-upload with equal-row-norm / additive row-local semantics does not rescue performance.

Therefore the chosen global-state probability allocation / Householder re-upload package is **not a credible primary explanation** for the failed SinD transfer.

## 2. Result

| Model | ADE | FDE | J | Best epoch | Graph params |
|---|---:|---:|---:|---:|---:|
| Row-local paper-math Quantum | 0.810507801 | 1.708257703 | 1.664636653 | 1 | 56,763 |
| Frozen full paper-math Quantum | 0.810536438 | 1.707538617 | 1.664305747 | 1 | 56,763 |

Row-local relative gain over the frozen full-paper reference:
- ADE: +0.0035%
- FDE: -0.0421%
- J: -0.0199%

Preregistered gates:
- primary rescue: both ADE/FDE lower and J gain >=3% -> **FAIL**
- strong rescue: both lower and J gain >=5% -> **FAIL**
- near-no-effect: |J change| <=1% -> **PASS**

## 3. Integrity

Both runs use:
- SinD, 0 dB, seed2026
- deterministic 4096 train windows
- all 1880 validation windows
- 20 epochs, batch32, LR 3e-4
- single j=3, D=6, k=3, depth3
- same SubsetFeatureBuilder
- same graph-conditioned EquivariantSubsetAdjacency
- same compound W(theta)
- same final JointRegisterMixer
- same conditional probability-weighted full-complex 1-RDM incidence readout
- same fixed GPT-2 / Tokenizer / trajectory+token objective
- identical train-index hash
- identical downstream initialization hash
- identical graph parameter count
- test closed.

The row-local run completed 20/20 epochs, 2560 steps, with no failure file, OOM, deadlock or non-finite result.

## 4. Learning curve

The row-local model's best validation point is epoch 1. Training loss continues to decrease, while validation J worsens through later epochs; epoch20 J is 1.781802.

This reproduces the same early-generalization failure pattern as the frozen full paper-math model. More training exposure is not the missing explanation.

## 5. Mechanisms now eliminated as primary explanations

The accumulated controlled diagnostics now make the following poor primary explanations for the ~8.34% J gap between the intermediate Quantum and full paper-math Quantum:

1. feature-builder differences: weighted classical J changes only ~0.081%;
2. final joint mixer M alone: zero-M inference perturbation is negligible;
3. probability-weighted versus uniform readout alone: inference perturbation is negligible;
4. probability mass lost outside the readable factor sector: ~99.4% remains readable;
5. global subset probability allocation / controlled-Householder re-upload: row-local replacement changes J only ~0.020%.

## 6. Main remaining structural difference

The highest-priority unresolved mechanism is now **A(G), the node/subset message-passing operator**.

Intermediate `RajPaperQGNNCore` uses a weighted Johnson Hamiltonian followed by a matrix exponential `exp(iH)`.

The full paper-math prototype uses a canonical ordered product of graph-conditioned node-pair Givens rotations.

These are both norm-preserving and graph-conditioned but are not the same unitary. Non-commuting pair rotations make the operator composition materially different.

The next diagnostic should therefore change only this operator-composition semantics while keeping V/W/M/readout fixed.

No full-train, five-SNR or test-set experiment is authorised.
