# Round 4 Raj Paper-Native QGNN Findings — 2026-09-19

## 1. Status

Round4 changed the comparison policy from “new quantum route must first beat the previously hand-strengthened ICCT Adaptive Classical” to **paper-native primary baselines first**, while retaining ICCT strong classical models as secondary references.

For the Raj et al. 2026 route, the primary classical family is:
- JohnsonGIN on the same subset lift (main matched baseline);
- GIN on the original graph;
- PPGN / 3-WL-style higher-order reference.

The test split remains closed. All results below are validation-only diagnostics / model selection evidence, not final test claims.

## 2. Main quantum route selected for convergence test

Current strongest Raj-inspired ICCT candidate:

**Weighted multi-j subset QGNN**

- every scene runs both j=2 and j=3 quantum subset branches;
- j=2 preserves pair-subset identity; j=3 preserves triplet-subset identity;
- SinD history-only physical relation weights directly control weighted Johnson quantum mixing;
- each branch includes trainable fixed-particle-number embedding evolution based on the Raj paper's compound/RBS construction;
- per-agent outputs of the two quantum branches are fused locally;
- same Motion-Token / GPT-2 downstream as all matched baselines;
- graph module parameters: 126,535.

This is a paper-inspired task adaptation, not a claim of verbatim reproduction of the full paper's final cross-register joint mixer.

Primary matched baseline:

**multi-j JohnsonGIN**

- same j=2+j=3 subset lift;
- same downstream GPT-2 / tokenizer / loss / training exposure;
- graph module parameters: 179,201;
- no deliberate parameter weakening.

## 3. Why 20 epochs were preregistered

At 12 epochs the weighted multi-j quantum model still trailed JohnsonGIN by roughly 1–2%, while both learning curves were improving. `ROUND4_RAJ_CONVERGENCE_PREREG_20260919.md` therefore froze one convergence check:

- SinD 0 dB;
- 4096 train windows;
- all 1880 validation windows;
- batch 32;
- 20 epochs;
- no architecture / LR / loss / width / token / data change;
- paired fresh initialization;
- test closed.

Gate: Quantum must beat JohnsonGIN on **both ADE and FDE**.

## 4. Seed2026 result — gate passed

| Model | ADE | FDE | J=ADE+0.5FDE |
|---|---:|---:|---:|
| Weighted multi-j Quantum | **0.516458** | **1.102781** | **1.067849** |
| multi-j JohnsonGIN | 0.525952 | 1.129782 | 1.090843 |

Quantum relative improvement over JohnsonGIN:

- ADE: **+1.805%**
- FDE: **+2.390%**
- J: **+2.108%**

Quantum best epoch = 19; JohnsonGIN best epoch = 20. Both completed the same 20-epoch exposure.

## 5. Other paper-native baselines — seed2026 / 20 epochs

| Model | ADE | FDE | J |
|---|---:|---:|---:|
| Weighted multi-j Quantum | **0.516458** | **1.102781** | **1.067849** |
| JohnsonGIN | 0.525952 | 1.129782 | 1.090843 |
| PPGN-style higher-order | 0.579594 | 1.151496 | 1.155342 |
| GIN | 0.590572 | 1.191985 | 1.186565 |

Quantum relative improvement:

vs PPGN:
- ADE +10.893%
- FDE +4.231%
- J +7.573%

vs GIN:
- ADE +12.550%
- FDE +7.484%
- J +10.005%

Therefore JohnsonGIN is the hardest paper-native baseline in this current SinD adaptation and remains the main matched comparator.

## 6. Seed2027 independent paired confirmation — gate passed again

The confirmation protocol was preregistered in `ROUND4_RAJ_SEED2027_CONFIRMATION_PREREG_20260919.md` before the result was complete.

| Model | ADE | FDE | J |
|---|---:|---:|---:|
| Weighted multi-j Quantum | **0.530895** | **1.148234** | **1.105012** |
| multi-j JohnsonGIN | 0.550423 | 1.189891 | 1.145368 |

Quantum relative improvement over JohnsonGIN:

- ADE: **+3.548%**
- FDE: **+3.501%**
- J: **+3.523%**

Again, Quantum wins both primary metrics.

## 7. Two-seed summary

Mean best-validation metrics across seed2026 and seed2027:

| Model | mean ADE | mean FDE | mean J |
|---|---:|---:|---:|
| Weighted multi-j Quantum | **0.523676** | **1.125508** | **1.086430** |
| multi-j JohnsonGIN | 0.538187 | 1.159837 | 1.118106 |

Relative improvement using the two-seed mean metrics:

- ADE: **+2.696%**
- FDE: **+2.960%**
- J: **+2.833%**

This is a repeated positive paper-native signal, not yet the original ~5% target.

## 8. Representation bottleneck audit

History-only probe target: frozen strong-classical graph interaction residual. Test closed.

Validation R²:
- raw physical pair/triplet representation: ~0.853;
- quantum control angles: ~0.863;
- rich direct quantum observables: ~0.836;
- relation-carrying readout input (~500D): ~0.922;
- compressed quantum interaction 64D: ~0.824;
- local-only 64D: ~0.872;
- final local+quantum 64D: ~0.919.

Interpretation:
- no evidence of catastrophic information loss at the original beta/gamma encoding step;
- relation-carrying quantum representation already contains substantial interaction information;
- the final 64D representation retains most information available in the 500D pre-readout representation when combined with the local branch;
- the main failure of earlier RC-HQGNN is therefore not cleanly attributable to a single encoding/readout bottleneck;
- the large improvement after moving to subset-preserving multi-j representation indicates that **representation identity + graph-conditioned subset quantum evolution** is the more productive direction.

## 9. Scientific interpretation

What is supported:
1. A Raj-inspired higher-order subset representation is substantially better suited to SinD than the earlier RC-HQGNN representation.
2. With sufficient equal training exposure, the current Weighted multi-j Quantum model beats its main matched paper-native JohnsonGIN baseline on both ADE and FDE for seed2026 and seed2027.
3. The same seed2026 Quantum model also beats GIN and PPGN-style paper-native references.
4. Quantum uses fewer graph-module trainable parameters than JohnsonGIN in this implementation.

What is **not** yet supported:
1. No final test-set quantum advantage claim — test remains untouched.
2. No full 15,802-train-window confirmation yet; current evidence uses 4096 train windows per seed.
3. No five-SNR result yet.
4. No claim that the exact Raj paper architecture itself has been reproduced verbatim; the current method is an ICCT adaptation inspired by its subset / graph-conditioned / fixed-particle-number design.
5. No claim of universal classical superiority reversal; ICCT historical strong classical references remain secondary results and must be reported honestly.
6. The two-seed mean advantage is around 2.7–3.0%, still below the original ~5% ambition.

## 10. Current decision

Do **not** abandon the Raj route now.

The preregistered stop condition for Raj was “if Quantum cannot beat JohnsonGIN on both ADE and FDE after the fixed convergence check, stop and switch to SQM-GNN.” That condition is not met: Quantum passes in both seed2026 and seed2027.

Next work should be confirmation rather than another architecture search:
- freeze Weighted multi-j Quantum as the current Round4 main QGNN candidate;
- preserve JohnsonGIN as the primary matched baseline;
- preserve GIN / PPGN as additional paper-native baselines;
- keep prior ICCT Adaptive Classical as a secondary strong reference;
- next confirmation should increase evidence strength (additional seed and/or full-train paired run) before five-SNR / test use.

Machine-readable summary:
`reports/qgnn/ROUND4_RAJ_TWO_SEED_RESULTS_20260919.json`
