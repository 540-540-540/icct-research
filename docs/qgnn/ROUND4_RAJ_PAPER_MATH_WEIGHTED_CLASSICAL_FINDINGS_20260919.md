# Round4 Raj Paper-Math Weighted-Classical Diagnostic Findings — 2026-09-19

## 1. Decision

The preregistered weighted/no-local classical diagnostic completed successfully.

Frozen preregistration:
`docs/qgnn/ROUND4_RAJ_PAPER_MATH_WEIGHTED_CLASSICAL_PREREG_20260919.md`
commit: `7189d76`.

The result shows that the very large gap in the first paper-math transfer was **mostly, but not fully, explained by the task-interface/local-history advantage of the original Johnson comparator**.

It does not establish a substrate-only classical advantage.

## 2. Result

| Model | ADE | FDE | J = ADE + 0.5 FDE | Best epoch | Graph params |
|---|---:|---:|---:|---:|---:|
| Full paper-math Quantum, j=3 | 0.810536438 | 1.707538617 | 1.664305747 | 1 | 56,763 |
| Weighted/no-local Johnson, j=3 | 0.768920247 | 1.568731510 | 1.553286002 | 19 | 267,354 |
| Original `raj_johnson`, j=3 | 0.550160027 | 1.155770682 | 1.128045368 | 20 | 81,248 |

Quantum relative gain versus weighted/no-local Johnson:
- ADE: **-5.412%**
- FDE: **-8.848%**
- J: **-7.147%**

Therefore the weighted/no-local classical comparator still outperforms the frozen full paper-math Quantum.

## 3. What removing the local/interface support changed

Relative to the original `raj_johnson`, the weighted/no-local Johnson is worse by:
- ADE: **+39.763%**
- FDE: **+35.730%**
- J: **+37.697%**

This is a major effect.

The original first diagnostic showed Quantum roughly 47.5% worse in J than `raj_johnson`.
After removing the local-history residual / changing to the weighted no-local paper-style comparator, the remaining J gap is only about 7.15%.

Thus a large fraction of the apparent failure was caused by **task-interface support**, not by a clean quantum-versus-classical substrate difference.

## 4. Integrity

Both frozen Quantum and weighted/no-local classical:
- SinD, 0 dB
- seed2026
- deterministic 4096 train windows
- all 1880 validation windows
- 20 epochs
- batch32
- LR 3e-4
- single j=3
- same fixed GPT-2 / Tokenizer / trajectory+token objective
- identical train-index SHA256
- identical downstream trainable-initialization SHA256
- test closed
- completed 20/20 epochs

No OOM / deadlock / non-finite failure occurred.

## 5. Remaining confounds

The remaining ~7% gap must **not** be called a classical-substrate win.

### 5.1 Parameter capacity is not matched

Graph parameters:
- full paper-math Quantum: 56,763
- weighted/no-local Johnson: 267,354

The classical diagnostic has about **4.71x** as many graph-module trainable parameters.

This is acceptable for the preregistered causal diagnostic, but not for isolating the substrate effect.

### 5.2 Loader / representation path is not identical

The weighted/no-local Johnson uses `ContinuousSubsetLoader`.

The full paper-math Quantum uses:
- `SubsetFeatureBuilder`
- globally normalized factored state
- controlled Householder re-upload
- compound W
- final joint-register M
- conditional projection
- 1-RDM incidence readout

Thus the comparison changes several blocks simultaneously.

### 5.3 Per-agent readout is an ICCT adaptation

Raj j>=2 paper tasks are graph-level. SinD needs one output per vehicle.
The incidence subset-to-agent interface is therefore outside the original paper task definition.

## 6. Learning behavior

The full paper-math Quantum is best at epoch1 and validation degrades while training loss keeps falling.

Weighted/no-local Johnson continues improving until epoch19.

This strengthens the interpretation that the frozen full paper-math transfer has a representation/generalization mismatch on SinD, rather than simply requiring more optimization steps.

## 7. Important historical clue

A historical `raj_paper_quantum` run on the same deterministic seed2026 / 4096-window subset obtained:
- ADE 0.765478402
- FDE 1.572725152
- J 1.551840978
- best epoch 9 of 12

This is numerically almost tied with the new weighted/no-local Johnson J=1.553286.

However this historical run used an older `raj_paper.py` source hash and only 12 epochs. It is **contextual evidence only**, not a valid paired causal conclusion.

The clue is nevertheless important: the simpler Raj intermediate, which has weighted Johnson mixing + compound evolution + additive re-upload + conditional RDM but **no final joint-register M**, transferred much better than the new full paper-math prototype.

## 8. Next mechanism diagnostic

The next highest-value experiment is a fresh, preregistered 20-epoch run of the current `raj_paper_quantum` against the already frozen weighted/no-local `raj_paper_johnson` result under the same seed2026 / 4096 protocol.

Purpose:
- test whether the simpler intermediate Raj quantum route still approximately matches the weighted/no-local classical comparator;
- determine whether the severe degradation appears specifically when adding the new paper-math loader/global-state/final-M/conditional-projection path.

This remains a mechanism diagnostic. It does not replace the primary paper-native baseline policy.

A later capacity-matched classical diagnostic may be used to isolate parameter-count effects, but it must be labelled an ablation and must not replace the unweakened classical baseline in final reporting.

No full-train, five-SNR or test-set experiment is authorised by this result.

