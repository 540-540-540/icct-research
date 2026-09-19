# Round4 Raj Intermediate Weighted-Matched Diagnostic Findings — 2026-09-19

## 1. Decision

The fresh current-source 20-epoch intermediate Raj Quantum run passes the preregistered **Quantum-positive** gate against the frozen weighted/no-local classical comparator.

Preregistration commit: `37bf941`.

This is a mechanism result, not a replacement of the accepted multi-j main model.

## 2. Paired result

| Model | ADE | FDE | J | Best epoch | Graph params |
|---|---:|---:|---:|---:|---:|
| Intermediate Raj Quantum (`raj_paper_quantum`) | **0.754217271** | **1.542620952** | **1.525527747** | 9 | **48,195** |
| Weighted/no-local Johnson (`raj_paper_johnson`) | 0.768920247 | 1.568731510 | 1.553286002 | 19 | 267,354 |

Quantum relative gain:
- ADE: **+1.912%**
- FDE: **+1.664%**
- J: **+1.787%**

Preregistered gates:
- supporting ADE/FDE near-match: PASS;
- Quantum-positive (ADE and FDE both lower, J gain >=1%): **PASS**;
- clear-negative: FAIL.

The classical graph module has about **5.55x** as many trainable graph parameters as the intermediate Quantum module.

This parameter difference must be reported; it strengthens the parameter-efficiency observation but prevents a simplistic capacity-matched substrate claim.

## 3. Integrity

Both runs:
- SinD, 0 dB
- seed2026
- deterministic 4096 train windows
- all 1880 validation windows
- batch32
- LR 3e-4
- depth3
- single j=3
- same fixed GPT-2 / Tokenizer / trajectory+token loss
- identical train-index SHA256
- identical downstream trainable-initialization SHA256
- test closed
- 20/20 epochs completed
- no OOM / deadlock / non-finite failure.

## 4. Comparison with the full paper-math prototype

Frozen full paper-math Quantum:
- ADE 0.810536438
- FDE 1.707538617
- J 1.664305747
- best epoch 1
- graph params 56,763

The simpler intermediate Quantum is better than the full paper-math prototype by:
- ADE: **6.948%**
- FDE: **9.658%**
- J: **8.338%**

This is the central mechanism finding.

The weighted quantum subset dynamics and compound embedding evolution are therefore **not sufficient to explain the full-paper prototype failure**. A simpler implementation containing those ingredients can beat the weighted/no-local classical comparator under the same SinD protocol.

## 5. What differs between intermediate and full paper-math Quantum

Intermediate `raj_paper_quantum`:
- `ContinuousSubsetLoader`
- per-subset normalized amplitudes
- weighted Johnson quantum mixing
- compound/Cayley W
- additive layerwise data re-upload + row renormalization
- conditional 1-RDM
- incidence readout
- no final cross-register M
- no local-history residual.

Full paper-math `raj_paper_math_quantum`:
- `SubsetFeatureBuilder`
- one globally normalized factored quantum state
- graph-conditioned Givens adjacency
- compound/Cayley W
- controlled orthogonal Householder re-upload
- final total-weight joint-register M
- conditional projection back to the (j,k) sector
- probability-weighted 1-RDM incidence readout
- no local-history residual.

Thus the degradation enters somewhere in this transfer package, not in the generic idea of weighted quantum message passing itself.

## 6. Joint-mixer / projection diagnostics

### 6.1 Sector survival

On 64 validation scenes using the full paper-math best checkpoint:
- pre-joint norm^2 mean: ~1.0000005
- joint norm^2 mean: ~1.0000005
- readable (j=3,k=3) factor-sector fraction mean: **0.99407**
- minimum: **0.99292**
- maximum: **0.99591**

Therefore the final M does **not** destroy performance by moving most probability mass into unreadable sectors.

### 6.2 Subset concentration

At the full paper-math best checkpoint, per active vehicle:
- mean available triplets: **19.31**
- mean effective probability-weighted triplets: **9.24**
- effective fraction: **48.51%**
- mean maximum single-triplet weight: **28.16%**
- observed maximum: **69.73%**

At random initialization, the effective fraction is already only **44.10%** and the mean maximum weight is **31.39%**.

Therefore this concentration is largely structural to the global-state/projection construction rather than something created late by training.

It remains a mechanism clue, not yet a proven causal failure.

### 6.3 Counterfactual inference sensitivity

On the first 128 validation scenes at the full paper-math best checkpoint:
- trained-full J: 1.84627884
- set M angles to zero: 1.84629090
- replace probability-weighted readout with uniform incidence readout: 1.84633206
- both changes: 1.84634558

These are inference-time counterfactuals without retraining, so they are not performance ablations.

However the changes are tiny.

Graph-feature sensitivity on 64 scenes:
- graph feature norm: ~8.005
- zero-M feature delta norm: ~0.094, cosine ~0.99993
- uniform-readout delta norm: ~0.235, cosine ~0.99941

This makes it unlikely that final M or probability weighting **alone** explains the 8.34% performance gap.

## 7. Current leading hypothesis

The evidence now points upstream, primarily toward:
- global-state construction versus per-subset state normalization;
- controlled orthogonal Householder re-upload versus additive row-local data re-upload;
- and secondarily the small feature-definition difference between `SubsetFeatureBuilder` and `ContinuousSubsetLoader`.

This is a hypothesis to be tested by single-factor ablations.

## 8. Scientific interpretation

Supported:
1. The Raj official WL climb is reproducible.
2. Same-j JohnsonGIN shares the WL climb; the paper does not promise universal Q > Johnson predictive error.
3. The current accepted multi-j model is not a full-paper reproduction.
4. The full paper-math SinD adaptation is a poor transfer.
5. A simpler current-source Raj intermediate Quantum **does** beat the weighted/no-local classical comparator on both ADE and FDE.
6. Therefore blindly increasing paper fidelity is not a valid optimisation strategy for SinD.

Not supported:
- that the quantum substrate universally beats classical models;
- that the final M is harmful by itself;
- that projection concentration is causal;
- that the full-paper mathematical architecture is wrong;
- that the paper's graph-level architecture should transfer unchanged to per-agent trajectory prediction.

## 9. Next step

Freeze these results before any new variant.

Next diagnostic priority:
1. isolate feature-builder difference;
2. isolate global-state / controlled re-upload versus row-local additive re-upload;
3. keep final M/readout fixed where possible;
4. use small preregistered mechanism runs before another full 4096 comparison.

The accepted main model remains `raj_weighted_multij_quantum`; no full-train, five-SNR or test-set experiment is authorised here.

