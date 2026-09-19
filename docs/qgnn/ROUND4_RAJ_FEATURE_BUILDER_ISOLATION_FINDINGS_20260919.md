# Round4 Raj Feature-Builder Isolation Findings — 2026-09-19

## 1. Decision

The preregistered feature-builder isolation **passes strong feature-equivalence**.

Changing only `ContinuousSubsetLoader` to `SubsetFeatureBuilder` does not materially change the weighted/no-local Johnson result.

Therefore the feature-definition difference is **not a credible primary explanation** for the ~8.34% J degradation of the full paper-math Quantum relative to the successful intermediate Quantum.

## 2. Result

| Model | ADE | FDE | J | Best epoch | Graph params |
|---|---:|---:|---:|---:|---:|
| SubsetFeatureBuilder weighted Johnson | 0.765164297 | 1.573716227 | 1.552022410 | 19 | 258,950 |
| ContinuousSubsetLoader weighted Johnson | 0.768920247 | 1.568731510 | 1.553286002 | 19 | 267,354 |

SubsetFeatureBuilder relative change:
- ADE: +0.488%
- FDE: -0.318%
- J: +0.081%

The metrics split direction, while J differs by only 0.081%.

Preregistered gates:
- |J difference| <= 1%: **PASS**
- |ADE| <=2%, |FDE| <=2%, |J| <=1%: **PASS**
- feature-material J degradation >3%: **FAIL**

## 3. Integrity

Both arms use:
- SinD, 0 dB, seed2026
- deterministic 4096 train windows
- all 1880 validation windows
- 20 epochs, batch32, LR 3e-4
- single j=3, depth3
- graph-conditioned WeightedJohnsonMix
- hidden128, same classical update/readout family
- no local-history residual
- same fixed GPT-2 / Tokenizer / trajectory+token loss
- identical train-index hash
- identical downstream initialization hash
- test closed.

The new model completed 20/20 epochs without OOM, deadlock or non-finite failure.

## 4. Feature difference was small by construction

`SubsetFeatureBuilder` and `ContinuousSubsetLoader` already share:
- subset own-history mean/std
- 16D physical-edge mean/std
- risk mean/std.

The Continuous loader adds risk max/min plus mean distance/closing terms.

The latter distance/closing means are duplicates of components already present in the 16D edge mean. Thus only risk extrema are genuinely new information.

The experiment confirms that these feature additions are not the source of the major transfer gap.

## 5. Mechanism state after this result

Eliminated as primary explanations:
- loss of probability mass in final M: readable sector retains ~99.4%;
- final M alone: inference/graph-feature sensitivity is tiny;
- probability-weighted readout alone: inference sensitivity is tiny;
- feature-builder difference: J changes only ~0.08%.

Still strongly implicated:
1. **global two-register state construction / subset probability allocation**;
2. **controlled orthogonal Householder re-upload** versus row-local additive re-upload.

These are now the next single-factor target.

## 6. Next step

Build an isolated row-local/additive loader ablation inside the otherwise fixed paper-math core:
- same `SubsetFeatureBuilder`
- same graph-conditioned adjacency
- same compound W
- same final joint M
- same conditional 1-RDM readout
- change only state construction / data re-upload semantics.

This diagnostic must be preregistered before performance is read.

No full-train, five-SNR or test-set experiment is authorised.
