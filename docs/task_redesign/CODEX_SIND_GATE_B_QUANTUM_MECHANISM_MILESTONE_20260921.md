# SinD Gate B+ — Quantum Mechanism Milestone

Date: 2026-09-21
Scope: development validation only; test was not accessed.

## Fair comparison contract

The primary scoreboard is fixed to models with the same `N<=8` information budget:

1. Raj QGNN;
2. Strong Graph;
3. Matched Johnson;
4. Large Johnson.

Self and Pool remain task-necessity and interaction-complexity controls. AllGraph is retained only as a
neighborhood-budget/task-definition ablation and is not a primary Raj performance hurdle.

All primary runs below use the same SinD-IC4 data and split, target definition, 2 s history, 4 s future,
input state, 36,643-sample training population, fixed batch-order seed `20260921`, EMA `0.999`, FDE loss
weight `0.75`, maximum 50 epochs, and validation-IC4-J patience 8. Seeds 2026 and 2027 are development
seeds; no confirmation or test claim is made.

ADE and FDE are co-primary evaluation metrics. A model is called better only when both are lower under
the paired seed comparison. J (`ADE + 0.5 * FDE`) is a predeclared auxiliary scalar used only to select
one validation checkpoint and drive patience; it is not treated as independent evidence or the headline
paper result. Any ADE/FDE trade-off must be reported as a trade-off regardless of J.

## Mechanism diagnosis before the change

The original Full-Multiscale Raj had three reproducible problems:

- the j2/j3 quantum latents were low-rank and shared a dominant subspace;
- j3 received weaker gradients;
- Raj beat Strong Graph at 0.5–1.5 s, crossed near 2 s, and lost increasingly at 3–4 s.

This localized the main bottleneck to the quantum-latent-to-long-horizon interface rather than to the
absence of useful quantum interaction information.

## Quantum-side change

`horizon_multiscale` preserves the Raj j2 and j3 interaction cores and adds only:

- branch-specific normalization and projection;
- a horizon-conditioned j2/j3 gate;
- a horizon-specific quantum residual into the trajectory decoder interface.

The change has 479,501 parameters. Matched Johnson uses the identical multi-order interface and decoder
with only the interaction operator replaced; Large Johnson widens the classical subset hidden state to
128 and has 753,863 parameters.

## Primary development results

| Model | Seed | ADE | FDE | J | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|---:|
| Strong Graph N<=8 | 2026 | 1.043666 | 2.707381 | 2.397356 | 23 | 31 |
| Strong Graph N<=8 | 2027 | 1.028331 | 2.664979 | 2.360821 | 22 | 30 |
| Matched Johnson N<=8 | 2026 | 1.081756 | 2.900521 | 2.532016 | 22 | 30 |
| Matched Johnson N<=8 | 2027 | 1.067938 | 2.886683 | 2.511280 | 25 | 33 |
| Large Johnson N<=8 | 2026 | 1.087370 | 2.944943 | 2.559841 | 13 | 21 |
| Large Johnson N<=8 | 2027 | 1.071024 | 2.906961 | 2.524505 | 25 | 33 |
| Raj horizon-multiscale | 2026 | 1.069516 | 2.874257 | 2.506645 | 22 | 30 |
| Raj horizon-multiscale | 2027 | 1.065716 | 2.873779 | 2.502606 | 21 | 29 |

Two-seed mean ADE / FDE:

- Raj: `1.067616 / 2.874018`;
- Matched Johnson: `1.074847 / 2.893602`;
- Large Johnson: `1.079197 / 2.925952`;
- Strong Graph: `1.035999 / 2.686180`.

Auxiliary two-seed mean J (sample standard deviation):

- Raj: `2.504625 (0.002856)`;
- Matched Johnson: `2.521648 (0.014663)`;
- Large Johnson: `2.542173 (0.024987)`;
- Strong Graph: `2.379089 (0.025834)`.

Raj beats Matched Johnson on both primary metrics in both seeds: ADE improves by 1.131% / 0.208% and
FDE improves by 0.905% / 0.447% for seeds 2026 / 2027. Raj also beats Large Johnson on both primary
metrics in both seeds: ADE improves by 1.642% / 0.496% and FDE by 2.400% / 1.141%, despite Large
Johnson having substantially more capacity.

Raj does not beat Strong Graph. Its ADE is 2.477% / 3.636% worse and its FDE is 6.164% / 7.835% worse
for seeds 2026 / 2027. The larger FDE deficit is the central remaining performance problem.

## Updated mechanism evidence

- Horizon-wise ADE and FDE both show that Raj's relative advantage is concentrated at short horizons and
  reverses at longer horizons. The remaining failure is specifically long-horizon propagation; J is only
  a compact locator for the approximate 2–2.5 s crossover.
- The branch-specific projections reduce j2/j3 linear CKA from 0.813 to 0.096 in seed 2026 and from
  0.572 to 0.225 in seed 2027. The new readout therefore removes substantial multi-order redundancy.
- Inference ablations show both branches are indispensable: removing either branch raises 4 s J from
  about 2.50 to 3.13–3.88; removing both raises it to 4.53–4.63.
- The learned softmax gates reverse their j2/j3 horizon roles across seeds. At 4 s, seed 2026 uses
  approximately 99.8% j2 while seed 2027 uses approximately 99.6% j3. Branch-projection gradient ratios
  reverse with them. This is the next stability bottleneck.

## Non-competitive gate ablation

An equal-capacity independent bounded gate was tested to prevent branch suppression. It obtained J
`2.516959 / 2.500763` (mean `2.508861`, sample SD `0.011452`). It still beats Matched and Large Johnson
in both seeds, but is worse in mean and stability than the competitive gate. It is rejected.

The result means that horizon specialization is useful, but unconstrained branch-role swapping is not.
The next mechanism change should preserve specialization and add a more direct quantum-to-long-horizon
trajectory interface; it should not merely increase generic decoder capacity or continue gate-only search.

## Direct long-horizon interface ablations

Two targeted follow-ups tested whether the remaining 3--4 s deficit came from an overly indirect
quantum-to-trajectory interface. Both used paired restart controls and the same development seeds.

The direct quantum trajectory head adds horizon-conditioned j2/j3 displacement corrections to the
shared decoder output. It obtained ADE / FDE `1.067196 / 2.875439` (seed 2026) and
`1.065763 / 2.875184` (seed 2027), versus paired restart-control results
`1.066077 / 2.873306` and `1.064070 / 2.870547`. It is worse on both primary metrics in both seeds and
is rejected.

The quantum kinematic head predicts quantum-conditioned velocity and acceleration corrections and
integrates them over the forecast horizon. It obtained ADE / FDE `1.071361 / 2.887409` (seed 2026)
and `1.069223 / 2.883040` (seed 2027), versus fixed-dropout controls
`1.068811 / 2.875800` and `1.068588 / 2.879544`. It is also worse on both primary metrics in both
seeds and is rejected.

These paired failures narrow the diagnosis: simply exposing projected quantum latents more directly
to the 4 s output does not solve the bottleneck. The next test therefore targets the measured low-rank
latent representation itself rather than adding another output head.

## Effective-rank intervention

A spectral-entropy regularizer was applied directly to the j2, j3, q2, and q3 quantum latents. The
paired control and two predeclared weights used the same fixed batch order, fixed dropout stream,
training population, optimizer recipe, maximum 40 epochs, and patience 8.

| Variant | Seed | ADE | FDE | J | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|---:|
| no rank regularization | 2026 | 1.076179 | 2.876621 | 2.514490 | 26 | 34 |
| no rank regularization | 2027 | 1.076417 | 2.914878 | 2.533856 | 25 | 33 |
| rank weight 0.01 | 2026 | 1.052121 | 2.879113 | 2.491677 | 32 | 40 |
| rank weight 0.01 | 2027 | 1.088109 | 2.936293 | 2.556256 | 23 | 31 |
| rank weight 0.05 | 2026 | 1.047644 | 2.832925 | 2.464107 | 23 | 31 |
| rank weight 0.05 | 2027 | 1.061169 | 2.848614 | 2.485476 | 25 | 33 |

Weight 0.01 is seed-unstable and is rejected. Weight 0.05 lowers both ADE and FDE relative to the
original horizon-multiscale Raj in both seeds: by 2.045% / 1.438% in seed 2026 and by
0.427% / 0.876% in seed 2027. Its two-seed mean ADE / FDE is `1.054407 / 2.840769`, making it the
current Raj candidate.

The intervention changes the measured representation in the intended direction. Across seeds, control
q2 effective rank is `3.074 / 3.025` and q3 rank is `3.002 / 3.274`; weight 0.05 raises them to
`73.171 / 72.880` and `16.512 / 31.480`. Raw j2 rank rises from `3.117 / 3.266` to
`5.237 / 5.082`, while raw j3 rank rises from `3.072 / 3.657` to `9.359 / 6.730`. The performance
gain therefore coincides with direct removal of the diagnosed quantum-latent rank collapse.

This is not yet a complete representation cure. q2 total variance falls from `693.4 / 537.3` in the
controls to `1.83 / 4.94` with weight 0.05, while q3 total variance is `264.8 / 14.6` across the two
seeds. The regularizer reliably flattens the spectrum, but a new scale imbalance remains. The evidence
supports "rank collapse was alleviated and performance improved"; it does not support claiming that all
quantum information loss has been eliminated.

A paired continuation test asked whether rank regularization could repair an already trained low-rank
checkpoint. Rank-finetuned ADE / FDE was `1.069767 / 2.877419` and `1.066398 / 2.871039`, versus
restart controls `1.069441 / 2.873430` and `1.066773 / 2.875989`. Seed 2026 worsens on both metrics;
seed 2027 improves only FDE. Post-hoc rank repair is therefore rejected: the useful intervention must
shape the representation from early training rather than attempt to recover it after collapse.

Rank-regularized Raj still does not beat Strong Graph. Its ADE is 0.381% / 3.193% worse and its FDE is
4.637% / 6.891% worse for seeds 2026 / 2027. Long-horizon utilization remains the dominant gap.

## Node-level multi-order quantum readout

The next intervention moved the bottleneck upstream from the final trajectory head to the interaction
readout itself. `multiscale_quantum_attention` forms target--neighbor messages independently in the j2
and j3 quantum representation spaces, pools each order separately, and fuses them only after the two
quantum interaction summaries have been preserved. It contains no classical cross-agent message-passing
path.

| Quantum readout | Seed | ADE | FDE | J | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|---:|
| sample-conditioned horizon gate | 2026 | 1.088841 | 2.924805 | 2.551244 | 27 | 35 |
| sample-conditioned horizon gate | 2027 | 1.057261 | 2.837701 | 2.476111 | 24 | 32 |
| shared quantum-latent attention | 2026 | 1.048823 | 2.702255 | 2.399950 | 25 | 33 |
| shared quantum-latent attention | 2027 | 1.061257 | 2.772081 | 2.447297 | 20 | 28 |
| order-separated multiscale quantum attention | 2026 | 1.040333 | 2.692310 | 2.386488 | 32 | 40 |
| order-separated multiscale quantum attention | 2027 | 0.999491 | 2.666342 | 2.332662 | 34 | 40 |

The sample-conditioned gate is rejected. Node-level quantum readout is the effective intervention, and
preserving j2/j3 messages separately is more stable than compressing both orders into one shared message.
The order-separated model beats Strong Graph on both ADE and FDE in seed 2026 by 0.319% / 0.557%. In
seed 2027 it improves ADE by 2.805% but misses FDE by `0.001363 m` (0.051%), so this seed remains an
explicit trade-off rather than a win.

Across the two development seeds, order-separated Raj has mean ADE / FDE `1.019912 / 2.679326`, versus
Strong Graph `1.035999 / 2.686180`; the mean improvements are 1.553% / 0.255%. This is the first Raj
candidate to beat Strong Graph on both mean primary metrics, but the per-seed stability rule is not yet
satisfied because of the seed-2027 FDE margin.

The two-seed sample standard deviations make the instability location more specific. Strong Graph has
ADE / FDE standard deviations `0.010843 / 0.029983` (CV `1.047% / 1.116%`), whereas the
order-separated Raj has `0.028880 / 0.018362` (CV `2.832% / 0.685%`). Raj's endpoint error is not
the more seed-sensitive quantity; its average-displacement error is. Because order, dropout stream, and
training population are already fixed across these development seeds, the remaining difference is
primarily initialization sensitivity. This motivates stabilizing the quantum neighbor readout rather than
increasing the FDE loss weight again.

A direct rank-loss transfer to the node-level attention readouts did not preserve the earlier gain:

| Readout + rank weight 0.05 | Seed | ADE | FDE | J | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|---:|
| shared quantum-latent attention | 2026 | 1.033373 | 2.661235 | 2.363990 | 26 | 34 |
| shared quantum-latent attention | 2027 | 1.018693 | 2.684056 | 2.360721 | 21 | 29 |
| order-separated quantum attention | 2026 | 1.037490 | 2.700538 | 2.387759 | 22 | 30 |
| order-separated quantum attention | 2027 | 1.025509 | 2.675981 | 2.363499 | 24 | 32 |

The shared version beats Strong Graph on both primary metrics only in seed 2026; its seed-2027 FDE is
0.716% worse. The order-separated version loses FDE in both seeds. Rank regularization is therefore not
a portable cure for readout instability and is rejected for the current attention interface. The
unregularized order-separated model remains the best worst-seed candidate.

A dual readout then combined the shared cross-order quantum message with the two order-separated
quantum messages. It produced `1.030729 / 2.623526` in seed 2026, beating Strong Graph on ADE/FDE by
1.240% / 3.097%, but produced `1.032797 / 2.693998` in seed 2027, losing by 0.434% / 1.089%.
This is a larger seed-dependent role reversal than the unregularized order-separated model. It shows
that the shared quantum view can materially improve endpoint prediction, but competing shared and
order-specific readouts are initialization-sensitive. The dual readout is rejected by the worst-seed
rule; its seed-2026 result is not treated as a method win.

Top-5 EMA checkpoint averaging was tested on the unregularized order-separated model to distinguish
within-run checkpoint noise from initialization sensitivity. The averaged epochs were
`25,26,23,24,22` for seed 2026 and `32,36,35,31,29` for seed 2027. Averaged ADE/FDE was
`1.052034 / 2.728177` and `0.999956 / 2.616751`, respectively. Averaging makes seed 2027 a clear
two-metric win but makes seed 2026 lose both metrics, increasing rather than reducing the cross-seed
role reversal. Checkpoint averaging is rejected as the stability fix. The remaining instability is
between optimization basins, not merely selection noise among neighboring late checkpoints.

A fixed 50/50 residual of quantum-message mean pooling and attention pooling was then tested. It
produced ADE/FDE `1.050153 / 2.716184` in seed 2026 and `1.029584 / 2.729259` in seed 2027.
Although the ADE spread is smaller, FDE degrades in both seeds and the model does not beat Strong Graph.
The fixed residual is rejected: uniform quantum-message averaging suppresses useful selective neighbor
information rather than stabilizing it.

The subsequent attention diagnostic isolates a cross-order role reversal. In seed 2026, normalized j2/j3
attention entropy is `0.699 / 0.698`; replacing j2 or j3 attention with mean pooling raises J from
`2.386` to `2.510 / 2.573`, so j3 selectivity is more important. In seed 2027, entropy becomes
`0.616 / 0.854`; the same ablations raise J from `2.333` to `2.595 / 2.404`, so j2 becomes dominant
while j3 becomes diffuse. The pooled j2/j3 representations remain complementary (linear CKA
`0.255 / 0.204`), ruling out simple branch redundancy. The next intervention therefore targets
cross-order attention-role stability rather than representation removal or fixed pooling.

### Attention-role balance control

An auxiliary loss that directly matched the normalized j2 and j3 attention entropies was tested at
weights 0.05 and 0.20. This was a deliberately narrow test of whether the cross-seed role reversal could
be removed by making the two order-specific selectors similarly concentrated.

| Balance weight | Seed | ADE | FDE | J | Best epoch | Stop epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 2026 | 1.050450 | 2.706080 | 2.403490 | 24 | 32 |
| 0.05 | 2027 | 1.049806 | 2.779758 | 2.439684 | 29 | 37 |
| 0.20 | 2026 | 1.078048 | 2.778869 | 2.467482 | 22 | 30 |
| 0.20 | 2027 | 1.016667 | 2.661875 | 2.347605 | 34 | 40 |

Weight 0.05 degrades both seeds. Weight 0.20 makes seed 2027 a two-metric win over Strong Graph but
substantially degrades seed 2026, increasing rather than reducing initialization sensitivity. The
forward-distribution constraint is rejected. Complementary orders are allowed to specialize differently;
subsequent stability interventions act on quantum optimization rather than forcing matched attention
entropy.

## Post-projection normalization control

A paired control tested whether the scale imbalance diagnosed above could be removed by applying
LayerNorm after the j2 and j3 quantum projections. All runs retained the fixed data order, fixed dropout
stream, 40-epoch cap, and patience 8.

| Variant | Seed | ADE | FDE | J | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|---:|
| ranknorm, no rank loss | 2026 | 1.074997 | 2.884883 | 2.517439 | 26 | 34 |
| ranknorm, no rank loss | 2027 | 1.072136 | 2.872220 | 2.508247 | 23 | 31 |
| ranknorm + rank weight 0.05 | 2026 | 1.053591 | 2.859352 | 2.483267 | 22 | 30 |
| ranknorm + rank weight 0.05 | 2027 | 1.051535 | 2.811619 | 2.457345 | 22 | 30 |

RankNorm reduces the between-seed spread but is substantially worse than the order-separated quantum
attention model on both primary metrics. It is rejected. The result narrows the mechanism diagnosis:
the quantum branches do suffer from scale imbalance, but forcibly erasing their amplitude information
after projection also removes useful signal. Subsequent variants therefore preserve amplitude and act on
branch-specific readout or spectral utilization instead.

## Decision

Current layer status:

- Raj < Matched Johnson: **PASS on both development seeds**;
- Raj < Large Johnson: **PASS on both development seeds**;
- Raj < Strong Graph N<=8: **FAIL**;
- frozen independent confirmation: **not started**;
- test: **sealed**.

The order-separated multiscale quantum-attention Raj is the current best quantum-mechanism candidate, but it is not yet the final
paper method and no top-classical claim is justified.
