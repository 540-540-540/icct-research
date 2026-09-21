# SinD Gate B+ — Literature-Guided Method Adoption Note

Date: 2026-09-22
Scope: development validation only; test remains sealed.

## Problem matched to the literature

The current Raj candidate preserves separate j2 and j3 quantum interaction representations and wins
the two-seed mean ADE/FDE against Strong Graph, but its branch roles reverse across initialization seeds.
The literature search therefore targets multi-order interaction preservation and trainability, not generic
trajectory-model capacity.

## Adopt, adapt, or reject

| Source | Transferable idea | ICCT decision |
|---|---|---|
| [GroupNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_GroupNet_Multiscale_Hypergraph_Neural_Networks_for_Trajectory_Prediction_With_Relational_CVPR_2022_paper.html) | Preserve pair-wise and group-wise relations at multiple scales and learn their representations end to end. | **Adopted principle.** Keep j2 and j3 separate through the quantum neighbor readout; do not prematurely average them. Do not import its CVAE or future-conditioned posterior because the current task is deterministic, strict history-only forecasting. |
| [HighGraph, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.html) | Aggregate relations from multiple social depths while retaining degree-specific features before combination. | **Adopted principle.** Treat multi-order structure as the mechanism, while retaining the existing SinD physical-history graph rather than copying its pedestrian collision kernel. |
| [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | Balance learning through gradient magnitudes and relative training rates rather than by forcing forward representations to look alike. | **Conditional next step.** Use only if branch-specific gradient diagnostics demonstrate persistent imbalance. The current model has one shared trajectory objective, so a literal multitask implementation would require auxiliary branch objectives and is not yet justified. |
| [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Project conflicting task gradients instead of changing the forward representation. | **Conditional next step.** Applicable only after defining scientifically meaningful j2/j3 auxiliary objectives and measuring negative gradient cosine. Not adopted speculatively. |
| [McClean et al., Nature Communications 2018](https://www.nature.com/articles/s41467-018-07090-4) | Deep randomly initialized parameterized quantum circuits can have vanishing-gradient regions; structured starts deserve explicit testing. | **Diagnostic motivation.** Our reduced D=6 compound evolution is much smaller than the asymptotic setting, so no barren-plateau claim is made. |
| [Grant et al., Quantum 2019](https://quantum-journal.org/papers/q-2019-12-09-214/) | Initialize trainable quantum blocks near an identity-effective circuit to improve the initial optimization landscape. | **Adapted now.** Test a fixed, small, nonzero near-identity initialization only for the compound quantum evolutions. Exact zero angles were rejected in preflight because the zero-initialized trajectory head blocks upstream gradients on update 1; after one decoder update both random and near-identity quantum angles receive nonzero gradients. |
| [Skolik et al., Quantum Machine Intelligence 2021](https://link.springer.com/article/10.1007/s42484-020-00036-4) | Grow or unfreeze quantum layers progressively so fewer quantum parameters compete at once. | **Reserved follow-up.** Use only if near-identity initialization does not stabilize paired seeds; it is more invasive and changes training dynamics without changing model capacity. |
| [TNT, CoRL 2020](https://proceedings.mlr.press/v155/zhao21b.html) | Predict target states first, then generate trajectories conditioned on those targets. | **Reserved long-horizon follow-up.** If bounded initialization tests fail, add one quantum-conditioned endpoint and feed that prediction to the deterministic trajectory decoder. Do not import TNT's map, multimodal sampling, or target-selection stack. |

## Current experimental decision

The first literature-guided experiment is the smallest quantum-specific intervention:

1. retain the order-separated j2/j3 quantum attention architecture;
2. keep all data, split, N<=8 budget, loss, optimizer, EMA, patience, and decoder unchanged;
3. replace only random compound-evolution angles with a deterministic near-identity pattern in
   `[-0.01, 0.01]`;
4. compare paired development seeds 2026 and 2027 on ADE and FDE separately.

This experiment tests whether quantum-evolution initialization is the source of the observed basin and
branch-role instability. It does not claim that the current model has a barren plateau, and it does not
use literature as justification for test-set access or baseline weakening.

Scale 0.01 reduced cross-seed ADE/FDE spread but degraded both means and did not produce paired
two-metric wins. It is rejected as the final recipe. Scales 0.05 and 0.10 form the bounded final test of
the initialization hypothesis; failure of both closes this branch and activates the TNT-inspired
quantum-conditioned endpoint experiment.

Both scales failed the paired-seed rule. Scale 0.05 yielded a very strong seed-2027 result but degraded
seed 2026 and increased ADE/FDE dispersion; scale 0.10 degraded both means. The initialization branch
is closed. The next experiment implements the predeclared TNT transfer: a history-only quantum
interaction target head predicts the 4 s endpoint, and its prediction conditions the shared trajectory
decoder. The future endpoint appears only in the auxiliary supervised loss and never enters history.
