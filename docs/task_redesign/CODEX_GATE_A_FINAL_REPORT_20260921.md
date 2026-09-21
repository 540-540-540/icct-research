# Gate A final report: strict-causal task and interaction necessity

## Decision

**PASS** for Gate A (Task / Interaction Necessity).

This decision means that history-side neighbor information has reproducible predictive value beyond a strong target-only TCN and a target-only capacity control. It does **not** mean that a GNN is irreplaceable, that the interaction proxy is causal traffic truth, or that Raj QGNN has passed Gate B.

## Source and causal contract

The available SinD histories are not used because their positions/states are tied to an RTS-smoothed release pipeline and no pre-smoothing observations are available in the project. Automatum is deprecated and was not revisited.

The fallback is the repository's Lankershim trajectory table (`SHA-256 ec10b87...47d180`). The official FHWA description states that synchronized video was transcribed into per-vehicle locations every 0.1 s. In this benchmark, those published per-frame `Local_X/Y` values are the observation boundary. All downstream construction is prefix-only:

- position history uses only rows timestamped `<=t0`;
- velocity is an OLS estimate from the current and at most four preceding contiguous rows;
- 0 dB three-BS sensing is same-frame only, with no temporal smoother;
- membership, interaction edges, neighbor order, ego frame, and strata use only history-side sensed states;
- rows after `t0` can populate only `future_xy` and `future_mask`;
- history and labels are constrained to their own split, and masked future entries are exactly zero;
- no test cache is materialized and the loader rejects `test`.

The guarantee is exact with respect to the published per-frame coordinate table and all repository-side processing. The historical NGVIDEO transcription implementation is not distributed here, so no stronger claim is made about undocumented processing inside the data publisher's original extraction software.

## Dataset

User-selected split: 80% train / 10% validation / 10% sealed test, with 6 s gaps and conservative cross-record `source_group` purge.

| Split | Samples | Origins | IC targets | 4 s endpoint visible |
|---|---:|---:|---:|---:|
| train | 114,740 | 1,642 | 62,033 | 107,186 |
| validation | 11,695 | 193 | 5,564 | 10,940 |

There are at most 53 nodes inside the 50 m history-defined context. Full2/IC2 and Full4/IC4 are evaluation views of the same samples and frozen membership; future completeness never controls inclusion.

## Matched controls

All five models share a node TCN, ego transform, CV residual anchor, 40-step decoder, masked `ADE + 0.5 FDE` objective, optimizer budget, and validation IC4 origin-macro checkpoint rule.

| Model | Information/control role | Parameters |
|---|---|---:|
| Self | target only | 244,578 |
| Own | target-only capacity control | 302,178 |
| Pool | all-neighbor context pooling | 302,690 |
| Graph | target + 7, edge-aware messages | 319,203 |
| All Graph | every <=50 m neighbor, same parameters | 319,203 |

## Two-seed validation results

### IC4 (primary)

| Seed | Model | ADE m | FDE m | J m |
|---:|---|---:|---:|---:|
| 2026 | Self | 2.5562 | 5.4925 | 5.3024 |
| 2026 | Own | 2.5792 | 5.5433 | 5.3509 |
| 2026 | Pool | 2.1152 | 4.3516 | 4.2911 |
| 2026 | Graph | 2.0699 | 4.3394 | 4.2396 |
| 2026 | All Graph | **1.9286** | **3.9483** | **3.9027** |
| 2027 | Self | 2.5379 | 5.4368 | 5.2563 |
| 2027 | Own | 2.5283 | 5.3934 | 5.2250 |
| 2027 | Pool | 2.1440 | 4.4140 | 4.3510 |
| 2027 | Graph | 2.1189 | 4.3889 | 4.3133 |
| 2027 | All Graph | **1.9230** | **3.9585** | **3.9022** |

Relative to Self, the 8-node Graph reduces IC4 ADE/FDE by 19.0%/21.0% (2026) and 16.5%/19.3% (2027). All Graph reduces them by 24.6%/28.1% and 24.2%/27.2%. Own explains essentially none of the gain. Pool establishes that neighbor information itself is useful; All Graph remains better than Pool with positive origin-cluster bootstrap lower bounds for both ADE and FDE in both seeds.

The limited Graph's small advantage over Pool is not uniformly significant under origin-cluster bootstrap (especially FDE), so these experiments do not support “message passing is indispensable.” The robust conclusion is narrower: neighbors are necessary for the achieved accuracy, and retaining all legal neighbors matters.

### Interaction concentration and capacity limit

All Graph versus Self gains are stronger for IC4 than Full4 in both seeds. IC4 gains are 24.6%/28.1% and 24.2%/27.2% ADE/FDE; Full4 gains are 19.5%/21.9% and 19.2%/21.6%. The all-neighbor model also beats the 8-node graph by 6.8%/9.0% and 9.2%/9.8% on IC4, so the future Raj interface must disclose its eight-node information cap.

Fixed strata remain directionally consistent. All Graph versus Self ADE/FDE reductions are approximately 24-25%/27-30% for `k>=2`, 20%/22% for stronger closing, and 21%/23-24% for CPA-positive samples across the two seeds.

## Gate B readiness

Gate B is prepared but not started. `configs/gate_b_raj_migration.json` freezes the Raj input as target plus the first seven history-ranked neighbors, the matched Johnson/classical comparator as the 8-node Graph, the all-neighbor graph as an information-cap reference, the 20->40 masked target label, paired seeds, origin-macro checkpointing, and sealed test. Raj architecture, PennyLane conversion, and formal quantum training remain untouched.

## Artifacts

- Data/provenance: `data/task_redesign/lankershim_gate_a_v1/manifest.json`
- Causal/preflight: `reports/task_redesign/gate_a_preflight.json`
- Seeds: `reports/task_redesign/gate_a_seed2026/summary.json`, `gate_a_seed2027/summary.json`
- Paired bootstrap: `reports/task_redesign/GATE_A_PAIRED_BOOTSTRAP_20260921.json`
- Machine decision: `reports/task_redesign/GATE_A_FINAL_SUMMARY_20260921.json`

The formal test remains closed. Gate A is passed on the bounded two-seed validation pilot; any paper claim should retain that scope until a separately authorized sealed-test evaluation.
