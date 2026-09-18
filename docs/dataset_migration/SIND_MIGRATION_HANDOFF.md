# SinD high-interaction migration handoff

Status: **GO / migration accepted**

This document hands the completed dataset and controlled-sensing migration back to the QGNN / prediction mainline. It supersedes dataset-selection assumptions in the earlier replacement handoff, but it does **not** redesign the final QGNN.

## Executive decision

The selected source is the directly downloadable **SinD public Changchun + Xi'an recordings** from the official SinD repository.

Why GO:

- real-world drone-derived road-user trajectories;
- stable source timestamps at nominal ~10 Hz;
- direct world-frame `x,y,vx,vy`;
- long continuous recordings, not isolated clips;
- high-interaction supply is materially higher than Automatum before model-facing selection;
- enough data to create 20-history / 20-future windows;
- 3-BS controlled sensing can be re-laid out with full multi-BS coverage;
- raw download is only about 164 MB for the two selected vehicle CSVs.

The key preselection evidence is:

| Metric | SinD public Changchun+Xi'an | Automatum |
|---|---:|---:|
| Source candidate windows satisfying high-interaction gate | 83.22% | 34.77% |
| Model-facing windows | 19,768 | 12,394 |
| Model-facing targets with >=3 active neighbors | 38.83% | 4.69% |

This supports replacing Automatum for the interaction-focused QGNN experiments.
## Candidate-screening outcome

The engineering rule was tightened during this task: if the actual trajectory files could not be directly acquired in-session, the candidate was not allowed to become the migration target.

- **SinD public samples**: direct official GitHub data, continuous recordings, sufficient scale, strong interaction evidence -> **GO**.
- **InterHub interaction_multi derivative**: directly readable and strongly interaction-focused, but the inspected cache clips are only 40 frames each and the available event set is too small to serve as the main training corpus -> no-go as primary dataset; still useful as a future stress-test source.
- **Official INTERACTION / inD / Waymo full data**: not taken into engineering migration because the required full files were not directly obtainable under the user's access rule in this run.
- **Full SinD 2.0**: full data/semantic labels require application; this migration intentionally uses only the public official recordings.
- **FLUID**: not taken forward because selective acquisition was not practical under the run's direct-access/download-size constraint; no claim is made that its interaction quality is inferior.

The final decision is therefore based on measured data, not on paper descriptions alone.

---

# A. Dataset

## A.1 Identity and source

- Name used in this project: **SinD public high-interaction subset**
- Official repository: `https://github.com/SOTIF-AVLab/SinD`
- Pinned source commit: `930e4dea78d924c6e9a58ff8e378331f93bba8ec`
- Selected recordings:
  - Changchun: `Data/Changchun/changchun_pudong_507_009/Veh_smoothed_tracks.csv`
  - Xi'an: `Data/Xi'an/Xi'an_412_m1/Veh_smoothed_tracks.csv`
Raw SHA256:

- Changchun: `f3011d7dc1786f940981a9d49c7f83c7860beda9d14ed0e06c995f7b7e590692`
- Xi'an: `1bf5d8450577249beb9c97bc210e7510babf0796ba817f9a88c64d19a5a5e4a7`

The download is implemented by:

`tools/data_preprocessing/download_sind_public.py`

Raw data remain under `data/` and are not committed to Git.

## A.2 License

The pinned SinD repository LICENSE is headed **CC0 1.0 Universal**, but its Statement of Purpose explicitly says the dataset is not permitted for commercial purposes and repeatedly excludes commercial uses.

Project policy for this branch:

> Treat the public trajectory data as non-commercial research data unless the SinD maintainers clarify the license wording.

Do not redistribute the raw CSVs from this repository.

## A.3 Raw scale and formal road-user types

| Scene | Raw CSV bytes | All rows | All tracks | Formal four-wheel rows after quality filtering | Formal tracks |
|---|---:|---:|---:|---:|---:|
| Changchun | 129,987,853 | 432,224 | 1,362 | 407,599 | 1,244 |
| Xi'an | 33,613,874 | 116,836 | 423 | 103,692 | 361 |

Formal types are:

`car / bus / truck`

Bicycle, motorcycle and tricycle trajectories are not part of the frozen prediction/sensing contract.
## A.4 Cleaning rules

Raw audit checks:

- required fields present;
- no duplicate `(track_id, frame_id)`;
- finite `x,y,vx,vy`;
- strictly increasing source timebase;
- continuous frame IDs inside each retained track;
- position finite-difference motion checked against provided velocity.

Quality filter:

> Exclude a complete track if its maximum adjacent position/velocity consistency error exceeds 5 m/s.

Exactly two original tracks are excluded:

- Changchun original track ID `747`;
- Xi'an original track ID `141`.

This threshold was fixed during preprocessing and is not selected using test prediction performance.

Internal model-facing vehicle IDs are deterministic scene-local integer IDs. The mapping to original SinD track IDs is written to:

`data/sind/vehicle_id_mapping.csv`

and is metadata only.

## A.5 Timebase

No fixed `dt=3/29.97` assumption is used for SinD.

The task is nominally 10 Hz, but timestamps are taken from source `timestamp_ms`.

Audited source cadence:

| Scene | median dt | min dt | max dt | Resampled |
|---|---:|---:|---:|---|
| Changchun | 0.1001001001 s | 0.1001001001 s | 0.1001001001 s | No |
| Xi'an | 0.1001001001 s | 0.1001001001 s | 0.1001001001 s | No |

`history_timestamp[20]` stores the source-derived timestamps directly.
# B. Prediction task

Frozen task:

- history: 20 frames;
- future: 20 frames;
- state: `[x,y,vx,vy]`;
- maximum graph size: 8 vehicles;
- source timebase: stable nominal ~10 Hz, preserved from timestamps.

Final sample counts:

| Split | Samples |
|---|---:|
| train | 15,802 |
| val | 1,880 |
| test | 2,086 |
| total | **19,768** |

The split target is chronological approximately 8:1:1 per recording, with 40-frame exclusion gaps around boundaries.

Leakage protection additionally removes every vehicle that spans across a split boundary plus its 40-frame guard region from **all model-facing samples**.

Final cross-split physical-vehicle overlap:

- train / val: 0
- train / test: 0
- val / test: 0

## B.1 High-interaction gate

A window is retained only if, at the last history frame, at least one vehicle has two or more **active neighbors**.

An active relation is:

[
d_{ij}le 30,m
quadlandquad
left(
c_{ij}>0.5,m/s
;lor;
(0<TCPA_{ij}le4,s land DCPA_{ij}le10,m)
ight).
]

All gate and selection decisions use history only.
## B.2 N > 8 rule

Vehicles are never silently truncated.

For `N <= 8`:

- keep every eligible non-guard vehicle in the window.

For `N > 8`:

- emit exactly one local graph for that global window;
- choose the focal vehicle from the last history frame by:
  1. highest active-neighbor degree;
  2. more neighbors within 20 m;
  3. larger summed closing rate over active edges;
  4. smaller active-edge DCPA;
  5. deterministic vehicle ID tie-break;
- choose the remaining seven vehicles by focal-related:
  1. active relation first;
  2. smaller DCPA;
  3. larger closing rate;
  4. smaller distance;
  5. vehicle ID.

The future is never used.

Metadata records:

- `original_num_vehicles`;
- `selection_mode`;
- `focal_vehicle_id`.

In the final model-facing set, **90.73%** of windows originate from global windows with `N>8`. This is intentional and must be remembered when interpreting results.
# C. Interaction statistics

The comprehensive frozen report is:

`reports/sind/sind_interaction_audit.json`

The compact comparison is:

`reports/sind/interaction_audit_comparison.json`

## C.1 Vehicle-scale comparison

| Metric | SinD | Automatum |
|---|---:|---:|
| N >= 4 | 99.94% | 51.80% |
| N >= 5 | 99.25% | 28.80% |
| N >= 6 | 98.16% | 12.90% |

## C.2 Spatial pairs per window

| Radius | SinD | Automatum |
|---:|---:|---:|
| <=10 m | 5.726 | 0.728 |
| <=15 m | 9.029 | 1.448 |
| <=20 m | 12.230 | 2.097 |
| <=25 m | 15.371 | 2.755 |
| <=30 m | 18.779 | 3.389 |
| <=45 m | 23.625 | 4.865 |

## C.3 Closing pairs per window

| Closing threshold | SinD | Automatum |
|---:|---:|---:|
| >0 m/s | 19.173 | 3.028 |
| >0.5 m/s | 15.141 | 2.680 |
| >1.0 m/s | 13.719 | 2.491 |
## C.4 CPA-risk window coverage

| Condition | SinD | Automatum |
|---|---:|---:|
| at least one DCPA <=2 m | 17.57% | 7.12% |
| at least one DCPA <=5 m | 74.52% | 32.69% |
| at least one DCPA <=10 m | 93.56% | 51.34% |

CPA is a constant-relative-velocity history-state diagnostic over the next 4 s; it is not a future-GT selector.

## C.5 Multi-neighbor interaction

Fraction of target nodes:

| Active-neighbor count | SinD | Automatum |
|---|---:|---:|
| >=2 | 60.22% | 18.23% |
| >=3 | 38.83% | 4.69% |
| >=4 | 23.96% | 0.87% |

## C.6 Explicit higher-order active components

Fraction of windows whose active-edge graph contains a connected component of at least:

| Component size | SinD | Automatum |
|---|---:|---:|
| 3 vehicles | 100.00% | 32.13% |
| 4 vehicles | 93.83% | 14.06% |
| 5 vehicles | 87.12% | 5.49% |

## C.7 Interpretation

The data support a **GO** decision for interaction-focused experiments.

However, two statements must remain distinct:

1. **Underlying source supply is richer:** before model-facing high-interaction selection, the measured high-interaction candidate fraction is 83.22% for the selected SinD source versus 34.77% for Automatum.
2. **The final SinD task is deliberately interaction-focused:** every retained window passes the history-only gate.

Therefore, later QGNN gains must be established against strong classical models on exactly the same SinD splits.
# D. 3-BS controlled ISAC

Research boundary remains unchanged from Automatum Route-B:

- controlled sensing;
- no real detector;
- no identity association failure;
- no temporal tracker;
- no smoothing/Kalman stage;
- vehicle identity is used only to generate/alignment-cache the controlled estimate and is not a model feature.

Sensing chain:

[
[x,y,v_x,v_y]_{GT}
ightarrow
3	ext{-BS noisy }[r,	heta,v_r]
ightarrow
	ext{covariance-weighted position fusion}
ightarrow
	ext{multi-BS radial-velocity recovery}
ightarrow
[hat x,hat y,hat v_x,hat v_y].
]

The measurement noise law, range/FOV rules and weak velocity prior are transferred unchanged from frozen Automatum Route-B. Test is never used for geometry search or SNR calibration.

Formal SNR levels:

[
[-10,-5,0,5,10];dB.
]

## D.1 Frozen BS geometry

Scene 0 — Changchun:

- center: `[-21.957584, -0.158642]`
- BS1: `[42.994321, 37.341358]`, boresight `-150 deg`
- BS2: `[-86.909489, 37.341358]`, boresight `-30 deg`
- BS3: `[-21.957584, -75.158642]`, boresight `90 deg`

Scene 1 — Xi'an:

- center: `[-12.036871, 24.263565]`
- BS1: `[13.614639, 94.740512]`, boresight `-110 deg`
- BS2: `[-85.897453, 11.239952]`, boresight `10 deg`
- BS3: `[36.172199, -33.189768]`, boresight `130 deg`
Visibility/fusion constants:

- range: 5–150 m;
- FOV half-angle: 70 deg;
- BS/vehicle height difference: 5 m;
- velocity prior sigma: 200 m/s;
- geometry search scope: train only.

Train-only geometry audit:

- both scenes: 100% sampled train points have >=2 visible BS;
- Changchun: about 97.95% have all 3 BS;
- Xi'an: about 99.99% have all 3 BS.

Geometry figures:

- `reports/sind/plots/sind_scene0_3bs_geometry.png`
- `reports/sind/plots/sind_scene1_3bs_geometry.png`

## D.2 Held-out test sensing error

| SNR | Position RMSE | Velocity RMSE |
|---:|---:|---:|
| -10 dB | 1.175 m | 1.176 m/s |
| -5 dB | 0.660 m | 0.661 m/s |
| 0 dB | 0.371 m | 0.372 m/s |
| 5 dB | 0.208 m | 0.209 m/s |
| 10 dB | 0.117 m | 0.118 m/s |

The five points are **monotonically decreasing**, not assumed to be linearly related to SNR.

Test unique history states:

- 31,556 total;
- 514 use 2 BS;
- 31,042 use all 3 BS;
- 0 use only 1 BS;
- velocity recovery rank-2 fraction: 100%.

The sensing error is small enough not to dominate the downstream prediction task.
# E. Model-facing contract

Formal loader:

`frontend.sind_prediction_dataset.SinDPredictionDataset`

Per sample:

```text
history_state       [20, 8, 4] float32
future_state        [20, 8, 4] float32
vehicle_mask        [8]        bool
history_timestamp   [20]       float64

scene_id            metadata only
start_frame         metadata only
vehicle_ids         metadata only
original_num_vehicles metadata only
selection_mode      metadata only
focal_vehicle_id    metadata only
```

Semantics:

- `history_state = [x_hat,y_hat,vx_hat,vy_hat]` from frozen SinD sensing cache;
- `future_state = [x,y,vx,vy]` from SinD GT;
- padding states are strictly zero;
- padding vehicle IDs are `-1`;
- scene / vehicle / focal IDs must not be model features by default.

Each split has one `sensing_cache.npz` containing all five SNR levels. Cache state array shape is:

`[5, M, 4]`

where `M` is the number of unique history states for that split.
# F. Git and reproduction

Target branch: **`sind`**

Frozen base commit inherited from QGNN handoff:

`a9b3ae630c8bb46d07d228e33a5896d6c43fb3af`

Frozen SinD migration implementation commit:

`0d541aaee86ecd063f80f53eab07bb45ada24c5e`

Primary implementation/config/report paths:

- `configs/sind_controlled_isac.json`
- `configs/sind_prediction.json`
- `frontend/controlled_isac/sind_frontend.py`
- `frontend/sind_prediction_dataset.py`
- `tools/data_preprocessing/download_sind_public.py`
- `tools/data_preprocessing/audit_sind_raw.py`
- `tools/data_preprocessing/build_sind_high_interaction.py`
- `tools/data_preprocessing/validate_sind_high_interaction.py`
- `tools/data_preprocessing/search_sind_geometry.py`
- `tools/data_preprocessing/freeze_sind_geometry.py`
- `tools/data_preprocessing/build_sind_isac_cache.py`
- `tools/data_preprocessing/validate_sind_isac_and_dataset.py`
- `tools/data_preprocessing/audit_sind_interaction.py`
- `tools/data_preprocessing/validate_sind_migration.py`

Full command chain is documented in:

`docs/dataset_migration/SIND_MIGRATION_REPRODUCE.md`

Acceptance frozen in the final branch:

- GT validation: **56/56 PASS**
- ISAC + loader validation: **782/782 PASS**
- integrated migration validation: **43/43 PASS**

Raw data and generated `data/` arrays are not committed.
# G. Remaining risks / unresolved questions

1. **Public subset, not full SinD 2.0.** The migration covers two public continuous recordings, not the full multi-city release.
2. **Interaction-focused distribution.** The final dataset deliberately removes low-interaction windows. It is suitable for testing interaction modeling, but it is not a natural-frequency traffic benchmark.
3. **Local-subgraph dependence.** 90.73% of final windows originate from global `N>8` windows; the history-only focal/local selection rule is therefore an important part of the benchmark definition.
4. **Four-wheel vehicles only.** Vulnerable road users and powered two/three-wheel vehicles are outside the frozen task.
5. **Controlled sensing remains idealized.** Detection, association and identity failures are still excluded by design.
6. **Transferred sensing calibration.** Noise parameters come from the frozen Automatum controlled-sensing design rather than hardware-specific SinD radar calibration.
7. **QGNN advantage is not established by this migration.** This dataset creates more room for interaction models; whether QGNN outperforms a strong Classical GNN / higher-order GNN must be measured separately.
8. **Selection sensitivity should be an ablation.** At minimum, later experiments should compare the frozen high-interaction set against a less aggressively filtered or unfiltered SinD subset to quantify how much model ranking depends on the gate.

## Final handoff statement

The dataset-replacement question is closed as **GO for SinD public Changchun + Xi'an** under the current ICCT research definition.

The downstream QGNN / Classical GNN / GPT-2+LoRA pipeline may now treat the frozen SinD model-facing interface as the new data source. No downstream model should read GT history in formal experiments; formal history comes from the selected SNR slice of the frozen sensing cache.
