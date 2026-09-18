# SinD migration reproduction guide

This guide reproduces the frozen high-interaction SinD branch from public source files.
Raw trajectories and generated NPZ/cache arrays are intentionally ignored by Git.

## Frozen source

- Dataset repository: https://github.com/SOTIF-AVLab/SinD
- Source commit: `930e4dea78d924c6e9a58ff8e378331f93bba8ec`
- Public recordings:
  - Changchun: `Data/Changchun/changchun_pudong_507_009/Veh_smoothed_tracks.csv`
  - Xi'an: `Data/Xi'an/Xi'an_412_m1/Veh_smoothed_tracks.csv`
- Raw SHA256:
  - Changchun: `f3011d7dc1786f940981a9d49c7f83c7860beda9d14ed0e06c995f7b7e590692`
  - Xi'an: `1bf5d8450577249beb9c97bc210e7510babf0796ba817f9a88c64d19a5a5e4a7`

The SinD repository LICENSE is headed CC0 1.0 but explicitly states that the dataset is not permitted for commercial purposes. Treat this branch as non-commercial research use unless the dataset maintainers clarify otherwise.

## Environment

The validated server environment is:

```text
Ubuntu
Python: /home/dell/YrM/envs/ICCT
2 x RTX 4090 24 GB
```

The preprocessing and controlled sensing pipeline is CPU-capable. The sensing-cache builder supports multiple CPU workers.
## Reproduce from scratch

Run from the repository root.

```bash
python tools/data_preprocessing/download_sind_public.py
python tools/data_preprocessing/audit_sind_raw.py
python tools/data_preprocessing/build_sind_high_interaction.py
python tools/data_preprocessing/validate_sind_high_interaction.py
```

The GT builder writes `data/sind/.BUILD_COMPLETE`. A normal second invocation will skip the frozen build. To intentionally rebuild:

```bash
python tools/data_preprocessing/build_sind_high_interaction.py --force
```

Search and freeze the 3-BS geometry using train only:

```bash
python tools/data_preprocessing/search_sind_geometry.py
python tools/data_preprocessing/freeze_sind_geometry.py
python tools/data_preprocessing/plot_sind_geometry.py
```

Build all three split caches. Every cache contains the five SNR levels `[-10,-5,0,5,10]` dB:

```bash
python tools/data_preprocessing/build_sind_isac_cache.py --split all --workers 12
python tools/data_preprocessing/validate_sind_isac_and_dataset.py
```
## Interaction audit and final validation

The comprehensive SinD interaction audit also compares against the existing Automatum formal data. Pass a repository root that contains `data/automatum_t_crossing`:

```bash
python tools/data_preprocessing/audit_sind_interaction.py \
  --automatum-root /path/to/icct-research
```

The compact comparison utility expects the Automatum split directory directly:

```bash
python tools/data_preprocessing/audit_sind_vs_automatum.py \
  --automatum-root /path/to/icct-research/data/automatum_t_crossing/splits
```

Then run the integrated migration acceptance check:

```bash
python tools/data_preprocessing/validate_sind_migration.py
```

Frozen acceptance in this branch:

```text
GT validation:             56 / 56 PASS
ISAC + loader validation: 782 / 782 PASS
Migration validation:      43 / 43 PASS
```

## Model-facing smoke test

```python
from frontend.sind_prediction_dataset import SinDPredictionDataset

ds = SinDPredictionDataset(
    split="train",
    snr_db=0.0,
    root_dir=".",
    return_tensors=False,
)
item = ds[0]
print(item["history_state"].shape)      # (20, 8, 4)
print(item["future_state"].shape)       # (20, 8, 4)
print(item["vehicle_mask"].shape)       # (8,)
print(item["history_timestamp"].shape)  # (20,)
```
## Generated paths

```text
data/sind/raw/<city>/Veh_smoothed_tracks.csv
data/sind/vehicle_id_mapping.csv
data/sind/splits/{train,val,test}/trajectories.csv
data/sind/splits/{train,val,test}/samples.npz
data/sind/isac/{train,val,test}/sensing_cache.npz
data/sind/isac/{train,val,test}/sensing_manifest.json

reports/sind/sind_raw_audit.json
reports/sind/sind_dataset_build.json
reports/sind/sind_dataset_validation.json
reports/sind/sind_geometry_search.json
reports/sind/sind_isac_cache_report.json
reports/sind/sind_isac_validation.json
reports/sind/sind_interaction_audit.json
reports/sind/interaction_audit_comparison.json
reports/sind/sind_migration_validation.json
reports/sind/plots/sind_scene0_3bs_geometry.png
reports/sind/plots/sind_scene1_3bs_geometry.png
```

Do not use scene ID, vehicle ID, focal vehicle ID, original vehicle count, or selection mode as model features unless a future experiment explicitly changes the frozen contract.
