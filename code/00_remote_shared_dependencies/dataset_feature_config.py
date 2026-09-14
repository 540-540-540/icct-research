"""
Helpers for dataset feature layout configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

LEGACY_BS_COUNT = 3
LEGACY_DIST_DIM = 16
LEGACY_VEL_DIM = 16
LEGACY_ANG_DIM = 8
LEGACY_RD_MAP_SHAPE = (2, 8, 8)
LEGACY_RA_MAP_SHAPE = (2, 8, 4)
LEGACY_DT = 0.05
LEGACY_BS_POSITIONS = np.array([[0.0, 0.0], [150.0, 0.0], [75.0, 150.0]], dtype=np.float32)


@dataclass(frozen=True)
class DatasetFeatureConfig:
    bs_count: int = LEGACY_BS_COUNT
    dist_dim: int = LEGACY_DIST_DIM
    vel_dim: int = LEGACY_VEL_DIM
    ang_dim: int = LEGACY_ANG_DIM
    rd_map_shape: tuple[int, int, int] = LEGACY_RD_MAP_SHAPE
    ra_map_shape: tuple[int, int, int] = LEGACY_RA_MAP_SHAPE
    measurement_dim: int = 5
    dt: float = LEGACY_DT
    bs_positions: np.ndarray = field(default_factory=lambda: LEGACY_BS_POSITIONS.copy())

    @property
    def per_bs_dim(self) -> int:
        return int(self.dist_dim + self.vel_dim + self.ang_dim)

    @property
    def rf_total_dim(self) -> int:
        return int(self.bs_count * self.per_bs_dim)

    @property
    def range_complex_len(self) -> int:
        return int(self.dist_dim // 2)

    @property
    def vel_complex_len(self) -> int:
        return int(self.vel_dim // 2)

    @property
    def angle_complex_len(self) -> int:
        return int(self.ang_dim // 2)


LEGACY_FEATURE_CONFIG = DatasetFeatureConfig()


def _scalar(npz_obj, key: str, default):
    if key not in npz_obj.files:
        return default
    value = npz_obj[key]
    if np.isscalar(value):
        return value.item()
    if getattr(value, "shape", ()) == ():
        return value.item()
    return value


def load_feature_config(root_path: str | None = None, snr: int | None = None, data_dir: str | None = None) -> DatasetFeatureConfig:
    if data_dir is None:
        if root_path is None or snr is None:
            raise ValueError("Either data_dir or both root_path and snr must be provided.")
        data_dir = os.path.join(root_path, f"{snr}dB")

    config_path = os.path.join(data_dir, "feature_config.npz")
    if not os.path.exists(config_path):
        return LEGACY_FEATURE_CONFIG

    npz_obj = np.load(config_path)
    try:
        bs_count = int(_scalar(npz_obj, "bs_count", LEGACY_BS_COUNT))
        dist_dim = int(_scalar(npz_obj, "dist_dim", LEGACY_DIST_DIM))
        vel_dim = int(_scalar(npz_obj, "vel_dim", LEGACY_VEL_DIM))
        ang_dim = int(_scalar(npz_obj, "ang_dim", LEGACY_ANG_DIM))
        rd_map_shape = tuple(int(v) for v in np.asarray(_scalar(npz_obj, "rd_map_shape", LEGACY_RD_MAP_SHAPE)).tolist())
        ra_map_shape = tuple(int(v) for v in np.asarray(_scalar(npz_obj, "ra_map_shape", LEGACY_RA_MAP_SHAPE)).tolist())
        measurement_dim = int(_scalar(npz_obj, "measurement_dim", 5))
        dt = float(_scalar(npz_obj, "dt", LEGACY_DT))
        if "bs_positions" in npz_obj.files:
            bs_positions = np.asarray(npz_obj["bs_positions"], dtype=np.float32)
        else:
            bs_positions = LEGACY_BS_POSITIONS.copy()
    finally:
        npz_obj.close()

    return DatasetFeatureConfig(
        bs_count=bs_count,
        dist_dim=dist_dim,
        vel_dim=vel_dim,
        ang_dim=ang_dim,
        rd_map_shape=rd_map_shape,
        ra_map_shape=ra_map_shape,
        measurement_dim=measurement_dim,
        dt=dt,
        bs_positions=bs_positions,
    )


def apply_feature_config_to_args(args, feature_cfg: DatasetFeatureConfig):
    setattr(args, "dist_dim", int(feature_cfg.dist_dim))
    setattr(args, "vel_dim", int(feature_cfg.vel_dim))
    setattr(args, "ang_dim", int(feature_cfg.ang_dim))
    setattr(args, "bs_feature_dim", int(feature_cfg.per_bs_dim))
    setattr(args, "enc_in", int(feature_cfg.rf_total_dim))
    setattr(args, "rf_total_dim", int(feature_cfg.rf_total_dim))
    setattr(args, "dataset_dt", float(feature_cfg.dt))
    return args
