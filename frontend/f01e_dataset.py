"""F01-E formal SNR-aware dataset loader (F01E-AUDIT-08, final guard F01E-FINALIZE-11R2).

The caller selects only (root, split, snr_db); the loader binds the matching per-SNR input and
label files and hard-rejects mismatched-SNR pairing, the legacy generic labels and any F01-D path.
The full metadata header must additionally carry the finalized IDENTITY_SWITCH_PM1_MASK
sanitization contract; pre-final F01-E roots are rejected instead of being silently accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ALLOWED_SPLITS = ("train", "V_select", "V_confirm", "test")
ALLOWED_SNR_DB = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
INPUT_FIELDS = ("state_hat", "track_exists", "detected", "timestamp")
LABEL_FIELDS = ("future_position", "label_valid")
MODEL_INPUT_KEYS = ("state_hat", "track_exists", "detected", "timestamp", "origin_eligible")
LEGACY_LABEL_PATTERN = "labels/{split}.npz"
FINAL_METADATA_CONTRACT = {
    "finalized": True,
    "label_alignment": "origin-safe contiguous segment",
    "label_sanitization": "IDENTITY_SWITCH_PM1_MASK",
    "mask_radius_origins": 1,
    "mask_scope": "same_continuous_segment_only",
}


def require_final_metadata(document: dict, path) -> None:
    """Formal F01-E path only accepts the finalized, sanitized dataset (F01E-FINALIZE-11R2)."""
    failures = [f"{key}={document.get(key)!r} (expected {value!r})"
                for key, value in FINAL_METADATA_CONTRACT.items() if document.get(key) != value]
    if failures:
        raise RuntimeError(f"F01-E metadata is not the finalized contract ({path}): "
                           + "; ".join(failures))


def snr_name(snr_db: float) -> str:
    return "snr_" + str(int(snr_db)).replace("-", "m")


def origin_eligibility(exists: np.ndarray) -> np.ndarray:
    """[S,20,8] -> [S,8] origin eligibility (SENS-FREEZE-09).

    Eligible iff the origin frame is alive and the contiguous alive run ending at the origin covers
    at least 3 history frames. Separate lives of the same slot are never summed.
    """
    if exists.shape != (exists.shape[0], 20, 8):
        raise ValueError("exists must be [S,20,8]")
    run = np.cumprod(exists[:, ::-1, :], axis=1).sum(axis=1)
    return exists[:, -1, :] & (run >= 3)


def _read_inputs(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(INPUT_FIELDS):
            raise ValueError(f"Input cache fields must match the model-only whitelist: {path}")
        arrays = {key: payload[key] for key in INPUT_FIELDS}
    state, exists, detected, timestamp = (arrays["state_hat"], arrays["track_exists"],
                                          arrays["detected"], arrays["timestamp"])
    if state.ndim != 4 or state.shape[1:] != (20, 8, 4) or state.dtype != np.float32:
        raise ValueError("Expected float32 [S,20,8,4] state")
    if exists.shape != state.shape[:-1] or detected.shape != exists.shape \
            or exists.dtype != np.bool_ or detected.dtype != np.bool_:
        raise ValueError("Invalid input masks")
    if timestamp.shape != state.shape[:2] or timestamp.dtype != np.float64:
        raise ValueError("Invalid timestamps")
    if not np.isfinite(state).all() or not np.isfinite(timestamp).all() or np.any(detected & ~exists):
        raise ValueError("Invalid finite values or mask relation")
    if np.any(state[~exists] != 0):
        raise ValueError("Unavailable state must be zero padded")
    if not np.allclose(np.diff(timestamp, axis=1), 0.1, rtol=0, atol=1e-6):
        raise ValueError("Nonuniform history clock")
    return arrays


def _read_labels(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(LABEL_FIELDS):
            raise ValueError(f"Unexpected label fields: {path}")
        labels = {key: payload[key] for key in LABEL_FIELDS}
    future, valid = labels["future_position"], labels["label_valid"]
    if future.ndim != 4 or future.shape[1:] != (20, 8, 2) or future.dtype != np.float32:
        raise ValueError("Expected float32 [S,20,8,2] future_position")
    if valid.shape != future.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError("Invalid label_valid")
    if not np.isfinite(future).all() or np.any(future[~valid] != 0):
        raise ValueError("Invalid label padding")
    return labels


class F01EDataset:
    """Formal F01-E loader: same-SNR input/label binding with hard guards."""

    def __init__(self, root, split: str, snr_db: float, label_snr_db: float,
                 inputs: dict, labels: dict, metadata: list):
        self.root = Path(root)
        self.split = split
        self.snr_db = float(snr_db)
        self.label_snr_db = float(label_snr_db)
        self.input_path = self.root / "inputs" / f"{split}_{snr_name(self.snr_db)}.npz"
        self.label_path = self.root / "labels" / f"{split}_{snr_name(self.label_snr_db)}.npz"
        self.arrays = inputs
        self.label_arrays = labels
        self.metadata = metadata
        self.eligible = origin_eligibility(inputs["track_exists"])
        label_any = labels["label_valid"].any(axis=1)
        violations = label_any & ~self.eligible
        if np.any(violations):
            sample, slot = np.argwhere(violations)[0]
            raise RuntimeError(f"Label exists on an origin-ineligible slot: sample {int(sample)}, "
                               f"slot {int(slot)}; dead-slot labels are forbidden")

    @classmethod
    def from_paths(cls, root, split: str, snr_db: float, label_snr_db: float | None = None) -> "F01EDataset":
        root = Path(root)
        if "f01d" in str(root).lower():
            raise RuntimeError("F01-D is an archived oracle reference; the formal loader refuses it")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}")
        if float(snr_db) not in ALLOWED_SNR_DB:
            raise ValueError(f"snr_db must be one of {ALLOWED_SNR_DB}")
        if label_snr_db is None:
            raise RuntimeError("Legacy generic labels/ are reference-only and rejected for formal F01-E")
        if float(label_snr_db) not in ALLOWED_SNR_DB:
            raise ValueError(f"label_snr_db must be one of {ALLOWED_SNR_DB}")
        if float(label_snr_db) != float(snr_db):
            raise ValueError(f"same-SNR binding required: input {snr_db} vs labels {label_snr_db}")
        input_path = root / "inputs" / f"{split}_{snr_name(snr_db)}.npz"
        label_path = root / "labels" / f"{split}_{snr_name(label_snr_db)}.npz"
        if label_path.name == LEGACY_LABEL_PATTERN.format(split=split).split("/")[-1]:
            raise RuntimeError("Refusing the legacy generic label file")
        if not input_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing F01-E cache files: {input_path} / {label_path}")
        metadata_path = root / "metadata" / f"{split}.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing F01-E finalized metadata header: {metadata_path}")
        document = json.loads(metadata_path.read_text())
        require_final_metadata(document, metadata_path)
        metadata = document["samples"]
        return cls(root, split, snr_db, label_snr_db, _read_inputs(input_path), _read_labels(label_path),
                   metadata)

    @classmethod
    def from_root(cls, root, split: str, snr_db: float) -> "F01EDataset":
        return cls.from_paths(root, split, snr_db, label_snr_db=snr_db)

    def __len__(self) -> int:
        return int(self.arrays["state_hat"].shape[0])

    def __getitem__(self, index: int) -> dict:
        state = self.arrays["state_hat"][index].copy()
        exists = self.arrays["track_exists"][index].copy()
        detected = self.arrays["detected"][index].copy()
        timestamp = self.arrays["timestamp"][index].copy()
        model_input = {"state_hat": state, "track_exists": exists, "detected": detected,
                       "timestamp": timestamp, "origin_eligible": self.eligible[index].copy()}
        labels = {"future_position": self.label_arrays["future_position"][index].copy(),
                  "label_valid": self.label_arrays["label_valid"][index].copy()}
        evaluation_only = self.metadata[index] if index < len(self.metadata) else {}
        return {"model_input": model_input, "labels": labels, "evaluation_only": evaluation_only}