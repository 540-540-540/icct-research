"""Run frozen module selfchecks and B-domain smoke tests (P0 logic, kept out of modules)."""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

from frontend.sensing import coords, detector, simulator, waveform  # noqa: E402


def waveform_check() -> dict:
    w = waveform.PaperWaveform()
    array = waveform.ArrayConfig()
    return {"df": w.df, "elements": array.elements, "range_resolution": w.range_resolution}


def simulator_check() -> dict:
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    _common.assert_height_alignment(config, geometry)
    w, array = _common.build_objects(config)
    echo = _common.synthesize([[0.0, 0.0]], [[5.0, -3.0]], [7], geometry["stations"], geometry["boresights"],
                              w, array, config, 20.0, 0, 0, "cuda:0")
    if echo["Y"].shape != (3, array.elements, w.K, w.N) or echo["W"].shape != echo["Y"].shape:
        raise AssertionError("unexpected simulator shapes")
    if not bool((echo["alpha"] > 0).any()):
        raise AssertionError("simulator produced no visible target")
    return {"Y": list(echo["Y"].shape), "visible": echo["visible"].tolist()}


def detector_check() -> dict:
    config = _common.load_frontend_config()
    geometry = _common.load_geometry_config()
    w, array = _common.build_objects(config)
    echo = _common.synthesize([[0.0, 0.0]], [[5.0, -3.0]], [7], geometry["stations"], geometry["boresights"],
                              w, array, config, 20.0, 0, 0, "cuda:0")
    result = detector.process_baseline(echo["Y"][0], echo["X"][0], 0, w, array, config,
                                       station=geometry["stations"][0],
                                       boresight=float(geometry["boresights"][0]),
                                       height=geometry["height_difference_m"])
    if not result["detections"]:
        raise AssertionError("detector smoke produced no detection")
    keys = {"candidates_before_cap", "local_maxima", "nms_suppressed", "interpolation_geometry_rejected",
            "overflow", "returned_count", "multiplier"}
    if set(result["counters"]) != keys:
        raise AssertionError(f"unexpected counters: {result['counters']}")
    return {"detections": len(result["detections"]), "counters": result["counters"]}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("smoke tests require CUDA per frozen design")
    results = {}
    for name, function in (("waveform", waveform_check), ("coords", coords.selfcheck),
                           ("simulator", simulator_check), ("detector", detector_check)):
        try:
            results[name] = {"passed": True, "detail": function()}
        except Exception:  # noqa: BLE001
            results[name] = {"passed": False, "traceback": traceback.format_exc()}
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    if not all(entry["passed"] for entry in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()