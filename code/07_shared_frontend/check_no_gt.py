"""T2 no-GT detection test: static import/API audit plus dynamic GT-isolation checks."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import _common  # noqa: E402

OUT = ROOT / "reports/f01e/t2_no_gt"
B_DOMAIN_FILES = ["frontend/sensing/detector.py", "frontend/sensing/coords.py"]
FORBIDDEN_IMPORTS = ("echo_source", "scene_manifest", "source_states", "source_key")
FORBIDDEN_PARAMS = ("target", "targets", "truth", "gt", "vehicle", "vehicles", "source_key",
                    "source_keys", "target_count", "n_t")


def static_audit() -> dict:
    findings = {}
    for name in B_DOMAIN_FILES:
        text = (ROOT / name).read_text()
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        bad_imports = [module for module in imports if any(token in module for token in FORBIDDEN_IMPORTS)]
        bad_text = [token for token in ("source_states",) if token in text]
        bad_params = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                for argument in [*node.args.args, *node.args.kwonlyargs]:
                    lowered = argument.arg.lower()
                    if any(token == lowered for token in FORBIDDEN_PARAMS):
                        bad_params.append({"function": node.name, "parameter": argument.arg})
        findings[name] = {"bad_imports": bad_imports, "forbidden_text": bad_text, "bad_params": bad_params,
                          "imports": imports}
    return findings


def build_echo(positions, velocities, keys, episode, frame, snr_ref=25.0):
    config = _common.load_frontend_config()
    waveform, array = _common.build_objects(config)
    geometry = _common.load_geometry_config()
    echo = _common.synthesize(positions, velocities, keys, geometry["stations"], geometry["boresights"],
                              waveform, array, config, snr_ref, episode, frame, "cuda:0")
    return config, waveform, array, geometry, echo


def detection_key(detection):
    return (round(detection["r_m"], 6), round(detection["vr_mps"], 6), round(detection["u"], 9))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("T2 requires CUDA per frozen design")
    from frontend.sensing import detector as detector_module

    checks = {}
    lut = _common.load_lut(ROOT)
    static = static_audit()
    checks["static_imports_and_api"] = {
        "passed": all(not entry["bad_imports"] and not entry["forbidden_text"] and not entry["bad_params"]
                      for entry in static.values()),
        "detail": static}

    poisoned = types.ModuleType("frontend.echo_source")

    def _boom(*args, **kwargs):
        raise AssertionError("B-domain code must not touch the GT source")

    poisoned.__getattr__ = _boom
    saved = sys.modules.get("frontend.echo_source")
    sys.modules["frontend.echo_source"] = poisoned
    sys.modules.pop("frontend.sensing.detector", None)
    try:
        import importlib

        fresh = importlib.import_module("frontend.sensing.detector")
        config, waveform, array, geometry, echo = build_echo([[10.0, 5.0]], [[4.0, 2.0]], [1], 9300, 0)
        result = fresh.process_baseline(echo["Y"][0], echo["X"][0], 0, waveform, array, config,
                                        station=geometry["stations"][0],
                                        boresight=float(geometry["boresights"][0]), covariance_lut=lut)
        sentinel_ok = len(result["detections"]) >= 1
    except Exception as error:  # noqa: BLE001
        sentinel_ok = False
        result = {"error": repr(error)}
    finally:
        if saved is not None:
            sys.modules["frontend.echo_source"] = saved
        else:
            sys.modules.pop("frontend.echo_source", None)
    checks["import_sentinel"] = {"passed": sentinel_ok, "detail": {"detections": len(result.get("detections", []))}}

    torch.cuda.empty_cache()
    order_a = [[10.0, 5.0], [-30.0, 40.0], [25.0, -20.0]]
    order_b = order_a[::-1]
    velocity = [[4.0, 2.0], [-3.0, 5.0], [2.0, -4.0]]
    config, waveform, array, geometry, echo_a = build_echo(order_a, velocity, [11, 22, 33], 9301, 0)
    _, _, _, _, echo_b = build_echo(order_b, velocity[::-1], [33, 22, 11], 9301, 0)
    station, boresight = geometry["stations"][0], float(geometry["boresights"][0])
    detections_a = detector_module.process_baseline(echo_a["Y"][0], echo_a["X"][0], 0, waveform, array, config,
                                                    station=station, boresight=boresight,
                                                    covariance_lut=lut)["detections"]
    detections_b = detector_module.process_baseline(echo_b["Y"][0], echo_b["X"][0], 0, waveform, array, config,
                                                    station=station, boresight=boresight,
                                                    covariance_lut=lut)["detections"]
    keys_a = {detection_key(detection) for detection in detections_a}
    keys_b = {detection_key(detection) for detection in detections_b}
    permutation_ok = len(keys_a) == len(detections_a) and keys_a == keys_b
    checks["permutation_invariance"] = {"passed": permutation_ok,
                                        "detail": {"count_a": len(detections_a), "count_b": len(detections_b),
                                                   "sets_equal": keys_a == keys_b}}

    whitelist = set(detector_module.DETECTION_FIELDS)
    keys_clean = all(set(detection) <= whitelist for detection in detections_a)
    forbidden_keys = [key for detection in detections_a for key in detection
                      if any(token in key.lower() for token in ("source", "target", "vehicle", "truth"))]
    checks["detection_payload_whitelist"] = {"passed": keys_clean and not forbidden_keys,
                                             "detail": {"fields": sorted(set().union(*[set(d) for d in detections_a]))
                                                        if detections_a else [], "forbidden": forbidden_keys}}

    _, _, _, _, noise_echo = build_echo([], [], [], 9302, 0, snr_ref=25.0)
    noise_detections = detector_module.process_baseline(noise_echo["Y"][0], noise_echo["X"][0], 0, waveform,
                                                        array, config, station=station,
                                                        boresight=boresight, covariance_lut=lut)["detections"]
    _, _, _, _, one_echo = build_echo([order_a[0]], [velocity[0]], [11], 9303, 0)
    one_detections = detector_module.process_baseline(one_echo["Y"][0], one_echo["X"][0], 0, waveform, array,
                                                      config, station=station, boresight=boresight,
                                                      covariance_lut=lut)["detections"]
    counts_follow = (len(noise_detections) <= 2 and len(one_detections) >= 1
                     and len(detections_a) >= 2)
    checks["count_follows_observation"] = {"passed": counts_follow,
                                           "detail": {"noise_only": len(noise_detections),
                                                      "one_target": len(one_detections),
                                                      "three_targets": len(detections_a)}}

    if lut is None:
        checks["covariance_depends_only_on_observed_q"] = {"passed": False,
                                                           "detail": "covariance LUT missing; run calibrate_covariance first"}
    else:
        from frontend.sensing import coords

        covariance_ok = True
        detail = []
        for detection in detections_a:
            rebuilt = coords.covariance_from_lut(detection["r_m"], detection["u"], detection["peak_to_noise_db"],
                                                 boresight, lut, config["detector"]["covariance_floor_m2"])
            match = torch.allclose(torch.tensor(detection["C_xy"], dtype=torch.float64), rebuilt, atol=1e-12)
            covariance_ok &= match
            detail.append({"q_db": detection["peak_to_noise_db"], "matches_lut": bool(match)})
        checks["covariance_depends_only_on_observed_q"] = {"passed": covariance_ok, "detail": detail}

        sigmas = [(entry.get("q_max_db"), entry["sigma_r_m"], entry["sigma_u"]) for entry in lut["bins"]]
        monotone = all(sigmas[index][1] >= sigmas[index + 1][1] and sigmas[index][2] >= sigmas[index + 1][2]
                       for index in range(len(sigmas) - 1))
        checks["lut_sigma_monotone_conservative"] = {"passed": monotone, "detail": sigmas}

    summary = {"test": "T2", "checks": checks,
               "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in B_DOMAIN_FILES}}
    passed = all(entry["passed"] for entry in checks.values())
    _common.write_json(OUT / "summary.json", summary)
    _common.write_json(OUT / "checks.json", {"test": "T2", "passed": passed, "checks": checks})
    print(json.dumps({"T2": "PASS" if passed else "FAIL", "checks": {key: entry["passed"] for key, entry in checks.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()