"""FTEST-01: one-shot final-test evaluation of two frozen best checkpoints.

``--prepare`` verifies code, GPUs, model construction and checkpoint identity without
opening any test array.  ``--run --resume`` is the only path allowed to open test
inputs and labels.  No optimizer or training entry point is constructed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_bottleneck import metrics


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_native(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path, block=2**20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configuration(path):
    cfg = read_json(path)
    if cfg["schema_version"] != "FTEST-01-v1" or cfg["split"] != "test":
        raise ValueError("Final-test schema or split changed")
    if cfg["snr_db"] != [5, 10, 15, 20] or cfg["prediction_steps"] != 20:
        raise ValueError("Frozen final-test matrix changed")
    if set(cfg["models"]) != {"QGNN", "GNN"}:
        raise ValueError("Exactly the frozen QGNN and GNN are required")
    cfg["_path"] = str(Path(path).resolve())
    cfg["_sha256"] = sha256(path)
    return cfg


def locations(cfg):
    return ROOT / cfg["report_root"] / cfg["run_id"], ROOT / cfg["cache_root"] / cfg["run_id"]


def log(report, message):
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    with (report / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_model(cfg, name, device):
    from prediction import training
    from experiments.qgnn_directed_context import run as directed
    from experiments.qgnn_motionframe import run as motion
    spec = cfg["models"][name]
    model = (directed.Predictor("original", directed.configuration()) if spec["kind"] == "original"
             else motion.Predictor("gnn", motion.configuration()))
    payload = torch.load(ROOT / spec["path"], map_location="cpu", weights_only=False)
    if int(payload["progress"]["epoch"]) != spec["epoch"]:
        raise ValueError(f"{name} checkpoint epoch changed")
    training.load_weights(model, payload["model"])
    model.requires_grad_(False)
    return model.to(device).eval(), len(payload["model"])


def test_file_metadata(cfg):
    records = {}
    for relative, expected_size in cfg["expected_test_file_sizes"].items():
        path = ROOT / relative
        info = path.stat()
        if not path.is_file() or info.st_size != expected_size:
            raise ValueError(f"Test asset presence/size changed: {relative}")
        records[relative] = {"bytes": info.st_size, "content_opened": False}
    return records


def prepare(cfg):
    report, cache = locations(cfg)
    if (report / "summary.json").exists():
        raise FileExistsError("This one-shot final test is already complete")
    report.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (report / "tables").mkdir(exist_ok=True)
    (report / "source_snapshot").mkdir(exist_ok=True)
    records = test_file_metadata(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs are required")
    checkpoints = {}
    for name, spec in cfg["models"].items():
        path = ROOT / spec["path"]
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(f"{name} checkpoint SHA256 changed")
        model, keys = load_model(cfg, name, spec["device"])
        checkpoints[name] = {**spec, "bytes": path.stat().st_size, "model_keys": keys, "sha256_verified": True}
        del model
        torch.cuda.empty_cache()
    from experiments.qgnn_bottleneck.tests.test_metrics import run as metric_tests
    code_paths = [
        "experiments/qgnn_final_test/evaluate.py", "experiments/qgnn_bottleneck/metrics.py",
        "experiments/qgnn_directed_context/run.py", "experiments/qgnn_directed_context/graph.py",
        "experiments/qgnn_motionframe/run.py", "prediction/training.py", "prediction/temporal.py",
        "prediction/classical.py", "frontend/symbol_dataset.py", "configs/qgnn_final_test.json",
    ]
    code_hashes = {path: sha256(ROOT / path) for path in code_paths}
    for relative in code_paths:
        destination = report / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    manifest = {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "prepared_utc": utc_now(),
        "configuration_sha256": cfg["_sha256"], "configuration": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "purpose": "one-shot frozen-checkpoint final-test inference; no selection and no training",
        "models": checkpoints, "code_sha256": code_hashes, "test_file_metadata": records,
        "test_content_opened": False, "labels_opened": False, "optimizer_created": False,
        "optimizer_steps_executed": 0, "new_training_started": False,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "metric_unit_tests": metric_tests(),
    }
    write_json(report / "manifest.json", manifest)
    write_json(report / "access_log.json", {"prepared_utc": utc_now(), "test_content_opened": False,
                                              "labels_opened": False, "note": "Only path metadata was stat'ed during preparation."})
    write_json(report / "status.json", {"execution_status": "prepared", "updated_utc": utc_now(),
                                         "test_opened": False, "completed_cells": []})
    (report / "PROTOCOL.md").write_text(
        "# FTEST-01 最终测试协议\n\n"
        "固定比较师兄原始 QGNN 第 75 轮 best 与匹配强 GNN 第 82 轮 best；"
        "一次性评估 test 的 864 个 origin 和 5/10/15/20 dB。"
        "测试结果不得用于再选模型、调参或继续训练。ADE、FDE 分开报告，J 仅保留为辅助字段。\n",
        encoding="utf-8")
    log(report, "PREPARED: checkpoints/models/GPUs verified; test contents and labels remain unopened")
    return report


class SealedTestDataset:
    def __init__(self, cfg, report):
        from frontend.symbol_dataset import SharedPredictionInputs
        self.split = "test"
        self.loaders = {snr: SharedPredictionInputs(ROOT / f"data/f01d/inputs/test_snr_{snr}.npz")
                        for snr in cfg["snr_db"]}
        self.n = len(self.loaders[20])
        if self.n != cfg["test_origins"]:
            raise ValueError(f"Expected {cfg['test_origins']} test origins, found {self.n}")
        with np.load(ROOT / "data/f01d/labels/test.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"future_position", "label_valid"}:
                raise ValueError("Unexpected test-label fields")
            self.labels = {key: saved[key] for key in saved.files}
        future, valid = self.labels["future_position"], self.labels["label_valid"]
        if future.shape != (self.n, 20, 8, 2) or valid.shape != future.shape[:-1] or valid.dtype != bool:
            raise ValueError("Invalid test supervision shape/type")
        if not np.isfinite(future[valid]).all():
            raise ValueError("Non-finite valid test supervision")
        reference = self.loaders[20].arrays
        for loader in self.loaders.values():
            if (len(loader) != self.n or not np.array_equal(reference["timestamp"], loader.arrays["timestamp"])
                    or not np.array_equal(reference["track_exists"], loader.arrays["track_exists"])):
                raise ValueError("Test SNR origin/eligibility alignment failed")
        write_json(report / "access_log.json", {"opened_utc": utc_now(), "test_content_opened": True,
            "labels_opened": True, "opened_split": "test", "inputs": [5, 10, 15, 20],
            "metadata_json_opened": False, "purpose": "authorized one-shot frozen final evaluation"})

    def batch(self, ids, snr, device):
        ids = np.asarray(ids, dtype=np.int64)
        samples = [self.loaders[snr][int(i)] for i in ids]
        fields = ("state_hat", "standardized_state", "track_exists", "detected")
        inputs = {key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=device)
                  for key in fields}
        return inputs


def cache_contract(cfg, manifest, name, snr):
    return {"schema_version": cfg["schema_version"], "configuration_sha256": cfg["_sha256"],
            "model": name, "snr_db": snr, "checkpoint_sha256": manifest["models"][name]["sha256"],
            "code_sha256": manifest["code_sha256"], "test_file_size": cfg["expected_test_file_sizes"][f"data/f01d/inputs/test_snr_{snr}.npz"]}


def infer_cell(cfg, manifest, dataset, model, name, snr, cache, resume, report):
    target = cache / f"{name}__test__snr{snr}.npz"
    sidecar = target.with_suffix(".json")
    contract = cache_contract(cfg, manifest, name, snr)
    if resume and target.is_file() and sidecar.is_file() and read_json(sidecar) == contract:
        with np.load(target, allow_pickle=False) as saved:
            if set(saved.files) == {"prediction", "origin_eligible"} and saved["prediction"].shape == (dataset.n, 20, 8, 2):
                log(report, f"RESUME {name} SNR={snr}: verified completed cache reused")
                return saved["prediction"], saved["origin_eligible"]
    device = next(model.parameters()).device
    predictions, eligibilities = [], []
    batches = (dataset.n + cfg["inference_batch"] - 1) // cfg["inference_batch"]
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, dataset.n, cfg["inference_batch"]), 1):
            ids = np.arange(start, min(start + cfg["inference_batch"], dataset.n))
            inputs = dataset.batch(ids, snr, device)
            output = model(**inputs)
            predictions.append(output["prediction"].detach().cpu().numpy().astype(np.float32))
            eligibilities.append(output["origin_eligible"].detach().cpu().numpy().astype(bool))
            if batch_index == 1 or batch_index % 5 == 0 or batch_index == batches:
                elapsed = time.perf_counter() - started
                log(report, f"RUN {name} SNR={snr}: {batch_index}/{batches} batches, {min(start + len(ids), dataset.n)}/{dataset.n} origins, {elapsed:.1f}s")
    prediction, eligible = np.concatenate(predictions), np.concatenate(eligibilities)
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction, origin_eligible=eligible)
    temporary.replace(target)
    write_json(sidecar, contract)
    return prediction, eligible


def summarize(cfg, predictions, dataset, report):
    metric_rows, scene_table = [], []
    denominator_identity = {}
    paired = []
    for snr in cfg["snr_db"]:
        q_prediction, q_eligible = predictions[("QGNN", snr)]
        g_prediction, g_eligible = predictions[("GNN", snr)]
        if not np.array_equal(q_eligible, g_eligible):
            raise ValueError(f"QGNN/GNN origin eligibility differs at SNR={snr}")
        denominator_identity[str(snr)] = True
        by_model = {}
        for name, prediction in (("QGNN", q_prediction), ("GNN", g_prediction)):
            rows = metrics.evaluate_metrics(prediction, dataset.labels["future_position"],
                                            dataset.labels["label_valid"], q_eligible)
            final = next(row for row in rows if row["horizon_steps"] == 20)
            metric_rows.append({"model": name, "split": "test", "snr_db": snr, **final})
            scenes = metrics.scene_rows(prediction, dataset.labels["future_position"],
                                        dataset.labels["label_valid"], q_eligible)
            by_model[name] = scenes
            scene_table.extend({"model": name, "split": "test", "snr_db": snr, **row} for row in scenes)
        for q, g in zip(by_model["QGNN"], by_model["GNN"]):
            if (q["origin"], q["ade_targets"], q["fde_targets"]) != (g["origin"], g["ade_targets"], g["fde_targets"]):
                raise ValueError("Paired scene denominator mismatch")
            paired.append({"origin": q["origin"], "snr_db": snr,
                           "ade_targets": q["ade_targets"], "fde_targets": q["fde_targets"],
                           "QGNN_ADE": q["ADE"], "GNN_ADE": g["ADE"],
                           "QGNN_FDE": q["FDE"], "GNN_FDE": g["FDE"],
                           "QGNN_minus_GNN_ADE": None if q["ADE"] is None else q["ADE"] - g["ADE"],
                           "QGNN_minus_GNN_FDE": None if q["FDE"] is None else q["FDE"] - g["FDE"]})
    overall = {}
    for name in ("QGNN", "GNN"):
        rows = [row for row in metric_rows if row["model"] == name]
        ade, fde = np.mean([row["ADE"] for row in rows]), np.mean([row["FDE"] for row in rows])
        overall[name] = {"ADE": float(ade), "FDE": float(fde), "J_auxiliary": float(ade + .5 * fde),
                         "aggregation": "scene macro within SNR, then equal macro over four SNRs"}
    q, g = overall["QGNN"], overall["GNN"]
    comparison = {
        "QGNN_minus_GNN_ADE": q["ADE"] - g["ADE"], "QGNN_minus_GNN_FDE": q["FDE"] - g["FDE"],
        "QGNN_improvement_percent_ADE": 100 * (1 - q["ADE"] / g["ADE"]),
        "QGNN_improvement_percent_FDE": 100 * (1 - q["FDE"] / g["FDE"]),
        "interpretation": "Positive improvement means QGNN error is lower; ADE and FDE are the reported outcomes.",
    }
    write_csv(report / "tables/metrics_by_snr.csv", metric_rows)
    write_csv(report / "tables/per_scene.csv", scene_table)
    write_csv(report / "tables/paired_scene_comparison.csv", paired)
    return {"overall": overall, "comparison": comparison, "common_denominators_verified": all(denominator_identity.values()),
            "denominator_identity_by_snr": denominator_identity, "test_origins": dataset.n,
            "snr_db": cfg["snr_db"], "model_selection_after_test": False, "J_is_auxiliary_only": True}


def run(cfg, resume):
    report, cache = locations(cfg)
    manifest_path = report / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run --prepare before opening the test set")
    if (report / "summary.json").exists():
        raise FileExistsError("One-shot final test already completed; overwrite refused")
    manifest = read_json(manifest_path)
    if manifest["configuration_sha256"] != cfg["_sha256"] or manifest["test_content_opened"]:
        raise ValueError("Prepared manifest is not an unopened match for this configuration")
    for name, spec in cfg["models"].items():
        if sha256(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"{name} checkpoint changed after preparation")
    status = {"execution_status": "running", "updated_utc": utc_now(), "test_opened": True, "completed_cells": []}
    write_json(report / "status.json", status)
    log(report, "FINAL TEST OPEN AUTHORIZED: loading the sealed test split; no later selection or training")
    dataset = SealedTestDataset(cfg, report)
    models, state_before = {}, {}
    for name, spec in cfg["models"].items():
        models[name], _ = load_model(cfg, name, spec["device"])
        state_before[name] = tensor_digest(models[name].state_dict())
        log(report, f"LOADED {name}: frozen epoch={spec['epoch']} on {spec['device']}")
    predictions = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="final-test") as pool:
        for snr in cfg["snr_db"]:
            jobs = {name: pool.submit(infer_cell, cfg, manifest, dataset, models[name], name, snr,
                                      cache, resume, report) for name in ("QGNN", "GNN")}
            for name, job in jobs.items():
                predictions[(name, snr)] = job.result()
            for name in ("QGNN", "GNN"):
                status["completed_cells"].append({"model": name, "snr_db": snr})
            status["updated_utc"] = utc_now()
            write_json(report / "status.json", status)
    state_after = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    if state_after != state_before:
        raise ValueError("Frozen model state changed during final inference")
    checkpoint_after = {name: sha256(ROOT / spec["path"]) for name, spec in cfg["models"].items()}
    if any(checkpoint_after[name] != spec["sha256"] for name, spec in cfg["models"].items()):
        raise ValueError("Checkpoint file changed during final inference")
    summary = summarize(cfg, predictions, dataset, report)
    summary.update({"schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "status": "complete",
                    "completed_utc": utc_now(), "split": "test", "checkpoints_unchanged": True,
                    "model_states_unchanged": True, "optimizer_created": False,
                    "optimizer_steps_executed": 0, "new_training_started": False})
    write_json(report / "summary.json", summary)
    lines = ["# QGNN 与 GNN 最终测试结果", "", "固定 best 检查点、864 个 test origin、5/10/15/20 dB；测试后不再选模或调参。", "",
             "| 模型 | ADE (m) | FDE (m) |", "|---|---:|---:|",
             *[f"| {name} | {summary['overall'][name]['ADE']:.6f} | {summary['overall'][name]['FDE']:.6f} |" for name in ("QGNN", "GNN")], "",
             f"QGNN 相对 GNN：ADE 改善 {summary['comparison']['QGNN_improvement_percent_ADE']:.3f}%，"
             f"FDE 改善 {summary['comparison']['QGNN_improvement_percent_FDE']:.3f}%。",
             "ADE 与 FDE 分开作为最终结果；J 未用于最终结论。四档 SNR 使用共同样本与共同分母。"]
    (report / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report / "status.json", {"execution_status": "complete", "updated_utc": utc_now(),
                                         "test_opened": True, "completed_cells": status["completed_cells"]})
    log(report, "COMPLETE: final test summary and paired tables written")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qgnn_final_test.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = configuration(args.config)
    if args.prepare:
        if args.resume:
            parser.error("--resume applies only to --run")
        print(json.dumps({"status": "prepared", "report": str(prepare(cfg)), "test_opened": False}, indent=2))
    else:
        run(cfg, args.resume)


if __name__ == "__main__":
    main()
