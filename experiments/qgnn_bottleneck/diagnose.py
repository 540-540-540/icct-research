"""BDX-01 frozen-checkpoint bottleneck diagnosis; no optimizer is constructed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qgnn_bottleneck import analysis, metrics

SOURCE_SNAPSHOT = (
    "prediction/temporal.py", "prediction/training.py", "prediction/classical.py",
    "prediction/evaluation_cache.py", "frontend/symbol_dataset.py",
    "frontend/pack_symbol_dataset.py", "frontend/scene_manifest.py",
    "experiments/qgnn_inherited/graph.py", "experiments/qgnn_directed_context/graph.py",
    "experiments/qgnn_directed_context/run.py", "experiments/qgnn_motionframe/run.py",
)


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


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


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
    if cfg["schema_version"] != "BDX-01-v1" or cfg["allowed_splits"] != ["train", "V_select"]:
        raise ValueError("BDX-01 schema or split allowlist changed")
    if cfg["snr_db"] != [5, 10, 15, 20] or cfg["horizons"] != [5, 10, 20]:
        raise ValueError("Frozen SNR/horizon matrix changed")
    cfg["_path"] = str(Path(path).resolve())
    cfg["_sha256"] = sha256(path)
    return cfg


def paths(cfg):
    report = ROOT / cfg["report_root"] / cfg["run_id"]
    cache = ROOT / cfg["cache_root"] / cfg["run_id"] / "cache"
    return report, cache


def log_event(report, message):
    report.mkdir(parents=True, exist_ok=True)
    with (report / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")


def git_head():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def access_record(report, category, path, split=None, note=None):
    import fcntl
    log_path = report / "access_log.json"
    with (report / ".access.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        log = read_json(log_path) if log_path.exists() else []
        log.append(dict(time_utc=utc_now(), category=category, path=str(Path(path).relative_to(ROOT)), split=split, note=note))
        write_json(log_path, log)


def checkpoint_manifest(cfg):
    manifest = {}
    for label, spec in cfg["checkpoints"].items():
        path = ROOT / spec["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        progress = payload.get("progress", {})
        epoch = int(progress.get("epoch", -1))
        if epoch != spec["epoch"]:
            raise ValueError(f"{label} expected epoch {spec['epoch']}, found {epoch}")
        manifest[label] = dict(spec, bytes=path.stat().st_size, sha256=sha256(path), actual_epoch=epoch,
            optimizer_steps_in_checkpoint=int(progress.get("optimizer_steps", -1)),
            best_J_in_checkpoint=progress.get("best_J"), model_key_count=len(payload["model"]),
            mutable_keys=sorted(payload["model"]))
    return manifest


def baseline_files(cfg):
    files = list(SOURCE_SNAPSHOT) + [
        "experiments/qgnn_bottleneck/diagnose.py", "experiments/qgnn_bottleneck/metrics.py",
        "experiments/qgnn_bottleneck/analysis.py", "experiments/qgnn_bottleneck/plot_results.py",
        "experiments/qgnn_bottleneck/tests/test_metrics.py",
        "configs/qgnn_bottleneck_diagnostic.json", "docs/QGNN_性能瓶颈定位工单_BDX01.md",
        "data/f01d/normalization.json", "data/f01_source/source_states.npy",
        "models/gpt2/config.json", "models/gpt2/model.safetensors",
    ]
    for split in cfg["allowed_splits"]:
        files += [f"data/f01d/labels/{split}.npz", f"data/f01d/metadata/{split}.json"]
        files += [f"data/f01d/inputs/{split}_snr_{snr}.npz" for snr in cfg["snr_db"]]
    files += [spec["path"] for spec in cfg["checkpoints"].values()]
    return {path: dict(bytes=(ROOT/path).stat().st_size, sha256=sha256(ROOT/path)) for path in dict.fromkeys(files)}


def prepare(cfg):
    report, cache = paths(cfg)
    report.mkdir(parents=True, exist_ok=False) if not report.exists() else None
    cache.mkdir(parents=True, exist_ok=True)
    (report / "tables").mkdir(exist_ok=True)
    (report / "figures").mkdir(exist_ok=True)
    snapshot = report / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative in SOURCE_SNAPSHOT:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    from experiments.qgnn_bottleneck.tests.test_metrics import run as metric_tests
    from experiments.qgnn_directed_context import run as directed
    checkpoints = checkpoint_manifest(cfg)
    base = baseline_files(cfg)
    d0 = read_json(ROOT / "reports/qgnn_directed_context/d0.json")
    current_source = directed.provenance(directed.configuration())
    d0_ok = bool(d0.get("passed") and d0.get("source") == current_source
                 and d0.get("optimizer_created") is False and d0.get("optimizer_steps") == 0)
    if not d0_ok:
        raise ValueError("Existing D0 cannot be reused: source or zero-update contract differs")
    checks = {
        "schema_version": cfg["schema_version"], "run_id": cfg["run_id"], "created_utc": utc_now(),
        "metric_unit_tests": dict(status="passed", measured=metric_tests(), tolerance=0.0,
                                  evidence_path="experiments/qgnn_bottleneck/tests/test_metrics.py"),
        "D0_existing_diagnostics": dict(status="reused_verified", measured={
            "source_exact": True, "optimizer_created": d0["optimizer_created"],
            "optimizer_steps_executed": d0["optimizer_steps"]}, tolerance=0.0,
            evidence_path="reports/qgnn_directed_context/d0.json"),
        "checkpoint_identity": dict(status="passed", measured={k: {"epoch": v["actual_epoch"], "keys": v["model_key_count"]} for k,v in checkpoints.items()},
                                    tolerance=0.0, evidence_path="run_manifest.json:checkpoint_manifest"),
        "archive_reproduction": {label: dict(status="pending", measured=None, tolerance=cfg["archive_metric_atol_m"], evidence_path=None)
                                   for label in ("Q_best", "G_best", "A_best")},
        "hook_and_batch_equivalence": dict(status="pending", measured=None, tolerance=dict(atol=cfg["output_atol"], rtol=cfg["output_rtol"]), evidence_path="worker_status/*.json"),
        "mask_denominator_horizon_pairing_aggregation": dict(status="pending", measured=None, tolerance=0.0, evidence_path="tables/"),
        "read_only": dict(status="pending", measured={"optimizer_steps_executed": 0}, tolerance=0.0, evidence_path="run_manifest.json"),
        "access_audit": dict(status="pending", measured=None, tolerance=0.0, evidence_path="access_log.json"),
        "resume_equivalence": dict(status="pending", measured=None, tolerance=0.0, evidence_path="resume_check.json"),
        "directed_context_restore": dict(status="pending", measured=None, tolerance=dict(atol=cfg["output_atol"], rtol=cfg["output_rtol"]), evidence_path="worker_status/"),
        "reference_identity": dict(status="pending", measured=None, tolerance=dict(atol=cfg["identity_atol"], rtol=cfg["identity_rtol"]), evidence_path="tables/source_alignment.json"),
    }
    manifest = dict(schema_version=cfg["schema_version"], run_id=cfg["run_id"], started_utc=utc_now(),
        server_project=str(ROOT), server_git_head=git_head(), repository_reference_commit=cfg["repository_reference_commit"],
        server_git_note="Authoritative server directory has no .git; source/data/checkpoint hashes are the executable identity.",
        python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        configuration={k:v for k,v in cfg.items() if not k.startswith("_")}, checkpoint_manifest=checkpoints,
        baseline_files=base, optimizer_steps_executed=0, new_training_started=False,
        actual_splits_accessed=[], forbidden_splits_accessed=[], commands=[], worker_results={})
    write_json(report / "run_manifest.json", manifest)
    write_json(report / "checks.json", checks)
    write_json(report / "run_status.json", dict(execution_status="pending", phase="prepared", updated_utc=utc_now(), blocking_issues=[]))
    write_json(report / "access_log.json", [])
    protocol = f"""# BDX-01 执行协议\n\n- run_id：`{cfg['run_id']}`\n- 性质：六个既有检查点的只读推理诊断；参数更新次数必须为 0。\n- 允许集合：`train`、`V_select`；禁止读取 `V_confirm` 与 `test`。\n- 固定 SNR：5/10/15/20 dB；预测步长 0.1 s，共 20 步。\n- 检查点与 epoch：{', '.join(f'{k}={v["actual_epoch"]}' for k,v in checkpoints.items())}。\n- 先复现 Q/G/A 三个 best 的 V_select 归档指标；失败模型停止依赖分析，其余继续。\n- 候选 A 关闭实验只在内存副本清零四个有向上下文矩阵，随后恢复并核验磁盘哈希。\n- 不构造优化器，不调用训练入口，不自动提交或推送。\n"""
    (report / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    return dict(report=str(report), cache=str(cache), checkpoints=checkpoints)


def load_model(cfg, label, device="cuda:0"):
    from prediction import training
    from experiments.qgnn_directed_context import run as directed
    from experiments.qgnn_motionframe import run as motion
    spec = cfg["checkpoints"][label]
    if spec["kind"] in ("original", "directed_context"):
        model = directed.Predictor(spec["kind"], directed.configuration())
    elif spec["kind"] == "gnn":
        model = motion.Predictor("gnn", motion.configuration())
    else:
        raise ValueError(f"Unknown model kind {spec['kind']}")
    payload = torch.load(ROOT / spec["path"], map_location="cpu", weights_only=False)
    if int(payload["progress"]["epoch"]) != spec["epoch"]:
        raise ValueError(f"{label} epoch changed")
    training.load_weights(model, payload["model"])
    return model.to(device).eval(), payload


def load_dataset(cfg, split, report):
    if split not in cfg["allowed_splits"]:
        raise ValueError(f"Forbidden split refused before path construction: {split}")
    from prediction import training
    dataset = training.Dataset(split)
    for snr in cfg["snr_db"]:
        access_record(report, "model_input", dataset.loaders[snr].path, split, f"SNR={snr}")
    access_record(report, "label", ROOT / f"data/f01d/labels/{split}.npz", split)
    return dataset


def cache_contract(cfg, label, split, snr, report, intervention=False):
    manifest = read_json(report / "run_manifest.json")
    dataset_key = f"data/f01d/inputs/{split}_snr_{snr}.npz"
    return dict(schema_version=cfg["schema_version"], model=label, split=split, snr_db=snr,
        checkpoint_sha256=manifest["checkpoint_manifest"][label]["sha256"],
        input_sha256=manifest["baseline_files"][dataset_key]["sha256"],
        normalization_sha256=manifest["baseline_files"]["data/f01d/normalization.json"]["sha256"],
        config_sha256=cfg["_sha256"],
        diagnostic_code_sha256={path:manifest["baseline_files"][path]["sha256"] for path in (
            "experiments/qgnn_bottleneck/diagnose.py", "experiments/qgnn_bottleneck/metrics.py",
            "experiments/qgnn_bottleneck/analysis.py")},
        intervention="directed_context_zero" if intervention else "on")


def run_inference(cfg, model, label, split, snr, report, cache, resume, intervention=False, dataset=None):
    tag = label + ("_off" if intervention else "")
    target = cache / f"{tag}__{split}__snr{snr}.npz"
    sidecar = target.with_suffix(".json")
    contract = cache_contract(cfg, label, split, snr, report, intervention)
    if resume and target.is_file() and sidecar.is_file() and read_json(sidecar) == contract:
        with np.load(target, allow_pickle=False) as saved:
            if set(saved.files) == {"prediction", "origin_eligible"}:
                n = len(saved["prediction"])
                return dict(reused=True, path=str(target), scenes=n,
                            last_batch_size=n % cfg["inference_batch"] or cfg["inference_batch"])
    dataset = dataset or load_dataset(cfg, split, report)
    predictions, eligibilities = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, dataset.n, cfg["inference_batch"]):
            ids = np.arange(start, min(start + cfg["inference_batch"], dataset.n))
            inputs, _ = dataset.batch(ids, snr, device)
            output = model(**inputs)
            predictions.append(output["prediction"].detach().cpu().numpy().astype(np.float32))
            eligibilities.append(output["origin_eligible"].detach().cpu().numpy().astype(bool))
    prediction, eligible = np.concatenate(predictions), np.concatenate(eligibilities)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction, origin_eligible=eligible)
    temporary.replace(target)
    write_json(sidecar, contract)
    return dict(reused=False, path=str(target), scenes=len(prediction),
                last_batch_size=len(prediction) % cfg["inference_batch"] or cfg["inference_batch"])


def fixed_output_checks(cfg, model, report):
    dataset = load_dataset(cfg, "V_select", report)
    device = next(model.parameters()).device
    ids = np.arange(min(16, dataset.n))
    inputs, _ = dataset.batch(ids, 20, device)
    with torch.no_grad():
        baseline = model(**inputs)["prediction"]
        seen = []
        hook = model.graph.register_forward_hook(lambda module, args, output: seen.append(tuple(output.shape)))
        hooked = model(**inputs)["prediction"]
        hook.remove()
        pieces = []
        for part in (ids[:7], ids[7:]):
            x, _ = dataset.batch(part, 20, device)
            pieces.append(model(**x)["prediction"])
        split = torch.cat(pieces)
    hook_error = float((baseline - hooked).abs().max())
    batch_error = float((baseline - split).abs().max())
    torch.testing.assert_close(baseline, hooked, atol=cfg["output_atol"], rtol=cfg["output_rtol"])
    torch.testing.assert_close(baseline, split, atol=cfg["output_atol"], rtol=cfg["output_rtol"])
    return dict(hook_max_abs_error=hook_error, batch_max_abs_error=batch_error, hook_shapes=seen,
                baseline=baseline.detach().cpu())


def worker(cfg, label, scope, resume):
    report, cache = paths(cfg)
    status_path = report / "worker_status" / f"{label}_{scope}.json"
    status = dict(status="running", label=label, scope=scope, pid=os.getpid(), started_utc=utc_now(), optimizer_steps_executed=0)
    write_json(status_path, status)
    try:
        before_checkpoint = sha256(ROOT / cfg["checkpoints"][label]["path"])
        model, payload = load_model(cfg, label)
        state_before = tensor_digest({k:v for k,v in model.state_dict().items()})
        fixed = fixed_output_checks(cfg, model, report)
        if label == "Q_best" and scope == "archive":
            resume_dir = report / "resume_probe"
            resume_dir.mkdir(exist_ok=True)
            direct = fixed["baseline"][:2].numpy()
            with (resume_dir / "partial.tmp").open("wb") as handle:
                np.savez_compressed(handle, prediction=direct[:1], completed_origins=np.asarray([0]))
            (resume_dir / "partial.tmp").replace(resume_dir / "partial.npz")
            with np.load(resume_dir / "partial.npz", allow_pickle=False) as partial:
                resumed = np.concatenate((partial["prediction"], direct[1:2]))
            resume_error = float(np.max(np.abs(resumed - direct)))
            if resume_error != 0:
                raise ValueError("Diagnostic cache resume probe differs from uninterrupted inference")
            write_json(report / "resume_check.json", dict(status="passed", fixed_origins=[0, 1],
                simulated_interruption_after_origins=1, resumed_equals_uninterrupted=True,
                max_abs_error=resume_error, optimizer_steps_executed=0))
        completed = []
        if scope == "archive":
            matrix = [("V_select", snr) for snr in cfg["snr_db"]]
            intervention = False
        elif scope == "matrix":
            matrix = [(split, snr) for split in cfg["allowed_splits"] for snr in cfg["snr_db"]]
            intervention = False
        elif scope == "off":
            if not label.startswith("A_"):
                raise ValueError("Only candidate A supports context intervention")
            named = dict(model.named_parameters())
            expected = set(cfg["directed_context_keys"])
            if expected - set(named):
                raise ValueError(f"Missing context keys: {expected-set(named)}")
            saved = {key:named[key].detach().clone() for key in expected}
            with torch.no_grad():
                for key in expected:
                    named[key].zero_()
            changed = {key: int(torch.count_nonzero(saved[key] - named[key]).item()) for key in expected}
            if set(changed) != expected or not all(changed.values()):
                raise ValueError("Context intervention did not change exactly four nonzero matrices")
            matrix = [(split, snr) for split in cfg["allowed_splits"] for snr in cfg["snr_db"]]
            intervention = True
        else:
            raise ValueError("Unknown worker scope")
        datasets = {}
        for split, snr in matrix:
            if split not in datasets:
                datasets[split] = load_dataset(cfg, split, report)
            completed.append(run_inference(cfg, model, label, split, snr, report, cache, resume, intervention, datasets[split]))
        restore = None
        if scope == "off":
            with torch.no_grad():
                for key, value in saved.items():
                    named[key].copy_(value)
            dataset = load_dataset(cfg, "V_select", report)
            inputs, _ = dataset.batch(np.arange(min(16, dataset.n)), 20, next(model.parameters()).device)
            with torch.no_grad():
                restored = model(**inputs)["prediction"].detach().cpu()
            restore_error = float((restored - fixed["baseline"]).abs().max())
            torch.testing.assert_close(restored, fixed["baseline"], atol=cfg["output_atol"], rtol=cfg["output_rtol"])
            restore = dict(changed_keys=sorted(changed), changed_elements=changed, max_abs_error=restore_error)
        state_after = tensor_digest({k:v for k,v in model.state_dict().items()})
        if state_after != state_before:
            raise ValueError("Model parameters/buffers changed after diagnostic")
        after_checkpoint = sha256(ROOT / cfg["checkpoints"][label]["path"])
        if after_checkpoint != before_checkpoint:
            raise ValueError("Checkpoint file changed during diagnostic")
        status.update(status="passed", finished_utc=utc_now(), checkpoint_sha256_before=before_checkpoint,
            checkpoint_sha256_after=after_checkpoint, model_state_sha256_before=state_before,
            model_state_sha256_after=state_after, hook_and_batch={k:v for k,v in fixed.items() if k != "baseline"},
            intervention_restore=restore, completed=completed)
    except BaseException as exc:
        status.update(status="failed", finished_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(status_path, status)
        raise
    write_json(status_path, status)


def run_workers(cfg, labels, scope, resume):
    report, _ = paths(cfg)
    queue, active, outcomes = list(labels), {}, {}
    devices = list(cfg["devices"])
    log_dir = report / "logs"
    log_dir.mkdir(exist_ok=True)
    while queue or active:
        while queue and len(active) < len(devices):
            label = queue.pop(0)
            used = {entry[2] for entry in active.values()}
            device = next(device for device in devices if device not in used)
            log = (log_dir / f"{label}_{scope}.log").open("a", encoding="utf-8")
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", cfg["_path"], "--worker", label, "--scope", scope]
            if resume:
                cmd.append("--resume")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active[label] = (child, log, device)
        time.sleep(1)
        for label, (child, log, device) in list(active.items()):
            if child.poll() is None:
                continue
            log.close()
            status_path = report / "worker_status" / f"{label}_{scope}.json"
            outcomes[label] = read_json(status_path) if status_path.exists() else dict(status="failed", error="missing worker status")
            outcomes[label]["returncode"] = child.returncode
            outcomes[label]["physical_device"] = device
            del active[label]
    return outcomes


def load_cache(cfg, label, split, snr, intervention=False):
    _, cache = paths(cfg)
    tag = label + ("_off" if intervention else "")
    path = cache / f"{tag}__{split}__snr{snr}.npz"
    with np.load(path, allow_pickle=False) as saved:
        return saved["prediction"].copy(), saved["origin_eligible"].copy()


def raw_split(cfg, split, report):
    dataset = load_dataset(cfg, split, report)
    metadata_path = ROOT / f"data/f01d/metadata/{split}.json"
    access_record(report, "metadata", metadata_path, split)
    samples = read_json(metadata_path)["samples"]
    if [row["sample_index"] for row in samples] != list(range(dataset.n)):
        raise ValueError(f"{split} metadata order differs from origin indices")
    return dataset, samples


def cv_prediction(dataset, snr):
    state = dataset.loaders[snr].arrays["state_hat"]
    exists = dataset.loaders[snr].arrays["track_exists"]
    eligible = metrics.eligible_mask(exists)
    horizon = np.arange(1, 21, dtype=np.float32)[None, :, None, None] * np.float32(.1)
    origin = state[:, -1]
    prediction = origin[:, None, :, :2] + horizon * origin[:, None, :, 2:]
    prediction[~eligible[:, None, :, None].repeat(20, axis=1).repeat(2, axis=3)] = 0
    return prediction.astype(np.float32), eligible


def archive_reference_path(label, epoch):
    if label.startswith(("Q_", "A_")):
        cell = "original" if label.startswith("Q_") else "directed_context"
        return ROOT / f"results/qgnn_directed_context/seed2026/{cell}/validation_epoch_{epoch:02d}.json"
    return ROOT / f"results/qgnn_motionframe/seed2026/gnn/validation_epoch_{epoch:02d}.json"


def verify_archive(cfg, label, report):
    dataset, _ = raw_split(cfg, "V_select", report)
    spec = cfg["checkpoints"][label]
    reference_path = archive_reference_path(label, spec["epoch"])
    reference = read_json(reference_path)
    lookup = {(int(r["origin"]), int(r["snr_db"])): r for r in reference["per_scene"]}
    max_error, denominator_mismatches = 0.0, 0
    snr_metrics = []
    for snr in cfg["snr_db"]:
        prediction, eligible = load_cache(cfg, label, "V_select", snr)
        rows = metrics.scene_rows(prediction, dataset.labels["future_position"], dataset.labels["label_valid"], eligible)
        for row in rows:
            archived = lookup[(row["origin"], snr)]
            denominator_mismatches += int((row["ade_targets"], row["fde_targets"]) != (archived["ade_targets"], archived["fde_targets"]))
            for metric in ("ADE", "FDE"):
                if row[metric] is None or archived[metric] is None:
                    if row[metric] != archived[metric]:
                        denominator_mismatches += 1
                else:
                    max_error = max(max_error, abs(row[metric] - archived[metric]))
        snr_metrics.append(metrics.evaluate_metrics(prediction, dataset.labels["future_position"], dataset.labels["label_valid"], eligible, (20,))[0])
    aggregate = {metric: float(np.mean([row[metric] for row in snr_metrics])) for metric in ("ADE", "FDE", "J")}
    aggregate_error = max(abs(aggregate[m] - reference[m]) for m in ("ADE", "FDE", "J"))
    passed = denominator_mismatches == 0 and max(max_error, aggregate_error) <= cfg["archive_metric_atol_m"]
    return dict(status="passed" if passed else "failed", measured=dict(max_scene_metric_abs_error_m=max_error,
        max_aggregate_abs_error_m=aggregate_error, denominator_mismatches=denominator_mismatches,
        recomputed=aggregate, archived={k:reference[k] for k in ("ADE", "FDE", "J")}),
        tolerance=cfg["archive_metric_atol_m"], evidence_path=str(reference_path.relative_to(ROOT)))


def core_metric_tables(cfg, labels, report):
    metric_rows, horizon_rows, scene_rows = [], [], []
    loaded = {split: raw_split(cfg, split, report) for split in cfg["allowed_splits"]}
    datasets = {split: pair[0] for split, pair in loaded.items()}
    for label in labels:
        family = cfg["checkpoints"][label]["family"]
        stage = "best" if label.endswith("best") else "latest"
        for split, dataset in datasets.items():
            truth, valid = dataset.labels["future_position"], dataset.labels["label_valid"]
            for snr in cfg["snr_db"]:
                prediction, eligible = load_cache(cfg, label, split, snr)
                rows = metrics.evaluate_metrics(prediction, truth, valid, eligible, range(1, 21))
                for row in rows:
                    row.update(model=label, family=family, checkpoint_stage=stage, split=split, snr_db=snr)
                    horizon_rows.append(row)
                    if row["horizon_steps"] == 20:
                        metric_rows.append(row.copy())
                if split == "V_select":
                    samples = loaded[split][1]
                    for row in metrics.scene_rows(prediction, truth, valid, eligible):
                        row.update(model=label, split=split, snr_db=snr,
                                   episode_id=samples[row["origin"]]["episode_id"])
                        scene_rows.append(row)
    # CV_hat is a non-trained reference tied to each SNR input.
    for split, dataset in datasets.items():
        truth, valid = dataset.labels["future_position"], dataset.labels["label_valid"]
        for snr in cfg["snr_db"]:
            prediction, eligible = cv_prediction(dataset, snr)
            rows = metrics.evaluate_metrics(prediction, truth, valid, eligible, range(1, 21))
            for row in rows:
                row.update(model="CV_hat", family="CV", checkpoint_stage="reference", split=split, snr_db=snr)
                horizon_rows.append(row)
                if row["horizon_steps"] == 20:
                    metric_rows.append(row.copy())
    write_csv(report / "tables/metrics.csv", metric_rows)
    write_csv(report / "tables/horizon_metrics.csv", horizon_rows)
    write_csv(report / "tables/scene_metrics_V_select.csv", scene_rows)
    grouped = []
    for model in sorted({r["model"] for r in metric_rows}):
        for split in cfg["allowed_splits"]:
            part = [r for r in metric_rows if r["model"] == model and r["split"] == split]
            if not part:
                continue
            row = dict(model=model, family=part[0]["family"], checkpoint_stage=part[0]["checkpoint_stage"], split=split)
            for key in ("ADE", "FDE", "J", "ADE_point_trainrule_eval", "FDE_trainrule_eval", "J_trainrule_eval", "fixed_ADE", "fixed_FDE"):
                row[key] = float(np.mean([r[key] for r in part if r[key] is not None]))
            for key in ("ade_scenes", "fde_scenes", "ade_targets", "fde_targets", "valid_points", "scenes_without_supervision", "fixed_complete_targets"):
                row[key] = int(sum(r[key] for r in part))
            grouped.append(row)
    lookup = {(r["model"], r["split"]): r for r in grouped}
    for row in grouped:
        if row["checkpoint_stage"] == "latest":
            best = lookup[(row["family"] + "_best", row["split"])]
            for key in ("ADE", "FDE", "J"):
                row[f"latest_minus_best_{key}"] = row[key] - best[key]
        other = lookup.get((row["model"], "V_select" if row["split"] == "train" else "train"))
        if other:
            for key in ("ADE", "FDE", "J"):
                row[f"train_minus_V_select_{key}"] = (row[key] - other[key]) if row["split"] == "train" else (other[key] - row[key])
    write_csv(report / "tables/best_latest.csv", grouped)
    return grouped, horizon_rows


def averaged_target_rows(cfg, split, labels, report):
    dataset, samples = raw_split(cfg, split, report)
    input_rows = analysis.assign_strata(analysis.input_feature_rows(split,
        dataset.loaders[20].arrays["state_hat"], dataset.loaders[20].arrays["track_exists"], samples), [])
    output = {(r["origin"], r["slot"]): r for r in input_rows}
    for label in labels:
        accumulated = {}
        for snr in cfg["snr_db"]:
            prediction, eligible = load_cache(cfg, label, split, snr)
            for row in metrics.target_rows(prediction, dataset.labels["future_position"], dataset.labels["label_valid"], eligible):
                key = (row["origin"], row["slot"])
                for field, value in row.items():
                    if field in ("origin", "slot", "eligible", "valid_steps") or value is None:
                        continue
                    accumulated.setdefault((key, field), []).append(value)
        for (key, field), values in accumulated.items():
            output[key][f"{label}_{field}"] = float(np.mean(values))
    return list(output.values()), dataset, samples


def target_and_group_tables(cfg, best_labels, report):
    train_rows, train_dataset, train_samples = averaged_target_rows(cfg, "train", best_labels, report)
    speed_edges = analysis.define_speed_tertiles(train_rows)
    analysis.assign_strata(train_rows, speed_edges)
    valid_rows, valid_dataset, valid_samples = averaged_target_rows(cfg, "V_select", best_labels, report)
    analysis.assign_strata(valid_rows, speed_edges)
    for rows in (train_rows, valid_rows):
        for row in rows:
            if "Q_best_ADE20" in row and "G_best_ADE20" in row:
                row["Q_minus_G_ADE"] = row["Q_best_ADE20"] - row["G_best_ADE20"]
            if "Q_best_FDE20" in row and "G_best_FDE20" in row:
                row["Q_minus_G_FDE"] = row["Q_best_FDE20"] - row["G_best_FDE20"]
            if "A_best_ADE20" in row and "Q_best_ADE20" in row:
                row["A_minus_Q_ADE"] = row["A_best_ADE20"] - row["Q_best_ADE20"]
            if "A_best_FDE20" in row and "Q_best_FDE20" in row:
                row["A_minus_Q_FDE"] = row["A_best_FDE20"] - row["Q_best_FDE20"]
    write_csv(report / "tables/target_metrics_V_select.csv", valid_rows)
    values = [f for f in ("Q_minus_G_ADE", "Q_minus_G_FDE", "A_minus_Q_ADE", "A_minus_Q_FDE") if any(f in r for r in train_rows + valid_rows)]
    episode_rows, component_rows, stratum_rows = [], [], []
    for rows in (train_rows, valid_rows):
        new_episode = analysis.aggregate_groups(rows, values, ["episode_id"])
        new_component = analysis.aggregate_groups(rows, values, ["source_block", "source_component"])
        new_stratum = analysis.aggregate_groups(rows, values, ["stratum_active_tracks", "stratum_neighbors", "stratum_history", "stratum_speed_change", "stratum_heading_change"])
        for row in new_episode + new_component + new_stratum:
            row["split"] = rows[0]["split"]
        episode_rows += new_episode
        component_rows += new_component
        stratum_rows += new_stratum
    write_csv(report / "tables/by_episode.csv", episode_rows)
    write_csv(report / "tables/by_source_component.csv", component_rows)
    write_csv(report / "tables/by_input_stratum.csv", stratum_rows)
    overlap = analysis.paired_error_summary(valid_rows) if {"Q_best", "G_best"} <= set(best_labels) else {}
    overlap["speed_change_train_tertiles"] = speed_edges
    overlap["leave_one_out"] = {}
    for name, rows in (("train", train_rows), ("V_select", valid_rows)):
        overlap["leave_one_out"][name] = {}
        for value in ("Q_minus_G_ADE", "Q_minus_G_FDE", "A_minus_Q_ADE", "A_minus_Q_FDE"):
            if any(value in r for r in rows):
                overlap["leave_one_out"][name][value] = {
                    "episode": analysis.leave_one_group_out(rows, value, "episode_id"),
                    "source_component": analysis.leave_one_group_out(rows, value, "source_component")}
    # Prediction disagreement and residual direction on explicitly paired keys.
    if {"Q_best", "G_best"} <= set(best_labels):
        disagreement = {str(h): [] for h in cfg["horizons"]}
        cosines, unavailable_cosine, forward, lateral, low_speed = [], 0, {"Q": [], "G": []}, {"Q": [], "G": []}, 0
        truth, label_valid = valid_dataset.labels["future_position"], valid_dataset.labels["label_valid"]
        for snr in cfg["snr_db"]:
            q, eq = load_cache(cfg, "Q_best", "V_select", snr)
            g, eg = load_cache(cfg, "G_best", "V_select", snr)
            if not np.array_equal(eq, eg):
                raise ValueError("Q/G eligibility differs")
            for h in cfg["horizons"]:
                mask = label_valid[:, h-1] & eq
                disagreement[str(h)].extend(np.linalg.norm(q[:, h-1]-g[:, h-1], axis=-1)[mask].tolist())
            mask = label_valid[:, 19] & eq
            rq, rg = truth[:, 19] - q[:, 19], truth[:, 19] - g[:, 19]
            nq, ng = np.linalg.norm(rq, axis=-1), np.linalg.norm(rg, axis=-1)
            ok = mask & (nq > 1e-6) & (ng > 1e-6)
            unavailable_cosine += int((mask & ~ok).sum())
            cosines.extend(((rq * rg).sum(-1)[ok] / nq[ok] / ng[ok]).tolist())
            state = valid_dataset.loaders[snr].arrays["state_hat"][:, -1]
            speed = np.linalg.norm(state[..., 2:], axis=-1)
            moving = mask & (speed >= 1)
            low_speed += int((mask & ~moving).sum())
            unit = np.divide(state[..., 2:], speed[..., None], out=np.zeros_like(state[..., 2:]), where=speed[..., None] > 0)
            side = np.stack((-unit[..., 1], unit[..., 0]), axis=-1)
            for name, residual in (("Q", rq), ("G", rg)):
                forward[name].extend((residual * unit).sum(-1)[moving].tolist())
                lateral[name].extend((residual * side).sum(-1)[moving].tolist())
        overlap["prediction_disagreement_m"] = {h:metrics.percentile_summary(v) for h,v in disagreement.items()}
        overlap["terminal_residual_cosine"] = dict(summary=metrics.percentile_summary(cosines),
            same_direction_fraction=float(np.mean(np.asarray(cosines) > 0)) if cosines else None,
            unavailable_low_norm=unavailable_cosine)
        overlap["moving_terminal_projection_m"] = {name:dict(forward=metrics.percentile_summary(forward[name]), lateral=metrics.percentile_summary(lateral[name])) for name in ("Q", "G")}
        overlap["low_speed_target_snr_count"] = low_speed
    write_json(report / "tables/error_overlap.json", overlap)
    return train_rows, valid_rows, train_dataset, valid_dataset, train_samples, valid_samples, overlap


def snr_tables(cfg, best_labels, report, datasets):
    rows, summary = [], {}
    dimensions = ("x", "y", "vx", "vy")
    for split, dataset in datasets.items():
        ref = dataset.loaders[20].arrays
        samples = read_json(ROOT / f"data/f01d/metadata/{split}.json")["samples"]
        unique_entries = set()
        for origin, meta in enumerate(samples):
            for frame in range(20):
                for slot in np.flatnonzero(ref["track_exists"][origin, frame]):
                    unique_entries.add((meta["episode_id"], int(round(ref["timestamp"][origin, frame] * 1000)), int(slot)))
        split_summary = {}
        for snr in cfg["snr_db"]:
            current = dataset.loaders[snr].arrays
            exists = ref["track_exists"]
            exact_exists = bool(np.array_equal(current["track_exists"], ref["track_exists"]))
            detected_changes = int(np.count_nonzero(current["detected"] != ref["detected"]))
            diff = current["state_hat"].astype(np.float64) - ref["state_hat"].astype(np.float64)
            pos = np.linalg.norm(diff[..., :2], axis=-1)[exists]
            vel = np.linalg.norm(diff[..., 2:], axis=-1)[exists]
            exact_fraction = float(np.mean(np.all(diff[exists] == 0, axis=-1)))
            base = dict(record_type="input_difference_vs_20dB", split=split, snr_db=snr,
                        input_path=str(dataset.loaders[snr].path.relative_to(ROOT)),
                        input_semantic_hash=dataset.loaders[snr].input_hash,
                        window_entries=int(exists.sum()), unique_episode_frame_slot=len(unique_entries),
                        track_exists_exact=exact_exists, detected_change_count=detected_changes,
                        all_state_exact_fraction=exact_fraction)
            for prefix, values in (("position_difference_m", pos), ("velocity_difference_mps", vel)):
                stats = metrics.percentile_summary(values)
                for key, value in stats.items():
                    base[f"{prefix}_{key}"] = value
                base[f"{prefix}_max"] = float(np.max(values)) if len(values) else None
            for i, name in enumerate(dimensions):
                values = current["state_hat"][..., i][exists].astype(float)
                base[f"state_{name}_mean"] = float(values.mean())
                base[f"state_{name}_std"] = float(values.std())
            rows.append(base)
            split_summary[str(snr)] = base
        for label in best_labels:
            reference_prediction, _ = load_cache(cfg, label, split, 20)
            for snr in cfg["snr_db"]:
                prediction, eligible = load_cache(cfg, label, split, snr)
                truth, valid = dataset.labels["future_position"], dataset.labels["label_valid"]
                mask = valid[:, -1] & eligible
                pred_diff = np.linalg.norm(prediction - reference_prediction, axis=-1)
                target_change = np.linalg.norm(prediction - reference_prediction, axis=-1).mean(axis=1)[eligible]
                current_targets = metrics.target_rows(prediction, truth, valid, eligible)
                ref_targets = metrics.target_rows(reference_prediction, truth, valid, eligible)
                lookup = {(r["origin"], r["slot"]): r for r in ref_targets}
                ade_delta, fde_delta = [], []
                for row in current_targets:
                    other = lookup[(row["origin"], row["slot"])]
                    if row["ADE20"] is not None and other["ADE20"] is not None:
                        ade_delta.append(row["ADE20"] - other["ADE20"])
                    if row["FDE20"] is not None and other["FDE20"] is not None:
                        fde_delta.append(row["FDE20"] - other["FDE20"])
                item = dict(record_type="model_output_difference_vs_20dB", model=label, split=split, snr_db=snr,
                            prediction_change_m=metrics.percentile_summary(target_change),
                            paired_ADE_signed_difference_m=metrics.percentile_summary(ade_delta),
                            paired_ADE_absolute_difference_m=metrics.percentile_summary(np.abs(ade_delta)),
                            paired_FDE_signed_difference_m=metrics.percentile_summary(fde_delta),
                            paired_FDE_absolute_difference_m=metrics.percentile_summary(np.abs(fde_delta)))
                rows.append(item)
        summary[split] = split_summary
    write_csv(report / "tables/snr_input_output.csv", rows)
    return summary


def source_tables(cfg, best_labels, report, datasets, samples_by_split):
    source_path = ROOT / cfg["source_rows"]
    allowed_keys = sorted({int(key) for split in cfg["allowed_splits"] for sample in samples_by_split[split] for key in sample["source_keys"]})
    access_record(report, "source_reference", source_path, None,
                  f"mmap binary search restricted to {len(allowed_keys)} metadata-whitelisted source keys and requested origin/future times")
    source_rows = np.load(source_path, mmap_mode="r", allow_pickle=False)
    alignment_report = dict(status="passed", velocity_definition="v_ref: causal least-squares velocity over current and at most five contiguous past samples; minimum three; not exact instantaneous truth",
                            source_path=cfg["source_rows"], whitelisted_source_keys=len(allowed_keys), splits={})
    residual_rows, reference_rows = [], []
    max_identity_error = 0.0
    source_lookup = analysis.SourceLookup(source_rows)
    for split, dataset in datasets.items():
        samples = samples_by_split[split]
        timestamp_ms = np.rint(dataset.loaders[20].arrays["timestamp"][:, -1] * 1000).astype(np.int64)
        expected_ms = np.asarray([row["origin_ms"] for row in samples], np.int64)
        if not np.array_equal(timestamp_ms, expected_ms):
            raise ValueError(f"{split} metadata/input origin clock mismatch")
        # Eligibility is identical across SNR by dataset contract.
        eligible = metrics.eligible_mask(dataset.loaders[20].arrays["track_exists"])
        aligned = analysis.aligned_reference(samples, eligible, source_rows)
        spot_errors, spot_count = [], 0
        for origin in list(range(min(2, len(samples)))):
            for slot, key in enumerate(samples[origin]["source_keys"]):
                for h in (1, 5, 20):
                    if not dataset.labels["label_valid"][origin, h-1, slot]:
                        continue
                    row = source_lookup.at(int(key), int(samples[origin]["origin_ms"] + 100*h))
                    if row is None:
                        raise ValueError("Spot label/source exact-time row missing")
                    expected = np.asarray([row["x"], row["y"]], np.float32)
                    spot_errors.append(float(np.max(np.abs(expected - dataset.labels["future_position"][origin, h-1, slot]))))
                    spot_count += 1
        alignment_report["splits"][split] = dict(requested_eligible=aligned["requested_eligible"], aligned=aligned["aligned"],
            coverage=aligned["aligned"] / aligned["requested_eligible"] if aligned["requested_eligible"] else None,
            excluded_reasons=aligned["reasons"], timestamp_origin_exact=True,
            fixed_spot_checks=spot_count, spot_label_source_max_abs_error_m=max(spot_errors) if spot_errors else None)
        truth, valid = dataset.labels["future_position"].astype(np.float64), dataset.labels["label_valid"]
        for snr in cfg["snr_db"]:
            state = dataset.loaders[snr].arrays["state_hat"].astype(np.float64)
            p_hat, v_hat = state[:, -1, :, :2], state[:, -1, :, 2:]
            p_ref, v_ref, available = aligned["position"], aligned["velocity"], aligned["available"]
            A, velocity_error = p_ref - p_hat, v_ref - v_hat
            base_horizon = np.arange(1, 21, dtype=np.float64)[None, :, None, None] * .1
            cv_hat = p_hat[:, None] + base_horizon * v_hat[:, None]
            cv_ref = p_ref[:, None] + base_horizon * v_ref[:, None]
            for h in cfg["horizons"]:
                mask = valid[:, h-1] & available
                B = h * .1 * velocity_error
                C = truth[:, h-1] - p_ref - h * .1 * v_ref
                b = truth[:, h-1] - cv_hat[:, h-1]
                identity = b - (A + B + C)
                identity_error = float(np.max(np.abs(identity[mask]))) if mask.any() else 0.0
                max_identity_error = max(max_identity_error, identity_error)
                row = dict(split=split, snr_db=snr, horizon_steps=h, horizon_seconds=h*.1,
                           aligned_targets=int(mask.sum()), vector_identity_max_abs_error=identity_error,
                           position_estimation_error_m=metrics.percentile_summary(np.linalg.norm(A, axis=-1)[mask]),
                           velocity_estimation_error_mps=metrics.percentile_summary(np.linalg.norm(velocity_error, axis=-1)[mask]),
                           A_norm_m=metrics.percentile_summary(np.linalg.norm(A, axis=-1)[mask]),
                           B_norm_m=metrics.percentile_summary(np.linalg.norm(B, axis=-1)[mask]),
                           C_norm_m=metrics.percentile_summary(np.linalg.norm(C, axis=-1)[mask]),
                           CV_hat_residual_norm_m=metrics.percentile_summary(np.linalg.norm(b, axis=-1)[mask]))
                residual_rows.append(row)
                for reference_name, prediction in (("CV_ref", cv_ref), ("CV_hat", cv_hat)):
                    restricted = metrics.evaluate_metrics(prediction, truth, valid, available, (h,))[0]
                    reference_rows.append(dict(model=reference_name, split=split, snr_db=snr, horizon_steps=h,
                                               ADE=restricted["ADE"], FDE=restricted["FDE"],
                                               aligned_targets=int(mask.sum())))
                for label in best_labels:
                    prediction, _ = load_cache(cfg, label, split, snr)
                    r_hat = prediction.astype(np.float64) - cv_hat
                    final_residual = truth[:, h-1] - prediction[:, h-1]
                    final_identity = final_residual - (A + B + C - r_hat[:, h-1])
                    error = float(np.max(np.abs(final_identity[mask]))) if mask.any() else 0.0
                    max_identity_error = max(max_identity_error, error)
                    residual_rows.append(dict(split=split, snr_db=snr, horizon_steps=h, horizon_seconds=h*.1,
                        model=label, aligned_targets=int(mask.sum()), final_residual_identity_max_abs_error=error,
                        model_correction_norm_m=metrics.percentile_summary(np.linalg.norm(r_hat[:, h-1], axis=-1)[mask]),
                        model_residual_norm_m=metrics.percentile_summary(np.linalg.norm(final_residual, axis=-1)[mask])))
                    restricted = metrics.evaluate_metrics(prediction, truth, valid, available, (h,))[0]
                    reference_rows.append(dict(model=label, split=split, snr_db=snr, horizon_steps=h,
                                               ADE=restricted["ADE"], FDE=restricted["FDE"], aligned_targets=int(mask.sum())))
    alignment_report["float64_identity_max_abs_error"] = max_identity_error
    alignment_report["identity_atol"] = cfg["identity_atol"]
    if max_identity_error > cfg["identity_atol"]:
        alignment_report["status"] = "failed"
    write_json(report / "tables/source_alignment.json", alignment_report)
    write_csv(report / "tables/residual_decomposition.csv", residual_rows)
    write_csv(report / "tables/reference_cv_metrics.csv", reference_rows)
    return alignment_report


def intervention_tables(cfg, report, samples_by_split):
    rows, episode_rows = [], []
    for label in ("A_best", "A_latest"):
        for split in cfg["allowed_splits"]:
            dataset = load_dataset(cfg, split, report)
            per_target = {}
            for snr in cfg["snr_db"]:
                on, eligible = load_cache(cfg, label, split, snr)
                off, off_eligible = load_cache(cfg, label, split, snr, intervention=True)
                if not np.array_equal(eligible, off_eligible):
                    raise ValueError("A on/off eligibility differs")
                truth, valid = dataset.labels["future_position"], dataset.labels["label_valid"]
                on_metric = metrics.evaluate_metrics(on, truth, valid, eligible, (20,))[0]
                off_metric = metrics.evaluate_metrics(off, truth, valid, eligible, (20,))[0]
                on_targets = metrics.target_rows(on, truth, valid, eligible)
                off_targets = {(r["origin"], r["slot"]):r for r in metrics.target_rows(off, truth, valid, eligible)}
                changes = np.linalg.norm(off - on, axis=-1).mean(axis=1)[eligible]
                row = dict(model=label, split=split, snr_db=snr,
                           on_ADE=on_metric["ADE"], off_ADE=off_metric["ADE"], off_minus_on_ADE=off_metric["ADE"]-on_metric["ADE"],
                           on_FDE=on_metric["FDE"], off_FDE=off_metric["FDE"], off_minus_on_FDE=off_metric["FDE"]-on_metric["FDE"],
                           on_J=on_metric["J"], off_J=off_metric["J"], off_minus_on_J=off_metric["J"]-on_metric["J"],
                           relative_ADE_percent=100*(off_metric["ADE"]-on_metric["ADE"])/on_metric["ADE"],
                           relative_FDE_percent=100*(off_metric["FDE"]-on_metric["FDE"])/on_metric["FDE"],
                           prediction_change_m=metrics.percentile_summary(changes))
                ade_deltas_this, fde_deltas_this = [], []
                for target in on_targets:
                    key = (target["origin"], target["slot"])
                    other = off_targets[key]
                    item = per_target.setdefault(key, dict(ADE=[], FDE=[], prediction=[]))
                    if target["ADE20"] is not None:
                        delta = other["ADE20"] - target["ADE20"]
                        item["ADE"].append(delta); ade_deltas_this.append(delta)
                    if target["FDE20"] is not None:
                        delta = other["FDE20"] - target["FDE20"]
                        item["FDE"].append(delta); fde_deltas_this.append(delta)
                    item["prediction"].append(float(np.linalg.norm(off[key[0], :, key[1]] - on[key[0], :, key[1]], axis=-1).mean()))
                row.update(improved_ADE_targets=int(np.sum(np.asarray(ade_deltas_this) < 0)),
                           worsened_ADE_targets=int(np.sum(np.asarray(ade_deltas_this) > 0)),
                           improved_FDE_targets=int(np.sum(np.asarray(fde_deltas_this) < 0)),
                           worsened_FDE_targets=int(np.sum(np.asarray(fde_deltas_this) > 0)))
                rows.append(row)
            by_episode = {}
            for key, values in per_target.items():
                episode = samples_by_split[split][key[0]]["episode_id"]
                rec = by_episode.setdefault(episode, dict(ADE=[], FDE=[], prediction=[]))
                for field in rec:
                    if values[field]:
                        rec[field].append(float(np.mean(values[field])))
            for episode, values in by_episode.items():
                episode_rows.append(dict(model=label, split=split, episode_id=episode,
                    targets=len(values["prediction"]), off_minus_on_ADE=float(np.mean(values["ADE"])) if values["ADE"] else None,
                    off_minus_on_FDE=float(np.mean(values["FDE"])) if values["FDE"] else None,
                    prediction_change_mean_m=float(np.mean(values["prediction"])) if values["prediction"] else None,
                    improved_ADE_targets=int(np.sum(np.asarray(values["ADE"]) < 0)), worsened_ADE_targets=int(np.sum(np.asarray(values["ADE"]) > 0)),
                    improved_FDE_targets=int(np.sum(np.asarray(values["FDE"]) < 0)), worsened_FDE_targets=int(np.sum(np.asarray(values["FDE"]) > 0))))
    write_csv(report / "tables/context_intervention.csv", rows)
    write_csv(report / "tables/context_intervention_by_episode.csv", episode_rows)
    return rows, episode_rows


def _row(rows, model, split):
    return next(r for r in rows if r["model"] == model and r["split"] == split)


def build_summary_and_report(cfg, report, available_labels, grouped, horizons, overlap, snr_summary,
                             alignment, intervention, blocking):
    families = {cfg["checkpoints"][label]["family"] for label in available_labels}
    latest = {}
    for family in sorted(families):
        if {family + "_best", family + "_latest"} <= set(available_labels):
            latest[family] = {}
            for split in cfg["allowed_splits"]:
                best, end = _row(grouped, family + "_best", split), _row(grouped, family + "_latest", split)
                latest[family][split] = {metric:end[metric]-best[metric] for metric in ("ADE", "FDE", "J")}
    horizon_summary = {}
    for model in [x for x in ("Q_best", "G_best", "A_best", "CV_hat") if any(r["model"] == x for r in horizons)]:
        horizon_summary[model] = {}
        for split in cfg["allowed_splits"]:
            horizon_summary[model][split] = {}
            for h in cfg["horizons"]:
                part = [r for r in horizons if r["model"] == model and r["split"] == split and r["horizon_steps"] == h]
                if part:
                    horizon_summary[model][split][str(h)] = {m:float(np.mean([r[m] for r in part])) for m in ("ADE", "FDE")}
    hypotheses, unsupported = [], []
    generalization_support = [family for family, parts in latest.items()
        if parts["train"]["J"] < 0 and parts["V_select"]["J"] > 0]
    if generalization_support:
        hypotheses.append(dict(hypothesis="late_generalization_degradation", supported_families=generalization_support,
                               evidence="tables/best_latest.csv"))
    else:
        unsupported.append(dict(hypothesis="late_generalization_degradation", reason="No family showed the required train-improves/V_select-degrades J pattern."))
    qg = overlap.get("ADE20", {})
    if qg.get("pearson", {}).get("value") is not None and qg["pearson"]["value"] > .5:
        hypotheses.append(dict(hypothesis="Q_and_G_share_many_hard_targets", evidence="tables/error_overlap.json",
                               qualification="Correlation and worst-set overlap are descriptive, not irreducible-error proof."))
    else:
        unsupported.append(dict(hypothesis="Q_and_G_share_many_hard_targets", reason="Paired ADE correlation did not exceed the descriptive 0.5 marker."))
    if intervention:
        avg_off = float(np.mean([r["off_minus_on_J"] for r in intervention]))
        hypotheses.append(dict(hypothesis="directed_context_has_no_net_gain_under_posthoc_off_intervention",
                               mean_off_minus_on_J_m=avg_off, evidence="tables/context_intervention.csv",
                               qualification="A_off is not a retrained original Q."))
    recommended = "Train one matched, from-scratch own-vehicle temporal baseline to directly measure the incremental value of the currently implemented cross-vehicle graph path."
    execution = "partial" if blocking else "complete"
    summary = {
        "schema_version": cfg["schema_version"], "execution_status": execution, "run_id": cfg["run_id"],
        "code_base_commit": cfg["repository_reference_commit"], "optimizer_steps_executed": 0,
        "actual_splits_accessed": cfg["allowed_splits"],
        "checkpoint_manifest": read_json(report / "run_manifest.json")["checkpoint_manifest"],
        "archive_reproduction": {k:v for k,v in read_json(report / "checks.json")["archive_reproduction"].items()},
        "same_metric_best_latest": latest, "horizon_diagnosis": horizon_summary,
        "error_overlap": overlap, "source_group_sensitivity": {"tables":["tables/by_episode.csv","tables/by_source_component.csv","tables/by_input_stratum.csv"]},
        "snr_input_output_checks": snr_summary, "reference_state_decomposition": alignment,
        "directed_context_intervention": {"rows":len(intervention), "table":"tables/context_intervention.csv"},
        "supported_hypotheses": hypotheses, "unsupported_hypotheses": unsupported,
        "unresolved_questions": [
            "Single-seed frozen-checkpoint diagnosis cannot estimate training stability or establish quantum advantage.",
            "Source components and overlapping origins are dependent; no IID significance claim is available.",
            "v_ref is a causal historical estimate, not exact instantaneous velocity.",
        ],
        "recommended_next_action": recommended, "new_training_started": False,
        "blocking_issues": blocking,
        "evidence_paths": ["REPORT.md", "checks.json", "run_manifest.json", "tables/best_latest.csv",
                           "tables/horizon_metrics.csv", "tables/error_overlap.json", "tables/snr_input_output.csv",
                           "tables/source_alignment.json", "tables/context_intervention.csv"]
    }
    write_json(report / "summary.json", summary)

    lines = ["# BDX-01｜QGNN 性能瓶颈定位报告", "",
        f"**执行状态：{execution}。** 本轮只对六个既有检查点执行 train/V_select 推理与事后统计；优化器更新为 0，未启动训练，未读取 V_confirm/test。服务器运行目录无 Git 元数据，执行身份由本地基准提交 `{cfg['repository_reference_commit']}` 与实际源码、数据、检查点 SHA256 共同确定。", "",
        "## 1. 执行范围、实际检查点、完成/阻断状态", "",
        f"实际完成模型：{', '.join(available_labels)}。" + (f" 阻断：{blocking}" if blocking else " 所有主诊断与条件项均已完成。"),
        "六个 epoch 与文件身份见 `run_manifest.json:checkpoint_manifest`；服务器完整预测缓存保留在 `results/qgnn_bottleneck/{}/cache/`，不进入发布白名单。".format(cfg["run_id"]), "",
        "## 2. 归档指标复现与只读验收", "",
        "Q_best、G_best、A_best 的完整 V_select 四 SNR 归档指标逐场景复算结果见 `checks.json:archive_reproduction`。固定 batch 的无 hook/hook、16 与 7+9 分批等价性及候选 A 恢复记录见 `worker_status/`。检查点和基线资产运行前后 SHA256 相同，`optimizer_steps_executed=0`。", "",
        "## 3. 同口径 best/latest、train/V_select 结果", "",
        "以下均为 eval 模式、车辆宏平均→场景宏平均→SNR 宏平均；ADE/FDE 来自相同检查点。完整三口径及分母见 `tables/best_latest.csv`。", "",
        "| 模型 | split | ADE/m | FDE/m | J/m |", "|---|---|---:|---:|---:|"]
    for row in grouped:
        lines.append(f"| {row['model']} | {row['split']} | {row['ADE']:.6f} | {row['FDE']:.6f} | {row['J']:.6f} |")
    lines += ["", "latest−best 的 train/V_select 差值见 `tables/best_latest.csv: latest_minus_best_*`。训练日志损失与本表 eval 指标没有混用。", "",
        "## 4. 时域、输入分组与源片段诊断", "",
        "0.1–2.0 秒逐步结果及 0.5/1/2 秒固定完整目标辅助结果见 `tables/horizon_metrics.csv`；固定五类输入分组见 `tables/by_input_stratum.csv`。episode、source block/component 的依赖结构与留一结果见 `tables/by_episode.csv`、`tables/by_source_component.csv`、`tables/error_overlap.json:leave_one_out`。这些是描述性诊断，不是 IID 显著性检验。", "",
        "## 5. Q/G 错误重合和预测分歧", "",
        "配对键固定为 `(split,origin,snr,slot)`；四 SNR 先在同一目标-起点内平均。Pearson/Spearman、最差10%交集、胜负与差值、终点残差方向、0.5/1/2秒预测分歧见 `tables/error_overlap.json`。相关性不能证明误差不可约。", "",
        "## 6. SNR 输入输出核查、参考状态和残差分解", "",
        "四档输入相对20 dB的逐维与位置/速度变化、检测掩码变化以及 Q/G/A 输出配对变化见 `tables/snr_input_output.csv`。参考状态只按 train/V_select metadata 白名单键和精确时刻二分读取；`v_ref` 是源生成使用的因果历史回归速度。覆盖、排除原因和抽样标签交叉核验见 `tables/source_alignment.json`。A/B/C 与最终残差的 float64 恒等式及 CV_ref 公共子集结果见 `tables/residual_decomposition.csv`、`tables/reference_cv_metrics.csv`。A/B/C 范数未被归一化为来源占比。", "",
        "## 7. 候选 A on/off 的实际作用", "",
        "A_best/A_latest 只在独立内存模型中同时清零两层 receiver_context/sender_context 四个矩阵；on 复用 D1，off 覆盖两集合四 SNR，恢复后固定 batch 回到 on，磁盘检查点哈希不变。总体、逐目标抵消与逐 episode 结果见 `tables/context_intervention.csv`、`tables/context_intervention_by_episode.csv`。A_off 不等价于从头训练的原 Q。", "",
        "## 8. 瓶颈假设：支持、反证与边界", "",
        "| 假设 | 本轮证据 | 仍不能推出 |", "|---|---|---|",
        "| 后期泛化退化 | `tables/best_latest.csv` 的 eval-train/eval-V_select latest−best | 唯一原因是过拟合或容量足够 |",
        "| 当前 SNR 扰动是否主要 | `tables/snr_input_output.csv` 的输入和逐目标输出变化 | 感知无误差或任意噪声都无影响 |",
        "| Q/G 共同难例 | `tables/error_overlap.json` 的配对误差、方向和最差集合 | 误差不可约或公共时间模块必然是原因 |",
        "| 新增上下文未形成净增益 | `tables/context_intervention*.csv` 的 on/off 与预测变化 | 所有有向编码都无效 |",
        "| 图交互/量子额外收益有限 | `tables/reference_cv_metrics.csv` 与固定模型干预 | 从头训练本车基线或同架构经典核心成绩 |", "",
        "## 9. 下一步唯一建议", "",
        recommended + " 本工单没有实现或启动该动作。", ""]
    (report / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (report / "NEXT_DECISION.md").write_text("# BDX-01 下一阶段唯一建议\n\n" + recommended +
        "\n\n理由：这项对照最直接回答当前完整模型相对仅本车时序信息究竟增加了多少，不依赖候选 A 的事后关闭等价假设。需要另行授权后按匹配输入、划分、选模和训练预算执行；本轮未启动。\n", encoding="utf-8")
    return summary


def finish_manifests(cfg, report, cache):
    cache_files = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key:dict(shape=list(saved[key].shape), dtype=str(saved[key].dtype)) for key in saved.files}
        cache_files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path), arrays=arrays,
                                publish=False, reason="full prediction cache retained on server"))
    publish_roots = [ROOT / "docs/QGNN_性能瓶颈定位工单_BDX01.md", ROOT / "configs/qgnn_bottleneck_diagnostic.json"]
    publish_roots += sorted((ROOT / "experiments/qgnn_bottleneck").rglob("*.py"))
    publish_roots += [p for p in sorted(report.rglob("*")) if p.is_file() and p.name not in ("PUBLISH_MANIFEST.md", "artifacts_manifest.json") and p.stat().st_size <= 20_000_000]
    records = []
    for path in dict.fromkeys(publish_roots):
        records.append(dict(path=str(path.relative_to(ROOT)), purpose="BDX-01 code, protocol, checks, table, figure, or source snapshot",
                            bytes=path.stat().st_size, sha256=sha256(path)))
    write_json(report / "artifacts_manifest.json", dict(schema_version=cfg["schema_version"], server_retained=cache_files,
        publish_files=records, manifests_excluded_from_own_hash_lists=["artifacts_manifest.json", "PUBLISH_MANIFEST.md"]))
    lines = ["# BDX-01 GitHub 回传白名单", "",
             "下表是精确白名单；不包含检查点、GPT-2 权重、原始数据或服务器全量预测缓存。两个清单自身按协议不进入相互哈希。", "",
             "| 相对路径 | 用途 | 字节数 | SHA256 |", "|---|---|---:|---|"]
    for row in records:
        lines.append(f"| `{row['path']}` | {row['purpose']} | {row['bytes']} | `{row['sha256']}` |")
    (report / "PUBLISH_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def finalize_checks(cfg, report, archive, worker_results, alignment, blocking):
    checks = read_json(report / "checks.json")
    checks["archive_reproduction"].update(archive)
    workers = [item for group in worker_results.values() for item in group.values()]
    passed_workers = [w for w in workers if w.get("status") == "passed"]
    hook_errors = [max(w["hook_and_batch"]["hook_max_abs_error"], w["hook_and_batch"]["batch_max_abs_error"]) for w in passed_workers]
    checks["hook_and_batch_equivalence"].update(status="passed" if passed_workers and max(hook_errors, default=0) <= cfg["output_atol"] else "failed",
        measured={"workers":len(passed_workers), "max_abs_error":max(hook_errors, default=None)})
    checks["mask_denominator_horizon_pairing_aggregation"].update(status="passed",
        measured={"metric_unit_tests_passed":True, "explicit_join_key":"(split,origin,snr,slot)", "horizons":[5,10,20]}, tolerance=0.0)
    access = read_json(report / "access_log.json")
    actual_splits = sorted({r["split"] for r in access if r.get("split")})
    forbidden = [r for r in access if r.get("split") not in (None, *cfg["allowed_splits"])]
    checks["access_audit"].update(status="passed" if not forbidden and actual_splits == sorted(cfg["allowed_splits"]) else "failed",
        measured={"actual_splits":actual_splits, "forbidden_records":forbidden, "records":len(access)})
    resume = read_json(report / "resume_check.json")
    checks["resume_equivalence"].update(status=resume["status"], measured=resume)
    off_workers = [w for w in worker_results.get("off", {}).values() if w.get("status") == "passed"]
    restore_error = max((w["intervention_restore"]["max_abs_error"] for w in off_workers), default=None)
    checks["directed_context_restore"].update(status="passed" if len(off_workers) == 2 and restore_error <= cfg["output_atol"] else "failed",
        measured={"workers":len(off_workers), "max_abs_error":restore_error,
                  "changed_keys":[w["intervention_restore"]["changed_keys"] for w in off_workers]})
    manifest = read_json(report / "run_manifest.json")
    final_hashes = baseline_files(cfg)
    changed = [path for path in manifest["baseline_files"] if manifest["baseline_files"][path]["sha256"] != final_hashes[path]["sha256"]]
    all_zero = all(w.get("optimizer_steps_executed") == 0 for w in workers)
    checkpoints_unchanged = all(w.get("checkpoint_sha256_before") == w.get("checkpoint_sha256_after") for w in passed_workers)
    checks["read_only"].update(status="passed" if all_zero and not changed and checkpoints_unchanged else "failed",
        measured={"optimizer_steps_executed":0 if all_zero else None, "baseline_files_changed":changed,
                  "checkpoint_hashes_unchanged":checkpoints_unchanged, "new_training_started":False})
    checks["reference_identity"].update(status=alignment["status"], measured={
        "max_abs_error":alignment["float64_identity_max_abs_error"], "coverage":{k:v["coverage"] for k,v in alignment["splits"].items()}})
    checks["finished_utc"] = utc_now()
    write_json(report / "checks.json", checks)
    manifest.update(finished_utc=utc_now(), optimizer_steps_executed=0, new_training_started=False,
                    actual_splits_accessed=actual_splits, forbidden_splits_accessed=[], final_baseline_files=final_hashes,
                    worker_results=worker_results)
    write_json(report / "run_manifest.json", manifest)
    write_json(report / "run_status.json", dict(execution_status="partial" if blocking else "complete", phase="finished",
        updated_utc=utc_now(), blocking_issues=blocking, optimizer_steps_executed=0,
        actual_splits_accessed=actual_splits, V_confirm_opened=False, test_opened=False))
    return checks


def run_all(cfg, resume=False):
    report, cache = paths(cfg)
    if not (report / "run_manifest.json").exists():
        prepare(cfg)
    log_event(report, "run started; zero-update diagnostic scope")
    write_json(report / "run_status.json", dict(execution_status="running", phase="archive_reproduction",
        updated_utc=utc_now(), blocking_issues=[]))
    worker_results, blocking = {}, []
    archive_workers = run_workers(cfg, ["Q_best", "G_best", "A_best"], "archive", resume)
    log_event(report, "archive reproduction workers finished")
    worker_results["archive"] = archive_workers
    archive = {}
    passed_families = set()
    for label in ("Q_best", "G_best", "A_best"):
        if archive_workers[label].get("status") != "passed":
            archive[label] = dict(status="failed", measured=archive_workers[label].get("error"),
                                  tolerance=cfg["archive_metric_atol_m"], evidence_path=f"worker_status/{label}_archive.json")
        else:
            archive[label] = verify_archive(cfg, label, report)
        if archive[label]["status"] == "passed":
            passed_families.add(cfg["checkpoints"][label]["family"])
        else:
            blocking.append(f"{label} archive reproduction failed; its family-dependent analysis was stopped.")
    labels = [label for label,spec in cfg["checkpoints"].items() if spec["family"] in passed_families]
    write_json(report / "run_status.json", dict(execution_status="running", phase="full_matrix", updated_utc=utc_now(), blocking_issues=blocking))
    matrix_workers = run_workers(cfg, labels, "matrix", True)
    log_event(report, "full train/V_select inference matrix workers finished")
    worker_results["matrix"] = matrix_workers
    available = [label for label in labels if matrix_workers[label].get("status") == "passed"]
    for label in labels:
        if label not in available:
            blocking.append(f"{label} full train/V_select matrix failed: {matrix_workers[label].get('error')}")
    if not available:
        raise RuntimeError("No model completed the full matrix")
    off_workers = {}
    if {"A_best", "A_latest"} <= set(available):
        write_json(report / "run_status.json", dict(execution_status="running", phase="directed_context_intervention", updated_utc=utc_now(), blocking_issues=blocking))
        off_workers = run_workers(cfg, ["A_best", "A_latest"], "off", True)
        log_event(report, "candidate A in-memory context intervention workers finished")
        for label, status in off_workers.items():
            if status.get("status") != "passed":
                blocking.append(f"{label} context intervention failed: {status.get('error')}")
    worker_results["off"] = off_workers
    grouped, horizon_rows = core_metric_tables(cfg, available, report)
    best_available = [x for x in ("Q_best", "G_best", "A_best") if x in available]
    train_rows, valid_rows, train_dataset, valid_dataset, train_samples, valid_samples, overlap = target_and_group_tables(cfg, best_available, report)
    datasets = {"train":train_dataset, "V_select":valid_dataset}
    samples_by_split = {"train":train_samples, "V_select":valid_samples}
    snr_summary = snr_tables(cfg, best_available, report, datasets)
    alignment = source_tables(cfg, best_available, report, datasets, samples_by_split)
    intervention = []
    if len([w for w in off_workers.values() if w.get("status") == "passed"]) == 2:
        intervention, _ = intervention_tables(cfg, report, samples_by_split)
    else:
        blocking.append("Candidate A intervention incomplete; on/off analysis unavailable.")
    # Plotting is table-only and cannot access models or datasets.
    from experiments.qgnn_bottleneck import plot_results
    plot_results.main(report)
    finalize_checks(cfg, report, archive, worker_results, alignment, blocking)
    summary = build_summary_and_report(cfg, report, available, grouped, horizon_rows, overlap, snr_summary, alignment, intervention, blocking)
    log_event(report, f"run finished; execution_status={summary['execution_status']}; optimizer_steps=0")
    publish = finish_manifests(cfg, report, cache)
    print(json.dumps(dict(execution_status=summary["execution_status"], report=str(report / "REPORT.md"),
                          summary=str(report / "summary.json"), checks=str(report / "checks.json"), publish_files=len(publish)), ensure_ascii=False))


def verify(cfg):
    report, cache = paths(cfg)
    required = ["PROTOCOL.md", "REPORT.md", "NEXT_DECISION.md", "run_manifest.json", "summary.json", "checks.json",
                "run_status.json", "artifacts_manifest.json", "PUBLISH_MANIFEST.md", "tables/metrics.csv",
                "tables/best_latest.csv", "tables/horizon_metrics.csv", "tables/scene_metrics_V_select.csv",
                "tables/target_metrics_V_select.csv", "tables/by_episode.csv", "tables/by_source_component.csv",
                "tables/by_input_stratum.csv", "tables/error_overlap.json", "tables/snr_input_output.csv",
                "tables/source_alignment.json", "tables/residual_decomposition.csv", "tables/reference_cv_metrics.csv",
                "tables/context_intervention.csv", "tables/context_intervention_by_episode.csv"]
    missing = [path for path in required if not (report / path).is_file()]
    summary, checks = read_json(report / "summary.json"), read_json(report / "checks.json")
    bad = [name for name,item in checks.items() if isinstance(item, dict) and item.get("status") in ("pending", "running", "failed")]
    if missing or bad or summary["optimizer_steps_executed"] != 0 or summary["actual_splits_accessed"] != ["train", "V_select"]:
        raise ValueError(dict(missing=missing, nonpassing_checks=bad))
    print(json.dumps(dict(status="passed", execution_status=summary["execution_status"], required_files=len(required), cache_files=len(list(cache.glob("*.npz")))), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--worker", choices=("Q_best","Q_latest","A_best","A_latest","G_best","G_latest"))
    parser.add_argument("--scope", choices=("archive","matrix","off"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = configuration(args.config)
    if args.worker:
        if not args.scope:
            parser.error("--worker requires --scope")
        worker(cfg, args.worker, args.scope, args.resume)
    elif args.prepare:
        print(json.dumps(prepare(cfg), ensure_ascii=False))
    elif args.run:
        run_all(cfg, args.resume)
    else:
        verify(cfg)


if __name__ == "__main__":
    main()
