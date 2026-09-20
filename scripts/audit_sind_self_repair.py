#!/usr/bin/env python3
"""Replay the five completed Self runs on validation only; never train or use test."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_raj_residual_self import evaluate, subset_indices
from frontend.sind_prediction_dataset import SinDPredictionDataset
from frontend.sind_target_dataset import SinDTargetPredictionDataset
from prediction.qgnn_raj_pennylane import build_model, build_self_baseline

RUNS = [("legacy", "llm"), ("target", "llm"), ("target", "lstm"),
        ("target", "transformer"), ("target", "tcn")]


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = {"test_set_used": False, "training_started": False, "runs": []}
    output = ROOT / "reports/qgnn/sind_target_self_repair"
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    started = time.perf_counter()
    shared = None
    for dataset, kind in RUNS:
        name = f"{dataset}_{kind}_ego_v1_0db_seed2026"
        folder = ROOT / "results/qgnn/self_repair_v1" / name
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        records = json.loads((folder / "training.json").read_text())
        assert summary["status"] == "COMPLETED", name
        assert config["phase"] == "self" and config["test_set_used"] is False
        assert summary["test_set_used"] is False
        assert [r["epoch"] for r in records] == list(range(21)), name
        assert summary["epochs_completed"] == 20
        assert config["seed"] == 2026 and config["snr"] == 0
        assert config["train_limit"] is None and config["val_limit"] is None
        assert summary["global_step"] == (9880 if dataset == "legacy" else 25000)
        # Check the already-recorded provenance contract, not a new file hash scheme.
        for relative, expected in config["source_sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
        if dataset == "target":
            for split, expected in config["data_manifest"].items():
                actual = json.loads((ROOT / "data/sind/target_views_v1" / split / "manifest.json").read_text())
                assert actual == expected, split
            protocol = {k: config[k] for k in ("seed", "snr", "epochs", "batch_size",
                       "lr", "token_weight", "train_limit", "val_limit", "data_manifest")}
            if shared is None:
                shared = protocol
            else:
                assert shared == protocol, name
            assert summary["supervision"] == "center_only"
        best_record = min(records, key=lambda r: r["validation"]["selection_score"])
        assert summary["selected_checkpoint"] == {"epoch": best_record["epoch"], **best_record["validation"]}
        if dataset not in datasets:
            cls = SinDTargetPredictionDataset if dataset == "target" else SinDPredictionDataset
            datasets[dataset] = cls("val", 0.0, ROOT, True)
        data = datasets[dataset]
        ids = subset_indices(len(data), None, config["seed"] + 1)
        loader = DataLoader(Subset(data, ids.tolist()), batch_size=config["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        model = (build_model(seed=config["seed"], phase="self", self_frame="ego_v1")
                 if kind == "llm" else build_self_baseline(kind, seed=config["seed"])).to(device)
        replay = {}
        for checkpoint in ("best", "last"):
            saved = torch.load(folder / f"{checkpoint}.pt", map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(saved["model_state"], strict=False)
            allowed = lambda key: key.startswith("core.") or (key.startswith("llm.base.gpt2.") and "lora_" not in key)
            assert not unexpected and all(allowed(k) for k in missing), (missing, unexpected)
            if checkpoint == "last":
                assert saved["next_epoch"] == 21 and saved["next_batch"] == 0
                assert saved["records"] == records and saved["global_step"] == summary["global_step"]
                reference = records[-1]["validation"]
            else:
                reference = best_record["validation"]
                assert saved["validation"] == reference
            actual = evaluate(model, loader, device)
            deltas = {k: abs(actual[k] - reference[k]) for k in ("ADE", "FDE", "selection_score")}
            assert max(deltas.values()) < 2e-5, (name, checkpoint, deltas)
            assert actual["samples"] == summary["validation_samples"]
            replay[checkpoint] = {"metrics": actual, "absolute_difference": deltas}
            del saved
        last = records[-1]["validation"]
        at15 = records[15]["validation"]
        entry = {
            "run": name, "dataset": dataset, "model": kind,
            "status": summary["status"], "train_samples": summary["train_samples"],
            "validation_samples": summary["validation_samples"],
            "global_step": summary["global_step"], "elapsed_seconds": summary["elapsed_seconds"],
            "initial_validation": summary["initial_validation"],
            "selected_checkpoint": summary["selected_checkpoint"],
            "last_validation": last, "best_is_last_epoch": best_record["epoch"] == 20,
            "epoch15_to20_improvement_percent": {
                k: 100 * (at15[k] - last[k]) / at15[k] for k in ("ADE", "FDE")
            },
            "cv_to_best_improvement_percent": {
                k: 100 * (summary["initial_validation"][k] - best_record["validation"][k])
                / summary["initial_validation"][k] for k in ("ADE", "FDE")
            },
            "replay": replay, "records": records,
            "provenance_verified": True,
        }
        if dataset == "target":
            indices = np.flatnonzero(data.origin_context_count[data.origin_index] > 8)[:32]
            batch = next(iter(DataLoader(Subset(data, indices.tolist()), batch_size=32)))
            history = batch["history_state"].to(device)
            mask = batch["target_mask"].to(device)
            timestamps = batch["history_timestamp"].to(device)
            assert torch.all(mask[:, 0]) and not torch.any(mask[:, 1:])
            model.eval()
            with torch.no_grad():
                baseline = model(history, mask, timestamps)["prediction"][:, :, 0]
                changed = history.clone()
                changed[:, :, 1:] = changed[:, :, 1:].flip(2) + 123.0
                after = model(changed, mask, timestamps)["prediction"][:, :, 0]
                delta = float((baseline - after).abs().max())
            assert delta == 0.0, (name, delta)
            entry["self_neighbor_isolation_max_difference"] = delta
            entry["real_n_gt_8_samples_checked"] = len(indices)
        report["runs"].append(entry)
        print(json.dumps({k: entry[k] for k in ("run", "selected_checkpoint", "epoch15_to20_improvement_percent")}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["acceptance"] = "PASS"
    (output / "completed_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = ["# SinD 逐目标 Self 完整训练验收", "",
            "五组均完成 20 epoch；best/last 均在完整 val 回放，未训练、未访问 test。",
            "新数据四模型共享逐目标数据、seed 2026、0 dB、batch 256、25,000 updates。",
            "指标先在每个 origin 内对目标等权平均，再对 origin 等权平均。单位为米。", "",
            "| 数据 | 模型 | ADE | FDE | 选中 epoch | 末轮 ADE | 末轮 FDE |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for e in report["runs"]:
        b, last = e["selected_checkpoint"], e["last_validation"]
        rows.append(f'| {e["dataset"]} | {e["model"]} | {b["ADE"]:.6f} | {b["FDE"]:.6f} | {b["epoch"]} | {last["ADE"]:.6f} | {last["FDE"]:.6f} |')
    rows += ["", "## 验收结论", "",
             "- 保存的源码来源与数据 manifest 一致；完整 0–20 轮记录、更新次数和检查点选择一致。",
             "- 五组 best/last 的全量 val 回放与记录一致，ADE/FDE 允许误差 2e-5 m。",
             "- 四个新数据模型在各 32 条真实 N>8 样本上改变邻车，Self 中心输出差值均为零。",
             "- 新数据 LLM 已优于 CV，但仍明显落后于经典 Self；不能据此建立 LLM 优于经典模型的论断。",
             "- 新数据 TCN、LSTM、LLM 均在最后一轮选中，不能声称已收敛；Transformer 最佳为第 15 轮。",
             "- 旧数据 LLM 只用于坐标接口修复验证；新旧数据目标集合和聚合不同，不混排行。",
             "- 本轮证明逐目标 Self 路径和实际学习有效；邻车交互的收益尚待下一批受控实验。",
             "", "完整曲线、初始 CV、回放误差与来源检查见 completed_audit.json。", ""]
    (output / "completed_audit.md").write_text("\n".join(rows), encoding="utf-8")
    print(f"PASS: report saved to {output}; elapsed={report['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
