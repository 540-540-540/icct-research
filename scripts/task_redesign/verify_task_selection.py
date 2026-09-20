#!/usr/bin/env python3
"""Verify IC4 artifacts without training or parsing any trajectory state labels.
Raw CSVs are read as opaque bytes for SHA256 only. This is NOT an end-to-end
causality test of SinD's upstream smoothing or a pass of research Gates S/G/Q.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = "e94d61989333f0b93b1ddf25f4423c08defb49e9"
PRIOR = "8da14521467b96bb23dc93c51d52704a2b9b7a8d"
BRANCH = "task-redesign-gpt6pro-20260920"
OUT = ROOT / "reports/task_redesign"
sys.path.insert(0, str(ROOT / "scripts/task_redesign"))
from audit_history_only import pair_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).strip()


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    if ROOT != Path("/home/dell/YrM/ICCT_task_redesign_gpt6pro"):
        raise RuntimeError("Run only from the authorized isolated worktree")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Unexpected task branch; refusing to write")
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    git("merge-base", "--is-ancestor", PRIOR, "HEAD")
    summary = load("optimal_task_selection_20260920.json")
    audit = load("history_only_audit_20260920.json")
    supplement = load("audit_supplement_20260920.json")
    prereg = load("design_preregistration_20260920.json")
    checks: dict[str, bool] = {}
    evidence = {}
    for name, expected in supplement["evidence_files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError("Evidence path leaves the task worktree")
        actual = sha256(path)
        evidence[name] = actual
        checks["evidence:" + name] = actual == expected["sha256"]
    raw_hashes = {}
    for city, expected in audit["raw_sources"].items():
        path = ROOT / "data/task_redesign/raw" / city / "Veh_smoothed_tracks.csv"
        raw_hashes[city] = sha256(path)
        checks["raw_sha256:" + city] = raw_hashes[city] == expected["sha256"]
    checks["prereg_hash_unchanged"] = (
        sha256(OUT / "design_preregistration_20260920.json") == audit["prereg_sha256"]
    )
    th = summary["thresholds"]
    checks["summary_thresholds_match_prereg"] = all([
        th["radius_m"] == prereg["radius_m"],
        th["dcpa_limit_m"] == prereg["dcpa_m"],
        th["minimum_closing_speed_mps"] == prereg["closing_mps"],
        th["tcpa_seconds_open_closed"] == [0, prereg["cpa_threshold_s"]],
        th["following_max_center_headway_s"] == prereg["following_headway_s"],
        th["following_max_lateral_m"] == prereg["following_lateral_m"],
        th["following_heading_max_degrees"] == prereg["following_heading_degrees"],
    ])
    compressed = OUT / "history_only_membership_20260920.csv.gz"
    checks["gzip_sha256"] = sha256(compressed) == supplement["membership_gzip"]["sha256"]
    plain = gzip.decompress(compressed.read_bytes())
    checks["manifest_sha256"] = hashlib.sha256(plain).hexdigest() == audit["membership_sha256"]
    reader = csv.DictReader(io.StringIO(plain.decode("utf-8")))
    allowed = {"scene", "split", "t0", "id", "horizon", "n_active", "n50", "k", "cpa", "following", "future_points", "endpoint"}
    checks["only_expected_metadata_columns"] = set(reader.fieldnames or []) == allowed
    rows = list(reader)
    checks["manifest_row_count"] = len(rows) == supplement["membership_gzip"]["rows"]
    keys = [(r["scene"], r["split"], r["t0"], r["id"], r["horizon"]) for r in rows]
    checks["unique_keys"] = len(keys) == len(set(keys))
    checks["no_test_rows"] = all(r["split"] in {"train", "val"} for r in rows)
    checks["bounded_future_counts"] = all(0 <= int(r["future_points"]) <= int(r["horizon"]) for r in rows)
    checks["zero_future_targets_retained"] = any(int(r["future_points"]) == 0 for r in rows)
    groups = {}
    for r in rows:
        key = (r["scene"], r["split"], r["t0"], r["id"])
        groups.setdefault(key, set()).add(tuple(r[k] for k in ["n_active", "n50", "k", "cpa", "following"]))
    checks["shared_history_metrics_horizon_independent"] = all(len(v) == 1 for v in groups.values())
    tracks = {s: {(r["scene"], r["id"]) for r in rows if r["split"] == s} for s in ["train", "val"]}
    checks["train_val_track_disjoint"] = not bool(tracks["train"] & tracks["val"])
    recounted = {}
    for split in ["train", "val"]:
        recounted[split] = {}
        for horizon in [20, 30, 40]:
            rr = [r for r in rows if r["split"] == split and int(r["horizon"]) == horizon]
            critical = [r for r in rr if int(r["k"]) >= 1]
            values = {
                "targets": len(rr),
                "origins": len({(r["scene"], r["t0"]) for r in rr}),
                "critical_targets": len(critical),
                "critical_origins": len({(r["scene"], r["t0"]) for r in critical}),
                "multi2_targets": sum(int(r["k"]) >= 2 for r in rr),
                "critical_endpoint_observed": sum(r["endpoint"] == "True" for r in critical),
            }
            recounted[split][str(horizon)] = values
            old = audit["split_horizon"][split][str(horizon)]
            checks[f"recount:{split}:{horizon}"] = all(values[k] == old[k] for k in values)
            valid_bounds = True
            for r in rr:
                city = "changchun" if r["scene"] == "0" else "xian"
                lo, hi = summary["split_policy"]["outer_bounds"][city][split]
                valid_bounds &= lo <= int(r["t0"]) - 19 and int(r["t0"]) + horizon <= hi
            checks[f"bounds:{split}:{horizon}"] = bool(valid_bounds)
        expected = summary["expected_sample_counts"]
        values = recounted[split]["40"]
        checks["summary_counts:" + split] = (
            values["targets"] == expected["aux_" + split]
            and values["critical_targets"] == expected["main_" + split]
            and values["critical_origins"] == expected["main_" + split + "_origins"]
            and values["critical_endpoint_observed"] == expected["main_" + split + "_endpoint_available"]
        )
    crossing = np.array([[0., 0., 5., 0.], [10., -10., 0., 5.], [25., 25., 0., 0.]])
    e = pair_metrics(crossing)[0]
    checks["crossing_detected"] = bool(e[0, 1] and e[1, 0])
    follow = pair_metrics(np.array([[0., 0., 10., 0.], [10., 0., 10., 0.]]))[0]
    checks["following_directed"] = bool(follow[0, 1] and not follow[1, 0])
    checks["stationary_density_not_interaction"] = not bool(pair_metrics(np.array([[0., 0., 0., 0.], [3., 0., 0., 0.]]))[0].any())
    checks["diverging_not_interaction"] = not bool(pair_metrics(np.array([[0., 0., -1., 0.], [10., 0., 1., 0.]]))[0].any())
    z = crossing.copy(); z[:, :2] += [100., -80.]
    checks["translation_invariant"] = bool(np.array_equal(e, pair_metrics(z)[0]))
    theta = .71
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    z = np.concatenate((crossing[:, :2] @ rotation, crossing[:, 2:] @ rotation), 1)
    checks["rotation_invariant"] = bool(np.array_equal(e, pair_metrics(z)[0]))
    order = [2, 0, 1]
    checks["permutation_equivariant"] = bool(np.array_equal(pair_metrics(crossing[order])[0], e[np.ix_(order, order)]))
    checks["tcpa_distinct_from_ttc"] = bool(abs(pair_metrics(crossing)[6][0, 1] - 2) < 1e-12 and abs(pair_metrics(crossing)[8][0, 1] - (2 - 5 / math.sqrt(50))) < 1e-12)
    checks["gates_not_misreported_as_passed"] = (
        summary["gate_S"]["status"] == "NOT_ESTABLISHED"
        and summary["gate_G"]["status"] == "NOT_RUN_ON_NEW_TASK"
        and summary["gate_Q"]["status"] == "CLOSED"
    )
    report = {
        "schema": "icct_task_redesign_revalidation_v1", "date": "2026-09-20",
        "base_qgnn_commit": BASE, "prior_research_commit": PRIOR,
        "temporary_branch": BRANCH, "worktree": str(ROOT),
        "scope": {"raw_trajectory_states_parsed": False, "future_labels_opened": False,
                  "model_training": False, "quantum_execution": False,
                  "end_to_end_source_causality_proved": False},
        "checks": checks, "check_count": len(checks), "passed": all(checks.values()),
        "evidence_sha256": evidence, "raw_sha256": raw_hashes,
        "membership_rows": len(rows), "recounted": recounted,
        "release_status": "STRICT_ONLINE_RELEASE_BLOCKED_PENDING_SOURCE_AVAILABILITY",
        "meaning": "Integrity, metadata recount, and synthetic selector checks only; not research Gate S/G/Q success.",
    }
    destination = OUT / "revalidation_20260920.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + chr(10), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["check_count", "passed", "membership_rows", "release_status"]}, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Verification failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
