"""Descriptive G1 validation diagnostics. Never reads the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def subset(rows, fn):
    return [r for r in rows if fn(r)]


def summarize(ref_rows, model_rows, selector):
    ref = subset(ref_rows, selector)
    cur = subset(model_rows, selector)
    if len(ref) != len(cur) or not ref:
        raise ValueError("Row alignment/empty stratum")
    ra, rf = mean(ref, "ADE"), mean(ref, "FDE")
    ca, cf = mean(cur, "ADE"), mean(cur, "FDE")
    total_ref_ade = sum(float(r["ADE"]) for r in ref_rows)
    total_ref_fde = sum(float(r["FDE"]) for r in ref_rows)
    return {
        "scenes": len(ref),
        "reference_ADE": ra,
        "model_ADE": ca,
        "ADE_gain_percent": 100.0 * (1.0 - ca / ra),
        "reference_FDE": rf,
        "model_FDE": cf,
        "FDE_gain_percent": 100.0 * (1.0 - cf / rf),
        "reference_ADE_error_share": sum(float(r["ADE"]) for r in ref) / total_ref_ade,
        "reference_FDE_error_share": sum(float(r["FDE"]) for r in ref) / total_ref_fde,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ref_payload = json.loads(Path(args.reference).read_text())
    ref_rows = ref_payload["rows"]
    strata = {
        "all": lambda r: True,
        "N_2_3": lambda r: int(r["vehicle_count"]) <= 3,
        "N_4_5": lambda r: 4 <= int(r["vehicle_count"]) <= 5,
        "N_6_8": lambda r: int(r["vehicle_count"]) >= 6,
        "close_pairs_30m_ge_2": lambda r: int(r["close_pairs_30m"]) >= 2,
        "closing_pairs_30m_gt_0p5_ge_1": lambda r: int(r["closing_pairs_30m_gt_0p5"]) >= 1,
        "min_cpa_lt_10m": lambda r: float(r["min_cpa_distance_m"]) < 10.0,
        # Diagnostic only; this was not preregistered before first G1 result.
        "diagnostic_high_combo": lambda r: (
            int(r["close_pairs_30m"]) >= 2
            and int(r["closing_pairs_30m_gt_0p5"]) >= 1
        ),
    }
    output = {
        "reference": ref_payload["summary"]["model"],
        "snr_db": ref_payload["summary"]["snr_db"],
        "note": "Validation-only descriptive diagnostic; no independence/significance claim.",
        "test_set_used": False,
        "models": {},
    }
    for model_path in args.models:
        payload = json.loads(Path(model_path).read_text())
        rows = payload["rows"]
        if [r["dataset_index"] for r in rows] != [r["dataset_index"] for r in ref_rows]:
            raise ValueError("Evaluation rows are not aligned")
        output["models"][payload["summary"]["model"]] = {
            name: summarize(ref_rows, rows, selector)
            for name, selector in strata.items()
        }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

