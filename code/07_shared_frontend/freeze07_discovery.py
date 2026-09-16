"""SENS-FREEZE-07 discovery: F01-D cache contract, downstream loader, pre-freeze manifest.

Read-only on data/; writes reports/f01e/freeze_07/discovery_raw.json (no long-term doc) and
/tmp/f01d_pre_freeze_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/dell/YrM/ICCT")
OUT = ROOT / "reports/f01e/freeze_07"
KEYWORDS = ("f01d", "cache", "dataset", "loader", "pack", "symbol")


def file_manifest(base: Path) -> dict:
    manifest = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(base))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[relative] = {"size_bytes": path.stat().st_size, "sha256": digest}
    return manifest


def main() -> None:
    import numpy as np

    report = {"discovery": {}, "contract_probe": {}, "candidate_code": [], "f01d": {}}
    f01d = ROOT / "data/f01d"
    if f01d.exists():
        listing = []
        for path in sorted(f01d.rglob("*")):
            if path.is_file():
                listing.append({"path": str(path.relative_to(f01d)),
                                "size": path.stat().st_size})
        report["f01d"]["present"] = True
        report["f01d"]["file_count"] = len(listing)
        report["f01d"]["files"] = listing[:60]
        for candidate in ("inputs", "labels", "metadata", "sequences", "diagnostics"):
            directory = f01d / candidate
            if directory.exists():
                names = sorted(p.name for p in directory.iterdir())[:12]
                report["f01d"][f"{candidate}_listing"] = names
        probe = None
        for directory, pattern in (("inputs", "*.npz"), ("labels", "*.npz")):
            directory = f01d / directory
            if directory.exists():
                files = sorted(directory.glob(pattern))
                if files:
                    with np.load(files[0], allow_pickle=True) as payload:
                        probe = {f"{directory.name}:{files[0].name}": {
                            key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
                            for key in payload.files}}
                        report["contract_probe"].update(probe)
        manifest = file_manifest(f01d)
        Path("/tmp/f01d_pre_freeze_manifest.json").write_text(
            json.dumps({"base": str(f01d), "files": manifest}, indent=1))
        report["f01d"]["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        report["f01d"]["manifest_files"] = len(manifest)
    else:
        report["f01d"]["present"] = False

    candidates = []
    for base in (ROOT / "frontend", ROOT / "prediction", ROOT / "experiments", ROOT / "code",
                 ROOT / "scripts"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(errors="ignore").lower()
            if any(keyword in path.name.lower() for keyword in KEYWORDS) or \
                    "f01d" in text or "state_hat" in text or "track_exists" in text:
                candidates.append(str(path.relative_to(ROOT)))
    report["candidate_code"] = sorted(candidates)[:40]
    try:
        import frontend.symbol_dataset as symbol_dataset

        report["downstream_loader"] = {
            "module": "frontend.symbol_dataset",
            "public": [name for name in dir(symbol_dataset) if not name.startswith("_")][:40]}
    except Exception as error:  # noqa: BLE001
        report["downstream_loader"] = {"module": "frontend.symbol_dataset", "error": repr(error)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "discovery_raw.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({"f01d_present": report["f01d"].get("present"),
                      "f01d_files": report["f01d"].get("file_count"),
                      "contract_probe_keys": list(report["contract_probe"].keys()),
                      "candidate_code": report["candidate_code"][:15],
                      "loader": report["downstream_loader"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()