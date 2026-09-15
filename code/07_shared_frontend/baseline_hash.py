"""Record SHA256 baselines of the frozen oracle assets before any new frontend work."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/f01e/baseline_hashes.json"

DATA_FILES = [f"data/f01d/inputs/{split}_snr_{snr}.npz"
              for split in ("train", "V_select")
              for snr in (5, 10, 15, 20)]
DATA_FILES += ["data/f01d/labels/train.npz", "data/f01d/labels/V_select.npz",
               "data/f01d/normalization.json",
               "data/f01d/metadata/train.json", "data/f01d/metadata/V_select.json"]
CONFIG_FILES = ["configs/symbol_frontend.json"]
ENTRY_FILES = ["scripts/run_f01d_gpu.py", "scripts/rebuild_f01d_gpu.py"]
SENSING_FILES = ["frontend/symbol_level_gpu.py", "frontend/symbol_level.py",
                 "frontend/generate_symbol_gpu.py", "frontend/echo_source.py",
                 "frontend/ofdm_echo.py", "frontend/detector.py", "frontend/jpda.py",
                 "frontend/tracker.py", "frontend/pack_symbol_dataset.py",
                 "frontend/symbol_dataset.py"]
ALL_FILES = DATA_FILES + CONFIG_FILES + ENTRY_FILES + SENSING_FILES


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = {"purpose": "frozen oracle baseline hashes before REBUILD-03A", "generated_unix": time.time(),
              "root": str(ROOT), "files": {}, "missing": []}
    for name in ALL_FILES:
        path = ROOT / name
        if not path.exists():
            report["missing"].append(name)
            continue
        stat = path.stat()
        report["files"][name] = {"sha256": digest(path), "size": stat.st_size, "mtime": stat.st_mtime}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(OUT.relative_to(ROOT)), "hashed": len(report["files"]),
                      "missing": report["missing"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()