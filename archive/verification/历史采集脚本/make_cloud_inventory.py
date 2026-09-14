import hashlib
import json
from pathlib import Path

root = Path("/home/js_cn/sensing")
directories = [
    "data", "dataset_lankershim_clean_v1", "results", "双图研究框架",
    "gpt2", "preserved_experiments", "paper_figures", "paper_figures_nature",
    "figures", "assets", "figure_tools", "logs", "newnew_figure", "dataset",
    "deepmimo_generate_data", "diagnostics/core_message_seed2026_v2",
]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

result = {"root": str(root), "directories": {}, "key_files": {}}
for relative in directories:
    directory = root / relative
    files = [p for p in directory.rglob("*") if p.is_file()]
    result["directories"][relative] = {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }
for relative in [
    "Lankershim_Vehicle_Trajectories.csv",
    "data/multitarget_lankershim_v1.npz",
    "data/multitarget_lankershim_h20_p40_matched_v1.npz",
    "gpt2/pytorch_model.bin",
]:
    path = root / relative
    result["key_files"][relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
Path("/tmp/icct_cloud_inventory.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
