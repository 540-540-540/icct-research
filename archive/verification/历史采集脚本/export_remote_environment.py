from importlib import metadata
from pathlib import Path
import platform
import sys

lines = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    if name:
        lines.append(f"{name}=={dist.version}")
Path(sys.argv[1]).write_text("\n".join(sorted(set(lines), key=str.lower)) + "\n", encoding="utf-8")

if len(sys.argv) > 2:
    import torch
    try:
        import pennylane
        pl_version = pennylane.__version__
    except Exception as exc:
        pl_version = f"unavailable: {exc}"
    info = [
        f"platform={platform.platform()}",
        f"python={sys.version}",
        f"executable={sys.executable}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"pennylane={pl_version}",
        f"cuda_available={torch.cuda.is_available()}",
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
    ]
    Path(sys.argv[2]).write_text("\n".join(info) + "\n", encoding="utf-8")
