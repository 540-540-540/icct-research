"""Read-only inventory for extending the trajectory prediction horizon."""
from pathlib import Path
import json
import numpy as np

root = Path('/home/js_cn/sensing')
cache = root / 'data/multitarget_lankershim_v1.npz'
with np.load(cache, allow_pickle=False) as data:
    print(json.dumps({name: list(data[name].shape) for name in data.files}, indent=2))
    metadata = json.loads(str(data['metadata'].item()))
    print(json.dumps(metadata, indent=2))
    source = Path(metadata['source_csv'])
    if not source.is_absolute():
        source = root / source
    print(json.dumps({'resolved_source': str(source), 'source_exists': source.exists(),
                      'source_bytes': source.stat().st_size if source.exists() else None}))

for pattern in ('*lankershim*', '*multitarget*dataset*.py', '*trajectory*dataset*.py'):
    for path in sorted(root.rglob(pattern)):
        if path.is_file() and len(path.relative_to(root).parts) <= 5:
            print(path.relative_to(root))
