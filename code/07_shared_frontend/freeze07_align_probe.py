"""SENS-FREEZE-07 dry-run alignment inspection: permutation vs set mismatch (read-only)."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/dell/YrM/ICCT")
metadata = json.loads((ROOT / "data/f01e_dryrun/metadata/train.json").read_text())
stats = Counter()
examples = []
for sample in metadata["samples"]:
    reference = sample["slot_alignment"]
    by_snr = sample["slot_alignment_by_snr"]
    reference_slots = {int(entry["slot"]): int(entry["source_key"]) for entry in reference}
    for name, mapping in by_snr.items():
        mapping = {int(slot): int(key) for slot, key in mapping.items()}
        if mapping == reference_slots:
            stats[f"identical_{name}"] += 1
            continue
        if set(mapping.values()) == set(reference_slots.values()):
            stats[f"permutation_{name}"] += 1
        else:
            stats[f"setdiff_{name}"] += 1
            if len(examples) < 5 and name == "-5.0":
                examples.append({"sample": sample["sample_index"], "reference": reference_slots,
                                 "snr_m5": mapping})
    stats[f"slots_ref"] += len(reference_slots)
    stats[f"slots_m5"] += len(by_snr["-5.0"])
    stats[f"windows"] += 1
print(json.dumps(dict(stats), indent=1))
print(json.dumps(examples, indent=1)[:1500])