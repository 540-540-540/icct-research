"""Read-only class-balance check for automatically generated intent labels."""
import sys
from pathlib import Path
import torch

ROOT = Path('/home/js_cn/sensing')
HERE = ROOT / '双图研究框架/未来词QGNN_LLM实验_2026-09-09'
BASE = ROOT / 'diagnostics/core_message_seed2026_v2'
sys.path[:0] = [str(HERE), str(BASE), str(ROOT)]

from run_future_token_qgnn_llm import load_banks
from run_intent_token_planner import intent_labels

bank, _, _, _, _, _ = load_banks()
labels = intent_labels(bank['history'], bank['future'], bank['mask'])
counts = {
    name: [
        torch.bincount(value[:, :, stage][bank['mask']], minlength=3).tolist()
        for stage in range(3)
    ]
    for name, value in labels.items()
}
print(counts)
