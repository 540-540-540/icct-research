"""Run frozen module selfchecks with full tracebacks (P0 smoke)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontend.sensing import coords, detector, simulator, waveform  # noqa: E402


def main() -> None:
    results = {}
    for name, module in (("waveform", waveform), ("coords", coords), ("simulator", simulator),
                         ("detector", detector)):
        try:
            if module.__name__ == "frontend.sensing.waveform":
                value = {"df": waveform.PaperWaveform().df, "elements": waveform.ArrayConfig().elements}
            else:
                value = module.selfcheck() if hasattr(module, "selfcheck") else "no selfcheck"
            results[name] = {"passed": True, "detail": value}
        except Exception:  # noqa: BLE001
            results[name] = {"passed": False, "traceback": traceback.format_exc()}
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    if not all(entry["passed"] for entry in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()