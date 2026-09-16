"""Lightweight byte-level audit for Matplotlib PDF font embedding."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for raw_path in args.paths:
        path = Path(raw_path)
        data = path.read_bytes()
        type3 = b"/Subtype /Type3" in data
        embedded_truetype = b"/FontFile2" in data
        times_new_roman = b"TimesNewRoman" in data
        print(
            f"{path}: Type3={type3}, FontFile2={embedded_truetype}, "
            f"TimesNewRoman={times_new_roman}"
        )
        if type3 or not embedded_truetype or not times_new_roman:
            raise SystemExit(2)
    print("PDF font audit: PASS")


if __name__ == "__main__":
    main()
