#!/usr/bin/env python3
"""Download the exact public SinD recordings used by the ICCT migration."""
from __future__ import annotations
import argparse, hashlib, urllib.request
from pathlib import Path

COMMIT="930e4dea78d924c6e9a58ff8e378331f93bba8ec"
FILES={
    "changchun":{
        "url":f"https://media.githubusercontent.com/media/SOTIF-AVLab/SinD/{COMMIT}/Data/Changchun/changchun_pudong_507_009/Veh_smoothed_tracks.csv",
        "sha256":"f3011d7dc1786f940981a9d49c7f83c7860beda9d14ed0e06c995f7b7e590692",
    },
    "xian":{
        "url":f"https://media.githubusercontent.com/media/SOTIF-AVLab/SinD/{COMMIT}/Data/Xi%27an/Xi%27an_412_m1/Veh_smoothed_tracks.csv",
        "sha256":"1bf5d8450577249beb9c97bc210e7510babf0796ba817f9a88c64d19a5a5e4a7",
    },
}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="data/sind/raw")
    ap.add_argument("--verify-only",action="store_true")
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    root=Path(args.root)
    for city,meta in FILES.items():
        path=root/city/"Veh_smoothed_tracks.csv"
        if args.verify_only:
            if not path.exists():
                raise SystemExit(f"missing {path}")
        elif not path.exists() or args.force:
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix(".csv.part")
            tmp.unlink(missing_ok=True)
            print(f"downloading {city}: {meta['url']}",flush=True)
            urllib.request.urlretrieve(meta["url"],tmp)
            tmp.replace(path)
        actual=sha256(path)
        if actual!=meta["sha256"]:
            raise SystemExit(f"{city}: SHA mismatch {actual}")
        print(f"{city}: OK {path} sha256={actual}",flush=True)

if __name__=="__main__":
    main()
