#!/usr/bin/env python3
"""Freeze train-only SinD 3-BS geometry search results into the ISAC config."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--report",default="reports/sind/sind_geometry_search.json")
    ap.add_argument("--config",default="configs/sind_controlled_isac.json")
    args=ap.parse_args()
    report=Path(args.report)
    config=Path(args.config)
    geo=json.loads(report.read_text(encoding="utf-8"))
    if geo.get("scope")!="train_only":
        raise SystemExit("geometry report is not train_only")
    cfg=json.loads(config.read_text(encoding="utf-8"))
    for scene in cfg["scenes"]:
        sid=str(scene["scene_id"])
        block=geo["scenes"][sid]
        best=block["best"]
        scene["junction_center_xy_m"]=block["center_xy_m"]
        scene["stations_xy_m"]=best["stations"]
        scene["boresights_deg"]=best["boresights_deg"]
    cfg["geometry_freeze"]["search_scope"]="train only after raw-quality filtering"
    cfg["geometry_freeze"]["report"]=str(report).replace("\\","/")
    config.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    for scene in cfg["scenes"]:
        print({
            "scene_id":scene["scene_id"],
            "stations_xy_m":scene["stations_xy_m"],
            "boresights_deg":scene["boresights_deg"],
        })

if __name__=="__main__":
    main()
