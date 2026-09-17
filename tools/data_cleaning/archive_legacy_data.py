#!/usr/bin/env python3
"""Inventory and archive legacy derived data on the ICCT server (dry-run by default).

The formal data chain after this round is: raw source -> cleaning script -> canonical
trajectories. This tool moves files that are (a) derived, (b) regenerable, and
(c) not consumed by the current pipeline into a legacy archive, preserving the
relative directory structure. It never deletes anything and never touches model
checkpoints or formal experiment results.

Classification rules (first match wins, exact path components):
  A  keep in place:
     - data/Lankershim_Vehicle_Trajectories.csv        raw source
     - data/processed/**                               canonical clean dataset
     - data/f01_source/**                              current ISAC frontend input (A01)
     - data/f01e/**                                    current frozen shared frontend cache
     - data/prompt_bank/**                             small static prompt bank
  B  move to archive (regenerable legacy derived data):
     - data/dataset_lankershim_clean_v1/**             old clean dataset (marked legacy)
     - data/f01_frontend/**                            old F01-C production frontend output
     - data/f01_frontend_baseline/**                   old baseline frontend output
     - data/f01_echo_audit/**                          F01-B echo audit cache
     - data/f01d/**                                    F01-D intermediate cache
     - data/f01e_dryrun/**                             F01-E dry-run cache
     - data/multitarget_lankershim_v1.npz              old multi-target QGNN cache
     - data/multitarget_lankershim_h20_p40_matched_v1.npz
  C  keep, list for human confirmation (never moved here):
     - checkpoints/**                                  model checkpoints
     - .codex-work/**                                  diagnostic scratch outputs
     - any other target-extension file outside data/

Usage:
    python tools/data_cleaning/archive_legacy_data.py            # dry run, prints plan
    python tools/data_cleaning/archive_legacy_data.py --execute  # moves B files, writes manifest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

TARGET_EXT = {'.csv', '.npy', '.npz', '.parquet', '.pkl', '.pt'}
WALK_PRUNE = {'.git', 'archive', 'node_modules', '__pycache__', 'results', 'reports'}
KEEP_PREFIXES = [
    ('data/Lankershim_Vehicle_Trajectories.csv', 'raw source file'),
    ('data/processed', 'canonical clean dataset'),
    ('data/f01_source', 'current ISAC frontend input (A01 SourceEpisodes)'),
    ('data/f01e', 'current frozen shared frontend cache (BDX-01 lineage)'),
    ('data/prompt_bank', 'small static prompt bank'),
]
ARCHIVE_PREFIXES = [
    ('data/dataset_lankershim_clean_v1', 'old clean dataset; reports/exposure_manifest.json marks it legacy'),
    ('data/f01_frontend', 'old F01-C production frontend output; superseded by frozen f01e; regenerable'),
    ('data/f01_frontend_baseline', 'old baseline frontend output; superseded by frozen f01e; regenerable'),
    ('data/f01_echo_audit', 'F01-B echo audit cache; regenerable; only historical configs reference it'),
    ('data/f01d', 'F01-D intermediate cache; regenerable via scripts/run_f01d_gpu.py from raw'),
    ('data/f01e_dryrun', 'F01-E dry-run cache; regenerable; historical freeze scripts only'),
    ('data/multitarget_lankershim_v1.npz', 'old multi-target QGNN cache; no current pipeline reference'),
    ('data/multitarget_lankershim_h20_p40_matched_v1.npz', 'old long-horizon QGNN cache; no current pipeline reference'),
]
CONFIRM_PREFIXES = [
    ('checkpoints', 'model checkpoints; protected, never moved'),
    ('.codex-work', 'diagnostic scratch outputs; wait for human confirmation'),
]


def log(*args) -> None:
    print('[%7.1fs]' % (time.time() - START), *args, file=sys.stderr, flush=True)


def sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def human(size: int) -> str:
    value = float(size)
    for unit in ('B', 'K', 'M', 'G'):
        if value < 1024:
            return f'{value:.1f}{unit}'
        value /= 1024
    return f'{value:.1f}T'


def classify(relative: str):
    for prefix, reason in KEEP_PREFIXES:
        if relative == prefix or relative.startswith(prefix + '/'):
            return 'A', reason
    for prefix, reason in ARCHIVE_PREFIXES:
        if relative == prefix or relative.startswith(prefix + '/'):
            return 'B', reason
    for prefix, reason in CONFIRM_PREFIXES:
        if relative == prefix or relative.startswith(prefix + '/'):
            return 'C', reason
    return 'C', 'outside active data path; not covered by frozen classification rules'


def main() -> None:
    parser = argparse.ArgumentParser(description='Inventory/archive legacy ICCT data (dry-run default)')
    parser.add_argument('--root', default='/home/dell/YrM/ICCT')
    parser.add_argument('--archive-root', default='/home/dell/YrM/ICCT/archive/legacy_data_precanonical')
    parser.add_argument('--manifest', default='/home/dell/YrM/ICCT/reports/data_cleaning/legacy_data_manifest.csv')
    parser.add_argument('--summary', default='/tmp/legacy_data_summary.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--hash-limit-mb', type=int, default=8192,
                        help='skip SHA256 above this size (manifest gets empty hash)')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    archive_root = Path(args.archive_root).resolve()
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in WALK_PRUNE]
        base_path = Path(base)
        if archive_root == base_path or archive_root in base_path.parents:
            continue
        for name in files:
            path = base_path / name
            relative = path.relative_to(root).as_posix()
            if not (relative.startswith('data/') or path.suffix.lower() in TARGET_EXT):
                continue
            classification, reason = classify(relative)
            size = path.stat().st_size
            records.append({'path': path, 'relative': relative, 'size': size,
                            'classification': classification, 'reason': reason})
    stats = {key: {'files': 0, 'size': 0} for key in ('A', 'B', 'C')}
    for record in records:
        stats[record['classification']]['files'] += 1
        stats[record['classification']]['size'] += record['size']
    log('scanned', len(records), 'files;', {k: (v['files'], human(v['size'])) for k, v in stats.items()})
    for record in records:
        if record['classification'] == 'C':
            log('  C:', record['relative'], human(record['size']), '-', record['reason'])

    manifest_rows = []
    moved_files = 0
    moved_size = 0
    if args.execute:
        for record in records:
            if record['classification'] != 'B':
                continue
            source = record['path']
            target = archive_root / record['relative']
            if not source.exists():
                log('missing during move (skipped)', record['relative'])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise SystemExit(f'archive target already exists: {target}')
            digest = sha256(source) if record['size'] <= args.hash_limit_mb * (1 << 20) else ''
            shutil.move(str(source), str(target))
            manifest_rows.append({'original_path': record['relative'],
                                  'archive_path': target.as_posix(),
                                  'file_size': record['size'], 'sha256': digest,
                                  'classification': 'B', 'reason': record['reason']})
            moved_files += 1
            moved_size += record['size']
        for record in records:
            if record['classification'] != 'C':
                continue
            digest = sha256(record['path']) if record['size'] <= args.hash_limit_mb * (1 << 20) else ''
            manifest_rows.append({'original_path': record['relative'], 'archive_path': '',
                                  'file_size': record['size'], 'sha256': digest,
                                  'classification': 'C', 'reason': record['reason']})
        # remove directories left empty by the moves, scoped to the moved prefixes only
        moved_roots = set()
        for row in manifest_rows:
            if row['classification'] == 'B':
                moved_roots.add(root / Path(row['original_path']).parts[0] / Path(row['original_path']).parts[1])
        for moved_root in sorted(moved_roots, key=lambda p: len(p.parts), reverse=True):
            if not moved_root.is_dir():
                continue
            for base, dirs, files in os.walk(moved_root, topdown=False):
                base_path = Path(base)
                try:
                    if base_path.is_dir() and not any(base_path.iterdir()):
                        base_path.rmdir()
                except OSError:
                    pass
        if moved_files == 0 and manifest_path.exists():
            log('nothing to move; existing manifest left untouched', manifest_path)
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['original_path', 'archive_path', 'file_size',
                                                            'sha256', 'classification', 'reason'])
                writer.writeheader()
                writer.writerows(sorted(manifest_rows, key=lambda row: (row['classification'],
                                                                        row['original_path'])))
            log('manifest written', manifest_path, 'moved', moved_files)

    summary = {'root': str(root), 'archive_root': str(archive_root), 'execute': bool(args.execute),
               'stats': stats, 'moved_files': moved_files, 'moved_size': moved_size,
               'confirm_files': [{'path': record['relative'], 'size': record['size'],
                                  'reason': record['reason']}
                                 for record in records if record['classification'] == 'C']}
    Path(args.summary).write_text(json.dumps(summary, indent=1, default=float) + '\n', encoding='utf-8')
    print(json.dumps({'execute': summary['execute'], 'stats': stats,
                      'moved_files': moved_files, 'moved_size': human(moved_size)}, default=float))


if __name__ == '__main__':
    START = time.time()
    main()