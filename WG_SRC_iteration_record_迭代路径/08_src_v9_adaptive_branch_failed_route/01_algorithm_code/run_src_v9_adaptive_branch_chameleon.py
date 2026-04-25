#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError('Cannot locate root containing planetoid/data')


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'src_v9').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing src_v9 and scripts')


def main():
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)

    algo_path = project_root / 'src_v9' / 'algo1_adaptive_branch_pca_src_v9.py'
    data_base = drive_root / 'planetoid' / 'data' / 'chameleon'
    out_dir = project_root / 'scripts' / 'results_src_v9_adaptive_branch_chameleon'

    if not algo_path.exists():
        raise FileNotFoundError(
            f'Cannot find algorithm file: {algo_path}\n'
            f'Please put algo1_adaptive_branch_pca_src_v9.py into src_v9 first.'
        )

    cmd = [
        sys.executable,
        str(algo_path),
        '--dataset', 'chameleon',
        '--data-base', str(data_base),
        '--out-dir', str(out_dir),
    ]

    print('Running command:')
    print(' '.join(f'"{x}"' if ' ' in x else x for x in cmd))
    print()
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()


