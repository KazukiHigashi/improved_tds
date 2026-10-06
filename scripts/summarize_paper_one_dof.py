"""Summarize only the PCA/PLS one-DoF results in the frozen 15-task bundle.

The selected report hashes in the bundle manifest are verified before any
number is printed. This script does not rerun or modify the experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


TASKS = tuple(f'{category}_lite_{variant}'
              for category in ('scissors', 'push', 'spray')
              for variant in range(1, 6))


def summarize(bundle: Path) -> list[dict]:
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['tasks']) != set(TASKS):
        raise ValueError('paper bundle must contain exactly the selected 15 tasks')
    rows = []
    for task in TASKS:
        selection = manifest['tasks'][task]
        report_file = bundle/selection['selected_report']
        report_bytes = report_file.read_bytes()
        if (hashlib.sha256(report_bytes).hexdigest() !=
                selection['selected_report_sha256']):
            raise ValueError(f'selected report hash mismatch: {task}')
        report = json.loads(report_bytes)
        if report['task'] != task or report['fit_rows'] != 200:
            raise ValueError(f'task or success-posture count mismatch: {task}')
        if (report['trajectory_seconds'] != 5. or
                report['operation_hold_seconds'] != 2.):
            raise ValueError(f'one-DoF trajectory timing changed: {task}')
        target_low, target_high = map(float, report['target_range'])
        width = target_high-target_low
        if width <= 0:
            raise ValueError(f'invalid target interval: {task}')
        for method in ('pca', 'pls'):
            summary = report['configurations'][f'{method}_1d']['summary']
            if not summary['inverse_available']:
                raise ValueError(f'one-DoF inverse unavailable: {task}/{method}')
            inverse = summary['controller_comparison']['isotonic']
            if inverse['held_mae'] is None or inverse['trials'] != 72:
                raise ValueError(f'incomplete one-DoF inverse test: {task}/{method}')
            rows.append(dict(task=task, category=task.split('_')[0], method=method,
                inverse_domain_percent=100*summary['inverse_domain_fraction'],
                independent_coverage_percent=100*summary['independent_coverage'],
                normalized_held_mae_percent=100*inverse['held_mae']/width,
                held_trials=int(inverse['held']), total_trials=int(inverse['trials'])))
    return rows


def category_rows(rows: list[dict]) -> list[dict]:
    result = []
    for category in ('scissors', 'push', 'spray'):
        for method in ('pca', 'pls'):
            selected = [row for row in rows if row['category'] == category
                        and row['method'] == method]
            result.append(dict(category=category, method=method,
                inverse_domain_percent=mean(r['inverse_domain_percent'] for r in selected),
                independent_coverage_percent=mean(
                    r['independent_coverage_percent'] for r in selected),
                normalized_held_mae_percent=mean(
                    r['normalized_held_mae_percent'] for r in selected),
                held_trials=sum(r['held_trials'] for r in selected),
                total_trials=sum(r['total_trials'] for r in selected)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path,
        default=Path('analysis/paper15_paper_bundle_20260928'))
    parser.add_argument('--json', action='store_true', help='print machine-readable data')
    args = parser.parse_args()
    rows = summarize(args.bundle)
    if args.json:
        print(json.dumps(dict(by_task=rows, by_category=category_rows(rows)), indent=2))
        return
    print('Category Method Inverse-domain[%] Independent-coverage[%] '
          'Held-MAE[% of target width] Held/Trials')
    for row in category_rows(rows):
        print(f"{row['category']:9s} {row['method']:6s} "
              f"{row['inverse_domain_percent']:17.2f} "
              f"{row['independent_coverage_percent']:23.2f} "
              f"{row['normalized_held_mae_percent']:27.2f} "
              f"{row['held_trials']}/{row['total_trials']}")


if __name__ == '__main__':
    main()
