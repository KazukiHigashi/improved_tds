"""Detached four-worker campaign; finish with validation and HTML, no agent polling."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('analysis/strict_synergy_paper_v3_clipped/reports'))
    parser.add_argument('--output', type=Path, default=Path('analysis/adaptive_synergy_v1'))
    parser.add_argument('--pool', type=Path, default=Path('../tool-pose-pipeline/grasps_v3'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    reports = root/'reports'; reports.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path('src').resolve()), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    files = sorted(args.source.glob('*_seed*.json'))
    if len(files) != 75:
        raise ValueError(f'Expected 75 inputs; found {len(files)}')
    def status(state, **extra):
        (root/'status.json').write_text(json.dumps(dict(state=state, utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2), encoding='utf-8')
    status('running', total=len(files), workers=args.workers)
    def run(path):
        dest = reports/path.name
        if dest.exists():
            artifact = json.loads(dest.read_text(encoding='utf-8'))
            if artifact.get('pilot') or set(artifact['configurations']) != {'pca', 'pls'}:
                raise ValueError(f'Invalid existing result: {dest}')
            return path.name
        cmd = [sys.executable, '-m', 'improved_tds.experiments.evaluate_adaptive_synergy',
               '--strict-report', str(path.resolve()), '--pose-pool-dir', str(args.pool.resolve()), '--output', str(dest)]
        if path.stem.endswith('seed20260911'):
            cmd += ['--render-dir', str(root/'images'/path.stem.split('_seed')[0])]
        with (reports/(path.stem+'.log')).open('w', encoding='utf-8') as stdout, (reports/(path.stem+'.err.log')).open('w', encoding='utf-8') as stderr:
            subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=True)
        return path.name
    try:
        failures = []; complete = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run, path): path.name for path in files}
            for future in as_completed(futures):
                try:
                    future.result(); complete += 1
                except Exception as exc:
                    failures.append(dict(report=futures[future], error=str(exc)))
                status('running', total=len(files), complete=complete, failures=failures)
        if failures:
            status('failed', complete=complete, failures=failures)
            return 1
        status('summarizing', complete=complete)
        with (root/'summary.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-m', 'improved_tds.experiments.summarize_adaptive_synergy',
                            '--report-dir', str(reports), '--output-dir', str(root/'summary'), '--expected-reports', '75'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('completed', complete=complete, html=str(root/'summary'/'index.html'))
    except Exception:
        status('failed', error=traceback.format_exc())
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
