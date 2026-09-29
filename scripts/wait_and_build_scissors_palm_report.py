"""Build the readable five-task HTML once the background TDS run completes."""

import argparse
from pathlib import Path
import subprocess
import sys
import time
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--analysis-root',type=Path,required=True)
    p.add_argument('--timeout-hours',type=float,default=30.)
    a=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    marker=a.run_root.parent/(a.run_root.name+'_tds_completion/status.json')
    deadline=time.monotonic()+a.timeout_hours*3600
    while time.monotonic()<deadline:
        try:
            state=json.loads(marker.read_text(encoding='utf-8'))['stage']
        except (FileNotFoundError,json.JSONDecodeError):
            state=None
        if state=='completed':
            subprocess.run([sys.executable,str(project/'scripts/build_scissors_palm_all5_report.py'),
                            '--analysis-root',str(a.analysis_root),'--run-root',str(a.run_root)],
                           cwd=project,check=True)
            return
        if state=='failed':
            raise RuntimeError('five-task TDS run failed; readable report not built')
        time.sleep(60)
    raise TimeoutError('five-task TDS run did not complete before report timeout')


if __name__=='__main__':
    main()
