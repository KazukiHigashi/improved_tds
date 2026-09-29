"""Wait for the five Scissors palm-sphere runs, then launch matched direct TDS.

The background helper checks on disk every minute. It is not an interactive
agent monitor and consumes no model turns while training is in progress.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--timeout-hours',type=float,default=24.)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    run_root=args.run_root.resolve()
    analysis_root=args.analysis_root.resolve()
    helper_root=run_root.parent/(run_root.name+'_tds_completion')
    if helper_root.exists() or analysis_root.exists():
        raise FileExistsError('helper or analysis output already exists')
    helper_root.mkdir()

    def status(stage,**extra):
        (helper_root/'status.json').write_text(json.dumps(dict(stage=stage,
            utc=datetime.now(timezone.utc).isoformat(),run_root=str(run_root),
            analysis_root=str(analysis_root),**extra),indent=2),encoding='utf-8')

    try:
        deadline=time.monotonic()+args.timeout_hours*3600
        while True:
            saved=json.loads((run_root/'status.json').read_text(encoding='utf-8'))['tasks']
            if any(str(v).startswith('failed') for v in saved.values()):
                raise RuntimeError(f'learning or posture collection failed: {saved}')
            if len(saved)==5 and all(v=='completed' for v in saved.values()):
                break
            if time.monotonic()>deadline:
                raise TimeoutError(f'five-task run did not finish in {args.timeout_hours} hours')
            status('waiting_for_five_success_databases',tasks=saved)
            time.sleep(60)
        for variant in range(1,6):
            task=f'scissors_lite_{variant}'
            with np.load(run_root/task/'success_postures.npz',allow_pickle=False) as z:
                meta=json.loads(z['metadata_json'].item())
                if (not meta['complete'] or len(z['q'])!=200 or not z['success'].all()
                        or meta['contract']['grasp_criterion']!=
                        'hinge_center_inside_palm_attached_original_sphere'):
                    raise ValueError(f'{task}: invalid new-criterion 200-success database')
        status('tds_evaluation')
        cmd=[sys.executable,str(project/'scripts/run_reach_tds_direct.py'),
             '--source-root',str(run_root),'--output',str(analysis_root),
             '--tasks',*[f'scissors_lite_{i}' for i in range(1,6)],'--workers','4']
        (helper_root/'tds_command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        with (helper_root/'tds.log').open('w',encoding='utf-8') as out, \
                (helper_root/'tds.err.log').open('w',encoding='utf-8') as err:
            subprocess.run(cmd,cwd=project,env=os.environ.copy(),stdout=out,stderr=err,check=True)
        saved=json.loads((analysis_root/'status.json').read_text(encoding='utf-8'))['jobs']
        if any(v!='completed' for v in saved.values()):
            raise RuntimeError(f'TDS evaluation incomplete: {saved}')
        status('completed',report=str(analysis_root/'index.html'))
    except Exception as exc:
        status('failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
