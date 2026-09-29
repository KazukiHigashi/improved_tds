"""Open the current Scissors success-posture recollection in MuJoCo Viewer.

Example from improved_tds project root:
    python scripts/view_reach_collection.py --variant 4 --episodes 30 --speed 1

No database is written. The frozen Best Policy and collection rules match the
completed September 23 reach campaign. Pass --bin 9 to watch high-angle trials.
"""
import argparse
from pathlib import Path

from improved_tds.experiments.collect_reach_successes import main as collect


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=int,choices=(3,4,5),default=4)
    p.add_argument('--run-root',type=Path,default=Path('runs/scissors_reach_20260923'))
    p.add_argument('--control-version',choices=('simple','smooth'),default='simple')
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--per-bin',type=int,default=10)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--print-every',type=int,default=10)
    p.add_argument('--bin',type=int,choices=range(10),help='restrict preview to one target-angle bin')
    p.add_argument('--pose',type=int,help='restrict preview to one authored initial pose')
    p.add_argument('--pause-on-success',action='store_true')
    p.add_argument('--hold-seconds',type=float,default=0.,help='preview-only fixed-command hold after an accepted success')
    p.add_argument('--no-render',action='store_true',help='check the same logic without opening a window')
    a=p.parse_args(argv)
    if a.episodes<1 or a.per_bin<1 or a.speed<=0 or a.hold_seconds<0:p.error('positive episodes, quota and speed required')
    task=f'scissors_lite_{a.variant}'
    model=a.run_root/task/'best_success_policy.zip'
    pool=a.run_root/'pose_pool'
    if not model.is_file():p.error(f'model not found: {model}')
    if not (pool/f'{task}.npz').is_file():p.error(f'pose pool not found: {pool/task}.npz')
    forwarded=['--env-id',f'ScissorsVariant{a.variant}-v0','--model',str(model),
        '--pose-pool-dir',str(pool),'--preview','--max-episodes',str(a.episodes),
        '--per-bin',str(a.per_bin),'--seed',str(a.seed if a.seed is not None else 9300+a.variant),
        '--speed',str(a.speed),'--print-every',str(a.print_every),
        '--scissors-control-version',a.control_version]
    if a.bin is not None:forwarded+=['--preview-bin',str(a.bin)]
    if a.pose is not None:forwarded+=['--initial-pose-index',str(a.pose)]
    if a.pause_on_success:forwarded.append('--pause-on-success')
    if a.hold_seconds:forwarded+=['--preview-hold-seconds',str(a.hold_seconds)]
    if a.no_render:forwarded.append('--no-render')
    return collect(forwarded)


if __name__=='__main__':
    raise SystemExit(main())
