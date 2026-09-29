"""Summarize paired palm-sphere Scissors auxiliary-contact reward pilots."""
import argparse
import json
from pathlib import Path

import numpy as np


def report_row(folder,variant,weight):
    history=json.loads((folder/'validation_history.json').read_text(encoding='utf-8'))
    entry=history['history'][-1]
    auxiliary=[e.get('mean_reward_terms',{}).get('reward_contact_auxiliary',0.)
               for e in entry.get('episode_metrics',[])]
    return dict(variant=variant,contact_auxiliary_weight=weight,timesteps=entry['timesteps'],
        episodes=entry['episodes'],success_rate=entry['success_rate'],
        final_success_rate=entry['final_success_rate'],
        grasp_maintenance_rate=entry['grasp_maintenance_rate'],
        target_coverage_fraction=entry['target_coverage_fraction'],
        min_target_success_rate=entry['min_target_success_rate'],
        mean_abs_error=entry['mean_abs_error'],
        numerical_failure_episodes=entry['numerical_failure_episodes'],
        mean_contact_auxiliary_reward=float(np.mean(auxiliary)) if auxiliary else 0.)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/scissors_palm_aux_ablation_20260926'))
    p.add_argument('--partial',action='store_true')
    a=p.parse_args(argv)
    manifest=json.loads((a.root/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((a.root/'status.json').read_text(encoding='utf-8'))['jobs']
    if not a.partial and any(value!='completed' for value in status.values()):
        p.error('ablation is incomplete; use --partial only for a provisional report')
    rows=[]
    for variant in manifest['variants']:
        for weight in manifest['weights']:
            if status[f'{variant}:{weight:g}']!='completed':
                continue
            folder=a.root/f'scissors_lite_{variant}'/f'contact_{str(weight).replace(".","p")}'
            rows.append(report_row(folder,variant,weight))
    result=dict(protocol=manifest['protocol'],complete=len(rows)==len(status),
                note='One paired seed per variant is a pilot, not evidence of a general learning benefit.',
                rows=rows)
    (a.root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    header='| Task | Contact weight | Ever 5-step success | Final-step success | Palm grasp steps | Min target success | Mean angle error [rad] |'
    lines=['# Palm-sphere auxiliary-contact reward ablation','',
        'Same seed and settings within each variant; only contact auxiliary weight differs.',
        'A single seed cannot establish a general learning benefit.','',header,
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| Scissors {r["variant"]} | {r["contact_auxiliary_weight"]:g} | '
            f'{100*r["success_rate"]:.1f}% | {100*r["final_success_rate"]:.1f}% | '
            f'{100*r["grasp_maintenance_rate"]:.1f}% | '
            f'{100*r["min_target_success_rate"]:.1f}% | {r["mean_abs_error"]:.3f} |')
    (a.root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
