"""Validate every predetermined split before running simulations; export exact Fit X."""
import argparse
import json
from pathlib import Path
import numpy as np
from improved_tds.experiments.evaluate_strict_synergy_control import balanced_angle_success_indices, _reference_deltas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    with np.load(a.database,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    meta=json.loads(str(data.pop('metadata_json')))
    if meta.get('success_only') and not data['success'].all():
        raise ValueError('Success-only database contains failed posture rows')
    if not meta['strict_success_window'] or meta['representative']!='last' or meta['pose_count']!=32:
        raise ValueError('incompatible database protocol')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(database=str(a.database),raw_rows=len(data['q']),successful_rows=int(data['success'].sum()),splits={})
    failures=[]
    for seed in range(20260911,20260916):
        poses=np.random.default_rng(seed).permutation(32)
        fit,cal,test=map(np.sort,(poses[:16],poses[16:24],poses[24:]))
        try:
            ids=balanced_angle_success_indices(data,fit,100,seed,meta['target_range'],10)
            _,_,reference_poses=_reference_deltas(data,cal)
            np.savez_compressed(a.output_dir/f'fit_seed{seed}.npz',
                **{k:v[ids] for k,v in data.items()},source_rows=ids,
                metadata_json=np.asarray(json.dumps(dict(source=str(a.database),fit_poses=fit.tolist(),
                    calibration_poses=cal.tolist(),test_poses=test.tolist(),seed=seed,rows=100))))
            report['splits'][str(seed)]=dict(complete=True,rows=100,bin_counts=[10]*10,reference_poses=reference_poses)
        except ValueError as exc:
            failures.append(seed);report['splits'][str(seed)]=dict(complete=False,error=str(exc))
    (a.output_dir/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if failures:raise ValueError(f'Incomplete Fit splits: {failures}; see audit.json. No quota relaxation.')
    print('All five splits have 100 unique, success-only, angle-balanced Fit rows.',flush=True)


if __name__=='__main__':main()
