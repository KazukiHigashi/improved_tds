"""Evaluate per-task PLS on the guarded few-shot run's identical test resets.

This is an oracle reference: its fit uses the held-out tool's 200 successful
postures. The fitted model never enters few-shot selection or calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pose_conditioning import CATEGORIES, fit_operation, read_category
from few_shot_axis_adaptation import (PROBE_FRACTIONS, evaluate_command,
                                      fit_probe_response, inverse_score, probe_scores)
from simulate_initial_pose_conditioning import create_env, fit_angle_inverse, fit_local
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


def evaluate_tool(category, variant, bundle, source, root):
    name=f"{category}_lite_{variant}"
    protocol=json.loads((root/f"{name}.json").read_text(encoding="utf-8"))
    if protocol["task"]!=name or len(protocol["physical"])!=15:
        raise ValueError(f"incomplete guarded trial set: {name}")
    blocks=read_category(bundle,category)
    held=blocks[variant-1]
    known=[b for i,b in enumerate(blocks) if i!=variant-1]
    local_ref,local_axis,local_inverse,local_bounds=fit_local(held)
    common_ref,common_axis=fit_operation(known)
    common_inverse,common_bounds=fit_angle_inverse(known,common_ref,common_axis)
    env,pool,key,lo,hi=create_env(name,bundle,source,58200+variant)
    low,high,_=_control_bounds(env)
    records=[]
    try:
        if not np.allclose([lo,hi],protocol["target_range"]):
            raise ValueError(f"target range changed: {name}")
        # Replay one common-axis trial as a deterministic protocol audit.
        first=protocol["physical"][0]
        fraction=float(first["target_fraction"])
        raw=(fraction-common_inverse[0])/common_inverse[1]
        score=float(np.clip(raw,*common_bounds))
        check,_=evaluate_command(env,int(first["pose"]),key,float(first["target_angle"]),
            72000+int(first["pose"]),category,common_ref,common_axis,score,low,high)
        if not np.isclose(check["achieved"],first["baseline_achieved"],
                          rtol=0.,atol=1e-3*(hi-lo)):
            raise ValueError(f"replayed common-axis baseline differs: {name}: "
                             f"{check['achieved']} versus {first['baseline_achieved']}")
        calibration=[]
        cal_pose=int(protocol["calibration_pose"])
        for fraction,score in zip(PROBE_FRACTIONS,probe_scores(local_inverse,local_bounds)):
            outcome,joint_clipped=evaluate_command(env,cal_pose,key,lo+.5*(hi-lo),
                72000+cal_pose,category,local_ref,local_axis,float(score),low,high)
            calibration.append(dict(score=float(score),nominal_fraction=float(fraction),
                achieved_angle=float(outcome["achieved"]),joint_clipped=joint_clipped,
                admitted=bool(outcome["held"] and not outcome["failed"]
                    and not outcome["failure_numerical"] and not outcome["terminated"]
                    and not outcome["truncated"] and np.isfinite(outcome["achieved"]))))
        response=fit_probe_response(calibration,lo,hi)
        for row in protocol["physical"]:
            pose=int(row["pose"])
            target=float(row["target_angle"])
            fraction=float(row["target_fraction"])
            raw=(fraction-local_inverse[0])/local_inverse[1]
            score=float(np.clip(raw,*local_bounds))
            outcome,clipped=evaluate_command(env,pose,key,target,72000+pose,category,
                                               local_ref,local_axis,score,low,high)
            achieved=float(outcome["achieved"])
            if response is None:
                calibrated_outcome,calibrated_clipped=outcome,clipped
                calibrated_inverse_clipped=bool(not np.isclose(raw,score))
            else:
                calibrated_score,calibrated_inverse_clipped=inverse_score(target,response)
                calibrated_outcome,calibrated_clipped=evaluate_command(env,pose,key,target,
                    72000+pose,category,local_ref,local_axis,calibrated_score,low,high)
            calibrated_achieved=float(calibrated_outcome["achieved"])
            records.append(dict(pose=pose,target_fraction=fraction,target_angle=target,
                achieved=achieved,mae_fraction=abs(achieved-target)/(hi-lo),
                supported=bool(outcome["valid"]),
                final_supported=bool(outcome["held"] and not outcome["failed"]),
                joint_clipped=clipped,inverse_clipped=bool(not np.isclose(raw,score)),
                calibrated_achieved=calibrated_achieved,
                calibrated_mae_fraction=abs(calibrated_achieved-target)/(hi-lo),
                calibrated_supported=bool(calibrated_outcome["valid"]),
                calibrated_joint_clipped=calibrated_clipped,
                calibrated_inverse_clipped=calibrated_inverse_clipped))
    finally:
        env.close()
    result=dict(protocol="guarded_split_task_specific_pls_oracle_v1",task=name,
        reference_type="heldout_tool_success_database_oracle_not_fewshot",
        source_trial_file=str((root/f"{name}.json").resolve()),
        calibration_pose=protocol["calibration_pose"],test_poses=protocol["test_poses"],
        target_range=[lo,hi],calibration_probes=calibration,
        calibration_response=response,
        calibrated_fallback_to_db=response is None,rows=records)
    (root/f"{name}_local_pls.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                                encoding="utf-8")
    print(name,"task PLS MAE",round(float(np.mean([r["mae_fraction"] for r in records])),3),
          "task PLS + physical MAE",round(float(np.mean([
              r["calibrated_mae_fraction"] for r in records])),3),
          "supported",sum(r["supported"] for r in records),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,default=Path("analysis/paper15_paper_bundle_20260928"))
    p.add_argument("--source",type=Path,default=Path("runs/paper15_unified_3m_slew_20260927"))
    p.add_argument("--root",type=Path,default=Path("analysis/few_shot_guarded_20261008"))
    p.add_argument("--categories",nargs="+",choices=CATEGORIES,default=CATEGORIES)
    p.add_argument("--variant",type=int,choices=range(1,6))
    args=p.parse_args()
    _register_tool_manip_gym_environments()
    for category in args.categories:
        for variant in ([args.variant] if args.variant else range(1,6)):
            evaluate_tool(category,variant,args.bundle,args.source,args.root)


if __name__=="__main__":
    main()
