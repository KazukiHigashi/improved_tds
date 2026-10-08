"""New-pose validation of common-axis-first, budgeted few-shot adaptation.

The 75% measured-coverage guard was chosen after exploratory phase 1, so this
is a different-reset-pose validation, not an unseen-tool confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pose_conditioning import CATEGORIES, fit_operation, read_category
from few_shot_axis_adaptation import (PROBE_FRACTIONS, choose_candidate,
                                      evaluate_command, fit_probe_response,
                                      inverse_score, probe_scores)
from simulate_initial_pose_conditioning import (TARGET_FRACTIONS, create_env,
                                                fit_angle_inverse, fit_local)
from improved_tds.experiments.evaluate_two_dof_control import _control_bounds
from improved_tds.experiments.train_sb3 import _register_tool_manip_gym_environments


COMMON_COVERAGE_STOP = .75


def split_poses(pool_count, variant):
    old_test = set(np.random.default_rng(59320+variant).choice(
        pool_count, size=3, replace=False).tolist())
    old_cal = next(p for p in range(pool_count) if p not in old_test)
    remaining = [p for p in range(pool_count) if p not in old_test and p != old_cal]
    if len(remaining) < 4:
        raise ValueError("not enough unused reset poses for validation")
    chosen = np.random.default_rng(86233+variant).choice(remaining,size=4,replace=False)
    return int(chosen[0]), [int(p) for p in chosen[1:]]


def decide(models):
    common = models["common"]["response"]
    if common is not None and common["coverage_fraction"] >= COMMON_COVERAGE_STOP:
        return "common", True
    return choose_candidate(models, 0., 1.), False


def run_tool(category, variant, bundle, source, output):
    blocks=read_category(bundle,category)
    held=blocks[variant-1]
    known=[b for i,b in enumerate(blocks) if i!=variant-1]
    mean, axis=fit_operation(known)
    common_inverse,common_bounds=fit_angle_inverse(known,mean,axis)
    packages={"common":(mean,axis,common_inverse,common_bounds)}
    packages.update({b["name"]:fit_local(b) for b in known})
    env,pool,key,lo,hi=create_env(held["name"],bundle,source,58200+variant)
    low,high,_=_control_bounds(env)
    cal_pose,test_poses=split_poses(pool.pose_count,variant)
    models={}
    probes_attempted=0
    try:
        # The common axis is always the first three probes. The other four
        # known-tool axes are touched only when the common response is weak.
        for name,(reference,direction,inverse,bounds) in packages.items():
            if name!="common":
                common=models["common"]["response"]
                if common is not None and common["coverage_fraction"]>=COMMON_COVERAGE_STOP:
                    break
            probes=[]
            for fraction,score in zip(PROBE_FRACTIONS,probe_scores(inverse,bounds)):
                outcome,joint_clipped=evaluate_command(
                    env,cal_pose,key,lo+.5*(hi-lo),72000+cal_pose,category,
                    reference,direction,float(score),low,high)
                admitted=bool(outcome["held"] and not outcome["failed"]
                    and not outcome["failure_numerical"] and not outcome["terminated"]
                    and not outcome["truncated"] and np.isfinite(outcome["achieved"]))
                probes.append(dict(candidate=name,score=float(score),
                    nominal_fraction=float(fraction),achieved_angle=float(outcome["achieved"]),
                    admitted=admitted,strict_hold_valid=bool(outcome["valid"]),
                    joint_clipped=joint_clipped,numerical=bool(outcome["failure_numerical"])))
                probes_attempted+=1
            models[name]=dict(reference=reference.tolist(),axis=direction.tolist(),
                probes=probes,response=fit_probe_response(probes,lo,hi))
        selected,stopped_early=decide(models)
        records=[]
        for pose in test_poses:
            for fraction in TARGET_FRACTIONS:
                target=lo+fraction*(hi-lo)
                raw=(fraction-common_inverse[0])/common_inverse[1]
                baseline_score=float(np.clip(raw,*common_bounds))
                baseline,joint_clipped=evaluate_command(
                    env,pose,key,target,72000+pose,category,mean,axis,baseline_score,low,high)
                record=dict(pose=pose,target_fraction=float(fraction),target_angle=float(target),
                    baseline_achieved=float(baseline["achieved"]),
                    baseline_mae_fraction=abs(float(baseline["achieved"])-target)/(hi-lo),
                    baseline_supported=bool(baseline["valid"]),baseline_joint_clipped=joint_clipped,
                    selected_candidate=selected, fallback=selected is None)
                if selected is None:
                    record.update(adapted_achieved=record["baseline_achieved"],
                        adapted_mae_fraction=record["baseline_mae_fraction"],
                        adapted_supported=record["baseline_supported"],
                        adapted_joint_clipped=joint_clipped,inverse_clipped=bool(
                            not np.isclose(raw,baseline_score)))
                else:
                    model=models[selected]
                    score,inverse_clipped=inverse_score(target,model["response"])
                    outcome,clipped=evaluate_command(env,pose,key,target,72000+pose,category,
                        np.asarray(model["reference"]),np.asarray(model["axis"]),score,low,high)
                    record.update(adapted_achieved=float(outcome["achieved"]),
                        adapted_mae_fraction=abs(float(outcome["achieved"])-target)/(hi-lo),
                        adapted_supported=bool(outcome["valid"]),
                        adapted_joint_clipped=clipped,inverse_clipped=inverse_clipped)
                records.append(record)
        result=dict(protocol="few_shot_guarded_new_pose_v1",task=held["name"],
            training="none",common_coverage_stop=COMMON_COVERAGE_STOP,
            calibration_pose=cal_pose,test_poses=test_poses,
            previous_phase_poses_excluded=True,target_range=[lo,hi],
            selected_candidate=selected,stopped_after_common=stopped_early,
            probes_attempted=probes_attempted,models=models,physical=records)
        output.mkdir(parents=True,exist_ok=True)
        (output/f"{held['name']}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                                   encoding="utf-8")
        print(held["name"],"selected",selected,"probes",probes_attempted,
              "base",round(float(np.mean([r["baseline_mae_fraction"] for r in records])),3),
              "adapted",round(float(np.mean([r["adapted_mae_fraction"] for r in records])),3),
              flush=True)
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,default=Path("analysis/paper15_paper_bundle_20260928"))
    p.add_argument("--source",type=Path,default=Path("runs/paper15_unified_3m_slew_20260927"))
    p.add_argument("--output",type=Path,default=Path("analysis/few_shot_guarded_20261008"))
    p.add_argument("--categories",nargs="+",choices=CATEGORIES,default=CATEGORIES)
    p.add_argument("--variant",type=int,choices=range(1,6))
    args=p.parse_args()
    _register_tool_manip_gym_environments()
    for category in args.categories:
        for variant in ([args.variant] if args.variant else range(1,6)):
            run_tool(category,variant,args.bundle,args.source,args.output)


if __name__=="__main__":
    main()
