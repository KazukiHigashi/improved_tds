# Posture-generation task v1

Implemented 2026-09-17. This is a new task/observation version, not a continuation
of the September 15 campaign. Existing results and policies are retained.

## Purpose and shared criteria

Obtain physically supported, low-error Scissors postures spanning 0--1 rad.
All five variants use the same criteria and reward coefficients.
`PostureCriteria` is the single source for Scissors training and balanced collection:

| Criterion | Default |
|---|---:|
| Angle error | <= 0.03 rad |
| Continuous supported, low-speed hold | 0.4 s (10 control steps) |
| Absolute tool angular velocity | <= 0.10 rad/s |
| Hand bodies contacting handle geometry | >= 2 |
| Root displacement from authored reset | <= 0.10 m |
| Root orientation change from authored reset | <= 0.8 rad |
| Downward root displacement causing drop | > 0.08 m |
| Missing-contact grace | 0.12 s |

The motion/contact thresholds are provisional engineering criteria, NOT empirically
validated universal grasp thresholds. Contact support means distinct hand bodies,
not necessarily distinct fingers or force closure. Handle geometry is identified
by scissors_partN with N >= 2; the blade geometries are excluded. Audit this mapping
when adding a new model. Grace delays termination but never credits unsupported
steps as success. The root criterion permits articulation and measures drift from
each reset, rather than distance to a fixed scene marker.

Common reward weights remain broad/fine tracking 1+1, progress 1, grasp 1,
interaction .25, success 5, failure 25, control delta .1 and magnitude .002.
Success bonus is recomputed from the relabelled goal, current physical support and
speed, so HER does not reuse stale goal-dependent success flags. The physical
contact/velocity constraints are also required throughout the stable window.

Half of sampled training goals are from shuffled full-range bins (10 bins);
half come from the current easy-range curriculum. The curriculum reaches the
inner sampler through an explicit environment method. Each validation and the
final summary record per-worker *actual* sampled-goal counts and effective bounds.
The full-range half is balanced; the overall early curriculum is deliberately not
uniform. Explicit evaluation targets bypass this sampling.

## Fixes and reporting

Push/Spray embedded observation goal slot is zeroed, retaining observation shape;
the goal is supplied only in desired_goal. Old models still load, but feeding them
the new observation is not a valid unchanged-baseline comparison. Use the original
code/environment version to reproduce old metrics. Retrain for v1 comparisons.

best_policy and best_success_policy maximize aggregate success (error breaks ties).
best_pose_coverage_policy prioritizes the worst target-bin success, then coverage,
aggregate success and error. These are separate artifacts. Checkpoints and the
95%-twice-at-100k early-stop rule remain intact. Early stop is not DB acceptance.

Validation records target/pose trials, final success, normalized MAE, supported
angle extrema, grasp maintenance, longest success streak, reward terms and input
saturation. Normalized 1%, 3%, 5%, 7.5% tolerance curves use the hold criterion;
Scissors also records 0.01-rad precision success. The supported extrema are not
proof that all intermediate angles are stable or controllable.

## Pool qualification and bounded data collection

`audit_posture_pools` applies the same physical criteria to a static-pose hold.
The existing drive-qualified pool survived this 0.4 s test in 32/32, 32/32,
13/32, 6/32 and 3/32 cases respectively. See
`analysis/posture_design_v1/pool_audit.json`. Drive qualification alone is therefore
insufficient. Static qualification still does not guarantee target reachability.

`generate_posture_research_pools.ps1` generates a larger drive-qualified candidate
pool and selects 32 surviving poses per variant. It stops if the quota is unmet;
no variant-specific threshold relaxation or overwriting of old results occurs.
Candidate generation is bounded and may need a better authored grasp if it fails.

`collect_balanced_postures` first audits every initial pose at 11 goals including
both endpoints using the selected policy. This measures policy-conditioned reach,
not a mechanical upper bound. It then samples deficient bins until each of ten
bins has ten distinct supported postures or 6400 episodes are consumed. One actual
state at the end of the first successful window is saved per independent episode;
joint configurations are never averaged into an unverified synthetic posture.
Both requested and achieved angles must belong to the accepted bin. Postures
within 0.01 rad joint-space RMS of an accepted posture are rejected as duplicates.
All tasks have the same 100-row quota. Failures, duplicates and deficits remain in
the audit JSON. An unmet quota yields a `.partial.npz` and exit status 2, not a
full database. The bins adjacent to 0 and 1 must be populated; endpoint performance
is additionally reported by the audit grid, not inferred from bin membership.

Use disjoint initial-pose groups for fit/calibration/test in downstream experiments.
A 100-row DB is not automatically a 100-row fit split. Increase `--per-bin` when a
larger dataset is needed before splitting; the strict evaluator retains its
minimum-fit-size guard and recognizes the new task version. Never relax this guard
or merge held-out samples merely to achieve a requested fit count.

## Running

From the project root, with the existing FDMS virtual environment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_posture_research_pipeline.ps1 -RunName posture_research_v1
```

This generates/qualifies pools, validates the 32-pose quota, trains with 4 jobs x
4 environments, and collects equal-size datasets from coverage policies. Use a
fresh name; new task results are not mixed with the earlier campaign. Run it in
an independent hidden process for long jobs. This implementation turn runs only
tests/audits and short smoke learning, not the million-step campaign.

Threshold validation remains an experimental question. Inspect the shared pool
audit, target/pose audit and tolerance curves before interpreting higher success
rates as better physical manipulation. Compare fixes before introducing extra
optimizer/reward changes; the present patch does not claim measured improvement.
