# Tool control v2 (2026-09-17)

User requested stopping v1 and restarting ALL fifteen tasks with a uniform
2,000,000-step ceiling, including early stop. Completed v1 artifacts and interrupted
checkpoints remain under their original run directory. Observation changes require
fresh policies; this is not a weight-only continuation of v1.

Shared campaign settings: 4 concurrent jobs x 4 environments; learning rate 1e-4;
gradient steps 1; HER ratio 0.25; progress weight 5; grasp weight 0.25. Other common
weights remain unchanged. Scissors retains its fixed entropy coefficient 0.05;
Push/Spray retain automatic entropy tuning. No variant-specific reward tuning.

All categories get previous action in observations and goal channels normalized
by physical tool span inside the feature extractor. HER and datasets retain physical
units. Embedded Push/Spray goals remain masked. A bounded speed cost is activated
only within the vicinity of the relabelled goal:

`r_braking = -0.5 exp(-0.5 (error/(0.1*span))^2) min((velocity/span)^2, 1)`.

It does not replace failure penalties or reward motionlessness far from the goal.
Scissors additionally observes signed hinge velocity, reset-relative root position
and rotation, root spatial velocity, support count, digit/palm support indicators,
and elapsed missing-contact time. Success criteria (0.03 rad, low speed, 0.4 s
supported hold) and the initial full-range curriculum mixture remain unchanged.

Dynamic pool audit probes previously static-qualified candidates with gentle
opening/return control, ranks survival/support/travel using the same rule for all
variants, and saves 32 poses per variant. A failed scripted probe is recorded, not
treated as proof that a learned policy cannot operate the pose. The selected pool
does not certify complete range coverage. See its `dynamic_audit.json`.
The chosen mild probe amplitude is 0.10 of the normalized action scale. Selected
survivors out of 32 are 32, 32, 8, 32, 7 for variants 1--5. The remaining entries
in variants 3/5 are statically qualified, dynamically ranked candidates, not
certified dynamic successes. Learned stabilization is still required. The stronger
0.40 probe audit is also preserved for comparison.

Validation/checkpoints every 100,000 steps. Early stop remains aggregate success
>=95% twice consecutively. Save best_success_policy and best_pose_coverage_policy
separately. After all training, bounded 10x10 posture collection follows automatically.

`scripts/run_control_v2.ps1` uses a fresh run directory, snapshots src, and makes
workers and the subsequent collector import that snapshot. Old models are not
compatible with v2 observation dimensions. Do not compare reloaded v1 models on
v2 inputs as if the policy/environment were unchanged.

This is a combined engineering revision, not an ablation isolating each change.
An increase in success is not assumed; target-specific coverage and stable posture
quotas remain the actual research outcomes to check.
