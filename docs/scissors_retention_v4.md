# Scissors retention v4 — 2026-09-20

Scope: Scissors 3, 4, 5 only. New training from scratch, 3,000,000 total transitions
per task, three simultaneous SAC learners with four environments each. Existing
runs, pose pools and models are preserved. The copied v3 pose pools are identical
to their previous training inputs; pool hashes are recorded. No other tasks run.

## Mitigating sparse success from longer retention

Keep v3's dense tracking, progress, supported settling and safety terms, action
rate limiter, normalized goal features, learning rate 5e-5, entropy 0.02 and two
gradient updates. Keep full-range targets throughout (including stratified full
range sampling); do not sacrifice high-angle coverage to improve averages.

On 0.4 seconds of consecutive instantaneous success, freeze the actually applied
action. The learned requested action is ignored during this training phase. A
phase observation exposes remaining hold time, consecutive successful hold,
acquisition progress and this episode's required duration. Do not restore any
physics state or use per-target controllers. Release after the fixed-duration
probe; failures cannot count as retained success. Short probes can be attempted
again within the episode while the unchanged physical failure rules remain active.

The hold schedule uses the worker's transitions divided by its allocated budget
(global budget / n_envs), latched at reset:

| Global training steps | Required fixed-command probe |
|---|---|
| 0–600k | 0.4 s |
| 600k–1.2M | 0.8 s |
| 1.2M–1.8M | 1.2 s |
| 1.8M–3M | 2.0 s |

Shaping: +2 per successful fixed-command frame; +5 on completing the current
probe. No added reward on physical failure. The same formula, schedule and physical
criteria apply to every variant. Original angular tolerance 0.03 rad, maximum
tool speed 0.1 rad/s, minimum two support bodies, contact-loss grace and root-motion
limits are NOT relaxed. Episode horizon is 400 steps to accommodate acquisition
and retention. Only a completed two-second fixed-command probe receives the final
`is_stable_success` label and a successful terminal state. Curriculum successes
are logged separately and do not qualify for the final successful-pose DB.

The old 0.4-second RL success rate and new two-second frozen-command success rate
are different metrics: a lower number does not itself demonstrate worse learning.

## Validation and data validity

Validation disables the curriculum and ALWAYS requires two seconds of successful
fixed command, from the first 100k checkpoint. The unchanged fixed pose/full-range
goal grid is used every 100k; two successive >=95% evaluations trigger early stop.
Save 100k checkpoints, best_policy, best_success_policy and best_pose_coverage_policy
using the existing validation callback. Maximum steps are 3M, not additional steps
on a previous model. Policy observation changes require fresh weights.

HER is prohibited because relabelling a target would change the phase trigger,
ignored-action transitions and history-dependent success reward. Standard SAC
replay is used. `collect_posture_database --research-task
--scissors-control-version v4` builds the matching final two-second environment;
future collection must use this version to match the policy observation space.

The fixed-command phase is a posture-generation training constraint, not extra
online degrees of freedom for TDS operation. TDS remains a separate 1/2-DoF FF test.

Entry point: `scripts/run_scissors_retention_v4.py`. It freezes source and pools,
records commands and PIDs, and writes completion/failure status without interactive
agent monitoring. Training and validation logs remain in the run directory.
