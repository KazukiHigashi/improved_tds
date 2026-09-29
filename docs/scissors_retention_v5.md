# Scissors retention v5 — 2026-09-21

## Reassessment and scope

v4 Best-success Validation (352 trials per task): physical failures were 138,
197, 327 for Scissors 3,4,5; timeouts were 136,117,8. Contact failure dominated
Scissors 5. Two-second retention remained difficult even after angle acquisition.
v4 also ended on success while delivering positive near-goal reward to timeouts,
creating a structural incentive risk (not proof of deliberate reward exploitation).

Run ONLY Scissors 3--5, fresh weights, max 3M transitions per task, three learners
with four environments each. Preserve old runs and the same copied v3 pose pools.
All rules and reward coefficients are shared across variants. No online RL or
extra control degrees of freedom are being added to the TDS controller.

## Changes

1. Success no longer ends the episode. Once a two-second fixed-command hold is
   certified, keep that command to the common 400-step horizon or physical failure.
   Supported holding continues to receive reward; there is no artificial simulator
   success state. Record both attained two-second success and final success, plus
   physical failure after success. The original two-second attainment remains the
   primary Validation/Early Stop criterion; final survival is a separate diagnostic.
2. Only the initial, not-yet-acquired grasp has a 0.4 s contact acquisition window.
   After first supported grasp the original 0.12 s contact-loss grace applies.
   Drop, displacement and rotation failures are never suppressed. This is a task
   initialization change and NOT an apples-to-apples success-rate comparison to
   v4. A paired final evaluation also uses the original 0.12 s acquisition window.
3. Add a bounded partial-support reward (0.5 times support fraction) and bounded
   root-motion penalty (up to -0.25). Preserve original shared tracking/progress,
   angle tolerance 0.03 rad, instantaneous speed threshold 0.1 rad/s and two support
   bodies. Do not relax speed criteria without classifying vibration separately.
4. Start a fixed-command probe only after 0.4 s of success AND root speed <=0.05 m/s,
   angular speed <=0.5 rad/s, applied-command RMS change <=0.02. These are offline
   posture-generation readiness checks, not extra TDS runtime requirements.
5. Abort an unsuccessful pre-certification probe immediately; allow RL to correct
   again after a ten-step cooldown. Final certification still needs an uninterrupted
   two-second fixed-command window. After certification, remain open-loop even if
   success is subsequently lost, so long-horizon passive stability is measured.
6. A worker advances .4/.8/1.2/2-second training probes after >=70% success in its
   last20 current-stage attempts. Rules are identical, though progression can differ
   with competence. Each episode has 20% probability of the final two-second probe
   even early on. Validation/collection always uses two seconds, no curriculum.
7. MuJoCo bad-state warnings, clock rewinds, nonfinite state/reward/observations
   terminate with an explicit numerical-failure flag. A cached finite observation
   is only a terminal sentinel; physics is NOT restored. The collector drops the
   whole invalid episode; the replay buffer drops the corresponding vector row
   (safe sibling transitions too) rather than inventing replacement samples.
   Numerical failures count against success but have undefined angle error; MAE
   reports its valid observation count. This prevents fake post-reset states from
   being treated as valid trajectories or learning targets.

## Preserved training controls

SAC, no HER (goal-dependent phase history makes naive relabelling invalid), learning
rate 5e-5, entropy 0.02, two gradient updates. Full angle-range sampling is retained.
Every100k: checkpoint and 32 poses x11 targets Validation. Best success and best
coverage policies are stored separately. Two consecutive >=95% strict evaluations
trigger Early Stop. The 3M budget counts all four workers' transitions together.

## Checks and limitations

Unit tests cover identical criteria/observation shapes for all three variants,
success continuing without releasing the command, failed-probe recovery, readiness,
competence progression, initial-grace safety boundaries, clock-rewind detection,
invalid replay/trajectory exclusion and existing Validation behavior. Short learner
runs exercise all three tasks with four environments, optimization and model save.

This bundled correction is not an isolated ablation of each change. Retain v4 as a
baseline and compare matched acquisition-window evaluations before attributing gains
to learned stability. No guarantee of improved success is asserted before results.

Entry: `scripts/run_scissors_retention_v5.py`. Final matched evaluation is written
to each task's `acquisition_reference_evaluation.json`. During training, per-phase
failure counts, maximum fixed hold, numerical failures and post-success failures
are recorded in `validation_history.json`.
