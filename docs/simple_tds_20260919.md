# Simple feed-forward TDS protocol (2026-09-19)

## Research claim and limits

The operational controller uses a fixed preparation followed by one scalar tool
score. Push/Spray also have a scalar preparation/grasp direction: q = clip(q0 +
g*b + rho*a*w). Preparation ramps g from 0 to 1, then holds g=1 while rho is
swept. Scissors expose one tool scalar after a fixed preparation, not a claim
that the entire acquisition trajectory lies on one principal component.

There is NO online RL, inverse kinematics, state restoration, per-Test reference
selection, reset retry, target-specific grasp, or additional feedback controller.
The existing authored initial-pose pools remain an experimental constraint;
this is not an arbitrary-object-placement grasp acquisition experiment.

## Offline identification

- Preserve previous 16 Fit / 8 Calibration / 8 Test pose splits and five seeds.
- Use the SAME successful Fit100 and frozen PCA/PLS operation axes as the previous
  experiment. Do not relabel these poses as two-second-stable samples.
- Candidate fixed grasp offsets are zero and actual successful applied-command
  offsets from Calibration poses nearest target fraction 0.3. No synthetic
  coordinate-wise median. One candidate and one duration are chosen for the whole
  split, shared by PCA and PLS, not looked up by Test initial pose or target.
- Common preparation durations are 0.04, 0.2, 0.4 seconds. Rank by complete
  one-second held window after preparation, then longer duration, then smaller
  offset. All categories/variants use identical selection rules.
- Independent Calibration points span physical rho [-2,2], 17 points. Reset only
  to the original authored initial state; execute fixed preparation each time.
  This avoids early failure at an extreme censoring every subsequent point.
- Existing held-cohort/direction/range checks and isotonic inverse are retained.
  They decide only whether an inverse target controller is available, NOT whether
  a diagnostic Test sweep is run. No threshold is lowered to manufacture coverage.

## Evaluation

- Every Test pose receives an independent diagnostic sweep even if Calibration
  rejects the inverse map. Report actual response, valid fraction and missing data.
- Additionally run a continuous forward/reverse sweep with NO reset between
  points. Use the selected physical interval when available; otherwise the full
  diagnostic interval. This latter trajectory may terminate at its first extreme;
  its failure must not suppress the independent diagnostic sweep.
- Each operation point ramps for 0.4 seconds and holds the fixed command for two
  seconds. All holding frames must be supported and failure-free; ramp contact
  fraction is recorded separately. The unchanged environment contact grace and
  physical termination checks remain active throughout.
- A point is valid only if its preparation also completed a supported one-second
  window. No repeated attempts to rescue a failed Test preparation.
- Inverse target evaluation uses nine full-range goals; out-of-map targets clamp
  to endpoints through interpolation. Missing outcomes have undefined error, not
  zero. Report error observation counts and held counts alongside MAE.
- Overall procedure Coverage includes failed preparations as zero; this is not a
  statement of mechanical impossibility or the absence of successful RL poses.
- Coverage is clipped held min/max span, not guaranteed continuous reachability.
- HTML includes Fit100 projections, Calibration/Test diagnostic curves, continuous
  forward/reverse curves, side-by-side PCA/PLS, and images every 20 simulation steps
  for a predetermined Test pose and the first split. All frames are paginated.

## Retention audit and retraining decision

Prior Scissors success under RL is 0.4 s (10 steps at dt=0.04). The reused Fit
database therefore does not prove passive two-second stability. For the first
split only, audit the frozen policy on eight Calibration poses and fractions
0.1, 0.5, 0.9 without retries. Allow up to 200 RL steps to reach stable success;
then freeze the actually applied command for two seconds. Record precision-success
retention and grasp retention separately. A matched-reset continued-RL control
distinguishes general long-horizon instability from dependence on online feedback.
RL is an offline diagnostic only; the matched control is not a retry for success.

If RL reaches success but a frozen command loses grasp/precision, review the
posture-generation task and increase its hold duration consistently for all
Scissors variants, with a fixed-command retention test for DB admission. Merely
extending closed-loop RL success does not establish FF stability. If frozen
retention is good but fixed preparation is poor, prioritize acquisition trajectory
or initial-pose-pool design rather than training longer. Retraining is not started
automatically from an unverified assumption.

## Execution

`scripts/run_simple_tds.py` freezes source code, uses four single-environment
workers, preserves previous artifacts, records job failures, and regenerates HTML
after jobs complete. No interactive polling is required. Source-policy audit is
only in seed 20260911; no training process runs in this campaign.
