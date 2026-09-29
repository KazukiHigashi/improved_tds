# Adaptive synergy FF experiment (2026-09-13)

The original clipped-path results are preserved. The new campaign writes to
`analysis/adaptive_synergy_v1`. It reuses the 75 strict v3 reports and their
success-only, equal-count (100) Fit samples; no RL retraining or Test fitting.

Four arms: legacy fixed amplitude/cumulative-max inverse map; direction-only
selection/isotonic inverse map; fixed-grasp interval selection; joint grasp and
interval selection. Direction-only also changes the inverse estimator; its
difference from legacy must not be attributed exclusively to sign orientation.

All new calibration and test trials use the same reset seed for a given pose,
independent of score, arm or target. Each is an independent reset, 100-step
grasp ramp, 50-step operation ramp, 50-step hold. No online feedback. Target
error uses actual executions at nine evenly spaced full-range targets, not
interpolation of Test sweep results. Use the new legacy arm for comparisons
because earlier reports used score-dependent reset seeds and interpolated MAE.

The operation direction is unchanged except for selected scalar orientation.
Nominal amplitude equals the original actuator-half-range scale. The search
grid [-2,2] at .25 spacing includes amplitudes .5,1,1.5,2, asymmetric branches,
and branches that exclude the grasp reference. Identical clipped command
postures at all calibration poses are deduplicated, stopping saturation-only
expansion. The original [-1,1] arm uses 13 points. Test sweeps use 13 points.

Joint candidates include the original (preload, grasp); grasp scores
[-1,-.5,0,.5,1] at the original preload; and preload [0,.5,1] with zero grasp
score. Scissors uses only reference preload selection and one operation DoF.
Candidates are evaluated on the eight Calibration poses, never Test poses.

An interval must contain at least three grid points. At least 75% of calibration
poses must remain held and not failed at every point. The same fixed cohort is
used for all means in the selected interval. At least 75% of that cohort must
move in the chosen direction by more than 1% of target width; average endpoint
gain must be >=5%; target-overlapping mean range >=5%. Range/joint arms also
require each reverse increment <=2% and total reverse travel <=5% of target
width. Maximise calibration coverage, then held fraction, direction agreement,
then minimise calibration interpolated target MAE and reverse travel. Calibration
MAE is an in-sample selection diagnostic; Test MAE uses actual independent
executions. These are declared engineering thresholds, not
statistical significance criteria. A rejected interval is a calibration failure.

The selected interval maps to u in [-1,1], increasing with tool response. The
grasp reference can lie outside this interval and is separately marked. An
isotonic fit on the accepted raw response supplies the inverse map; plateaus
use mean score. No cumulative maxima except in the explicitly labelled legacy
arm. Target requests outside the map saturate at its endpoints; their errors
are still counted. A singleton legacy map is reported as inverse-map failure.

Coverage uses the held observed min/max interval clipped to the target range;
it is not evidence every intermediate angle is reachable. Aggregate coverage
and held rate include calibration failures as zero; MAE is conditional on
executed controllers and its sample count is displayed. Raw per-pose responses,
calibration choices, all target trial outcomes, and fit sample off-axis distances
remain in JSON. No significance claim is made from overlapping split samples.

Fit scatter uses the unclipped linear projection relative to each candidate
grasp, with the same selected interval mapping. Outside-domain samples are not
clamped. This projection differs from the old nearest clipped-path projection
and is not proof that executing a score reproduces that RL sample.

Run `scripts/run_adaptive_synergy_campaign.py` with the FDMS venv Python and
`PYTHONPATH=src`. Four workers run 75 task/split jobs; each worker uses one
environment and one BLAS thread. This is evaluation, not four-environment RL.
On completion the runner validates all outputs and generates HTML automatically.
`status.json` records completion or failure. Existing complete JSON outputs are
resumed; use a new output directory after changing the protocol.
