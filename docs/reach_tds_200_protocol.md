# Scissors reach-TDS: 200-posture, 5-second comparison

The 2026-09-25 protocol retains the former Scissors 3–5 smooth-control Best
Policies. It extends each completed 100-pose database to **200 successful
terminal poses** (20 in each of ten achieved-angle bins), preserving the old
database and recording the policy/pool hashes. Failed episodes are never added.

For PCA and PLS, final successful joint postures are centred around their
single global mean, not around authored initial-pose means. This makes PC1 the
leading variance direction of the complete success-posture set. The orientation
of the operation axis is still chosen to give positive database angle
correlation; orientation alone does not guarantee a positive physical response.

The physical operation-score diagnostic uses 17 points from
`1.5 * min(fit_rho)` to `1.5 * max(fit_rho)`. The protocol deliberately rejects
scores that do not straddle zero instead of silently applying an arbitrary
fixed interval. `fit_rho` is the projection of the change from the authored
initial posture and the selected fixed preparation onto the globally fitted
operation axis, divided by the nominal joint-range amplitude. Thus the axis
comes from global PCA, while the sweep coordinate remains the control-path
coordinate, not the raw PCA score `PC1 @ (q - global_mean)`.

Both the fixed preparation and each operation move are linear joint-command
ramps lasting five simulation seconds. The former is followed by a one-second
hold, and the latter by a two-second hold. No RL feedback is used during the
feed-forward evaluation. As before, a fixed preparation offset may involve
multiple joints, so this is not yet a strict reproduction of the original
single-PC trajectory from initial pose to reconstructed posture.

For every physically valid selected calibration interval, two inverse models
are fitted to the **same held-cohort mean responses**: (a) isotonic regression
and (b) a four-parameter clipped linear ramp, with the two plateau levels
estimated by least squares for each candidate breakpoint pair. The same test
poses, seeds and nine target angles are evaluated with both inverses. The
report shows calibration RMSE, observed target MAE and held-only target MAE
separately. If calibration yields no valid interval, neither inverse is claimed
to exist; the existing DB-only linear baseline remains explicitly labelled.

Run the collection continuation against a frozen Best Policy with
`python -m improved_tds.experiments.collect_reach_successes --extend-from
OLD_DB --output NEW_DB --per-bin 20` and the same model, pose pool, seed and
control version as the old DB. After all three 200-row databases are complete,
`scripts/run_reach_tds.py` defaults to the new root and writes a fresh HTML
report. The runner refuses an incomplete or 100-row input before creating its
output directory.
