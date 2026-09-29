# Full-range RL and synergy protocol (2026-09-14)

The earlier `unified_reward_4x4_v2`, strict v2/v3, and adaptive v1 artifacts
cover only part of the RL goal range in their RL training, validation, or
posture-fit stages. They remain preserved but must not be described as
full-range RL evidence.

The replacement protocol uses normalized target fractions 0.0 through 1.0.
Push and Spray sample the full range. Scissors expands its range in four
equal-duration curriculum stages while retaining earlier targets. Validation
always evaluates the fixed grid 0.0, 0.1, ..., 1.0 every
100,000 environment steps. Early stopping remains two consecutive evaluations
with aggregate success >=95%; the validation JSON stores aggregate and
per-target success/error, worst-bin success, and range coverage.

Scissors pose pools must be generated with drive validation under one common
criterion for variants 1--5. Grasp-only pools (`require_drive=false`) are
rejected by the training entry point.

After training, strict posture recollection uses 21 fixed targets
0.0, 0.05, ..., 1.0 and all 32 authored poses, producing 672 rows per task.
Only stable-success windows are eligible for PCA/PLS fitting. The common
100-row Fit set is selected in round-robin order across every target bin that
has a successful sample within the held-out Fit poses. Missing high-target
successes remain explicit in `database_success_by_target` and
`fit_success_by_target`; they are not silently imputed.

Fit, Calibration, and Test poses remain disjoint (16/8/8). The strict FF target
test uses 11 targets spanning the complete tool range. Adaptive Joint
calibration continues to choose its physical synergy interval using Calibration
poses only, and the selected controller is frozen for Test evaluation.

The new training run is `runs/unified_reward_fullrange_4x4_v1`. Downstream
outputs must use new analysis names. Comparisons with earlier results must note
that both RL policies and posture databases differ; this is a full protocol
replacement rather than an isolated synergy-range ablation.
