# Improved TDS: 15-task paper experiment snapshot (2026-09-29)

This document is the hand-off boundary before adding tendon-feedback grasp acquisition. It describes the implementation and **selected** results, not every historical trial. The local paper artifact is `analysis/paper15_paper_bundle_20260928/index.html`; its `manifest.json` records selected-source paths and SHA-256 hashes. `analysis/` and `runs/` are preserved locally but not committed to Git because the analysis tree alone is about 3.7 GB. A fresh clone therefore needs those artifacts restored separately before regenerating the report or loading policies.

## Current implementation

- Fifteen tasks: `scissors_lite_1..5`, `push_lite_1..5`, `spray_lite_1..5`. Training uses the unified 3-million-step campaign with validation and best-success-policy selection (`scripts/run_paper15_unified.py`, `src/improved_tds/experiments/train_sb3.py`). The selected campaign is `runs/paper15_unified_3m_slew_20260927`.
- Each task has a success-only database of 200 measured final hand postures. The collection and saved provenance are implemented in `src/improved_tds/experiments/collect_posture_database.py` and included in the paper bundle.
- PCA uses PC1 for tool operation and PC2 as the orthogonal grasp-axis candidate. PLS uses one component fitted to **measured** tool response for operation, then residual PCA for the orthogonal candidate. Both use global mean-centering of the 200 successful postures (`src/improved_tds/synergy/two_axis.py`). The second axis is a candidate, not established as a grasp synergy.
- Direct TDS reconstructs an absolute target posture from the successful-posture mean and operation score, plus a fixed second-axis score in the 2-DoF configuration. Each trial starts from a saved authored initial pose, takes 5 s for linear joint-command interpolation, then holds for 2 s. There is no online RL, preparation offset, or tool-angle feedback in this evaluation (`src/improved_tds/experiments/reach_tds_direct.py`).
- Calibration tests 17 operation scores from 1.5 times the observed minimum to 1.5 times the observed maximum, using 8 initial poses. A qualified approximately monotone interval is fitted with isotonic regression and a clipped linear ramp. Inverse control is tested at 9 unseen targets with 8 separate initial poses (`src/improved_tds/synergy/adaptive_range.py`).
- The selected Scissors TDS Calibration/Test uses a palm-attached acceptance sphere of radius 0.075 m; its RL and database success criterion keeps 0.05 m. Push/Spray TDS do not require contact-count gating, but exclude drop/tip; Spray floor contact remains enabled. The integrated report documents these differing acceptance conditions.
- The current 2-DoF candidate search tests `g ∈ {-1,-0.5,0,0.5,1}` on the 8 Calibration initial poses, selecting the largest count of valid 2-s holds and breaking ties by smallest `|g|`. In all 15 selected results, every candidate had 8/8 valid pretrials and `g=0` won the tie. Thus the selected 2-DoF **executed trajectory** equals its 1-DoF counterpart; independent grasp acquisition has not been demonstrated.
- The real-robot prototype (`scripts/real_experiment/operate_tds_jointtrajectory.py`) is a dry-run-first ROS1 `FollowJointTrajectory` one-DoF command with a reviewed joint map and real calibration. It does not yet sense tendon tension or stop a grasp sweep on force feedback.

## Selected results and interpretation

PLS 1-DoF means over five variants per category, calculated from the 15 selected `report.json` files. Percentages use each task's RL target-range width; held MAE only covers valid held inverse trials **inside** the qualified inverse domain.

| Category | Inverse-ready target range | Independent sweep coverage | Normalized isotonic held MAE |
| --- | ---: | ---: | ---: |
| Scissors | 91.8% | 92.2% | 4.24% |
| Push | 99.2% | 98.9% | 6.60% |
| Spray | 67.3% | 64.9% | 8.31% |

These values do not establish stable real-robot grasping or full-range inverse accuracy. The report HTML retains per-task PCA/PLS figures, 1/2-DoF configurations, historical comparisons, and trajectory images. The current `grasp_metric` used to orient the second axis is contact/grasp-state information, **not tendon tension** (`collect_posture_database.py`). Positive reported correlations are partly a consequence of sign orientation and are not evidence that increasing `g` monotonically increases real tendon load.

## Git boundary and next branch

This snapshot commits source, scripts, tests and documentation. Generated analyses, policies, image galleries, checkpoints, temporary Pytest directories and virtual environments stay outside Git; no paper artifact is deleted by ignoring it. Keep `manifest.json` with the local bundle (or archive the whole bundle externally) because a source-only clone cannot reproduce the numerical figures without the trained policies and databases.

The next branch should add a **separate experimental** force-gated grasp-acquisition mode. Preserve the present 15-task report and its configuration as the baseline. Before real motion, verify the installed Shadow Hand Lite ROS1 tendon/effort topic, motor-to-joint mapping, sign, units, sample freshness, noise and safe force limits. A short-step `g ∈ [-0.5,0.5]` acquisition with stop-on-validated-load may then fix `g*` while the operation score changes; its Calibration and inverse mapping must be measured again at that `g*` rather than reusing the `g=0` fit. Scissors should enter this mode only if an independent grasp direction is empirically supported.
