# PCA/PLS two-DoF tool-control experiment

## Axis definitions

Both conditions command an affine two-dimensional posture plane:

```text
q_command = q_reference + grasp_axis * sigma + operation_axis * rho
```

- PCA: operation axis = PC1; candidate grasp axis = PC2.
- PLS: operation axis = PLS1 fitted against achieved tool DoF; candidate
  grasp axis = PC1 of posture variation after orthogonally removing PLS1.

The operation axis is oriented toward increasing tool DoF. The candidate grasp
axis is oriented toward increasing the recorded grasp metric when that metric
varies. Orthogonality is enforced numerically. A residual-variance axis is not
assumed to be a real grasp synergy: its grasp correlation, online grasp sweep,
grasp improvement, and tool-state interference are all reported.

Push and Spray first calibrate and hold `sigma`, then identify and command
`rho`. Scissors follows the same code path for a controlled comparison, even
though its handles already constrain the tool.

The fixed affine reference interpolates between measured joint posture and the
policy's position-servo command posture. This preload is calibrated, not a
third online control variable. It is necessary because contact loads hold the
measured joints short of the command, while fully saturated policy commands can
leave no kinematic headroom for either synergy.

## Equal posture databases

Every task must have exactly 32 authored poses and uses the same 15 normalized
target fractions. One row is saved for every pose-target attempt, including a
failure, so every `.npz` contains exactly 480 rows. Failed cells are never
duplicated to make the successful analysis set look balanced. Axis fitting
takes at most one successful row per pose-target cell and reports the resulting
coverage.

The database stores both measured `q` and `command_q`. Measured posture is used
to fit PCA/PLS; command posture is used to reproduce a grasp with the position
servo.

## Fixed protocol

- 16 authored poses: axis fitting
- 8 authored poses: calibration/hyperparameter development
- 8 authored poses: held-out control evaluation
- 5 reference-preload candidates
- 9 grasp-coordinate candidates
- 13 operation-coordinate identification points
- 9 fixed Tool DoF targets in the trained 10--50% range
- nominal object placement for the primary comparison
- identical reset seeds for paired PCA and PLS trials

Targets outside a method's identified reachable interval are explicitly marked
unreachable and cannot count as successes. Drops and grasp losses also remain
failures rather than disappearing from the error table.

## Commands

Run all 15 tasks, with up to four task processes in parallel:

```powershell
.\scripts\run_two_dof_experiment.ps1
```

The script verifies all databases have 32 poses and 480 rows before starting
control evaluation. It writes per-task reports plus `summary.json` under
`analysis/two_dof_pca_pls_15tasks`.

The reduced implementation pilot is under `analysis/two_dof_pilot`. It is a
pipeline and feasibility check, not a paper result: it uses only three target
fractions and one held-out pose.
