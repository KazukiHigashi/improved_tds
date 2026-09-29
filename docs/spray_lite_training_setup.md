# Spray Lite SAC+HER training setup

## Environment IDs

- Training environments: `ShadowHandLiteSpray1-v0` through
  `ShadowHandLiteSpray5-v0`
- Authored-MJCF inspection environments: `ShadowHandLiteSpray1Pose-v0`
  through `ShadowHandLiteSpray5Pose-v0`

Training intentionally uses the canonical environments plus
`InitialPosePoolWrapper`.  The `*Pose-v0` IDs are for directly checking the
editor output; using those IDs for training would apply the authored hand-mount
offset twice when the pose pool is injected.

## Rebuild the initial-pose pools

From the workspace root in PowerShell:

```powershell
$python = ".\fdms_reconfiguration\.venv\Scripts\python.exe"
$env:PYTHONPATH = ".\tool-pose-pipeline;.\tool-manip-gym\gym-spray"
& $python -m toolpose.generate_spray_lite `
  --assets-dir ".\tool-manip-gym\gym-spray\gym_spray\assets\shadow_hand" `
  --out ".\tool-pose-pipeline\grasps" `
  --size 32
```

This creates:

```text
tool-pose-pipeline\grasps\spray_lite_spray1.npz
...
tool-pose-pipeline\grasps\spray_lite_spray5.npz
```

The generator starts from each `shadow_hand_spray*_lite_pose.xml`, searches
nearby hand targets and hand-mount transforms for a strict Stage-C-valid
center, then locally perturbs that center.  It does not relax the 4 mm
hand/tool penetration limit.

## Start all five training jobs

From `improved_tds`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_spray_lite_tasks.ps1
```

Defaults:

- SAC + HER
- 1,000,000 timesteps per spray variant
- random target angle from 5% to 50% of trigger range per episode
- at least two bottle contacts for success
- dense tracking, trigger-progress, grasp-contact, and index-proximity reward
- stable success after maintaining target and grasp for five consecutive steps
- terminal drop/tip penalty of 250
- checkpoints every 100,000 timesteps

The default reward is
`2.0 * tracking + 1.0 * trigger_progress + 2.0 * grasp +
0.25 * index_proximity + 5.0 * success`; a drop or tip overrides all terms
with `-250`. Each component is recorded in step `info`, and the complete
configuration is saved in the training summary for auditability.

Use `-TotalTimesteps`, `-CheckpointFreq`, `-BaseSeed`, `-RunName`,
`-SuccessHoldSteps`, or the `*RewardWeight`/`*Bonus`/`*Penalty` parameters to
override the run settings.

## Current pool QA

All five pools contain 32 poses with the 13 Lite actuators.  All 160 poses were
finite and remained non-dropped/non-tipped for a 20-step zero-action hold when
injected into the canonical training environments.  A simple fixed index-finger
drive moved spray1 for all poses, but did not move spray2-5; those variants begin
from stable near-contact configurations and rely on dense proximity/contact
shaping to acquire trigger contact during learning.
