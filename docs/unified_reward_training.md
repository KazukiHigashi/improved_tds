# Push / Spray / Scissors common-term reward training

The paper training campaign uses task-category reward functions, but every
category shares the following term meanings and default scales. There is no
variant-specific reward branch within Push, Spray, or Scissors.

| Term | Weight | Meaning |
|---|---:|---|
| `tracking` | 2.0 total | broad range kernel + fine tolerance kernel |
| `progress` | 1.0 | signed one-transition reduction in normalized target error |
| `grasp` | 1.0 | maintained grasp/contact score |
| `interaction` | 0.25 | thumb contact, trigger proximity, or scissors position hold |
| `success_bonus` | 5.0 | target reached while the grasp is valid |
| `control_delta_penalty` | -0.1 | normalized command change cost |
| `control_magnitude_penalty` | -0.002 | normalized command magnitude cost |
| `failure_penalty` | -25 | drop, tip, or loss of the scissors grasp |

Tracking and progress are gated by grasp quality. The gate retains 25% of the
signal before a full grasp, avoiding a zero-gradient contact-acquisition phase.
The failure penalty replaces the other terms on a failure step. Push and
Spray already terminate on drop/tip; the unified scissors wrapper likewise
terminates on grasp loss so the penalty is charged exactly once.

Every value needed by `compute_reward` is copied into `info`, including the
previous achieved goal and control costs. Therefore HER goal relabelling uses the
same reward equation as online collection.

For Push and Spray, the broad-kernel standard deviation is 35% of the tool
range and the fine kernel uses the task success tolerance. Scissors uses the
same weights with a category-specific broad standard deviation of 0.35 rad.
All Scissors variants use these exact parameters.

The campaign command is:

```powershell
.\scripts\train_unified_reward_campaign.ps1
```

It schedules at most four independent training jobs. Each job uses a
four-worker `SubprocVecEnv`. Push and Spray retain their common campaign
optimizer settings. Scissors uses one HER future goal, one gradient step,
learning rate `1e-4`, fixed entropy coefficient `0.05`, and the common
curriculum `[0,0.3] -> [0,0.5] -> [0,0.7] -> [0,1.0]`. Scissors training rejects
pose pools not marked `require_drive=true`; generate the shared pools first:

```powershell
.\scripts\generate_scissors_drivable_pools.ps1
```

Checkpoints and validation
run every 100,000 aggregate environment steps. Two consecutive validation
scores of at least 95% stop that task early; `best_policy.zip` is maintained
separately. Best-policy ranking first maximizes the worst target-bin success,
then range coverage, aggregate success, and median error.
