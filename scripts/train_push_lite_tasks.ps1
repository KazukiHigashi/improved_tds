param(
    [int]$TotalTimesteps = 1000000,
    [int]$CheckpointFreq = 100000,
    [int]$BaseSeed = 2000,
    [string]$RunName = "push_lite_1m",
    [double]$PushTrackingRewardWeight = 2.0,
    [double]$PushProgressRewardWeight = 1.0,
    [double]$PushGraspRewardWeight = 2.0,
    [double]$PushThumbContactRewardWeight = 0.25,
    [double]$PushSuccessBonus = 5.0,
    [double]$PushFailurePenalty = 250.0,
    [double]$PushGraspGateFloor = 0.25,
    [double]$TargetMinFraction = 0.05,
    [double]$TargetMaxFraction = 0.5,
    [int]$RequiredGraspContacts = 2,
    [int]$SuccessHoldSteps = 5,
    # the pools regenerated for the corrected 13-actuator lite hand, the same
    # directory the spray runs used
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [int]$NEnvs = 4,
    [int[]]$Variants = @(1, 2, 3, 4, 5)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$FailedVariants = @()

foreach ($Variant in $Variants) {
    $Pool = Join-Path $PoolRoot "push_lite_$Variant.npz"
    if (-not (Test-Path -LiteralPath $Pool)) {
        throw "Initial-pose pool not found: $Pool"
    }

    $TaskDir = Join-Path $RunRoot "push_lite_$Variant"
    New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null
    Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "running"
    $Arguments = @(
        "-m", "improved_tds.experiments.train_sb3",
        "--env-id", "ShadowHandLitePush$Variant-v0",
        "--total-timesteps", $TotalTimesteps,
        "--seed", ($BaseSeed + $Variant - 1),
        "--output", (Join-Path $TaskDir "model"),
        "--dataset", (Join-Path $TaskDir "training_candidates.npz"),
        "--checkpoints-dir", (Join-Path $TaskDir "checkpoints"),
        "--checkpoint-freq", $CheckpointFreq,
        "--n-envs", $NEnvs,
        "--gradient-steps", 2,
        "--pose-pool-dir", $PoolRoot,
        "--push-tracking-reward-weight", $PushTrackingRewardWeight,
        "--push-progress-reward-weight", $PushProgressRewardWeight,
        "--push-grasp-reward-weight", $PushGraspRewardWeight,
        "--push-thumb-contact-reward-weight", $PushThumbContactRewardWeight,
        "--push-success-bonus", $PushSuccessBonus,
        "--push-failure-penalty", $PushFailurePenalty,
        "--push-grasp-gate-floor", $PushGraspGateFloor,
        "--target-min-fraction", $TargetMinFraction,
        "--target-max-fraction", $TargetMaxFraction,
        "--required-grasp-contacts", $RequiredGraspContacts,
        "--success-hold-steps", $SuccessHoldSteps
    )
    try {
        # Windows PowerShell 5.1 turns any native stderr line into an ErrorRecord.
        # MuJoCo numerical warnings are non-fatal and reset the simulation state,
        # so only the Python process exit code determines task failure.
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python @Arguments *>&1 | Tee-Object -FilePath (Join-Path $TaskDir "train.log")
            $TrainingExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($TrainingExitCode -ne 0) {
            throw "training exited with code $TrainingExitCode"
        }
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "completed"
    }
    catch {
        $_ | Out-File -Append -FilePath (Join-Path $TaskDir "train.log")
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "failed"
        $FailedVariants += $Variant
    }
}

if ($FailedVariants.Count -gt 0) {
    throw "push training failed for variants: $($FailedVariants -join ', ')"
}
