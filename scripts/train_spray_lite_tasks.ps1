param(
    [int]$TotalTimesteps = 1000000,
    [int]$CheckpointFreq = 100000,
    [int]$BaseSeed = 3000,
    [string]$RunName = "spray_lite_1m",
    [double]$TargetMinFraction = 0.05,
    [double]$TargetMaxFraction = 0.5,
    [int]$RequiredGraspContacts = 2,
    [int]$SuccessHoldSteps = 5,
    [double]$TrackingRewardWeight = 2.0,
    [double]$ProgressRewardWeight = 1.0,
    [double]$GraspRewardWeight = 2.0,
    [double]$IndexProximityRewardWeight = 0.25,
    [double]$SuccessBonus = 5.0,
    [double]$FailurePenalty = 250.0,
    [double]$GraspGateFloor = 0.25,
    # which pose pools to train from, and which variants have one
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
    $Pool = Join-Path $PoolRoot "spray_lite_spray$Variant.npz"
    if (-not (Test-Path -LiteralPath $Pool)) {
        throw "Initial-pose pool not found: $Pool"
    }

    $TaskDir = Join-Path $RunRoot "spray_lite_$Variant"
    New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null
    Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "running"
    $Arguments = @(
        "-m", "improved_tds.experiments.train_sb3",
        "--env-id", "ShadowHandLiteSpray$Variant-v0",
        "--total-timesteps", $TotalTimesteps,
        "--seed", ($BaseSeed + $Variant - 1),
        "--output", (Join-Path $TaskDir "model"),
        "--dataset", (Join-Path $TaskDir "training_candidates.npz"),
        "--checkpoints-dir", (Join-Path $TaskDir "checkpoints"),
        "--checkpoint-freq", $CheckpointFreq,
        "--n-envs", $NEnvs,
        "--gradient-steps", 2,
        "--pose-pool-dir", $PoolRoot,
        "--target-min-fraction", $TargetMinFraction,
        "--target-max-fraction", $TargetMaxFraction,
        "--required-grasp-contacts", $RequiredGraspContacts,
        "--success-hold-steps", $SuccessHoldSteps,
        "--spray-tracking-reward-weight", $TrackingRewardWeight,
        "--spray-progress-reward-weight", $ProgressRewardWeight,
        "--spray-grasp-reward-weight", $GraspRewardWeight,
        "--spray-index-proximity-reward-weight", $IndexProximityRewardWeight,
        "--spray-success-bonus", $SuccessBonus,
        "--spray-failure-penalty", $FailurePenalty,
        "--spray-grasp-gate-floor", $GraspGateFloor
    )
    try {
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
    throw "spray training failed for variants: $($FailedVariants -join ', ')"
}
