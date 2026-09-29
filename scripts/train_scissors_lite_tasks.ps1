param(
    [int]$TotalTimesteps = 1000000,
    [int]$CheckpointFreq = 100000,
    [int]$BaseSeed = 4000,
    [string]$RunName = "scissors_lite_1m",
    # Scissors keeps its own reward: the environment is already goal conditioned
    # and the task needs no grasp, since the fingers start threaded through the
    # handles. Only the initial-pose pools were rebuilt, for the corrected
    # 13-actuator lite hand.
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    # charge for command jitter; 0 reproduces the earlier runs
    [double]$CtrlDeltaPenalty = 0.0,
    [string]$RewardType = "standard",
    # shift the scissors onto the base of the fingers, and how tight the angle
    # has to be to count as success
    [double]$ObjectDx = 0.0,
    [double]$AngleThreshold = 0.01,
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
    $Pool = Join-Path $PoolRoot "scissors_lite_$Variant.npz"
    if (-not (Test-Path -LiteralPath $Pool)) {
        throw "Initial-pose pool not found: $Pool"
    }

    $TaskDir = Join-Path $RunRoot "scissors_lite_$Variant"
    New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null
    Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "running"
    $Arguments = @(
        "-m", "improved_tds.experiments.train_sb3",
        "--env-id", "ScissorsVariant$Variant-v0",
        "--total-timesteps", $TotalTimesteps,
        "--seed", ($BaseSeed + $Variant - 1),
        "--output", (Join-Path $TaskDir "model"),
        "--dataset", (Join-Path $TaskDir "training_candidates.npz"),
        "--checkpoints-dir", (Join-Path $TaskDir "checkpoints"),
        "--checkpoint-freq", $CheckpointFreq,
        "--n-envs", $NEnvs,
        "--gradient-steps", 2,
        "--pose-pool-dir", $PoolRoot,
        "--scissors-ctrl-delta-penalty", $CtrlDeltaPenalty,
        "--scissors-object-dx", $ObjectDx,
        "--scissors-angle-threshold", $AngleThreshold,
        "--scissors-reward-type", $RewardType
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
    throw "scissors training failed for variants: $($FailedVariants -join ', ')"
}
