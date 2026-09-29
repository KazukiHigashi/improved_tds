param(
    [int]$TotalTimesteps = 2000000,
    [int]$MaxConcurrentTasks = 4,
    [int]$NEnvsPerTask = 4,
    [string]$RunName = "posture_research_v2",
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [string]$ScissorsPoolDir = "tool-pose-pipeline\grasps_posture_dynamic_v2_mild",
    [string]$SourceRoot = "",
    [string[]]$TaskNames = @(),
    [ValidateSet('v2','v3')][string]$ScissorsControlVersion = 'v2'
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
$ScissorsPoolRoot = Join-Path $WorkspaceRoot $ScissorsPoolDir
$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}
if ($MaxConcurrentTasks -lt 1 -or $NEnvsPerTask -lt 1) {
    throw "MaxConcurrentTasks and NEnvsPerTask must be positive"
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$env:PYTHONPATH = if ($SourceRoot) { $SourceRoot } else { Join-Path $ProjectRoot "src" }
$PackageRoots = @($env:PYTHONPATH)
foreach ($Package in @("gym-scissors","gym-slider","gym-spray","gym-push")) {
    $PackageRoots += Join-Path $WorkspaceRoot "tool-manip-gym\$Package"
}
$env:PYTHONPATH = $PackageRoots -join ';'

$Tasks = @()
foreach ($Variant in 1..5) {
    $Tasks += [pscustomobject]@{
        Name = "push_lite_$Variant"
        Category = "push"
        EnvId = "ShadowHandLitePush$Variant-v0"
        Pool = "push_lite_$Variant.npz"
        PoolRoot = $PoolRoot
        Seed = 2000 + $Variant - 1
    }
    $Tasks += [pscustomobject]@{
        Name = "spray_lite_$Variant"
        Category = "spray"
        EnvId = "ShadowHandLiteSpray$Variant-v0"
        Pool = "spray_lite_spray$Variant.npz"
        PoolRoot = $PoolRoot
        Seed = 3000 + $Variant - 1
    }
    $Tasks += [pscustomobject]@{
        Name = "scissors_lite_$Variant"
        Category = "scissors"
        EnvId = "ScissorsVariant$Variant-v0"
        Pool = "scissors_lite_$Variant.npz"
        PoolRoot = $ScissorsPoolRoot
        Seed = 4000 + $Variant - 1
    }
}

if ($TaskNames.Count -gt 0) {
    $KnownNames = @($Tasks | ForEach-Object { $_.Name })
    foreach ($Name in $TaskNames) {
        if ($Name -notin $KnownNames) { throw "Unknown task: $Name" }
    }
    $Tasks = @($Tasks | Where-Object { $_.Name -in $TaskNames })
}

foreach ($Task in $Tasks) {
    $Pool = Join-Path $Task.PoolRoot $Task.Pool
    if (-not (Test-Path -LiteralPath $Pool)) {
        throw "Initial-pose pool not found: $Pool"
    }
}

$Pending = [System.Collections.Queue]::new()
$Active = @()
$Failures = @()

Set-Content -LiteralPath (Join-Path $RunRoot "campaign_status.txt") -Value "running"

# Resume safely after a launcher interruption. A training summary is the
# authoritative completion marker; status.txt is only a human-readable view.
foreach ($Task in $Tasks) {
    $TaskDir = Join-Path $RunRoot $Task.Name
    $SummaryPath = Join-Path $TaskDir "model_training_summary.json"
    if (Test-Path -LiteralPath $SummaryPath) {
        $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
        $CompletedStatus = if ($Summary.stopped_early) { "early_stopped" } else { "completed" }
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value $CompletedStatus
        continue
    }
    $PidPath = Join-Path $TaskDir "pid.txt"
    $ExistingProcess = $null
    if (Test-Path -LiteralPath $PidPath) {
        $TaskProcessId = [int](Get-Content -Raw -LiteralPath $PidPath)
        $ExistingProcess = Get-Process -Id $TaskProcessId -ErrorAction SilentlyContinue
    }
    if ($ExistingProcess) {
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "running"
        $Active += [pscustomobject]@{
            Task = $Task
            Process = $ExistingProcess
            Dir = $TaskDir
        }
    }
    else {
        $Pending.Enqueue($Task)
    }
}

while ($Pending.Count -gt 0 -or $Active.Count -gt 0) {
    while ($Pending.Count -gt 0 -and $Active.Count -lt $MaxConcurrentTasks) {
        $Task = $Pending.Dequeue()
        $TaskDir = Join-Path $RunRoot $Task.Name
        New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "running"
        $Arguments = @(
            "-m", "improved_tds.experiments.train_sb3",
            "--env-id", $Task.EnvId,
            "--total-timesteps", $TotalTimesteps,
            "--seed", $Task.Seed,
            "--n-envs", $NEnvsPerTask,
            "--gradient-steps", 1,
            "--learning-rate", 0.0001,
            "--her-ratio", 0.25,
            "--output", (Join-Path $TaskDir "model"),
            "--dataset", (Join-Path $TaskDir "training_candidates.npz"),
            "--checkpoints-dir", (Join-Path $TaskDir "checkpoints"),
            "--checkpoint-freq", 100000,
            "--validation-freq", 100000,
            "--validation-target-fractions", '"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"',
            "--validation-success-threshold", 0.95,
            "--validation-consecutive", 2,
            "--best-policy-output", (Join-Path $TaskDir "best_policy"),
            "--validation-report", (Join-Path $TaskDir "validation_history.json"),
            "--pose-pool-dir", $Task.PoolRoot,
            "--target-min-fraction", 0.0,
            "--target-max-fraction", 1.0,
            "--success-hold-steps", 5,
            "--control-delta-penalty-weight", 0.1,
            "--control-magnitude-penalty-weight", 0.002,
            "--scissors-reward-type", "dense"
        )
        if ($Task.Category -eq "scissors") {
            # These settings are deliberately identical for Scissors 1--5.
            # Variants differ only in their geometry and drive-qualified poses.
            $Arguments += @(
                "--learning-rate", 0.0001,
                "--gradient-steps", 1,
                "--her-sampled-goals", 1,
                "--sac-ent-coef", 0.05,
                "--require-drivable-pool",
                "--scissors-coarse-tracking-weight", 1.0,
                "--scissors-fine-tracking-weight", 1.0,
                "--scissors-coarse-tracking-sigma", 0.35,
                "--scissors-progress-weight", 5.0,
                "--scissors-grasp-weight", 0.25,
                "--scissors-interaction-weight", 0.25,
                "--scissors-success-bonus", 5.0,
                "--scissors-failure-penalty", 25.0,
                "--scissors-grasp-gate-floor", 0.25
            )
            $Arguments += @('--scissors-control-version', $ScissorsControlVersion)
            if ($ScissorsControlVersion -eq 'v3') {
                # Full-range stationary goals; gentler optimizer/exploration.
                $Arguments += @('--learning-rate', 0.00005, '--gradient-steps', 2,
                    '--sac-ent-coef', 0.02)
            }
            else {
                $Arguments += @('--target-curriculum-max-fractions', '"0.3 0.5 0.7 1.0"')
            }
        }
        elseif ($Task.Category -eq "push") {
            # The 2.0 tracking budget is split internally into equal broad and
            # fine kernels, matching the 1.0 + 1.0 Scissors construction.
            $Arguments += @(
                "--push-tracking-reward-weight", 2.0,
                "--push-progress-reward-weight", 5.0,
                "--push-grasp-reward-weight", 0.25,
                "--push-thumb-contact-reward-weight", 0.25,
                "--push-success-bonus", 5.0,
                "--push-failure-penalty", 25.0,
                "--push-grasp-gate-floor", 0.25
            )
        }
        elseif ($Task.Category -eq "spray") {
            $Arguments += @(
                "--spray-tracking-reward-weight", 2.0,
                "--spray-progress-reward-weight", 5.0,
                "--spray-grasp-reward-weight", 0.25,
                "--spray-index-proximity-reward-weight", 0.25,
                "--spray-success-bonus", 5.0,
                "--spray-failure-penalty", 25.0,
                "--spray-grasp-gate-floor", 0.25
            )
        }
        $Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
            -RedirectStandardOutput (Join-Path $TaskDir "train.log") `
            -RedirectStandardError (Join-Path $TaskDir "train.err.log") `
            -WindowStyle Hidden -PassThru
        Set-Content -LiteralPath (Join-Path $TaskDir "pid.txt") -Value $Process.Id
        $Active += [pscustomobject]@{ Task = $Task; Process = $Process; Dir = $TaskDir }
    }

    Start-Sleep -Seconds 2
    $Remaining = @()
    foreach ($Job in $Active) {
        if ($Job.Process.HasExited) {
            $SummaryPath = Join-Path $Job.Dir "model_training_summary.json"
            if (Test-Path -LiteralPath $SummaryPath) {
                $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
                $CompletedStatus = if ($Summary.stopped_early) { "early_stopped" } else { "completed" }
                Set-Content -LiteralPath (Join-Path $Job.Dir "status.txt") -Value $CompletedStatus
            }
            else {
                Set-Content -LiteralPath (Join-Path $Job.Dir "status.txt") -Value "failed"
                $Failures += $Job.Task.Name
            }
        }
        else {
            $Remaining += $Job
        }
    }
    $Active = $Remaining
}

if ($Failures.Count -gt 0) {
    Set-Content -LiteralPath (Join-Path $RunRoot "campaign_status.txt") `
        -Value "failed: $($Failures -join ', ')"
    throw "Training failed for: $($Failures -join ', ')"
}
Set-Content -LiteralPath (Join-Path $RunRoot "campaign_status.txt") -Value "completed"
