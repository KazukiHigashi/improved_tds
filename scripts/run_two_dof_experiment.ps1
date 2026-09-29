param(
    [string]$RunName = "unified_reward_4x4_v2",
    [string]$AnalysisName = "two_dof_pca_pls_15tasks",
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [int]$MaxConcurrentTasks = 4
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName
$OutputRoot = Join-Path (Join-Path $ProjectRoot "analysis") $AnalysisName
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$TargetFractions = "0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$Tasks = @()
foreach ($Variant in 1..5) {
    $Tasks += [pscustomobject]@{
        Name = "push_lite_$Variant"
        EnvId = "ShadowHandLitePush$Variant-v0"
    }
    $Tasks += [pscustomobject]@{
        Name = "spray_lite_$Variant"
        EnvId = "ShadowHandLiteSpray$Variant-v0"
    }
    $Tasks += [pscustomobject]@{
        Name = "scissors_lite_$Variant"
        EnvId = "ScissorsVariant$Variant-v0"
    }
}

function Wait-ArtifactJobs([System.Collections.Queue]$Pending, [string]$Phase) {
    $Active = @()
    $Failures = @()
    while ($Pending.Count -gt 0 -or $Active.Count -gt 0) {
        while ($Pending.Count -gt 0 -and $Active.Count -lt $MaxConcurrentTasks) {
            $Job = $Pending.Dequeue()
            $Process = Start-Process -FilePath $Python -ArgumentList $Job.Arguments `
                -RedirectStandardOutput $Job.Stdout -RedirectStandardError $Job.Stderr `
                -WindowStyle Hidden -PassThru
            $Active += [pscustomobject]@{ Job = $Job; Process = $Process }
        }
        Start-Sleep -Seconds 2
        $Remaining = @()
        foreach ($Running in $Active) {
            if ($Running.Process.HasExited) {
                if (-not (Test-Path -LiteralPath $Running.Job.Artifact)) {
                    $Failures += $Running.Job.Name
                }
            }
            else {
                $Remaining += $Running
            }
        }
        $Active = $Remaining
    }
    if ($Failures.Count -gt 0) {
        throw "$Phase failed for: $($Failures -join ', ')"
    }
}

$CollectionQueue = [System.Collections.Queue]::new()
foreach ($Task in $Tasks) {
    $Model = Join-Path (Join-Path $RunRoot $Task.Name) "best_policy.zip"
    if (-not (Test-Path -LiteralPath $Model)) {
        throw "Best policy not found: $Model"
    }
    $Database = Join-Path $OutputRoot "$($Task.Name).npz"
    $Metadata = [System.IO.Path]::ChangeExtension($Database, ".json")
    if (Test-Path -LiteralPath $Database) {
        $Saved = Get-Content -Raw -LiteralPath $Metadata | ConvertFrom-Json
        if ($Saved.pose_count -ne 32 -or $Saved.rows -ne 672) {
            throw "Unbalanced existing database: $Database"
        }
        continue
    }
    $CollectionQueue.Enqueue([pscustomobject]@{
        Name = $Task.Name
        Artifact = $Database
        Stdout = Join-Path $OutputRoot "$($Task.Name)_collect.log"
        Stderr = Join-Path $OutputRoot "$($Task.Name)_collect.err.log"
        Arguments = @(
            "-m", "improved_tds.experiments.collect_posture_database",
            "--env-id", $Task.EnvId,
            "--model", $Model,
            "--pose-pool-dir", $PoolRoot,
            "--output", $Database,
            "--target-fractions", ('"' + $TargetFractions + '"'),
            "--expected-pose-count", 32,
            "--object-position-noise", 0.0
        )
    })
}
Wait-ArtifactJobs $CollectionQueue "posture collection"

# Enforce the same authored-pose and pose-target row counts before fitting.
foreach ($Task in $Tasks) {
    $Metadata = Join-Path $OutputRoot "$($Task.Name).json"
    $Saved = Get-Content -Raw -LiteralPath $Metadata | ConvertFrom-Json
    if ($Saved.pose_count -ne 32 -or $Saved.rows -ne 672) {
        throw "Database balance check failed: $Metadata"
    }
}

$EvaluationQueue = [System.Collections.Queue]::new()
foreach ($Task in $Tasks) {
    $Database = Join-Path $OutputRoot "$($Task.Name).npz"
    $Report = Join-Path $OutputRoot "$($Task.Name)_control.json"
    if (Test-Path -LiteralPath $Report) { continue }
    $EvaluationQueue.Enqueue([pscustomobject]@{
        Name = $Task.Name
        Artifact = $Report
        Stdout = Join-Path $OutputRoot "$($Task.Name)_control.log"
        Stderr = Join-Path $OutputRoot "$($Task.Name)_control.err.log"
        Arguments = @(
            "-m", "improved_tds.experiments.evaluate_two_dof_control",
            "--database", $Database,
            "--pose-pool-dir", $PoolRoot,
            "--output", $Report,
            "--methods", '"pca pls"',
            "--fit-poses", 16,
            "--calibration-poses", 8,
            "--reference-points", 5,
            "--grasp-points", 9,
            "--rho-points", 13,
            "--target-points", 9
        )
    })
}
Wait-ArtifactJobs $EvaluationQueue "two-DoF evaluation"
& $Python -m improved_tds.experiments.summarize_two_dof_control `
    --report-dir $OutputRoot --output (Join-Path $OutputRoot "summary.json")
if ($LASTEXITCODE -ne 0) {
    throw "two-DoF summary failed"
}
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "completed"
