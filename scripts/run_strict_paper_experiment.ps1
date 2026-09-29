param(
    [string]$RunName = "unified_reward_4x4_v2",
    [string]$AnalysisName = "strict_synergy_paper_v2",
    [string]$DatabaseSourceAnalysisName = "",
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [int]$MaxConcurrentTasks = 4,
    [int]$CommonFitRows = 100,
    [string]$TargetFractions = "0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName
$OutputRoot = Join-Path (Join-Path $ProjectRoot "analysis") $AnalysisName
$DatabaseRoot = if ($DatabaseSourceAnalysisName) {
    Join-Path (Join-Path (Join-Path $ProjectRoot "analysis") $DatabaseSourceAnalysisName) "databases"
} else {
    Join-Path $OutputRoot "databases"
}
$ReportRoot = Join-Path $OutputRoot "reports"
$SummaryRoot = Join-Path $OutputRoot "summary"
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$TargetFractionValues = @($TargetFractions -split '\s+' | Where-Object { $_ } | ForEach-Object { [double]$_ })
if ($TargetFractionValues.Count -lt 2 -or ($TargetFractionValues | Where-Object { $_ -lt 0.0 -or $_ -gt 1.0 })) {
    throw "TargetFractions must contain at least two values in [0,1]"
}
$ExpectedRows = 32 * $TargetFractionValues.Count
$Seeds = @(20260911, 20260912, 20260913, 20260914, 20260915)
New-Item -ItemType Directory -Force -Path $ReportRoot, $SummaryRoot | Out-Null
if (-not $DatabaseSourceAnalysisName) {
    New-Item -ItemType Directory -Force -Path $DatabaseRoot | Out-Null
}
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "running"

$Tasks = @()
foreach ($Variant in 1..5) {
    $Tasks += [pscustomobject]@{Name="push_lite_$Variant";EnvId="ShadowHandLitePush$Variant-v0"}
    $Tasks += [pscustomobject]@{Name="spray_lite_$Variant";EnvId="ShadowHandLiteSpray$Variant-v0"}
    $Tasks += [pscustomobject]@{Name="scissors_lite_$Variant";EnvId="ScissorsVariant$Variant-v0"}
}

function Run-Queue([System.Collections.Queue]$Queue, [string]$Phase) {
    $Active = @()
    $Failures = @()
    while ($Queue.Count -gt 0 -or $Active.Count -gt 0) {
        while ($Queue.Count -gt 0 -and $Active.Count -lt $MaxConcurrentTasks) {
            $Job = $Queue.Dequeue()
            $Started = Get-Date
            $Process = Start-Process -FilePath $Python -ArgumentList $Job.Arguments `
                -RedirectStandardOutput $Job.Stdout -RedirectStandardError $Job.Stderr `
                -WindowStyle Hidden -PassThru
            $Active += [pscustomobject]@{Job=$Job;Process=$Process;Started=$Started}
        }
        Start-Sleep -Seconds 2
        $Remaining = @()
        foreach ($Running in $Active) {
            if ($Running.Process.HasExited) {
                $Running.Process.WaitForExit()
                $Running.Process.Refresh()
                $Artifact = Get-Item -LiteralPath $Running.Job.Artifact -ErrorAction SilentlyContinue
                if ($null -eq $Artifact -or $Artifact.LastWriteTime -lt $Running.Started) {
                    $Failures += $Running.Job.Name
                }
            } else { $Remaining += $Running }
        }
        $Active = $Remaining
    }
    if ($Failures.Count -gt 0) { throw "$Phase failed for: $($Failures -join ', ')" }
}

$Collection = [System.Collections.Queue]::new()
if (-not $DatabaseSourceAnalysisName) { foreach ($Task in $Tasks) {
    $Database = Join-Path $DatabaseRoot "$($Task.Name).npz"
    $Metadata = [System.IO.Path]::ChangeExtension($Database, ".json")
    $Reuse = $false
    if ((Test-Path -LiteralPath $Database) -and (Test-Path -LiteralPath $Metadata)) {
        $Saved = Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json
        $SavedFractions = @($Saved.target_fractions | ForEach-Object { [double]$_ })
        $Reuse = $Saved.rows -eq $ExpectedRows -and $Saved.pose_count -eq 32 -and `
            $Saved.strict_success_window -and `
            (($SavedFractions -join ',') -eq ($TargetFractionValues -join ','))
    }
    if ($Reuse) { continue }
    $Model = Join-Path (Join-Path $RunRoot $Task.Name) "best_policy.zip"
    $Collection.Enqueue([pscustomobject]@{
        Name=$Task.Name;Artifact=$Database
        Stdout=Join-Path $DatabaseRoot "$($Task.Name).log"
        Stderr=Join-Path $DatabaseRoot "$($Task.Name).err.log"
        Arguments=@(
            "-m","improved_tds.experiments.collect_posture_database",
            "--env-id",$Task.EnvId,"--model",$Model,"--pose-pool-dir",$PoolRoot,
            "--output",$Database,"--target-fractions",('"'+$TargetFractions+'"'),
            "--expected-pose-count",32,"--object-position-noise",0.0,
            "--strict-success-window","--success-window-steps",5,"--overwrite"
        )
    })
} }
Run-Queue $Collection "strict posture collection"

foreach ($Task in $Tasks) {
    $Metadata = Get-Content -LiteralPath (Join-Path $DatabaseRoot "$($Task.Name).json") -Raw | ConvertFrom-Json
    if ($Metadata.rows -ne $ExpectedRows -or $Metadata.pose_count -ne 32 -or -not $Metadata.strict_success_window) {
        throw "Invalid strict database: $($Task.Name)"
    }
}

$Evaluation = [System.Collections.Queue]::new()
foreach ($Seed in $Seeds) {
    foreach ($Task in $Tasks) {
        $Report = Join-Path $ReportRoot "$($Task.Name)_seed$Seed.json"
        if (Test-Path -LiteralPath $Report) { continue }
        $Evaluation.Enqueue([pscustomobject]@{
            Name="$($Task.Name)/$Seed";Artifact=$Report
            Stdout=Join-Path $ReportRoot "$($Task.Name)_seed$Seed.log"
            Stderr=Join-Path $ReportRoot "$($Task.Name)_seed$Seed.err.log"
            Arguments=@(
                "-m","improved_tds.experiments.evaluate_strict_synergy_control",
                "--database",(Join-Path $DatabaseRoot "$($Task.Name).npz"),
                "--pose-pool-dir",$PoolRoot,"--output",$Report,
                "--split-seed",$Seed,"--common-fit-rows",$CommonFitRows,
                "--require-all-fit-targets"
            )
        })
    }
}
Run-Queue $Evaluation "strict held-out evaluation"
& $Python -m improved_tds.experiments.summarize_strict_synergy_control `
    --report-dir $ReportRoot --output-dir $SummaryRoot
if ($LASTEXITCODE -ne 0) { throw "strict summary failed" }
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "completed"
