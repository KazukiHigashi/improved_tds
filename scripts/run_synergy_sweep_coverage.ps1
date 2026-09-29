param(
    [string]$StrictAnalysisName = "strict_synergy_paper_v2",
    [string]$AnalysisName = "synergy_sweep_coverage_v1",
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [int]$MaxConcurrentTasks = 4,
    [int]$VisualSeed = 20260911
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$StrictReportRoot = Join-Path (Join-Path (Join-Path $ProjectRoot "analysis") $StrictAnalysisName) "reports"
$OutputRoot = Join-Path (Join-Path $ProjectRoot "analysis") $AnalysisName
$ReportRoot = Join-Path $OutputRoot "reports"
$ImageRoot = Join-Path $OutputRoot "images"
$SummaryRoot = Join-Path $OutputRoot "summary"
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
New-Item -ItemType Directory -Force -Path $ReportRoot, $ImageRoot, $SummaryRoot | Out-Null
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "running"

$Queue = [System.Collections.Queue]::new()
foreach ($StrictReport in Get-ChildItem -LiteralPath $StrictReportRoot -Filter "*_seed*.json" -File) {
    if ($StrictReport.BaseName -notmatch "^(.*)_seed(\d+)$") { continue }
    $Task = $Matches[1]
    $Seed = [int]$Matches[2]
    $Output = Join-Path $ReportRoot $StrictReport.Name
    if (Test-Path -LiteralPath $Output) { continue }
    $Arguments = @(
        "-m", "improved_tds.experiments.evaluate_synergy_sweep_coverage",
        "--strict-report", $StrictReport.FullName,
        "--pose-pool-dir", $PoolRoot,
        "--output", $Output,
        "--sweep-points", 13
    )
    if ($Seed -eq $VisualSeed) {
        $Arguments += @("--render-dir", (Join-Path $ImageRoot $Task))
    }
    $Queue.Enqueue([pscustomobject]@{
        Name="$Task/$Seed"; Artifact=$Output; Arguments=$Arguments
        Stdout=Join-Path $ReportRoot "$($StrictReport.BaseName).log"
        Stderr=Join-Path $ReportRoot "$($StrictReport.BaseName).err.log"
    })
}

$Active = @()
$Failures = @()
while ($Queue.Count -gt 0 -or $Active.Count -gt 0) {
    while ($Queue.Count -gt 0 -and $Active.Count -lt $MaxConcurrentTasks) {
        $Job = $Queue.Dequeue()
        $Started = Get-Date
        $Process = Start-Process -FilePath $Python -ArgumentList $Job.Arguments `
            -RedirectStandardOutput $Job.Stdout -RedirectStandardError $Job.Stderr `
            -WindowStyle Hidden -PassThru
        $Active += [pscustomobject]@{Job=$Job; Process=$Process; Started=$Started}
    }
    Start-Sleep -Seconds 2
    $Remaining = @()
    foreach ($Running in $Active) {
        if ($Running.Process.HasExited) {
            $Running.Process.WaitForExit()
            $Artifact = Get-Item -LiteralPath $Running.Job.Artifact -ErrorAction SilentlyContinue
            if ($null -eq $Artifact -or $Artifact.LastWriteTime -lt $Running.Started) {
                $Failures += $Running.Job.Name
            }
        } else {
            $Remaining += $Running
        }
    }
    $Active = $Remaining
}
if ($Failures.Count -gt 0) { throw "sweep evaluation failed for: $($Failures -join ', ')" }

& $Python -m improved_tds.experiments.summarize_synergy_sweep_coverage `
    --report-dir $ReportRoot --image-dir $ImageRoot --output-dir $SummaryRoot `
    --visual-seed $VisualSeed
if ($LASTEXITCODE -ne 0) { throw "sweep summary failed" }
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "completed"
