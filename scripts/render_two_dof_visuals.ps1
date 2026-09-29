param(
    [string]$AnalysisName = "two_dof_pca_pls_15tasks",
    [string]$OutputName = "two_dof_pca_pls_15tasks_visuals",
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [int]$MaxConcurrentTasks = 4,
    [switch]$PlotsOnly,
    [switch]$IndexOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$AnalysisRoot = Join-Path (Join-Path $ProjectRoot "analysis") $AnalysisName
$OutputRoot = Join-Path (Join-Path $ProjectRoot "analysis") $OutputName
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "running"

$Queue = [System.Collections.Queue]::new()
if (-not $IndexOnly) {
foreach ($Database in Get-ChildItem -LiteralPath $AnalysisRoot -Filter "*_lite_?.npz") {
    $Name = $Database.BaseName
    $Report = Join-Path $AnalysisRoot "${Name}_control.json"
    if (-not (Test-Path -LiteralPath $Report)) { throw "Missing report: $Report" }
    $TaskOutput = Join-Path $OutputRoot $Name
    $Arguments = @(
        "-m", "improved_tds.experiments.render_two_dof_visuals",
        "--database", $Database.FullName,
        "--report", $Report,
        "--pose-pool-dir", $PoolRoot,
        "--output", $TaskOutput
    )
    if ($PlotsOnly) { $Arguments += "--plots-only" }
    $Queue.Enqueue([pscustomobject]@{
        Name = $Name
        Artifact = Join-Path $TaskOutput "manifest.json"
        Stdout = Join-Path $OutputRoot "${Name}.log"
        Stderr = Join-Path $OutputRoot "${Name}.err.log"
        Arguments = $Arguments
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
        $Active += [pscustomobject]@{ Job = $Job; Process = $Process; Started = $Started }
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
        } else {
            $Remaining += $Running
        }
    }
    $Active = $Remaining
}
if ($Failures.Count -gt 0) { throw "Rendering failed for: $($Failures -join ', ')" }
}
$Cards = foreach ($TaskDir in Get-ChildItem -LiteralPath $OutputRoot -Directory | Sort-Object Name) {
    $Name = [System.Net.WebUtility]::HtmlEncode($TaskDir.Name)
    $PcaPanel = if (Test-Path -LiteralPath (Join-Path $TaskDir.FullName "pca\control_contact_sheet.png")) {
        "$Name/pca/control_contact_sheet.png"
    } else { "$Name/pca/operation_sweep_contact_sheet.png" }
    $PlsPanel = if (Test-Path -LiteralPath (Join-Path $TaskDir.FullName "pls\control_contact_sheet.png")) {
        "$Name/pls/control_contact_sheet.png"
    } else { "$Name/pls/operation_sweep_contact_sheet.png" }
    "<section><h2>$Name</h2><p><a href='$Name/index.html'>Open full-resolution panels and individual frames</a></p>" +
    "<img class='curve' src='$Name/operation_score_vs_tool_dof.png'>" +
    "<div><img src='$PcaPanel'><img src='$PlsPanel'></div></section>"
}
$Index = "<!doctype html><meta charset='utf-8'><title>PCA/PLS visual results</title>" +
    "<style>body{font-family:sans-serif;margin:24px}section{margin:36px 0}div{display:grid;grid-template-columns:1fr 1fr;gap:12px}" +
    "img{width:100%;border:1px solid #bbb}.curve{display:block;max-width:1120px;margin:12px 0}</style><h1>PCA/PLS one-/two-DoF visual results</h1>" +
    "<p>Each task graph includes all database rows, the successful fit-data mean, and the one-/two-DoF operation experiment. Open a task to inspect controls and score sweeps.</p>" +
    ($Cards -join "`n")
[System.IO.File]::WriteAllText((Join-Path $OutputRoot "index.html"), $Index)
Set-Content -LiteralPath (Join-Path $OutputRoot "status.txt") -Value "completed"
