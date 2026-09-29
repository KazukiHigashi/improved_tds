param(
    [int]$TrainingLauncherPid,
    [string]$RunName = "unified_reward_fullrange_4x4_v1",
    [string]$StrictAnalysisName = "strict_synergy_paper_v4_fullrange",
    [string]$CoverageAnalysisName = "synergy_sweep_coverage_v3_fullrange",
    [string]$AdaptiveAnalysisName = "adaptive_synergy_v2_fullrange"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName
$PipelineRoot = Join-Path (Join-Path $ProjectRoot "analysis") "full_range_pipeline_v1"
New-Item -ItemType Directory -Force -Path $PipelineRoot | Out-Null
$StatusPath = Join-Path $PipelineRoot "status.json"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

function Write-Status([string]$State, [string]$Detail = "") {
    [pscustomobject]@{
        state = $State
        detail = $Detail
        timestamp = (Get-Date).ToString("o")
        run = $RunName
        strict_analysis = $StrictAnalysisName
        coverage_analysis = $CoverageAnalysisName
        adaptive_analysis = $AdaptiveAnalysisName
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

try {
    Write-Status "waiting_for_training" "launcher pid $TrainingLauncherPid"
    if (Get-Process -Id $TrainingLauncherPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $TrainingLauncherPid
    }
    $TrainingStatus = (Get-Content -Raw -LiteralPath (Join-Path $RunRoot "campaign_status.txt")).Trim()
    if ($TrainingStatus -ne "completed") {
        throw "training did not complete: $TrainingStatus"
    }

    Write-Status "strict_full_range"
    & (Join-Path $PSScriptRoot "run_strict_paper_experiment.ps1") `
        -RunName $RunName -AnalysisName $StrictAnalysisName -MaxConcurrentTasks 4
    if ($LASTEXITCODE -ne 0) { throw "strict full-range experiment failed" }
    & $Python (Join-Path $PSScriptRoot "validate_strict_paper_experiment.py") `
        (Join-Path (Join-Path $ProjectRoot "analysis") $StrictAnalysisName)
    if ($LASTEXITCODE -ne 0) { throw "strict validation failed" }

    Write-Status "coverage_full_range"
    & (Join-Path $PSScriptRoot "run_synergy_sweep_coverage.ps1") `
        -StrictAnalysisName $StrictAnalysisName -AnalysisName $CoverageAnalysisName `
        -MaxConcurrentTasks 4 -VisualSeed 20260911
    if ($LASTEXITCODE -ne 0) { throw "coverage experiment failed" }

    Write-Status "adaptive_full_range"
    & $Python (Join-Path $PSScriptRoot "run_adaptive_synergy_campaign.py") `
        --source (Join-Path (Join-Path (Join-Path $ProjectRoot "analysis") $StrictAnalysisName) "reports") `
        --output (Join-Path (Join-Path $ProjectRoot "analysis") $AdaptiveAnalysisName) `
        --pool (Join-Path $WorkspaceRoot "tool-pose-pipeline\grasps_v3") --workers 4
    if ($LASTEXITCODE -ne 0) { throw "adaptive experiment failed" }

    Write-Status "completed" "full-range training, recollection, evaluation and HTML complete"
}
catch {
    Write-Status "failed" ($_ | Out-String)
    throw
}
