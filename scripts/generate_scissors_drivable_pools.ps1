param(
    [string]$OutputDir = "tool-pose-pipeline\grasps_v4_scissors_drive",
    [int]$PoolSize = 32,
    [int]$MaxCandidates = 2000,
    [double]$Sigma = 0.15,
    [int]$Seed = 5100
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$PipelineRoot = Join-Path $WorkspaceRoot "tool-pose-pipeline"
$Destination = Join-Path $WorkspaceRoot $OutputDir

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$env:PYTHONPATH = $PipelineRoot

foreach ($Variant in 1..5) {
    & $Python -m toolpose.generate `
        --out $Destination `
        --task scissors `
        --hand lite `
        --variant $Variant `
        --size $PoolSize `
        --max-candidates $MaxCandidates `
        --sigma $Sigma `
        --seed ($Seed + $Variant - 1) `
        --require-drive
    if ($LASTEXITCODE -ne 0) {
        throw "Drive-qualified pool generation failed for scissors variant $Variant"
    }
}

Write-Output "Drive-qualified Scissors pools written to $Destination"
