param([int]$CandidatePoolSize = 512, [int]$MaxCandidates = 40000)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$Raw = "tool-pose-pipeline\grasps_posture_candidates_v1"
$Qualified = "tool-pose-pipeline\grasps_posture_v1"
if ((Test-Path (Join-Path $WorkspaceRoot $Raw)) -or (Test-Path (Join-Path $WorkspaceRoot $Qualified))) { throw "Candidate/output directories already exist; preserve them and choose fresh paths in this script." }
& (Join-Path $PSScriptRoot "generate_scissors_drivable_pools.ps1") -OutputDir $Raw -PoolSize $CandidatePoolSize -MaxCandidates $MaxCandidates
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $Python -m improved_tds.experiments.audit_posture_pools --pool-dir (Join-Path $WorkspaceRoot $Raw) --qualified-dir (Join-Path $WorkspaceRoot $Qualified) --output (Join-Path $ProjectRoot "analysis\posture_pool_qualification_v1.json")
if ($LASTEXITCODE -ne 0) { throw "Posture qualification incomplete; inspect audit before training." }
