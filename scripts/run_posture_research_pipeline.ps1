param([string]$RunName = "posture_research_v1", [int]$TotalTimesteps = 1000000)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$RunRoot = Join-Path $ProjectRoot "runs\$RunName"
if (Test-Path -LiteralPath $RunRoot) { throw "Use a fresh run name; do not mix task versions: $RunRoot" }
$ScissorsPool = Join-Path $WorkspaceRoot "tool-pose-pipeline\grasps_posture_v1"
if (-not (Test-Path -LiteralPath $ScissorsPool)) {
    & (Join-Path $PSScriptRoot "generate_posture_research_pools.ps1")
}
& $Python -m improved_tds.experiments.audit_posture_pools --pool-dir $ScissorsPool --output (Join-Path $RunRoot "pool_audit.json")
if ($LASTEXITCODE -ne 0) { throw "Pool audit failed" }
$Audit = Get-Content (Join-Path $RunRoot "pool_audit.json") -Raw | ConvertFrom-Json
foreach ($Variant in 1..5) {
    $Entry = $Audit.PSObject.Properties[[string]$Variant].Value
    if ($Entry.pose_count -ne 32 -or $Entry.hold_survivors -ne 32) { throw "Scissors $Variant pool does not satisfy shared posture criteria." }
}
& (Join-Path $PSScriptRoot "train_unified_reward_campaign.ps1") -RunName $RunName -TotalTimesteps $TotalTimesteps -ScissorsPoolDir "tool-pose-pipeline\grasps_posture_v1"
$Incomplete = @()
foreach ($Category in @("push", "spray", "scissors")) {
    foreach ($Variant in 1..5) {
        $TaskDir = Join-Path $RunRoot "${Category}_lite_$Variant"
        $EnvId = switch ($Category) {
            "push" { "ShadowHandLitePush$Variant-v0" }
            "spray" { "ShadowHandLiteSpray$Variant-v0" }
            "scissors" { "ScissorsVariant$Variant-v0" }
        }
        $Pool = if ($Category -eq "scissors") { $ScissorsPool } else { Join-Path $WorkspaceRoot "tool-pose-pipeline\grasps_v3" }
        & $Python -m improved_tds.experiments.collect_balanced_postures --env-id $EnvId --model (Join-Path $TaskDir "best_pose_coverage_policy.zip") --pose-pool-dir $Pool --output (Join-Path $TaskDir "balanced_postures.npz")
        if ($LASTEXITCODE -ne 0) { $Incomplete += "${Category}_$Variant" }
    }
}
if ($Incomplete.Count) { throw "Posture quotas incomplete; inspect partial artifacts: $($Incomplete -join ', ')" }
