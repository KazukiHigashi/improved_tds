param([string]$RunName = "tool_control_v2_20260917", [int]$TotalTimesteps = 2000000)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$RunRoot = Join-Path $ProjectRoot "runs\$RunName"
if (Test-Path $RunRoot) { throw "Fresh v2 run required: $RunRoot" }
New-Item -ItemType Directory -Path $RunRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "src") -Destination (Join-Path $RunRoot "source_snapshot") -Recurse
# Workers import the immutable snapshot so later edits cannot mix experiment versions.
& (Join-Path $PSScriptRoot "train_unified_reward_campaign.ps1") -RunName $RunName -TotalTimesteps $TotalTimesteps -SourceRoot (Join-Path $RunRoot "source_snapshot")
Set-Content -LiteralPath (Join-Path $RunRoot "pipeline_status.txt") -Value "collecting"
$Incomplete = @()
foreach ($Category in @("push","spray","scissors")) {
    foreach ($Variant in 1..5) {
        $TaskDir=Join-Path $RunRoot "${Category}_lite_$Variant"
        $EnvId=switch($Category){"push"{"ShadowHandLitePush$Variant-v0"};"spray"{"ShadowHandLiteSpray$Variant-v0"};"scissors"{"ScissorsVariant$Variant-v0"}}
        $Pool=if($Category -eq "scissors"){Join-Path $WorkspaceRoot "tool-pose-pipeline\grasps_posture_dynamic_v2_mild"}else{Join-Path $WorkspaceRoot "tool-pose-pipeline\grasps_v3"}
        & $Python -m improved_tds.experiments.collect_balanced_postures --env-id $EnvId --model (Join-Path $TaskDir "best_pose_coverage_policy.zip") --pose-pool-dir $Pool --output (Join-Path $TaskDir "balanced_postures.npz")
        if($LASTEXITCODE -ne 0){$Incomplete += "${Category}_$Variant"}
    }
}
$FinalStatus=if($Incomplete.Count){"collection_incomplete: $($Incomplete -join ', ')"}else{"completed"}
Set-Content -LiteralPath (Join-Path $RunRoot "pipeline_status.txt") -Value $FinalStatus
