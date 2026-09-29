param([string]$RunName = 'scissors_control_v3_20260918')
$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunRoot = Join-Path $ProjectRoot "runs\$RunName"
if (Test-Path -LiteralPath $RunRoot) { throw "Use a fresh run directory: $RunRoot" }
New-Item -ItemType Directory -Path $RunRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'src') -Destination (Join-Path $RunRoot 'source_snapshot') -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'train_unified_reward_campaign.ps1') -Destination (Join-Path $RunRoot 'campaign_script.ps1')
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot '..')).Path
$PoolSource = Join-Path $WorkspaceRoot 'tool-pose-pipeline\grasps_posture_long_v3'
Copy-Item -LiteralPath $PoolSource -Destination (Join-Path $RunRoot 'pose_pool') -Recurse
$PoolRelative = "improved_tds\runs\$RunName\pose_pool"
$SourceRoot = Join-Path $RunRoot 'source_snapshot'
$Manifest = @{
    version='scissors_control_v3'; started_at=(Get-Date -Format o)
    tasks=@('scissors_lite_3','scissors_lite_4','scissors_lite_5')
    max_timesteps=2000000; n_envs_per_task=4; concurrent_tasks=3
    validation_interval=100000; checkpoint_interval=100000
    early_stop_threshold=0.95; early_stop_consecutive=2
    fresh_training=$true; source_snapshot=$SourceRoot
    pool=$PoolRelative
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunRoot 'launch_manifest.json')
& (Join-Path $PSScriptRoot 'train_unified_reward_campaign.ps1') `
    -RunName $RunName -TotalTimesteps 2000000 -MaxConcurrentTasks 3 -NEnvsPerTask 4 `
    -SourceRoot $SourceRoot -ScissorsControlVersion v3 `
    -ScissorsPoolDir $PoolRelative `
    -TaskNames @('scissors_lite_3','scissors_lite_4','scissors_lite_5')

# Keep a fixed old-pool comparison so reset selection cannot masquerade as
# improved generalization. This is separate from the early-stop validation.
$Python = Join-Path $WorkspaceRoot 'fdms_reconfiguration\.venv\Scripts\python.exe'
Set-Content -LiteralPath (Join-Path $RunRoot 'comparison_status.txt') -Value 'running'
foreach ($Variant in 3..5) {
    $TaskDir = Join-Path $RunRoot "scissors_lite_$Variant"
    & $Python -m improved_tds.experiments.evaluate_scissors_control_v3 `
        --variant $Variant --model (Join-Path $TaskDir 'best_success_policy.zip') `
        --pool-dir (Join-Path $WorkspaceRoot 'tool-pose-pipeline\grasps_posture_dynamic_v2_mild') `
        --output (Join-Path $TaskDir 'old_pool_comparison.json')
    if ($LASTEXITCODE -ne 0) {
        Set-Content -LiteralPath (Join-Path $RunRoot 'comparison_status.txt') -Value "failed: scissors_$Variant"
        throw "Fixed-pool comparison failed for Scissors $Variant"
    }
}
Set-Content -LiteralPath (Join-Path $RunRoot 'comparison_status.txt') -Value 'completed'
