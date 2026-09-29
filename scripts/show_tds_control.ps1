param(
    # One representative variant per task. Spray uses the checkpoint the
    # screening picked; push and scissors use the variants that trained well.
    [string]$SprayModel   = "runs\spray_lite_1m_graspfix_v5_20260816\spray_lite_1\checkpoints\sac_her_900000_steps.zip",
    [int]$SprayVariant    = 1,
    [string]$PushModel    = "runs\push_lite_1m_gate_v1_20260818\push_lite_3\model.zip",
    [int]$PushVariant     = 3,
    [string]$ScissorsModel = "runs\scissors_lite_1m_v1_20260818\scissors_lite_2\model.zip",
    [int]$ScissorsVariant = 2,
    [string]$PoolDir      = "tool-pose-pipeline\grasps_v3",
    # 1.0 plays the motion at the speed it is simulated; raise it to skim
    [double]$Speed        = 1.0,
    [string[]]$Tasks      = @("spray", "push", "scissors")
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Pool = Join-Path $WorkspaceRoot $PoolDir

$Cases = @{
    spray    = @{ model = $SprayModel;    variant = $SprayVariant }
    push     = @{ model = $PushModel;     variant = $PushVariant }
    scissors = @{ model = $ScissorsModel; variant = $ScissorsVariant }
}

foreach ($Task in $Tasks) {
    $Case = $Cases[$Task]
    $Model = Join-Path $ProjectRoot $Case.model
    if (-not (Test-Path -LiteralPath $Model)) {
        Write-Output "$Task : model not found, skipping ($Model)"
        continue
    }
    Write-Output ""
    Write-Output "######## $Task ########"
    & $Python -W ignore -m improved_tds.experiments.visualize_tds_control `
        --task $Task --variant $Case.variant --model $Model `
        --pose-pool-dir $Pool --speed $Speed
}
