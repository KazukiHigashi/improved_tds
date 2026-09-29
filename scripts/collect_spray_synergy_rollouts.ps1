param(
    [string]$RunName = "spray_lite_1m_graspfix_v5_20260816",
    # the pools the v5 run was actually trained from -- see the run's
    # model_training_summary.json, which records the pool path per variant
    [string]$PoolDir = "tool-pose-pipeline\grasps_v3",
    [string]$OutputName = "synergy_rollouts_v5",
    # 15 targets spanning the range trained on. With the object placement fixed a
    # repeat of the same cell is deterministic, so the episode budget buys
    # distinct tool states instead of repeats.
    [string]$TargetFractions = "0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00",
    [int]$EpisodesPerCell = 1,
    # A randomly placed object moves the tool DoF by an amount no hand synergy
    # could encode; fixing it makes the achieved tool state a function of the
    # hand's posture, which is what the synergy analysis measures.
    [double]$BottleXyNoise = 0.0,
    [int]$Seed = 7000,
    # SAC can collapse late in training and the saved final model is whatever it
    # collapsed into -- spray2 fell from 0.98 success at 700k steps to 0.22 by 1M.
    # These are the checkpoints screen_checkpoints.py picked; no variant's saved
    # final model won, so none of them is used.
    [string]$CheckpointSteps = "1=900000,2=300000,3=1000000,4=900000,5=700000",
    [int[]]$Variants = @(1, 2, 3, 4, 5)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
$Python = Join-Path $WorkspaceRoot "fdms_reconfiguration\.venv\Scripts\python.exe"
$PoolRoot = Join-Path $WorkspaceRoot $PoolDir
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName
$OutputRoot = Join-Path (Join-Path $ProjectRoot "analysis") $OutputName
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Skipped = @()

$PinnedSteps = @{}
foreach ($Pair in ($CheckpointSteps -split ",")) {
    if ($Pair.Trim() -eq "") { continue }
    $Parts = $Pair -split "="
    if ($Parts.Count -ne 2) { throw "bad -CheckpointSteps entry: $Pair" }
    $PinnedSteps[[int]$Parts[0]] = [int]$Parts[1]
}

foreach ($Variant in $Variants) {
    $TaskDir = Join-Path $RunRoot "spray_lite_$Variant"
    if ($PinnedSteps.ContainsKey($Variant)) {
        $Steps = $PinnedSteps[$Variant]
        $Model = Join-Path (Join-Path $TaskDir "checkpoints") "sac_her_${Steps}_steps.zip"
        Write-Output "spray_lite_${Variant}: using the $Steps-step checkpoint"
    }
    else {
        $Model = Join-Path $TaskDir "model.zip"
    }
    if (-not (Test-Path -LiteralPath $Model)) {
        Write-Output "spray_lite_${Variant}: no model yet, skipping"
        $Skipped += $Variant
        continue
    }
    $Output = Join-Path $OutputRoot "spray_lite_$Variant.npz"
    $Arguments = @(
        "-W", "ignore",
        "-m", "improved_tds.experiments.collect_synergy_rollouts",
        "--env-id", "ShadowHandLiteSpray$Variant-v0",
        "--model", $Model,
        "--pose-pool-dir", $PoolRoot,
        "--output", $Output,
        "--target-fractions", $TargetFractions,
        "--episodes-per-cell", $EpisodesPerCell,
        "--bottle-xy-noise", $BottleXyNoise,
        "--seed", ($Seed + $Variant),
        "--overwrite"
    )
    & $Python @Arguments *>&1 | Tee-Object -FilePath (Join-Path $OutputRoot "spray_lite_$Variant.log")
    if ($LASTEXITCODE -ne 0) {
        throw "rollout collection failed for spray_lite_$Variant"
    }
}

if ($Skipped.Count -gt 0) {
    Write-Output "skipped variants without a trained model: $($Skipped -join ', ')"
}
