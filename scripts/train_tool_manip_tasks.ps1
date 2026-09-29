param(
    [int]$TotalTimesteps = 100000,
    [int]$CheckpointFreq = 100000,
    [int]$BaseSeed = 1000,
    [string]$RunName = "tool_manip_100k"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot "..\.__venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$RunRoot = Join-Path (Join-Path $ProjectRoot "runs") $RunName
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$Tasks = @(
    @{ Name = "scissors_lite_1"; EnvId = "ScissorsVariant1-v0" },
    @{ Name = "scissors_lite_2"; EnvId = "ScissorsVariant2-v0" },
    @{ Name = "scissors_lite_3"; EnvId = "ScissorsVariant3-v0" },
    @{ Name = "scissors_lite_4"; EnvId = "ScissorsVariant4-v0" },
    @{ Name = "scissors_lite_5"; EnvId = "ScissorsVariant5-v0" },
    @{ Name = "slider_lite_1"; EnvId = "SliderVariant1-v0" },
    @{ Name = "slider_lite_2"; EnvId = "SliderVariant2-v0" },
    @{ Name = "slider_lite_3"; EnvId = "SliderVariant3-v0" },
    @{ Name = "slider_lite_4"; EnvId = "SliderVariant4-v0" },
    @{ Name = "slider_lite_5"; EnvId = "SliderVariant5-v0" },
    @{ Name = "spray_lite_push"; EnvId = "ShadowHandLitePushSpray-v0" }
)

for ($Index = 0; $Index -lt $Tasks.Count; $Index++) {
    $Task = $Tasks[$Index]
    $TaskDir = Join-Path $RunRoot $Task.Name
    New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null
    Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "running"

    $Arguments = @(
        "-m", "improved_tds.experiments.train_sb3",
        "--env-id", $Task.EnvId,
        "--total-timesteps", $TotalTimesteps,
        "--seed", ($BaseSeed + $Index),
        "--output", (Join-Path $TaskDir "model"),
        "--dataset", (Join-Path $TaskDir "training_candidates.npz"),
        "--checkpoints-dir", (Join-Path $TaskDir "checkpoints"),
        "--checkpoint-freq", $CheckpointFreq
    )
    try {
        & $Python @Arguments *>&1 | Tee-Object -FilePath (Join-Path $TaskDir "train.log")
        if ($LASTEXITCODE -ne 0) {
            throw "training exited with code $LASTEXITCODE"
        }
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "completed"
    }
    catch {
        $_ | Out-File -Append -FilePath (Join-Path $TaskDir "train.log")
        Set-Content -LiteralPath (Join-Path $TaskDir "status.txt") -Value "failed"
        throw
    }
}
