param(
    [ValidateSet("GPU", "CPU")]
    [string]$TargetDevice = "GPU"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $repoRoot "models\config.json"
$defaultOvmsExe = Join-Path $repoRoot "ovms\bin\ovms.exe"
$pythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"
$setupScript = Join-Path $repoRoot "setup_ovms.py"

if (-not (Test-Path $pythonExe)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "Python was not found. Activate the venv or add python to PATH."
    }
    $pythonExe = $pythonCommand.Source
}

if (Test-Path $defaultOvmsExe) {
    $ovmsExe = $defaultOvmsExe
} else {
    $ovmsCommand = Get-Command ovms -ErrorAction SilentlyContinue
    if ($null -eq $ovmsCommand) {
        throw "OVMS binary was not found. Run .\ovms\setupvars.ps1 or add ovms to PATH."
    }
    $ovmsExe = $ovmsCommand.Source
}

$env:OVMS_TARGET_DEVICE = $TargetDevice

Write-Host "Preparing OVMS assets for target device $TargetDevice"
& $pythonExe $setupScript
if ($LASTEXITCODE -ne 0) {
    throw "setup_ovms.py failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $configPath)) {
    throw "Missing OVMS config: $configPath. setup_ovms.py did not generate it."
}

Write-Host "Starting OVMS with config $configPath on target device $TargetDevice"
& $ovmsExe --config_path $configPath --rest_port 8000
