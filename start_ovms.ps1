$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $repoRoot "models\config.json"
$defaultOvmsExe = Join-Path $repoRoot "ovms\bin\ovms.exe"

if (-not (Test-Path $configPath)) {
    throw "Missing OVMS config: $configPath. Run `python setup_ovms.py` first."
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

Write-Host "Starting OVMS with config $configPath"
& $ovmsExe --config_path $configPath --rest_port 8000
