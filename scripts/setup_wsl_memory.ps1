# Size WSL2 memory/swap for large models WITHOUT starving Windows.
#
# On a 64 GB machine: WSL gets 48 GB, Windows keeps ~16 GB headroom.
# Over-allocating WSL (e.g. 56 GB) causes host thrashing and 100% disk pagefile I/O.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_memory.ps1
#   wsl --shutdown

param(
    [int]$WindowsReserveGB = 16,
    [int]$MaxWslGB = 0,
    [int]$SwapGB = 4
)

$ErrorActionPreference = "Stop"
$configPath = Join-Path $env:USERPROFILE ".wslconfig"

$totalBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$hostTotalGB = [math]::Floor($totalBytes / 1GB)

if ($MaxWslGB -le 0) {
    $wslGB = $hostTotalGB - $WindowsReserveGB
    if ($wslGB -lt 32) { $wslGB = 32 }
} else {
    $wslGB = $MaxWslGB
}

if ($wslGB -ge $hostTotalGB) {
    Write-Warning "WSL memory ($wslGB GB) >= host RAM ($hostTotalGB GB). Lowering by ${WindowsReserveGB} GB reserve."
    $wslGB = [math]::Max(32, $hostTotalGB - $WindowsReserveGB)
}

$content = @"
[wsl2]
memory=${wslGB}GB
swap=${SwapGB}GB
"@

Set-Content -Path $configPath -Value $content -Encoding UTF8

Write-Host "Host RAM:     $hostTotalGB GB"
Write-Host "WSL memory:   ${wslGB} GB  (Windows reserve ~${WindowsReserveGB} GB)"
Write-Host "WSL swap:     ${SwapGB} GB"
Write-Host ""
Write-Host "Wrote $configPath :"
Get-Content $configPath | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Next steps (required):"
Write-Host "  1. wsl --shutdown"
Write-Host "  2. Reopen WSL"
Write-Host "  3. bash scripts/switch_model.sh llama33 --context 8192"
Write-Host ""
Write-Host "Verify in WSL: grep MemTotal /proc/meminfo  (expect ~$($wslGB * 1024 * 1024) kB or close)"
