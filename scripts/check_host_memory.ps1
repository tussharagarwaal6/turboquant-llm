# Report Windows host memory pressure (for WSL preflight).
# Outputs KEY=VALUE lines parseable from bash.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_host_memory.ps1
# Exit 1 if host used >= 90% or free < 6 GB.

param(
    [int]$MinFreeGB = 6,
    [int]$MaxUsedPct = 90
)

$ErrorActionPreference = "SilentlyContinue"
$os = Get-CimInstance Win32_OperatingSystem
$total = [double]($os.TotalVisibleMemorySize * 1KB)
$free = [double]($os.FreePhysicalMemory * 1KB)
$used = $total - $free
$usedPct = if ($total -gt 0) { ($used / $total) * 100.0 } else { 0.0 }

$totalGB = [math]::Round($total / 1GB, 1)
$freeGB = [math]::Round($free / 1GB, 1)
$usedGB = [math]::Round($used / 1GB, 1)
$usedPctR = [math]::Round($usedPct, 0)

Write-Output "HOST_TOTAL_GB=$totalGB"
Write-Output "HOST_FREE_GB=$freeGB"
Write-Output "HOST_USED_GB=$usedGB"
Write-Output "HOST_USED_PCT=$usedPctR"

if ($freeGB -lt $MinFreeGB -or $usedPctR -ge $MaxUsedPct) {
    Write-Output "HOST_PRESSURE=high"
    exit 1
}
Write-Output "HOST_PRESSURE=ok"
exit 0
