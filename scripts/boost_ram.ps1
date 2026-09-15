# Best-effort Windows RAM reclaim before loading large CPU models.
# Safe to run from WSL via: powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\boost_ram.ps1
#
# Does NOT kill Cursor, WSL, or the user session. Trims working sets of known
# RAM hogs and optionally flushes the standby list when EmptyStandbyList is available.

$ErrorActionPreference = "SilentlyContinue"

function Write-RamLine {
    param([string]$Label, [double]$Bytes)
    $gb = [math]::Round($Bytes / 1GB, 2)
    Write-Host ("  {0,-12} {1,8} GB" -f $Label, $gb)
}

Write-Host "Windows RAM boost (best-effort)..."

# Report before
$osBefore = Get-CimInstance Win32_OperatingSystem
Write-Host "Before:"
Write-RamLine "FreePhysical" $osBefore.FreePhysicalMemory * 1KB
Write-RamLine "TotalVisible" $osBefore.TotalVisibleMemorySize * 1KB

# Trim working sets of safe-to-trim processes (not Cursor, not WSL core)
$trimTargets = @(
    "chrome",
    "msedge",
    "firefox",
    "opera",
    "brave",
    "Discord",
    "Spotify",
    "Teams",
    "OneDrive",
    "LM Studio",
    "ollama",
    "Code"
)

$trimmed = 0
foreach ($name in $trimTargets) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $_.MinWorkingSet = [IntPtr]::Zero
            $_.MaxWorkingSet = [IntPtr]::Zero
            $trimmed++
        } catch {
            # Requires SeDebugPrivilege for some processes; skip quietly.
        }
    }
}
Write-Host "  Trimmed working sets: $trimmed process(es)"

# Optional standby-list flush (Sysinternals EmptyStandbyList)
$eslPaths = @(
    "$env:ProgramFiles\EmptyStandbyList.exe",
    "$env:LOCALAPPDATA\EmptyStandbyList.exe",
    "$env:USERPROFILE\tools\EmptyStandbyList.exe"
)

$esl = $null
foreach ($p in $eslPaths) {
    if (Test-Path $p) {
        $esl = $p
        break
    }
}

if ($esl) {
    try {
        & $esl standbylist 2>$null
        Write-Host "  Standby list flushed via EmptyStandbyList"
    } catch {
        Write-Host "  Standby flush skipped (needs admin or tool unavailable)"
    }
} else {
    Write-Host "  Standby flush skipped (EmptyStandbyList.exe not found)"
}

# Report after
Start-Sleep -Milliseconds 500
$osAfter = Get-CimInstance Win32_OperatingSystem
Write-Host "After:"
Write-RamLine "FreePhysical" $osAfter.FreePhysicalMemory * 1KB
Write-RamLine "TotalVisible" $osAfter.TotalVisibleMemorySize * 1KB

$total = [double]($osAfter.TotalVisibleMemorySize * 1KB)
$free = [double]($osAfter.FreePhysicalMemory * 1KB)
if ($total -gt 0) {
    $usedPct = [math]::Round((($total - $free) / $total) * 100, 0)
    Write-Host ("  Used         {0,8} %" -f $usedPct)
    if ($usedPct -ge 90) {
        Write-Host "  WARN: Host RAM >= 90% - pagefile thrashing likely. Stop model and run setup_wsl_memory.ps1"
    }
}
