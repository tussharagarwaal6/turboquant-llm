# Launch Chrome with remote debugging for Browser Use (CDP attach).
#
# Chrome 136+ silently ignores --remote-debugging-port on the DEFAULT profile
# (security: prevents cookie theft). We use a dedicated persistent profile dir
# instead: browser\chrome-profile (in this repo, writable and persistent).
# Log in once there; cookies/sessions persist across runs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_chrome_cdp.ps1
#
# Optional env:
#   CHROME_CDP_PORT=9222
#   CHROME_USER_DATA_DIR=...   (must NOT be the default Chrome User Data path)
#   CHROME_PROFILE=Default

param(
    [switch]$NoRestart,
    [int]$Port = $(if ($env:CHROME_CDP_PORT) { [int]$env:CHROME_CDP_PORT } else { 9222 }),
    [string]$UserDataDir = $(
        if ($env:CHROME_USER_DATA_DIR) {
            $env:CHROME_USER_DATA_DIR
        } else {
            (Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "browser\chrome-profile")
        }
    ),
    [string]$Profile = $(if ($env:CHROME_PROFILE) { $env:CHROME_PROFILE } else { "Default" })
)

$ErrorActionPreference = "Stop"

$defaultChromeData = "$env:LOCALAPPDATA\Google\Chrome\User Data"
if ($UserDataDir.TrimEnd('\') -eq $defaultChromeData.TrimEnd('\')) {
    Write-Error @"
Chrome 136+ blocks remote debugging on the default profile path:
  $defaultChromeData

Use the dedicated BrowserUse profile (default: browser\chrome-profile) or set
CHROME_USER_DATA_DIR to any non-default writable path.
"@
    exit 1
}

function Test-Cdp {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri "$Url/json/version" -UseBasicParsing -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

$cdpUrl = "http://127.0.0.1:$Port"

if (Test-Cdp $cdpUrl) {
    Write-Host "Chrome CDP already available at $cdpUrl"
    (Invoke-WebRequest -Uri "$cdpUrl/json/version" -UseBasicParsing).Content
    exit 0
}

$chromeCandidates = @(
    "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)

$chrome = $chromeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) {
    Write-Error "Google Chrome not found."
    exit 1
}

New-Item -ItemType Directory -Force -Path $UserDataDir | Out-Null

# Only stop Chrome instances using OUR profile dir (leave daily Chrome alone).
$running = @(Get-Process -Name "chrome" -ErrorAction SilentlyContinue)
if ($running.Count -gt 0 -and -not $NoRestart) {
    Write-Host "Note: If CDP fails, close Chrome windows using profile: $UserDataDir"
}

Write-Host "Chrome:          $chrome"
Write-Host "CDP port:        $Port"
Write-Host "User data dir:   $UserDataDir  (persistent BrowserUse profile)"
Write-Host "Profile:         $Profile"
Write-Host ""

Start-Process -FilePath $chrome -ArgumentList @(
    "--remote-debugging-port=$Port",
    "--remote-debugging-address=127.0.0.1",
    "--user-data-dir=$UserDataDir",
    "--profile-directory=$Profile",
    "--no-first-run"
)

Write-Host "Waiting for CDP at $cdpUrl ..."
for ($i = 0; $i -lt 45; $i++) {
    if (Test-Cdp $cdpUrl) {
        Write-Host "OK: Chrome CDP is up."
        (Invoke-WebRequest -Uri "$cdpUrl/json/version" -UseBasicParsing).Content
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Error "Chrome started but CDP did not respond on $cdpUrl within 45s."
exit 1
