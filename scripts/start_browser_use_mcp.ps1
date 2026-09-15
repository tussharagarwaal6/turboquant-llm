# Start Browser Use MCP HTTP daemon for OpenCode (uses existing Chrome via CDP).
#
# Prerequisites:
#   1. Install once: powershell -ExecutionPolicy Bypass -File scripts\install_browser_use_mcp.ps1
#   2. Chrome with CDP: scripts\start_chrome_cdp.ps1  (close all Chrome first)
#   3. Local LLM on :8000: bash scripts/switch_model.sh qwen35-4b
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_browser_use_mcp.ps1
#
# OpenCode connects to: http://127.0.0.1:8383/mcp

param(
    [string]$CdpUrl = $(if ($env:BROWSER_CDP_URL) { $env:BROWSER_CDP_URL } else { "http://127.0.0.1:9222" }),
    [string]$LlmBase = $(if ($env:MCP_LLM_BASE_URL) { $env:MCP_LLM_BASE_URL } elseif ($env:OPENAI_API_BASE) { $env:OPENAI_API_BASE } else { "http://127.0.0.1:8000/v1" }),
    [string]$LlmModel = $(if ($env:MCP_LLM_MODEL_NAME) { $env:MCP_LLM_MODEL_NAME } elseif ($env:BROWSER_LLM_MODEL) { $env:BROWSER_LLM_MODEL } else { "qwen35-4b" }),
    [int]$Port = $(if ($env:MCP_SERVER_PORT) { [int]$env:MCP_SERVER_PORT } else { 8383 })
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$McpDir = Join-Path $RepoRoot "browser\mcp-browser-use"

function Test-Cdp {
    try {
        $r = Invoke-WebRequest -Uri "$CdpUrl/json/version" -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Path $McpDir)) {
    Write-Error "browser\mcp-browser-use not found. Run: scripts\install_browser_use_mcp.ps1"
    exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $uvBin = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $uvBin) {
        $env:Path = "$(Split-Path $uvBin);$env:Path"
    } else {
        Write-Error "uv not found. Run: scripts\install_browser_use_mcp.ps1"
        exit 1
    }
}

if (-not (Test-Cdp)) {
    Write-Host "WARNING: Chrome CDP not reachable at $CdpUrl" -ForegroundColor Yellow
    Write-Host "Starting MCP anyway (OpenCode can connect). Browser tasks need CDP:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start_chrome_cdp.ps1" -ForegroundColor Yellow
    Write-Host ""
}

$env:MCP_LLM_PROVIDER = "openai"
$env:MCP_LLM_MODEL_NAME = $LlmModel
$env:MCP_LLM_BASE_URL = $LlmBase
$env:OPENAI_API_KEY = $(if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "local" })
$env:MCP_BROWSER_CDP_URL = $CdpUrl
$env:MCP_BROWSER_HEADLESS = "false"
$env:MCP_AGENT_MAX_STEPS = $(if ($env:BROWSER_MAX_STEPS) { $env:BROWSER_MAX_STEPS } else { "25" })
$env:MCP_AGENT_USE_VISION = $(if ($env:BROWSER_USE_VISION) { $env:BROWSER_USE_VISION } else { "false" })
$env:MCP_SERVER_PORT = "$Port"
$env:MCP_SERVER_HOST = "127.0.0.1"

Write-Host "Browser Use MCP daemon (Saik0s / uv)"
Write-Host "  CDP URL:    $CdpUrl"
Write-Host "  LLM:        $LlmModel @ $LlmBase"
Write-Host "  MCP URL:    http://127.0.0.1:$Port/mcp"
Write-Host "  Dashboard:  http://127.0.0.1:$Port/dashboard"
Write-Host ""
Write-Host "Leave this window open. Restart OpenCode after the daemon starts."
Write-Host ""

Push-Location $McpDir
try {
    uv run mcp-server-browser-use server -f
} finally {
    Pop-Location
}
