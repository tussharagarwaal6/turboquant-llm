# Start Chrome CDP + Browser Use MCP for OpenCode (one command).
#
# Prerequisites:
#   - LLM on :8000 (WSL): bash scripts/switch_model.sh qwen35-4b
#   - Installed once: scripts\install_browser_use_mcp.ps1
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_browser_stack.ps1

$ErrorActionPreference = "Stop"
$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Step 1/2: Chrome with CDP ==="
& "$ScriptsDir\start_chrome_cdp.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Step 2/2: Browser Use MCP ==="
& "$ScriptsDir\start_browser_use_mcp.ps1"
