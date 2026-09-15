# Install Saik0s Browser Use MCP (HTTP daemon) via uv.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_browser_use_mcp.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$McpDir = Join-Path $RepoRoot "browser\mcp-browser-use"

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;" + $env:Path
}

Ensure-Uv

if (-not (Test-Path $McpDir)) {
    Write-Host "Cloning Saik0s/mcp-browser-use..."
    git clone --depth 1 https://github.com/Saik0s/mcp-browser-use.git $McpDir
}

Push-Location $McpDir
try {
    Write-Host "Syncing Python dependencies (uv sync)..."
    uv sync
    Write-Host "Installing Playwright Chromium..."
    uv run playwright install chromium
    Write-Host ""
    Write-Host "Installed. Verify:"
    Write-Host "  cd browser\mcp-browser-use"
    Write-Host "  uv run mcp-server-browser-use --help"
} finally {
    Pop-Location
}
