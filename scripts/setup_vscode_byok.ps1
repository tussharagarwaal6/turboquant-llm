# Wire turboquant-llm (:8000) into VS Code native Chat via Custom Endpoint BYOK.
#
# Writes user-local config only (AppData) — never to the repo or workspace.
#
# Prerequisites:
#   1. VS Code 1.122+ with GitHub Copilot Chat extension (for BYOK UI)
#   2. Local LLM running: bash scripts/switch_model.sh kat-npu --context 100000
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_vscode_byok.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_vscode_byok.ps1 -Insiders

param(
    [string]$LlmBase = $(if ($env:TURBOQUANT_LLM_BASE) { $env:TURBOQUANT_LLM_BASE } else { "http://127.0.0.1:8000/v1" }),
    [string]$ApiKey = "local",
    [switch]$Insiders,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$TemplatePath = Join-Path $RepoRoot "config\vscode-chatLanguageModels.template.json"
$SettingsRefPath = Join-Path $RepoRoot "config\vscode-settings.local.json"
$ProviderName = "Local (turboquant)"

function Get-VscodeChatModelsPath {
    param([switch]$UseInsiders)
    $appName = if ($UseInsiders) { "Code - Insiders" } else { "Code" }
    # VS Code reads BYOK providers from User\chatLanguageModels.json (array of groups).
    Join-Path $env:APPDATA "$appName\User\chatLanguageModels.json"
}

function Get-ModelsResponse {
    param([string]$BaseUrl)
    $modelsUrl = $BaseUrl.TrimEnd("/") + "/models"
    $resp = $null

    try {
        $resp = Invoke-RestMethod -Uri $modelsUrl -Method Get -TimeoutSec 5
    } catch {
        Write-Host "Windows HTTP timed out; trying WSL curl ..." -ForegroundColor Yellow
        try {
            $raw = wsl curl -sf --max-time 10 $modelsUrl 2>$null
            if ($raw) {
                $resp = $raw | ConvertFrom-Json
            }
        } catch {
            # fall through to error below
        }
    }

    if (-not $resp) {
        Write-Error "LLM not reachable at $modelsUrl. Start the server first:`n  bash scripts/switch_model.sh kat-npu --context 100000"
        exit 1
    }
    return $resp
}

function Get-ActiveModelInfo {
    param([string]$BaseUrl)
    $resp = Get-ModelsResponse -BaseUrl $BaseUrl
    $entry = $null

    if ($resp.data -and $resp.data.Count -gt 0) {
        $entry = $resp.data[0]
    } elseif ($resp.models -and $resp.models.Count -gt 0) {
        $entry = $resp.models[0]
    }

    if (-not $entry) {
        Write-Error "Unexpected /v1/models response from $($BaseUrl.TrimEnd('/') + '/models')"
        exit 1
    }

    $id = $null
    if ($entry.id) { $id = [string]$entry.id }
    elseif ($entry.name) { $id = [string]$entry.name }
    elseif ($entry.model) { $id = [string]$entry.model }

    $ctx = 100000
    if ($entry.meta -and $entry.meta.n_ctx) {
        $ctx = [int]$entry.meta.n_ctx
    }

    return [PSCustomObject]@{
        Id = $id
        ContextWindow = $ctx
    }
}

function Format-LocalModelName {
    param([string]$ModelId)
    if ([string]::IsNullOrWhiteSpace($ModelId)) { return "Local model" }
    $label = ($ModelId -replace "[/_]", " ").Trim()
    return ($label.Substring(0, 1).ToUpper() + $label.Substring(1)) + " (local)"
}

function Read-JsonArray {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return @()
    }
    $raw = Get-Content -Path $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return @()
    }
    $parsed = $raw | ConvertFrom-Json
    if ($parsed -is [System.Array]) {
        return @($parsed)
    }
    # Legacy single-object file from an earlier script version.
    return @($parsed)
}

function Write-JsonArray {
    param(
        [string]$Path,
        [array]$Items
    )
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    # ConvertTo-Json drops array brackets for single-element arrays; wrap explicitly.
    $json = ConvertTo-Json -InputObject @($Items) -Depth 20
    [System.IO.File]::WriteAllText($Path, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}

function Apply-ModelRuntime {
    param(
        [object]$Provider,
        [string]$ActiveModelId,
        [int]$ContextWindow
    )
    $clone = $Provider | ConvertTo-Json -Depth 20 | ConvertFrom-Json
    if ($clone.models -and $clone.models.Count -gt 0) {
        $model = $clone.models[0]
        $model.id = $ActiveModelId
        $model.name = Format-LocalModelName -ModelId $ActiveModelId
        $model.contextWindow = $ContextWindow
        # No maxOutputTokens: let VS Code and the server use the full context for long code gen.
        if ($model.PSObject.Properties["maxOutputTokens"]) {
            $model.PSObject.Properties.Remove("maxOutputTokens")
        }
    }
    return $clone
}

function Merge-ProviderEntry {
    param(
        [array]$Existing,
        [object]$TemplateProvider,
        [string]$ActiveModelId,
        [int]$ContextWindow
    )
    $merged = @()
    $replaced = $false
    foreach ($entry in $Existing) {
        if ($entry.name -eq $ProviderName) {
            $replaced = $true
            $merged += Apply-ModelRuntime -Provider $TemplateProvider -ActiveModelId $ActiveModelId -ContextWindow $ContextWindow
        } else {
            $merged += $entry
        }
    }
    if (-not $replaced) {
        $merged = @(Apply-ModelRuntime -Provider $TemplateProvider -ActiveModelId $ActiveModelId -ContextWindow $ContextWindow) + $merged
    }
    return ,$merged
}

if (-not (Test-Path $TemplatePath)) {
    Write-Error "Template not found: $TemplatePath"
    exit 1
}

Write-Host "Checking local LLM at $LlmBase ..."
$activeModel = Get-ActiveModelInfo -BaseUrl $LlmBase
Write-Host "Active model: $($activeModel.Id) (context $($activeModel.ContextWindow), no output cap)" -ForegroundColor Green

$templateRaw = Get-Content -Path $TemplatePath -Raw -Encoding UTF8
$templateProviders = @($templateRaw | ConvertFrom-Json)
$templateProvider = $templateProviders | Where-Object { $_.name -eq $ProviderName } | Select-Object -First 1
if (-not $templateProvider) {
    Write-Error "Template missing provider '$ProviderName' in $TemplatePath"
    exit 1
}

$targets = @()
if ($Insiders) {
    $targets += @{ Label = "VS Code Insiders"; Path = (Get-VscodeChatModelsPath -UseInsiders) }
} else {
    $targets += @{ Label = "VS Code"; Path = (Get-VscodeChatModelsPath) }
    $targets += @{ Label = "VS Code Insiders"; Path = (Get-VscodeChatModelsPath -UseInsiders) }
}

foreach ($target in $targets) {
    $destPath = $target.Path
    if ((Test-Path $destPath) -and -not $Force) {
        $existing = Read-JsonArray -Path $destPath
        $hasProvider = $existing | Where-Object { $_.name -eq $ProviderName }
        if ($hasProvider) {
            Write-Host "[$($target.Label)] Provider already present - syncing model and context." -ForegroundColor Yellow
        }
    }

    $existing = Read-JsonArray -Path $destPath
    $merged = Merge-ProviderEntry -Existing $existing -TemplateProvider $templateProvider -ActiveModelId $activeModel.Id -ContextWindow $activeModel.ContextWindow
    Write-JsonArray -Path $destPath -Items $merged
    Write-Host "[$($target.Label)] Wrote $destPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next steps in VS Code:" -ForegroundColor Cyan
Write-Host "  1. Reload Window (Developer: Reload Window)"
Write-Host "  2. Open Chat -> model picker -> Manage Language Models (gear)"
Write-Host "  3. When prompted for turboquantApiKey, enter: $ApiKey"
Write-Host "  4. Pin $($activeModel.Id) (local) and hide cloud Copilot models (eye icon)"
Write-Host "  5. Switch models mid-chat from the picker dropdown"
Write-Host ""
Write-Host "Optional user settings (copy from config\vscode-settings.local.json into"
Write-Host "  %APPDATA%\Code\User\settings.json):"
if (Test-Path $SettingsRefPath) {
    Get-Content $SettingsRefPath | ForEach-Object { Write-Host "  $_" }
}
Write-Host ""
Write-Host "After switching backend model (switch_model.sh), re-run this script or Reload Window."
