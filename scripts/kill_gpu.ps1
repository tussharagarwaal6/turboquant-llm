# Stop GPU-using apps for TurboQuant local inference (Windows + WSL2).

$ErrorActionPreference = "SilentlyContinue"

Write-Host "Stopping WSL TurboQuant server..."
wsl -d Ubuntu bash /mnt/c/dev/turboquant-llm/scripts/kill_gpu.sh

Write-Host "`nStopping common Windows GPU apps..."
$gpuApps = @(
    "LM Studio.exe",
    "ollama.exe",
    "ollama app.exe"
)
foreach ($app in $gpuApps) {
    if (Get-Process -Name ($app -replace '\.exe$','') -ErrorAction SilentlyContinue) {
        Write-Host "  Killing $app"
        taskkill /F /IM $app | Out-Null
    }
}

Write-Host "`nWindows GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
Write-Host "`nWindows GPU compute processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
