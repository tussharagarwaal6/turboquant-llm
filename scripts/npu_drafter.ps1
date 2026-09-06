<#
.SYNOPSIS
  Start the llama.cpp RPC server exposing the Intel NPU (or iGPU) as a drafting device.

.DESCRIPTION
  This process loads no model of its own -- it only exposes an OpenVINO-backed ggml
  device over TCP. The WSL2 llama-server attaches to it with
  --rpc <host>:<port> --spec-draft-device RPC0, so the same running drafter can be
  reused across target models.

  Leave this window open. Stop with Ctrl+C.

  Device notes:
    NPU    Static graph, stateless only. GGML_OPENVINO_CACHE_DIR is NOT supported
           and setting it makes model load crash.
    GPU.0  Intel Arc/Xe iGPU. Faster decode than the NPU on this machine.
    GPU.1  On this host this enumerates the RTX 5080 via OpenVINO; do not use it,
           the dGPU is already driven by CUDA from WSL.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\npu_drafter.ps1
  powershell -ExecutionPolicy Bypass -File scripts\npu_drafter.ps1 -Device GPU.0
#>
[CmdletBinding()]
param(
    [ValidateSet('NPU', 'GPU.0', 'GPU.1', 'CPU')]
    [string]$Device = 'NPU',
    [int]$Port = 50052,
    [string]$BindHost = '0.0.0.0',
    # NPU prefill is chunked with a fixed static shape; tune only if prefill fails.
    [int]$PrefillChunkSize = 256
)

$ErrorActionPreference = 'Stop'

$LlamaDir = Join-Path $env:USERPROFILE 'llamacpp-npu'
$binDir   = Join-Path $LlamaDir 'build\bin\Release'
$rpcExe   = Join-Path $binDir 'ggml-rpc-server.exe'

if (-not (Test-Path $rpcExe)) {
    Write-Host "Missing $rpcExe" -ForegroundColor Red
    Write-Host 'Run first: powershell -ExecutionPolicy Bypass -File scripts\setup_npu_drafter.ps1'
    exit 1
}

# The OpenVINO runtime DLLs ship inside the pip package and are not on PATH by
# default; without them the process dies instantly with STATUS_DLL_NOT_FOUND.
$ovLibs = python -c "import openvino,os; print(os.path.join(os.path.dirname(openvino.__file__),'libs'))"
$env:PATH = "$ovLibs;$env:USERPROFILE\opencl-sdk\install\bin;$env:PATH"

$env:GGML_OPENVINO_DEVICE = $Device
$env:GGML_OPENVINO_PREFILL_CHUNK_SIZE = "$PrefillChunkSize"

# Caching is unsupported on the NPU and causes a hard crash during model load.
if ($Device -eq 'NPU') {
    Remove-Item Env:\GGML_OPENVINO_CACHE_DIR -ErrorAction SilentlyContinue
}
else {
    $env:GGML_OPENVINO_CACHE_DIR = Join-Path $env:USERPROFILE '.cache\ov_llamacpp'
}

$wslHint = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.InterfaceAlias -like '*WSL*' } |
            Select-Object -First 1 -ExpandProperty IPAddress)

Write-Host 'llama.cpp OpenVINO drafting device' -ForegroundColor Cyan
Write-Host "  GGML_OPENVINO_DEVICE = $Device"
Write-Host "  prefill chunk        = $PrefillChunkSize"
Write-Host "  listening            = ${BindHost}:$Port"
if ($wslHint) { Write-Host "  WSL should reach this at $wslHint`:$Port" }
Write-Host ''
Write-Host 'Expect a device line naming OpenVINO below. If it says CPU, the NPU was' -ForegroundColor Yellow
Write-Host 'not selected and drafting will be slower than the CUDA target.' -ForegroundColor Yellow
Write-Host ''

& $rpcExe --host $BindHost --port $Port
