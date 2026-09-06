<#
.SYNOPSIS
  One-time Windows setup for the llama.cpp OpenVINO drafting device (Intel NPU / iGPU).

.DESCRIPTION
  Builds llama.cpp on Windows with the OpenVINO and RPC backends so that a small
  draft model can execute on the Intel NPU (or iGPU) and be consumed over RPC by
  the CUDA llama-server running in WSL2.

  The OpenVINO backend has a hard find_package(OpenCL) dependency and OpenVINO's
  own ocl_wrapper.hpp includes the C++ header CL/cl2.hpp, which the Khronos
  OpenCL-Headers repo does NOT ship. This script therefore assembles a minimal
  OpenCL SDK from three Khronos repos before configuring llama.cpp.

  MEASURED RESULT ON THIS MACHINE (Core Ultra 9 275HX, NPU 3720):
    Qwen3-0.6B-Q4_K_M   NPU   prefill 398 t/s   decode 10.1 t/s
    Qwen3-0.6B-Q4_K_M   iGPU  prefill 1128 t/s  decode 19.8 t/s
  Decode throughput is far too low for the drafter to accelerate a CUDA target,
  so this path is EXPERIMENTAL / opt-in only. See README "KAT NPU drafter".

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_npu_drafter.ps1
#>
[CmdletBinding()]
param(
    # Repo and filename of the draft model. Defaults to the KAT-compatible MTP head.
    [string]$DraftRepo = 'ggml-org/Qwen3.6-35B-A3B-GGUF',
    [string]$DraftFile = 'mtp-Qwen3.6-35B-A3B-Q4_0.gguf',
    [int]$RpcPort = 50052,
    # Set to skip the (slow) llama.cpp rebuild when only refreshing weights.
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

$LlamaDir  = Join-Path $env:USERPROFILE 'llamacpp-npu'
$OpenClDir = Join-Path $env:USERPROFILE 'opencl-sdk'
$OpenClPfx = Join-Path $OpenClDir 'install'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "WARN: $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- toolchain ---
Write-Step 'Locating Visual Studio C++ toolchain'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio Build Tools 2022 with the 'Desktop development with C++' workload."
}
$vsPath = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsPath) {
    throw "MSVC x64 C++ tools not installed. Run: winget install Microsoft.VisualStudio.2022.BuildTools --override '--quiet --add Microsoft.VisualStudio.Workload.VCTools'"
}
Write-Host "    VS: $vsPath"

$cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path $cmake)) {
    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
}
if (-not $cmake) { throw 'cmake.exe not found in Visual Studio or on PATH.' }
Write-Host "    cmake: $cmake"

# ----------------------------------------------------------------- openvino ---
Write-Step 'Verifying OpenVINO runtime and NPU visibility'
python -c "import openvino" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '    installing openvino via pip'
    python -m pip install --quiet openvino
}

$ovDir  = python -c "import openvino,os; print(os.path.join(os.path.dirname(openvino.__file__),'cmake'))"
$ovLibs = python -c "import openvino,os; print(os.path.join(os.path.dirname(openvino.__file__),'libs'))"
$devices = python -c "import openvino; print(','.join(openvino.Core().available_devices))"
Write-Host "    OpenVINO devices: $devices"
if ($devices -notmatch 'NPU') {
    Write-Warn 'NPU not visible to OpenVINO. Update the Intel NPU driver (Intel AI Boost in Device Manager).'
    Write-Warn 'The iGPU fallback (GGML_OPENVINO_DEVICE=GPU.0) may still work.'
}

# ------------------------------------------------------------- opencl sdk -----
# find_package(OpenCL) needs headers + an import library; OpenVINO additionally
# needs the C++ bindings (CL/cl2.hpp) which live in a separate Khronos repo.
Write-Step 'Assembling minimal OpenCL SDK'
New-Item -ItemType Directory -Force -Path $OpenClDir | Out-Null
Push-Location $OpenClDir
try {
    foreach ($repo in 'OpenCL-Headers', 'OpenCL-ICD-Loader', 'OpenCL-CLHPP') {
        if (-not (Test-Path $repo)) {
            Write-Host "    cloning $repo"
            git clone --depth 1 "https://github.com/KhronosGroup/$repo" 2>&1 | Out-Null
        }
    }

    if (-not (Test-Path (Join-Path $OpenClPfx 'include\CL\cl.h'))) {
        & $cmake -S OpenCL-Headers -B build-headers -G 'Visual Studio 17 2022' -A x64 "-DCMAKE_INSTALL_PREFIX=$OpenClPfx" | Out-Null
        & $cmake --install build-headers | Out-Null
    }

    if (-not (Test-Path (Join-Path $OpenClPfx 'lib\OpenCL.lib'))) {
        & $cmake -S OpenCL-ICD-Loader -B build-icd -G 'Visual Studio 17 2022' -A x64 `
            "-DCMAKE_INSTALL_PREFIX=$OpenClPfx" "-DCMAKE_PREFIX_PATH=$OpenClPfx" `
            "-DOPENCL_ICD_LOADER_HEADERS_DIR=$OpenClPfx\include" -DBUILD_TESTING=OFF | Out-Null
        & $cmake --build build-icd --config Release -j | Out-Null
        & $cmake --install build-icd --config Release | Out-Null
    }

    # cl2.hpp / opencl.hpp are header-only; OpenVINO's ocl_wrapper.hpp requires them.
    Copy-Item (Join-Path $OpenClDir 'OpenCL-CLHPP\include\CL\cl2.hpp')    -Destination (Join-Path $OpenClPfx 'include\CL') -Force
    Copy-Item (Join-Path $OpenClDir 'OpenCL-CLHPP\include\CL\opencl.hpp') -Destination (Join-Path $OpenClPfx 'include\CL') -Force
}
finally { Pop-Location }
Write-Host "    OpenCL SDK: $OpenClPfx"

# ----------------------------------------------------------------- llama.cpp --
if (-not $SkipBuild) {
    Write-Step 'Building llama.cpp with OPENVINO + RPC backends'
    if (-not (Test-Path $LlamaDir)) {
        git clone --depth 1 https://github.com/ggml-org/llama.cpp $LlamaDir 2>&1 | Out-Null
    }
    Push-Location $LlamaDir
    try {
        & $cmake -B build -G 'Visual Studio 17 2022' -A x64 `
            -DGGML_OPENVINO=ON -DGGML_RPC=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF `
            -DCMAKE_BUILD_TYPE=Release `
            "-DOpenVINO_DIR=$ovDir" `
            "-DOpenCL_INCLUDE_DIR=$OpenClPfx\include" `
            "-DOpenCL_LIBRARY=$OpenClPfx\lib\OpenCL.lib"
        if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }

        # The RPC server target is named ggml-rpc-server, not rpc-server.
        & $cmake --build build --config Release --target ggml-rpc-server llama-cli llama-bench -j
        if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
    }
    finally { Pop-Location }
}

$binDir = Join-Path $LlamaDir 'build\bin\Release'
foreach ($exe in 'ggml-rpc-server.exe', 'llama-cli.exe', 'llama-bench.exe') {
    if (-not (Test-Path (Join-Path $binDir $exe))) { throw "Missing build output: $exe" }
}
Write-Host "    binaries: $binDir"

# -------------------------------------------------------------- draft model ---
Write-Step "Resolving draft model $DraftFile"
$cacheRepo = 'models--' + ($DraftRepo -replace '/', '--')
$snapRoot  = Join-Path $env:USERPROFILE ".cache\huggingface\hub\$cacheRepo\snapshots"
$draftPath = $null
if (Test-Path $snapRoot) {
    $draftPath = Get-ChildItem $snapRoot -Recurse -Filter $DraftFile -ErrorAction SilentlyContinue |
                 Select-Object -First 1 -ExpandProperty FullName
}
if (-not $draftPath) {
    Write-Host "    not cached; downloading"
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    hf download $DraftRepo $DraftFile
    $draftPath = Get-ChildItem $snapRoot -Recurse -Filter $DraftFile -ErrorAction SilentlyContinue |
                 Select-Object -First 1 -ExpandProperty FullName
}
if (-not $draftPath) { throw "Could not resolve $DraftFile. Run: hf download $DraftRepo $DraftFile" }
Write-Host "    draft: $draftPath"

# ----------------------------------------------------------------- firewall ---
Write-Step "Firewall rule for RPC port $RpcPort"
$ruleName = "llamacpp-rpc-$RpcPort"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host '    rule already present'
}
else {
    # The RPC protocol is documented as insecure; scope it to the WSL NAT subnet.
    try {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $RpcPort -RemoteAddress 172.16.0.0/12 -Profile Any | Out-Null
        Write-Host '    created (scoped to 172.16.0.0/12)'
    }
    catch {
        Write-Warn "Could not create firewall rule (needs admin). Run this in an elevated shell:"
        Write-Warn "  New-NetFirewallRule -DisplayName '$ruleName' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $RpcPort -RemoteAddress 172.16.0.0/12"
    }
}

Write-Host ''
Write-Step 'Setup complete'
Write-Host "  Start the drafting device:  powershell -File scripts\npu_drafter.ps1"
Write-Host "  Then in WSL:                bash scripts/switch_model.sh kat-npu"
