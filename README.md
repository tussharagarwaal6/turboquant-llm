# TurboQuant vLLM Local Deployment

OpenAI-compatible local LLM server: **Qwen3-14B-AWQ** on vLLM with **TurboQuant** KV-cache compression (`turboquant_k8v4`), running in WSL2 Ubuntu with RTX 5080 GPU passthrough.

**Agent UI:** pair with [open-webui](https://github.com/tussharagarwaal6/openwebui) for web search, RAG, and code tools.

---

## Quick start (daily use)

After [one-time setup](#step-1--wsl2--gpu-verification) below, run the stack like this:

### 1. Start the LLM (this repo)

**WSL Ubuntu** — leave this terminal open:

```bash
cd /mnt/c/dev/turboquant-llm
bash scripts/serve.sh
```

Wait for `Application startup complete` (~1–3 min).

Verify:

```bash
curl http://localhost:8000/v1/models
bash scripts/test_tool_call.sh   # should print PASS
```

### 2. Start Open WebUI (Windows)

**PowerShell** (Docker Desktop running):

```powershell
cd c:\dev\open-webui
.\start.ps1
```

Open **http://localhost:3000** → enable **Web Search** and **Code Interpreter** in the chat Integrations (+) menu.

Full UI instructions: [open-webui README](https://github.com/tussharagarwaal6/openwebui#step-by-step-run-llm--open-webui).

### 3. Stop when done

| Component | Command |
|---|---|
| Open WebUI | `cd c:\dev\open-webui; .\stop.ps1` |
| TurboQuant | `Ctrl+C` in the WSL terminal, or `bash scripts/kill_gpu.sh` |

---

## One-time setup

### Prerequisites

- Windows 11, NVIDIA driver **>= 570** (verify: `nvidia-smi` on Windows)
- WSL2 with Ubuntu (verify: `wsl -l -v`)

## Step 1 — WSL2 + GPU verification

**Windows (PowerShell, admin if installing WSL):**

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --set-default-version 2
```

**Windows — confirm driver and GPU:**

```powershell
nvidia-smi
```

Expect RTX 5080 and driver >= 570.

**WSL2 Ubuntu — confirm GPU passthrough (do NOT install a Linux NVIDIA driver inside WSL):**

```bash
wsl -d Ubuntu
nvidia-smi
```

## Step 2 — Python environment (WSL2)

```bash
cd /mnt/c/dev/turboquant-llm

# Option A (recommended in plan): system packages
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-pip build-essential git curl
python3.12 -m venv .venv
source .venv/bin/activate

# Option B (if python3.12-venv unavailable): uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# ~/.local/bin/uv venv .venv --python 3.12
# source .venv/bin/activate
```

## Step 3 — PyTorch cu128 (>= 2.7.0, sm_120)

```bash
source /mnt/c/dev/turboquant-llm/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Verify sm_120:**

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('arch:', torch.cuda.get_arch_list()); assert 'sm_120' in torch.cuda.get_arch_list(), 'sm_120 missing'"
```

## Step 4 — vLLM >= 0.18.0 + FastAPI

```bash
pip install -r requirements.txt
```

**Verify TurboQuant dtype is supported:**

```bash
python -c "from vllm.config import CacheConfig; print('turboquant_k8v4' in CacheConfig.kv_cache_dtype.__args__ if hasattr(CacheConfig.kv_cache_dtype,'__args__') else 'check manually')"
```

If pip reports a **torch version conflict**, stop and resolve before changing versions.

## Step 5 — Download model

```bash
pip install "huggingface_hub[cli]"
hf download Qwen/Qwen3-14B-AWQ
```

Optional login for gated models: `hf auth login`

## Step 6 — Launch server

From WSL2 (recommended — use `scripts/serve.sh`):

```bash
cd /mnt/c/dev/turboquant-llm
bash scripts/serve.sh --context 32768
```

Other context lengths:

```bash
bash scripts/serve.sh 16384          # positional
bash scripts/serve.sh --context 8192
MAX_MODEL_LEN=32768 bash scripts/serve.sh
bash scripts/serve.sh --help
```

Engine args (via `serve.sh` env / flags):

- `model=Qwen/Qwen3-14B-AWQ`
- `quantization=awq`
- `kv_cache_dtype=turboquant_k8v4`
- `gpu_memory_utilization=0.88` (`--gpu-mem`)
- `max_model_len=16384` (`--context` / `MAX_MODEL_LEN`)
- `kv_offloading_size=8.0` (`--kv-offload`)
- `max_num_seqs=2`

`/v1/models` reports `max_model_len` and `context_length` for clients (e.g. Cursor).

First startup loads weights (~1–3 min). **Restart required** after changing context length.

## Step 7 — Verification

**From Windows PowerShell:**

```powershell
curl http://localhost:8000/v1/models
```

```powershell
curl -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen/Qwen3-14B-AWQ","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

**Streaming test:**

```powershell
curl -N -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen/Qwen3-14B-AWQ","messages":[{"role":"user","content":"Count to 3."}],"max_tokens":32,"stream":true}'
```

**Tool calling test (required for Open WebUI agent tools):**

Must be **POST**, not GET. From WSL:

```bash
bash scripts/test_tool_call.sh
```

Expect `finish_reason: "tool_calls"` and `search_web` in the response. Parser defaults to Hermes (`TOOL_CALL_PARSER=hermes`).

If you see `{"detail":"Method Not Allowed"}`, you hit the endpoint without `-X POST` (e.g. browser address bar).

### Connect any OpenAI-compatible chat client (Windows)

| Setting | Value |
|---------|-------|
| Base URL / API URL | `http://localhost:8000/v1` |
| API Key | any non-empty string (e.g. `local`) |
| Model | `Qwen/Qwen3-14B-AWQ` |

Works with Open WebUI, Chatbox, Jan, Continue, etc.

## Qwythos-9B GGUF (vision + reasoning + tools on :8000)

Alternative to Qwen3 on the **same port** (`8000`). Uses [empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF) via **llama.cpp** (default) or experimental **vLLM + vllm-gguf-plugin**. Supports **text, images (OCR/describe), reasoning, and tool calling**. Does **not** use TurboQuant.

**Note:** vLLM GGUF currently fails with `Unknown gguf model_type: qwen3_5` on plugin 0.0.5. The default `serve_qwythos.sh` uses **llama.cpp**, which is the [official model-card path](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF) for vision.

**VRAM:** only one `:8000` server at a time — Qwen3 **or** Qwythos, not both.

### One-time setup (WSL2)

```bash
source ~/turboquant-llm/.venv/bin/activate
pip install -r requirements-gguf.txt   # optional; only for QWYTHOS_RUNTIME=vllm experiments
bash scripts/download_qwythos.sh
```

Install llama.cpp (one-time, if not already installed):

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Downloads ~6.5 GB (Q4_K_M + mmproj vision encoder).

### Launch

```bash
bash scripts/kill_gpu.sh
bash scripts/serve_qwythos.sh --context 16384
```

**1M context** (YaRN baked into GGUF; uses CPU RAM for KV offload — needs substantial system RAM):

On a **64 GB** host, raise WSL memory first (one-time). Create or edit `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=56GB
swap=16GB
```

Then restart WSL (closes all WSL terminals):

```powershell
wsl --shutdown
```

Re-open Ubuntu, then:

```bash
bash scripts/kill_gpu.sh
bash scripts/serve_qwythos.sh --context 1048576
```

Startup may take several minutes while the KV cache is allocated.

**Switch back to Qwen3:**

```bash
bash scripts/switch_model.sh qwen --context 32768
```

Or use the helper:

```bash
bash scripts/switch_model.sh qwythos --context 16384
bash scripts/switch_model.sh qwen --context 32768
```

### Connect Open WebUI (same URL as Qwen3)

| Setting | Value |
|---------|-------|
| Base URL / API URL | `http://localhost:8000/v1` |
| API Key | any non-empty string (e.g. `local`) |
| Model | `qwythos-9b` |

Refresh models in Admin after starting Qwythos. Attach images in chat via **+** for describe/OCR.

### Verification

```bash
python scripts/check_qwythos.py
```

Recommended sampling (from model card): temperature 0.6, top_p 0.95, top_k 20. Reasoning may appear in a `reasoning` field.

### Qwythos troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing GGUF` on startup | Run `bash scripts/download_qwythos.sh` |
| GGUF load / weight mapping error | vLLM path blocked for qwen3_5 today; use default llama.cpp (`bash scripts/serve_qwythos.sh`) |
| Try experimental vLLM GGUF | `QWYTHOS_RUNTIME=vllm bash scripts/serve_qwythos.sh` (may fail until plugin adds Qwen3.5) |
| Model can't see images | Ensure mmproj is in the same folder as the text GGUF (`~/models/qwythos/`) |
| Open WebUI ignores images | Confirm model is `qwythos-9b`, not `Qwen/Qwen3-14B-AWQ` (text-only) |
| vLLM GGUF unstable | Default runtime is llama.cpp; see `scripts/serve_qwythos_llama.sh` |

### Context compaction (auto-summarize when near limit)

When a chat approaches the model context window, older turns are **summarized** and replaced with a `[CONVERSATION SUMMARY]` message so the conversation can continue without hitting the token cap.

**Qwythos (llama.cpp):** enabled by default. `serve_qwythos_llama.sh` starts llama on `:8002` and a lightweight proxy on `:8000` that compacts before forwarding.

```bash
# Default — compaction on (Open WebUI still uses :8000)
bash scripts/serve_qwythos.sh --context 32768

# Disable compaction
ENABLE_CONTEXT_COMPACTION=0 bash scripts/serve_qwythos.sh --context 32768
```

**TurboQuant Qwen3 (`app/server.py`):** compaction runs inside the vLLM server before each request (no extra proxy).

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_CONTEXT_COMPACTION` | `1` | Turn compaction on/off |
| `CONTEXT_COMPACTION_THRESHOLD_RATIO` | `0.85` | Compact when prompt exceeds this fraction of max context |
| `CONTEXT_COMPACTION_RETENTION_PERCENT` | `40` | Keep the newest ~40% of non-system messages verbatim |
| `CONTEXT_COMPACTION_MAX_SUMMARY_TOKENS` | `1024` | Max tokens for the summary generation call |
| `CONTEXT_COMPACTION_RESERVE_TOKENS` | `512` | Headroom below threshold before triggering |
| `ENABLE_TOOL_PRUNING` | `1` | Collapse completed tool turns into compact `[TOOL RESULT]` messages |
| `TOOL_PRUNE_MAX_RESULT_CHARS` | `2000` | Max chars kept per tool result after pruning |

**Tool pruning:** after a tool runs, the proxy/server replaces `assistant(tool_calls) + tool(result)` with a short `[TOOL RESULT: tool_name]` user message containing only the useful answer text. This saves context while keeping tool calling working. The active in-flight tool turn is left intact.

Compaction adds one extra model call when it runs (summary generation). Open WebUI RAG still injects retrieved chunks separately — compaction only affects the chat history sent to the model.

**Image OCR in Open WebUI:** pair with the [open-webui](https://github.com/tussharagarwaal6/openwebui) **Image OCR** tool (`tools/image_ocr_tool.py`), which calls Qwythos vision for chat image text extraction. PDF/scanned-document OCR still uses Tika in Open WebUI Knowledge.

## KAT-Coder-V2.5-Dev EXL3 (TabbyAPI on :8000)

Agentic coding model served from [P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq](https://huggingface.co/P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq) (~24 GB EXL3, 5.08 bpw + MTP draft head).

**Important:** This checkpoint is **EXL3 for ExLlamaV3**, not AWQ/NVFP4. It **cannot** use the vLLM TurboQuant server (`turboquant_k8v4`). Instead it runs via **TabbyAPI** with **Q8 KV cache** (ExLlamaV3’s KV compression). Tool format: `qwen3_coder`.

**VRAM:** only one `:8000` server at a time — Qwen3, Qwythos, **or** KAT. The quantizer recommends **2×24 GB**; on a **16 GB** RTX 5080 we offload MoE experts to CPU and start with `--context 8192`–`16384`.

### One-time setup (WSL2)

```bash
hf download P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq
bash scripts/setup_tabbyapi.sh
```

Weights are read from the Windows HF cache (`/mnt/c/Users/.../.cache/huggingface/hub/...`).

### Launch

```bash
bash scripts/kill_gpu.sh
bash scripts/switch_model.sh kat --context 16384
```

Startup may take several minutes while ~24 GB loads.

Verify:

```bash
curl http://localhost:8000/v1/models
MODEL=KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq bash scripts/test_tool_call.sh
```

**Switch back:**

```bash
bash scripts/switch_model.sh qwen --context 32768
bash scripts/switch_model.sh qwythos --context 16384
```

### Connect Open WebUI

| Setting | Value |
|---------|-------|
| Base URL / API URL | `http://localhost:8000/v1` |
| API Key | any non-empty string (e.g. `local`) |
| Model | `KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq` |

For reasoning, enable thinking in the request (`enable_thinking: true`). Default server config disables thinking for cleaner tool use.

### Long context (~200k)

**Intel GPU offload is not supported.** TabbyAPI/ExLlamaV3 uses **NVIDIA CUDA only** — there is no path to run this EXL3 checkpoint on Intel iGPU/Arc.

For ~200k context, the server spills KV pages to **system RAM** (not Intel GPU):

```bash
bash scripts/switch_model.sh kat --context 200000
```

This rounds to **199936** tokens (TabbyAPI requires multiples of 256), enables `cpu_moe_split_experts`, disables MTP drafting to save VRAM, and sets `sysmem_kv_cache=16384` MiB. Override spill size:

```bash
SYSMEM_KV_CACHE=32768 bash scripts/serve_kat.sh --context 200000
```

You have **54 GB WSL RAM** — enough in theory (~24 GB weights + ~8 GB Q8 KV at 200k), but startup will be **slow** and may still OOM on a 16 GB laptop GPU. The model card validated 258k on **2× RTX 3090 24 GB**, not a single 16 GB card.

If 200k fails on EXL3, alternatives are:

- Use **Qwythos** with `--context 200000` (llama.cpp CPU/GPU hybrid, already in this repo)
- Download a **GGUF** KAT build and run via llama.cpp with `-c 200000` (different weights, not your EXL3 file)

### KAT troubleshooting

| Symptom | Fix |
|---------|-----|
| `TabbyAPI not installed` | Run `bash scripts/setup_tabbyapi.sh` |
| Missing weights | `hf download P4pps3n/KAT-Coder-V2.5-Dev-MTP-exl3-5bpw-hq` |
| OOM on 16 GB | Lower context: `bash scripts/switch_model.sh kat --context 8192` |
| Slow on long context | Expected; model card tested up to 258k on 2×3090 with Q8 KV |
| Wanted vLLM TurboQuant | Not supported for EXL3 — use Qwen3 (`switch_model.sh qwen`) for TurboQuant |

## KAT-Coder GGUF + speculative decoding (`kat-npu` on :8000)

A second, **additive** way to serve KAT: GGUF on llama.cpp instead of EXL3 on TabbyAPI, so that
**speculative decoding** can be used to offset the cost of CPU MoE offload. The EXL3 path above is
untouched — `switch_model.sh kat` still works exactly as before.

Trunk: [offmonreal/KAT-Coder-V2.5-Dev-MaxQuality-MTP-GGUF](https://huggingface.co/offmonreal/KAT-Coder-V2.5-Dev-MaxQuality-MTP-GGUF)
(`KAT-Coder-V2.5-Dev_Q3_K_M_imatrix_MTP.gguf`, 18.1 GB). The `_MTP` suffix matters: this file has the
MTP draft head **bundled** as a 41st block, so drafting needs no second model file.

### Launch

```bash
bash scripts/link_kat_gguf.sh          # one-time; symlinks, downloads nothing
bash scripts/switch_model.sh kat-npu --context 16384
```

Startup takes a few minutes: 18 GB is read from the Windows filesystem over `drvfs`.

### Speculation modes

Selected with `SPEC_MODE` (or `--spec-mode`):

| Mode | What drafts | Needs |
|------|-------------|-------|
| `cuda` (default) | Bundled MTP head, on the RTX 5080 | nothing extra |
| `none` | Nothing; plain decode | nothing extra |
| `npu` | Separate MTP head on the Intel NPU, over RPC | Windows drafter + RPC build |

```bash
SPEC_MODE=none bash scripts/switch_model.sh kat-npu     # baseline
bash scripts/bench_kat_npu.sh                            # compare modes
```

`bench_kat_npu.sh` cold-starts the server once per configuration, sends a fixed coding prompt, and
writes prompt-eval tok/s, decode tok/s, and draft acceptance to `.bench_kat_npu.tsv`. It sweeps
`SPEC_N_SWEEP` (draft depth) and `MOE_SWEEP` (`--n-cpu-moe`), and skips `npu` automatically when the
drafter is not reachable. Keep a speculation mode only if it beats `none`.

`PROFILE=short|long|max` selects a context size (8k / 100k / 200k), a prompt length to measure at,
and an `--n-cpu-moe` range that brackets the optimum for that context:

```bash
PROFILE=long REPEATS=5 bash scripts/bench_kat_npu.sh
PROFILE=max MODES=cuda CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 bash scripts/bench_kat_npu.sh
```

Two details matter when reading its output. Decode tok/s swings by several tok/s between identical
requests, because this GPU also drives the Windows desktop — hence `REPEATS` and a median column
rather than single samples. And `PROMPT_TOKENS` prepends filler so decode is measured at a realistic
KV occupancy; measuring at an empty context flatters every configuration equally.

Measurement goes through `scripts/probe_llama_speed.py`, which also works against an already-running
server when you just want one number without a cold start:

```bash
python3 scripts/probe_llama_speed.py --prompt-tokens 16000 --n-predict 256
```

### Measured result

RTX 5080 16 GB, `--context 8192`, `--n-cpu-moe 16`, Q8 KV, 200-token completion:

| Mode | decode | vs baseline | draft acceptance |
|------|--------|-------------|------------------|
| `none` | 22.2 tok/s | — | — |
| `cuda` (`--spec-draft-n-max 2`) | **32.8 tok/s** | **1.48×** | **82%** (123/150) |

The 82% acceptance rate is the reason this works: KAT ships an MTP head trained against its own
trunk, so its drafts are almost always right. Model load takes ~1m40s for the 18 GB trunk.

### Long context (100k–200k)

Both 100k and 200k run on the single 16 GB card. The setting that decides whether they are fast or
unusable is how the model is split between GPU and CPU, not the context length.

Medians of 3–5 repeats, `SPEC_MODE=cuda`, `--spec-draft-n-max 2`, 17.7k-token prompt, 384-token
completion:

| Context | KV | `--n-cpu-moe` | decode | VRAM used | notes |
|---------|----|---------------|--------|-----------|-------|
| 100k | Q8 | auto-fit | 11.1 tok/s | 14.5 GB | previous default |
| 100k | Q8 | **12** | **19.2 tok/s** | 14.9 GB | recommended |
| 100k | Q4 | 10 | **21.8 tok/s** | 15.1 GB | fastest at 100k |
| 200k | Q8 | **16** | **18.5 tok/s** | 15.0 GB | recommended |
| 200k | Q4 | 12 | **22.0 tok/s** | 15.5 GB | fastest, ~0.8 GB headroom |

Speculation still earns its keep at long context: at 100k with `--n-cpu-moe 12`, `cuda` decodes
19.2 tok/s against 10.8 tok/s for `none` — **1.8×**, with 73% draft acceptance.

`serve_kat_npu.sh` applies these values automatically from `--context 65536` upward, so the launch is
just:

```bash
bash scripts/switch_model.sh kat-npu --context 200000
```

Set `N_CPU_MOE` or `N_GPU_LAYERS` to override the profile. Contexts below 64k keep the old auto-fit
behaviour, which is already near-optimal when KV is small.

**Why auto-fit was slow.** `--n-gpu-layers auto` balances its VRAM budget by moving whole layers to
the CPU, attention included. That is the wrong half to move: only a few experts fire per token, but
every token needs every attention layer. Keeping all layers on the GPU (`--n-gpu-layers all`) and
streaming only the MoE experts (`--n-cpu-moe N`) is nearly twice as fast at the same VRAM.

**Where the memory actually goes.** The weights dominate, not the KV cache. Going from 100k to 200k
costs well under a gigabyte of VRAM, and switching Q8 KV to Q4 frees only ~0.5 GB at 100k. Host RAM
peaks around 10 GB of resident memory for the parts of the 18 GB trunk kept off the GPU, so the 54 GB
WSL allocation is never the limit — the 16 GB of VRAM is.

**Leave the card ~1 GB free.** On Windows the driver pages VRAM out to host RAM instead of failing an
allocation, so an over-full card does not OOM, it just crawls. At `--n-cpu-moe 4` prompt eval fell
from ~1400 tok/s to **25 tok/s** with no error in the log. `bench_kat_npu.sh` refuses to benchmark a
configuration leaving less than `SPILL_MARGIN_MIB` (600) free, and reports it as `SPILL`.

**Q4 KV is a real but bounded win.** It buys 2–3 tok/s and costs some KV precision; draft acceptance
also drops (73% → 65% at 100k). Q8 is the default for that reason — use Q4 when throughput matters
more than long-range recall.

**Faster cold starts.** With `--load-mode none` the whole trunk is read on every start, and reading
18 GB from `/mnt/c` over `drvfs` takes ~1m40s. Copying it to ext4 once cuts that to **under 20s**:

```bash
mkdir -p ~/models/kat-gguf-local
cp -L ~/models/kat-gguf/KAT-Coder-V2.5-Dev_Q3_K_M_imatrix_MTP.gguf ~/models/kat-gguf-local/
TARGET_GGUF=~/models/kat-gguf-local/KAT-Coder-V2.5-Dev_Q3_K_M_imatrix_MTP.gguf \
  bash scripts/serve_kat_npu.sh --context 200000
```

This changes startup only; steady-state decode is identical, since the weights end up in RAM and
VRAM either way. It costs ~17 GB of ext4 disk.

**Verified at full depth.** The table above measures decode at a 17.7k-token prompt. A single
**173,958-token** prompt against `--context 200000` (Q8 KV, `--n-cpu-moe 16`) processes at 832 tok/s
and then decodes at **15.0 tok/s** with 67% draft acceptance — so the deep end costs roughly 3 tok/s
against the shallow measurement, not a collapse. VRAM peaked at 15.0 GB and resident host memory at
8.3 GB, both unchanged from the shallow run: `--ctx-size` reserves the KV cache up front, so a nearly
full context is no more expensive in memory than an empty one. Budget the prefill, though — 174k
tokens takes about 3.5 minutes before the first token appears.

### Why the NPU is not the default

The original goal was to draft on the **Intel AI Boost NPU** so drafting would cost no VRAM. It is
fully implemented and works, but **measurement says do not use it.** Speculative decoding only pays
off when the drafter is several times *faster* than the target; a drafter slower than the target
makes generation worse, because every rejected draft is wasted work.

Measured with `llama-bench` on `Qwen3-0.6B-Q4_K_M` (Core Ultra 9 275HX, NPU 3720):

| Device | prefill | **decode** |
|--------|---------|------------|
| NPU (`Intel(R) AI Boost`) | 398 t/s | **10.1 t/s** |
| Intel iGPU (`GPU.0`) | 1128 t/s | **19.8 t/s** |

The KAT trunk itself decodes at 22–33 tok/s on CUDA (table above), so a useful drafter would need to
sustain roughly **65–100 tok/s**. The NPU manages 10, and the real draft head (1.06 GB) is *larger*
than the 0.6 B model benchmarked, so it would be slower still — off by about an order of magnitude.
The NPU is built for prefill: high throughput on large static graphs, but poor autoregressive decode
because it is bandwidth-limited and runs stateless only. The iGPU fallback fails the same test.

Conclusion: `SPEC_MODE=cuda` (bundled head on the GPU) is the configuration that actually helps, and
it needs none of the Windows-side setup. `SPEC_MODE=npu` is kept as a working, opt-in experiment.

### Optional: the NPU drafting path

Two processes, started in this order.

**1. Windows — expose the NPU as a drafting device.** One-time build, then leave it running:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_npu_drafter.ps1
powershell -ExecutionPolicy Bypass -File scripts\npu_drafter.ps1          # or -Device GPU.0
```

`rpc-server` loads no model; it only exposes an OpenVINO device on TCP 50052, so it can stay up
across target-model switches.

**2. WSL2 — build llama.cpp with RPC, then serve.** The prebuilt `~/.local/bin/llama` has CUDA and
all `--spec-*` flags but **no RPC backend**, so the NPU path needs a source build:

```bash
bash scripts/setup_llamacpp_rpc.sh                  # needs the CUDA toolkit (~3 GB)
bash scripts/check_npu_rpc.sh                       # verifies RPC0 enumerates
SPEC_MODE=npu bash scripts/switch_model.sh kat-npu
```

`check_npu_rpc.sh` reports the Windows host IP, port reachability, and whether `RPC0` appears.

### Notes discovered while building this

- **Draft heads are family-locked.** An MTP head consumes the trunk's final hidden state, so it must
  match the trunk's embedding length *and* vocab. Verified with `scripts/gguf_compat.py`: both KAT and
  the Qwen3.6-35B-A3B head are `hidden=2048 vocab=248320`. `serve_kat_npu.sh` runs this check before
  starting `SPEC_MODE=npu`, because a mismatched head loads fine and then silently drafts tokens that
  are always rejected. Qwen3.6-**27B** is the counterexample: same 248320 vocab, but `hidden=5120`, so
  the 35B-A3B head is unusable with it.
- **The head is not a standalone model.** `llama-cli -m mtp-*.gguf` segfaults on every device. The
  head is only loadable through `--spec-type draft-mtp` attached to a trunk.
- **`--ctx-size-draft` does not exist** in current llama.cpp; the draft context follows the target's.
  Current flag names are `--spec-type`, `--spec-draft-model`, `--spec-draft-device`,
  `--spec-draft-n-max`. `--draft`/`--draft-max` were removed.
- **`--no-mmap` is deprecated** in favour of `--load-mode`. The scripts pass `--load-mode none`,
  which llama.cpp itself recommends when MoE tensors are overridden to CPU.
- **`--cache-ram` is not KV offload.** It sizes the *host-side prompt cache*, which lets a later
  request reuse an earlier prompt's KV instead of reprocessing it (the `selected slot by LCP
  similarity` lines in the log). The active context always lives in VRAM, so context length is a
  VRAM constraint, not a system-RAM one. Long contexts make each cached state large, hence
  `--cache-ram -1` past 131072 — otherwise the 8192 MiB default cannot hold even one.
- **`--n-gpu-layers` takes `auto`, `all`, or a number.** Not `-1` or `999`; the scripts pass `all`.
- **The OpenVINO backend needs an OpenCL SDK** that Windows does not ship. `setup_npu_drafter.ps1`
  assembles one from OpenCL-Headers, OpenCL-ICD-Loader, and OpenCL-CLHPP — the last is required
  because OpenVINO's `ocl_wrapper.hpp` includes `CL/cl2.hpp`, which the Headers repo omits.
- **`GGML_OPENVINO_CACHE_DIR` is unsupported on the NPU** and setting it crashes model load. The
  OpenVINO DLLs also ship inside the pip package and must be added to `PATH`, or the process exits
  immediately with `STATUS_DLL_NOT_FOUND`.
- The RPC protocol is unauthenticated, so the firewall rule is scoped to the WSL subnet.

### kat-npu troubleshooting

| Symptom | Fix |
|---------|-----|
| `couldn't bind HTTP server socket` on :8000 | Another server holds the port: `bash scripts/kill_gpu.sh` |
| Missing target GGUF | `bash scripts/link_kat_gguf.sh` |
| `SPEC_MODE=npu needs a llama.cpp build with the RPC backend` | `bash scripts/setup_llamacpp_rpc.sh` |
| `No RPC server at <ip>:50052` | Start `scripts\npu_drafter.ps1` on Windows; diagnose with `scripts/check_npu_rpc.sh` |
| `Draft head is INCOMPATIBLE` | Head does not match the trunk; check with `python3 scripts/gguf_compat.py show <file>` |
| Speculation makes decode slower | Expected for `npu`; use the default `SPEC_MODE=cuda` |
| Server responds but everything crawls (prompt eval ~25 tok/s), no error | VRAM is full and the driver is paging to host RAM. Raise `N_CPU_MOE` until `nvidia-smi` shows ~1 GB free |
| Decode much slower than the table above at 100k+ | `N_GPU_LAYERS` is probably `auto`; the long-context profile needs it unset or `all` |
| `Failed to initialize NVML` in WSL after the host sleeps | The dGPU dropped off the bus; reboot Windows (`wsl --shutdown` alone does not recover it) |

## Multimodal model (images + video + text)

A separate server serves **Qwen2.5-VL-7B-Instruct-AWQ** via vLLM's built-in OpenAI API on **port 8001**. It handles text, images, and video natively. It does **not** use TurboQuant.

**VRAM:** only one server can run at a time on the RTX 5080. Stop the text server first:

```bash
bash scripts/kill_gpu.sh
```

### Download (one-time, WSL2)

```bash
source ~/turboquant-llm/.venv/bin/activate
hf download Qwen/Qwen2.5-VL-7B-Instruct-AWQ
```

Approximate size: **~6.9 GB**.

### Launch

```bash
cd /mnt/c/dev/turboquant-llm
bash scripts/serve_vl.sh --context 16384
```

**Windows one-liner:**

```powershell
wsl -d Ubuntu bash -c "cd /mnt/c/dev/turboquant-llm && bash scripts/serve_vl.sh --context 16384"
```

Defaults: `MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct-AWQ`, `SERVED_MODEL_NAME=qwen2.5-vl`, port **8001**, `awq_marlin` quantization (required on sm_120).

Place local media under `media/` and reference with `file:///mnt/c/dev/turboquant-llm/media/yourfile.jpg`.

### Verification

```bash
python scripts/check_vl.py
```

**From Windows PowerShell:**

```powershell
curl http://localhost:8001/v1/models
```

```powershell
curl -X POST http://localhost:8001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen2.5-vl","messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"https://placehold.co/320x240.jpg"}}]}],"max_tokens":128}'
```

```powershell
curl -X POST http://localhost:8001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen2.5-vl","messages":[{"role":"user","content":[{"type":"text","text":"What happens in this video?"},{"type":"video_url","video_url":{"url":"https://samplelib.com/lib/preview/mp4/sample-5s.mp4"}}]}],"max_tokens":128}'
```

### Connect clients to the multimodal server

| Setting | Value |
|---------|-------|
| Base URL / API URL | `http://localhost:8001/v1` |
| API Key | any non-empty string (e.g. `local`) |
| Model | `qwen2.5-vl` |

For Open WebUI, point `OPENAI_API_BASE_URL` at `:8001/v1` instead of `:8000`.

**Cursor tunnel:** `PORT=8001 bash scripts/tunnel.sh` — then set Base URL to `<tunnel-host>/v1` and model `qwen2.5-vl`.

### Performance

Expected throughput on WSL2 + RTX 5080 (after warm-up):

| Workload | Typical decode speed |
|----------|---------------------|
| Text only | ~10–15 tok/s |
| Text + image | ~5–10 tok/s |

Run a local benchmark (bypasses Cursor tunnel latency):

```bash
python scripts/bench_vl.py
```

Tuning knobs (all via env on `serve_vl.sh`):

```bash
# More VRAM for KV cache, single-user Cursor (0.95 may fail on WSL2 — use 0.92)
GPU_MEMORY_UTILIZATION=0.92 MAX_NUM_SEQS=1 bash scripts/serve_vl.sh

# Faster text + small images (disables video profiling)
PERF_PROFILE=speed bash scripts/serve_vl.sh

# Fix stale torch compile cache if startup logs show cubin reload errors
CLEAR_VLLM_COMPILE_CACHE=1 bash scripts/serve_vl.sh
```

**WSL2 tip:** add `vmIdleTimeout=-1` to `%UserProfile%\.wslconfig` so WSL does not idle-shutdown mid-session.

**Hard limit:** `VLLM_USE_V2_MODEL_RUNNER=0` is required on WSL2 (no UVA). Native Linux would be ~10–20% faster. If local `bench_vl.py` is fast but Cursor feels slow, the Cloudflare tunnel is the bottleneck — not GPU allocation.

### Multimodal troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA OOM at startup | Lower `MM_PROCESSOR_KWARGS` max_pixels; try `ENFORCE_EAGER=1 bash scripts/serve_vl.sh` |
| `awq` / float16 error on sm_120 | Script uses `awq_marlin` by default — do not pass `--quantization awq` |
| Video works but KV cache is tiny | Defaults cap video profiling; set `LIMIT_MM` video count to 0 if you only need images |
| Attention backend errors | Try `ATTENTION_BACKEND=TRITON_ATTN bash scripts/serve_vl.sh` |
| `tool_choice "auto" requires --enable-auto-tool-choice` | Enabled by default in `serve_vl.sh`; restart server after pulling latest script |
| Slow responses in Cursor but fast locally | Tunnel adds RTT; run `python scripts/bench_vl.py` on localhost first |
| Unstable tok/s / cubin reload warnings at startup | `CLEAR_VLLM_COMPILE_CACHE=1 bash scripts/serve_vl.sh` |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `sm_120` not in `torch.cuda.get_arch_list()` | Reinstall torch from cu128 index only; confirm driver >= 570 |
| `nvidia-smi` fails inside WSL2 | Update Windows NVIDIA driver; run `wsl --update`; reboot |
| CUDA OOM at engine start | Lower `MAX_MODEL_LEN=8192` or `GPU_MEMORY_UTILIZATION=0.85`; close other GPU apps (e.g. LM Studio) |
| Generation stops mid-stream | KV cache full on GPU; defaults enable CPU offload (`KV_OFFLOADING_SIZE=8`). Try `MAX_NUM_SEQS=1` or raise `KV_OFFLOADING_SIZE=12` |
| `No space left on device` / `/dev/shm` full on WSL2 | Stale offload mmap from crashed runs: `rm -f /dev/shm/vllm_offload_*.mmap` after stopping server. Code uses `VLLM_USE_SIMPLE_KV_OFFLOAD=1` to avoid new mmap files |
| Startup fails with `madvise` / `Bad address` on WSL2 | Fixed in code via `VLLM_USE_SIMPLE_KV_OFFLOAD=1` (pinned RAM path, not `/dev/shm` mmap) |


## One liner to start the application
wsl -d Ubuntu bash -c "cd /mnt/c/dev/turboquant-llm && ~/turboquant-llm/.venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000"

## for openweb ui to access llm
wsl -d Ubuntu bash -c "export ENABLE_OLLAMA_API=false OPENAI_API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=local && ~/open-webui/.venv/bin/open-webui serve --host 0.0.0.0 --port 3000"

## Browser automation (Comet-like)

Uses [browser-use](https://github.com/browser-use/browser-use) (Playwright + your local LLM at `:8000`) to complete natural-language web tasks.

## start tunnel to use model from cursur
wsl -d Ubuntu bash -lc "sed -i 's/\r$//' /mnt/c/dev/turboquant-llm/scripts/tunnel.sh && cd /mnt/c/dev/turboquant-llm && bash scripts/tunnel.sh"
### once tunnel started then set the base url in cursur of openai as

<tunnel base url>/v1
api key - local
model name - exact that yopiu have

**Install once (WSL):**

```bash
source /mnt/c/dev/turboquant-llm/.venv/bin/activate
pip install -r browser/requirements-browser.txt
playwright install chromium
cp browser/.env.example browser/.env   # optional overrides
```

**Run a task:**

```bash
# Inline task
bash scripts/run_browser_agent.sh "Open https://example.com and return the H1 text"

# From a task file
bash scripts/run_browser_agent.sh --task-file browser/tasks/smoke_example.md
bash scripts/run_browser_agent.sh --task-file browser/tasks/api_docs_smoke.md
```

**Cursor-only (no install):** paste tasks from [`browser/cursor-tasks/`](browser/cursor-tasks/) into Agent mode — uses the built-in browser MCP.

**Env vars** (`browser/.env`): `OPENAI_API_BASE`, `BROWSER_LLM_MODEL`, `BROWSER_HEADLESS`, `BROWSER_USE_VISION`, `BROWSER_MAX_STEPS`.

**Note:** Qwen3-14B-AWQ works for simple smoke tasks; complex multi-site flows may need a stronger model or cloud API via the same env vars.
