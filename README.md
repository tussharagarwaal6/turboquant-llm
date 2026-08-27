# TurboQuant vLLM Local Deployment

OpenAI-compatible local LLM server: **Qwen3-14B-AWQ** on vLLM with **TurboQuant** KV-cache compression (`turboquant_k8v4`), running in WSL2 Ubuntu with RTX 5080 GPU passthrough.

## Prerequisites (Windows)

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

### Connect any OpenAI-compatible chat client (Windows)

| Setting | Value |
|---------|-------|
| Base URL / API URL | `http://localhost:8000/v1` |
| API Key | any non-empty string (e.g. `local`) |
| Model | `Qwen/Qwen3-14B-AWQ` |

Works with Open WebUI, Chatbox, Jan, Continue, etc.

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
