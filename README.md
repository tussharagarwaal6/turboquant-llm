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

From WSL2, project root:

```bash
cd /mnt/c/dev/turboquant-llm
source .venv/bin/activate
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Engine args (set in `app/server.py`):

- `model=Qwen/Qwen3-14B-AWQ`
- `quantization=awq`
- `kv_cache_dtype=turboquant_k8v4`
- `gpu_memory_utilization=0.90`
- `max_model_len=16384`

Override via env: `MODEL_ID`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`.

First startup loads weights (~1–3 min).

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

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `sm_120` not in `torch.cuda.get_arch_list()` | Reinstall torch from cu128 index only; confirm driver >= 570 |
| `nvidia-smi` fails inside WSL2 | Update Windows NVIDIA driver; run `wsl --update`; reboot |
| CUDA OOM at engine start | Lower `MAX_MODEL_LEN=8192` or `GPU_MEMORY_UTILIZATION=0.85`; close other GPU apps (e.g. LM Studio) |


## One liner to staret the application
wsl -d Ubuntu bash -c "cd /mnt/c/dev/turboquant-llm && GPU_MEMORY_UTILIZATION=0.92 MAX_MODEL_LEN=8192 ~/turboquant-llm/.venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000"

## for openweb ui to access llm
wsl -d Ubuntu bash -c "export ENABLE_OLLAMA_API=false OPENAI_API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=local && ~/open-webui/.venv/bin/open-webui serve --host 0.0.0.0 --port 3000"
