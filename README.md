# turboquant-llm

Local LLM serving and browser automation for OpenCode.

## Model switcher

One model on port **8000** at a time via `scripts/switch_model.sh`:

```bash
bash scripts/switch_model.sh qwen35-4b --context 262144   # fast GPU + vision
bash scripts/switch_model.sh gemma4 --context 100000      # MoE + vision
bash scripts/switch_model.sh kat-npu --context 100000     # KAT + spec decode
bash scripts/switch_model.sh llama33 --context 8192       # Llama 3.3 70B hybrid GPU+CPU
```

**Llama 3.3 70B:** one-time `bash scripts/download_llama33.sh` then `bash scripts/link_llama33_gguf.sh`. Default is **hybrid GPU offload** (`N_GPU_LAYERS=auto`) on the RTX 5080. On a **64 GB** host, run `scripts\setup_wsl_memory.ps1` (sets WSL to **48 GB**, leaves **16 GB** for Windows), then `wsl --shutdown`. Do **not** set WSL to 56 GB on 64 GB RAM — that causes 98% host usage and SSD pagefile thrashing. Check pressure: `powershell -File scripts\check_host_memory.ps1`. Benchmark: `bash scripts/bench_llama33.sh`.

---

## Browser Use + OpenCode

OpenCode talks to a **Browser Use MCP** HTTP daemon (`http://127.0.0.1:8383/mcp`), which drives Chrome via CDP. The LLM on port **8000** decides what the browser should do.

### One-time setup

From the repo root in **PowerShell**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_browser_use_mcp.ps1
```

This installs `uv`, clones [Saik0s/mcp-browser-use](https://github.com/Saik0s/mcp-browser-use) into `browser/mcp-browser-use/`, and installs Playwright Chromium.

OpenCode is configured in `%USERPROFILE%\.config\opencode\opencode.json` with the `browser-use` remote MCP and `browser-use_*` tools on the `build` agent. Restart OpenCode after changing that file.

**Chrome profile:** Browser automation uses a dedicated profile at `browser/chrome-profile/` (not your daily Chrome). Chrome 136+ blocks remote debugging on the default profile. Log into sites once in this window; cookies persist for later runs.

---

### Every session (before using browser tools in OpenCode)

You need **three things running**: local LLM, Chrome with CDP, and the Browser Use MCP daemon.

#### Option A — one PowerShell command (after LLM is up)

**Terminal 1 — WSL (LLM):**

```bash
bash scripts/switch_model.sh qwen35-4b --context 262144
```

Wait until the server is listening on `http://127.0.0.1:8000`.

**Terminal 2 — PowerShell (Chrome + MCP):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_browser_stack.ps1
```

Leave this window open.

**Then:** restart OpenCode (or start it if it was closed) so it connects to `http://127.0.0.1:8383/mcp`.

#### Option B — separate terminals

| Step | Where | Command |
|------|--------|---------|
| 1. LLM | WSL | `bash scripts/switch_model.sh qwen35-4b --context 262144` |
| 2. Chrome CDP | PowerShell | `powershell -ExecutionPolicy Bypass -File scripts\start_chrome_cdp.ps1` |
| 3. Browser Use MCP | PowerShell | `powershell -ExecutionPolicy Bypass -File scripts\start_browser_use_mcp.ps1` |
| 4. OpenCode | — | Restart OpenCode |

---

### Verify everything is up

**PowerShell:**

```powershell
# LLM
curl http://127.0.0.1:8000/v1/models

# Chrome CDP
curl http://127.0.0.1:9222/json/version

# Browser Use MCP
curl http://127.0.0.1:8383/api/health
```

**OpenCode:**

```powershell
opencode mcp list
```

You should see `browser-use` connected.

Dashboard (optional): [http://127.0.0.1:8383/dashboard](http://127.0.0.1:8383/dashboard)

---

### Optional environment variables

Set before `start_browser_use_mcp.ps1` or in your shell profile:

| Variable | Default | Purpose |
|----------|---------|---------|
| `BROWSER_CDP_URL` | `http://127.0.0.1:9222` | Chrome CDP endpoint |
| `MCP_LLM_MODEL_NAME` | `qwen35-4b` | Model name for browser agent |
| `MCP_LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Local OpenAI-compatible API |
| `BROWSER_USE_VISION` | `false` | Set to `1` for screenshot-based actions (requires vision/mmproj on the LLM) |
| `BROWSER_MAX_STEPS` | `25` | Max agent steps per task |

---

### Troubleshooting

- **MCP won't start / OpenCode can't connect:** Ensure terminal 2 is still running and `curl http://127.0.0.1:8383/api/health` returns `"status":"healthy"`.
- **Chrome "cannot read and write to its data directory":** Do not point CDP at the default Chrome profile. Use `scripts\start_chrome_cdp.ps1` (uses `browser\chrome-profile`).
- **CDP not on 9222:** Chrome 136+ ignores `--remote-debugging-port` on the default user-data path. The start script uses a separate profile on purpose.
- **Browser tasks fail but MCP connects:** Start Chrome CDP first (`start_chrome_cdp.ps1`), then confirm `http://127.0.0.1:9222/json/version` works.
