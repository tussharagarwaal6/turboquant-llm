#!/bin/bash
# Launch the TurboQuant vLLM OpenAI-compatible server from WSL2.
# Code lives on /mnt/c; the venv lives on the native WSL disk for speed.

VENV="$HOME/turboquant-llm/.venv"
APP_DIR=/mnt/c/dev/turboquant-llm

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B-AWQ}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

cd "$APP_DIR" || exit 1
exec "$VENV/bin/python" -m uvicorn app.server:app --host 0.0.0.0 --port 8000
