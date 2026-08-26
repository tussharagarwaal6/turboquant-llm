#!/bin/bash
VENV=/mnt/c/dev/turboquant-llm/.venv/bin/pip
"$VENV" install "vllm>=0.18.0" fastapi "uvicorn[standard]" "huggingface_hub[cli]" --timeout 1000 --retries 10
