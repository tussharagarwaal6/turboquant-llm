#!/bin/bash
# Stop TurboQuant/vLLM server processes inside WSL2.

echo "Stopping WSL GPU server processes..."
pkill -f 'uvicorn app.server' 2>/dev/null || true
pkill -f 'python -m uvicorn app.server' 2>/dev/null || true
pkill -f 'vllm serve' 2>/dev/null || true
pkill -f 'EngineCore' 2>/dev/null || true
pkill -f 'VllmWorker' 2>/dev/null || true
sleep 2

REMAINING=$(pgrep -af 'uvicorn app.server|EngineCore|VllmWorker' || true)
if [ -n "$REMAINING" ]; then
  echo "Force-killing remaining processes..."
  pkill -9 -f 'uvicorn app.server' 2>/dev/null || true
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  pkill -9 -f 'VllmWorker' 2>/dev/null || true
  sleep 1
fi

echo
echo "WSL GPU status:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
  || nvidia-smi
