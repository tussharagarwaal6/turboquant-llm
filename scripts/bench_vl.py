#!/usr/bin/env python3
"""Benchmark Qwen2.5-VL server throughput (prompt + decode tok/s)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("VL_API_BASE", "http://127.0.0.1:8001/v1").rstrip("/")
MODEL = os.environ.get("VL_MODEL", "qwen2.5-vl")
TIMEOUT = int(os.environ.get("VL_BENCH_TIMEOUT", "180"))
MAX_TOKENS = int(os.environ.get("VL_BENCH_MAX_TOKENS", "128"))
SAMPLE_IMAGE = "https://placehold.co/320x240.jpg"


def _gpu_stats() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        util, used, total = out.split(", ")
        return f"GPU util={util}%  VRAM={used}/{total} MiB"
    except Exception as exc:
        return f"GPU stats unavailable: {exc}"


def _post(payload: dict) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.perf_counter() - t0


def _bench(label: str, payload: dict) -> bool:
    try:
        data, elapsed = _post(payload)
        usage = data.get("usage") or {}
        prompt_toks = usage.get("prompt_tokens", 0)
        completion_toks = usage.get("completion_tokens", 0)
        prompt_tps = prompt_toks / elapsed if elapsed > 0 else 0.0
        decode_tps = completion_toks / elapsed if elapsed > 0 else 0.0
        print(
            f"{label:12}  {elapsed:6.2f}s  "
            f"prompt={prompt_toks:4d} ({prompt_tps:5.1f} tok/s)  "
            f"decode={completion_toks:4d} ({decode_tps:5.1f} tok/s)"
        )
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"{label:12}  FAIL HTTP {exc.code}: {detail[:200]}")
        return False
    except Exception as exc:
        print(f"{label:12}  FAIL: {exc}")
        return False


def main() -> int:
    print(f"VL benchmark  base={BASE_URL}  model={MODEL}")
    print(_gpu_stats())
    print()

    text_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a short paragraph about local LLM inference on consumer GPUs. "
                    "Include at least three sentences."
                ),
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    image_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": SAMPLE_IMAGE}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    results: list[bool] = []
    print("Warm-up...")
    results.append(_bench("warmup", text_payload))
    print()
    print("Timed runs (after warm-up):")
    results.append(_bench("text", text_payload))
    results.append(_bench("text-2", text_payload))
    results.append(_bench("image", image_payload))

    print()
    print(_gpu_stats())
    passed = sum(results)
    print(f"\n{passed}/{len(results)} runs OK")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
