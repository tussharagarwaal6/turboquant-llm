#!/usr/bin/env python3
"""Smoke test for the Qwen2.5-VL multimodal server on :8001."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("VL_API_BASE", "http://127.0.0.1:8001/v1").rstrip("/")
MODEL = os.environ.get("VL_MODEL", "qwen2.5-vl")
TIMEOUT = int(os.environ.get("VL_CHECK_TIMEOUT", "180"))

# Small public media for remote fetch tests.
SAMPLE_IMAGE = "https://placehold.co/320x240.jpg"
SAMPLE_VIDEO = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"


def _post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _content(data: dict) -> str:
    return data["choices"][0]["message"]["content"].strip()


def run_case(name: str, payload: dict) -> bool:
    try:
        data = _post("/chat/completions", payload)
        text = _content(data)
        preview = text[:120].replace("\n", " ")
        print(f"PASS  {name}: {preview!r}")
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL  {name}: HTTP {exc.code} {detail[:300]}")
        return False
    except Exception as exc:
        print(f"FAIL  {name}: {exc}")
        return False


def main() -> int:
    print(f"Checking VL API at {BASE_URL} (model={MODEL}, timeout={TIMEOUT}s)")
    results: list[bool] = []

    try:
        models = _get("/models")
        ids = [m["id"] for m in models.get("data", [])]
        print(f"Models: {ids}")
        results.append(MODEL in ids or bool(ids))
    except Exception as exc:
        print(f"FAIL  list models: {exc}")
        results.append(False)
        return 1

    results.append(
        run_case(
            "text",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                "max_tokens": 16,
            },
        )
    )

    results.append(
        run_case(
            "image_url",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in one short sentence."},
                            {"type": "image_url", "image_url": {"url": SAMPLE_IMAGE}},
                        ],
                    }
                ],
                "max_tokens": 128,
            },
        )
    )

    results.append(
        run_case(
            "video_url",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What happens in this short video? One sentence."},
                            {"type": "video_url", "video_url": {"url": SAMPLE_VIDEO}},
                        ],
                    }
                ],
                "max_tokens": 128,
            },
        )
    )

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
