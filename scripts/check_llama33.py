#!/usr/bin/env python3
"""Smoke test for the Llama 3.3 70B CPU GGUF server on :8000."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("LLAMA33_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("LLAMA33_MODEL", "llama33-70b")
TIMEOUT = int(os.environ.get("LLAMA33_CHECK_TIMEOUT", "600"))


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


def _message_text(data: dict) -> str:
    message = data["choices"][0]["message"]
    parts: list[str] = []
    for key in ("content", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def main() -> int:
    print(f"Checking Llama 3.3 70B API at {BASE_URL} (model={MODEL}, timeout={TIMEOUT}s)")

    try:
        models = _get("/models")
        ids = [m["id"] for m in models.get("data", [])]
        print(f"Models: {ids}")
        if MODEL not in ids and not ids:
            print("FAIL  list models: empty response")
            return 1
    except Exception as exc:
        print(f"FAIL  list models: {exc}")
        return 1

    try:
        data = _post(
            "/chat/completions",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly one word: ok",
                    }
                ],
                "max_tokens": 32,
                "temperature": 0.6,
                "top_p": 0.95,
            },
        )
        text = _message_text(data)
        preview = text[:120].replace("\n", " ")
        if "ok" not in text.lower():
            print(f"FAIL  text: unexpected response {preview!r}")
            return 1
        print(f"PASS  text: {preview!r}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL  text: HTTP {exc.code} {detail[:400]}")
        return 1
    except Exception as exc:
        print(f"FAIL  text: {exc}")
        return 1

    print("\n1/1 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
