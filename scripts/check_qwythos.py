#!/usr/bin/env python3
"""Smoke test for the Qwythos-9B GGUF server on :8000."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("QWYTHOS_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("QWYTHOS_MODEL", "qwythos-9b")
TIMEOUT = int(os.environ.get("QWYTHOS_CHECK_TIMEOUT", "300"))

SAMPLE_IMAGE = "https://placehold.co/320x240.jpg"
OCR_IMAGE = "https://placehold.co/600x200/png?text=HELLO+WORLD"


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


def run_case(name: str, payload: dict, *, check_fn=None) -> bool:
    try:
        data = _post("/chat/completions", payload)
        text = _message_text(data)
        if check_fn is not None and not check_fn(text, data):
            preview = text[:120].replace("\n", " ")
            print(f"FAIL  {name}: unexpected response {preview!r}")
            return False
        preview = text[:120].replace("\n", " ")
        print(f"PASS  {name}: {preview!r}")
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL  {name}: HTTP {exc.code} {detail[:400]}")
        return False
    except Exception as exc:
        print(f"FAIL  {name}: {exc}")
        return False


def main() -> int:
    print(f"Checking Qwythos API at {BASE_URL} (model={MODEL}, timeout={TIMEOUT}s)")
    results: list[bool] = []

    try:
        models = _get("/models")
        ids = [m["id"] for m in models.get("data", [])]
        print(f"Models: {ids}")
        results.append(MODEL in ids or bool(ids))
    except Exception as exc:
        print(f"FAIL  list models: {exc}")
        return 1

    results.append(
        run_case(
            "text",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                "max_tokens": 32,
                "temperature": 0.6,
                "top_p": 0.95,
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
                "temperature": 0.6,
                "top_p": 0.95,
            },
        )
    )

    results.append(
        run_case(
            "ocr",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all visible text verbatim from this image.",
                            },
                            {"type": "image_url", "image_url": {"url": OCR_IMAGE}},
                        ],
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.6,
                "top_p": 0.95,
            },
            check_fn=lambda text, _data: "hello" in text.lower(),
        )
    )

    results.append(
        run_case(
            "tool_choice_auto",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hi in one word."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 32,
                "temperature": 0.6,
                "top_p": 0.95,
            },
        )
    )

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
