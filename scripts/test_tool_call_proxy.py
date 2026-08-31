#!/usr/bin/env python3
"""Smoke test tool calling through the compaction proxy."""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("QWYTHOS_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("QWYTHOS_MODEL", "qwythos-9b")


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "image_ocr",
                "description": "Extract text from an image",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    first = post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Call image_ocr with no args to OCR the attached image."}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 256,
            "stream": False,
        }
    )
    message = first["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    print("First response finish:", first["choices"][0].get("finish_reason"))
    print("Tool calls:", len(tool_calls))

    if not tool_calls:
        print("WARN: model did not emit tool_calls:", message.get("content", "")[:200])
        return 0

    follow_up = post(
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "OCR the image"},
                message,
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"],
                    "content": json.dumps({"text": "HELLO WORLD", "model": MODEL}),
                },
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 256,
            "stream": False,
        }
    )
    text = follow_up["choices"][0]["message"].get("content", "")
    print("Follow-up OK:", text[:200].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
