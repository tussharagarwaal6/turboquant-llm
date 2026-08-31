"""OpenAI proxy with context compaction for llama.cpp or other backends."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.context_compaction import (
    maybe_compact_messages,
    prune_tool_messages,
    should_skip_compaction,
    summary_max_tokens,
)

logger = logging.getLogger("turboquant.compaction_proxy")

BACKEND_BASE = os.environ.get("COMPACTION_BACKEND_URL", "http://127.0.0.1:8002/v1").rstrip("/")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwythos-9b")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="Context Compaction Proxy")


async def _backend_summarize(prompt: str) -> str:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": summary_max_tokens(),
        "temperature": 0.3,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{BACKEND_BASE}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    message = data["choices"][0]["message"]
    for key in ("content", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class _ProxyTokenizer:
    """Token estimate fallback when we cannot tokenize locally."""

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(range(max(1, len(text) // 4)))

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: Any) -> str:
        parts: list[str] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                chunk_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"image", "image_url"}:
                        chunk_parts.append("[image]")
                    elif isinstance(item, dict):
                        chunk_parts.append(str(item.get("text") or item.get("content") or ""))
                    else:
                        chunk_parts.append(str(item))
                content = " ".join(chunk_parts)
            parts.append(f"{message.get('role')}: {content}")
        return "\n".join(parts)


_proxy_tokenizer = _ProxyTokenizer()


def _is_context_overflow(status_code: int, detail: str) -> bool:
    if status_code != 400:
        return False
    lowered = detail.lower()
    return (
        "exceed_context_size" in lowered
        or "exceeds the available context size" in lowered
        or "maximum context length" in lowered
        or ("context" in lowered and "exceed" in lowered)
    )


async def _sse_error_stream(status_code: int, detail: str) -> AsyncIterator[bytes]:
    payload = {
        "error": {
            "message": detail,
            "type": "backend_error",
            "code": status_code,
        }
    }
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _prepare_body(body: dict[str, Any], *, force: bool = False, aggressive: bool = False) -> dict[str, Any]:
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    pruned_messages, did_prune = prune_tool_messages(
        messages,
        active_tool_request=bool(body.get("tools")),
    )
    if did_prune:
        body = {**body, "messages": pruned_messages}
        messages = pruned_messages

    if should_skip_compaction(body):
        return body

    compacted_messages, did_compact = await maybe_compact_messages(
        messages,
        max_model_len=MAX_MODEL_LEN,
        tokenizer=_proxy_tokenizer,
        summarize=_backend_summarize,
        force=force,
        aggressive=aggressive,
    )
    if did_compact:
        body = {**body, "messages": compacted_messages}
        logger.info(
            "Forwarded compacted request with %d messages (force=%s aggressive=%s)",
            len(compacted_messages),
            force,
            aggressive,
        )
    return body


async def _iter_backend_stream(body: dict[str, Any]) -> tuple[int, str, AsyncIterator[bytes] | None]:
    client = httpx.AsyncClient(timeout=600.0)
    request = client.build_request("POST", f"{BACKEND_BASE}/chat/completions", json=body)
    response = await client.send(request, stream=True)
    if response.status_code >= 400:
        detail = (await response.aread()).decode("utf-8", errors="replace")
        await response.aclose()
        await client.aclose()
        return response.status_code, detail, None

    async def forward() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        except Exception as exc:
            logger.exception("Backend stream interrupted: %s", exc)
            payload = {"error": {"message": str(exc), "type": "stream_error", "code": 502}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            await response.aclose()
            await client.aclose()

    return response.status_code, "", forward()


@app.get("/sw.js")
async def service_worker_stub() -> PlainTextResponse:
    return PlainTextResponse("// no service worker\n", media_type="application/javascript")


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BACKEND_BASE}/models")
        response.raise_for_status()
        data = response.json()
    for item in data.get("data", []):
        if item.get("meta"):
            item["meta"]["n_ctx"] = MAX_MODEL_LEN
    return data


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = bool(body.get("stream"))

    for attempt in range(2):
        prepared = await _prepare_body(
            body,
            force=attempt > 0,
            aggressive=attempt > 0,
        )

        if stream:
            status_code, detail, stream_iter = await _iter_backend_stream(prepared)
            if status_code >= 400:
                if attempt == 0 and _is_context_overflow(status_code, detail):
                    logger.warning("Context overflow on stream; retrying with aggressive compaction")
                    continue
                logger.warning("Backend stream error %s: %s", status_code, detail[:300])
                return StreamingResponse(
                    _sse_error_stream(status_code, detail),
                    media_type="text/event-stream",
                )
            return StreamingResponse(stream_iter, media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{BACKEND_BASE}/chat/completions", json=prepared)
            if response.status_code >= 400:
                detail = response.text
                if attempt == 0 and _is_context_overflow(response.status_code, detail):
                    logger.warning("Context overflow; retrying with aggressive compaction")
                    continue
                raise HTTPException(status_code=response.status_code, detail=detail)
            return JSONResponse(response.json())

    raise HTTPException(status_code=400, detail="Request still exceeds context after compaction")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
