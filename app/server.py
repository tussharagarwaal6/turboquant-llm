"""OpenAI-compatible FastAPI server backed by vLLM AsyncLLMEngine with TurboQuant KV cache."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# WSL2 GPU passthrough does not expose UVA, which the V2 model runner requires.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

# Native KV offload uses /dev/shm mmap + madvise, which fails on WSL2 (errno 14).
# SimpleCPUOffloadConnector uses pinned host RAM instead.
os.environ.setdefault("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")

# The pip CUDA toolkit ships nvcc 13.3 against 13.0 headers, which CCCL rejects, so
# FlashInfer's JIT sampler cannot build. TurboQuant itself only needs Triton.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# TurboQuant kernels are compiled at runtime, so nvcc must be discoverable. The
# CUDA toolkit is only present as a pip package here, not under /usr/local/cuda.
if "CUDA_HOME" not in os.environ:
    try:
        import nvidia

        for _root in nvidia.__path__:
            _match = next(
                (c for c in sorted(Path(_root).glob("cu*")) if (c / "bin" / "nvcc").is_file()),
                None,
            )
            if _match is not None:
                os.environ["CUDA_HOME"] = str(_match)
                break
    except ImportError:
        pass

# ninja and nvcc are invoked by bare name during the kernel build, and the engine
# runs in a subprocess that inherits this environment rather than an activated venv.
_bin_dirs = [str(Path(sys.executable).parent)]
if "CUDA_HOME" in os.environ:
    _bin_dirs.append(str(Path(os.environ["CUDA_HOME"]) / "bin"))
os.environ["PATH"] = os.pathsep.join([*_bin_dirs, os.environ.get("PATH", "")])

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-14B-AWQ")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_ID)


def _env_float(key: str, default: str) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: str) -> int:
    return int(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().strip("\r").lower() in {"1", "true", "yes", "on"}


logger = logging.getLogger("turboquant.server")

REQUESTED_MAX_MODEL_LEN = _env_int("MAX_MODEL_LEN", "16384")
# Set after engine init; may be lower than requested when VRAM is tight.
effective_max_model_len = REQUESTED_MAX_MODEL_LEN


def _env_explicit(key: str) -> bool:
    return os.environ.get(key) is not None


def _auto_gpu_memory_utilization(requested_max_len: int) -> float:
    if _env_explicit("GPU_MEMORY_UTILIZATION"):
        return _env_float("GPU_MEMORY_UTILIZATION", "0.88")
    if requested_max_len <= 16384:
        return 0.88
    if requested_max_len <= 32768:
        return 0.92
    return 0.95


def _auto_kv_offloading_size(requested_max_len: int) -> float:
    """Scale CPU/system-RAM KV spillover with requested context length."""
    if _env_explicit("KV_OFFLOADING_SIZE"):
        return _env_float("KV_OFFLOADING_SIZE", "8.0")
    # ~8 GiB base plus ~1 GiB per 4k tokens above 8k (32768 -> 16 GiB).
    return max(8.0, requested_max_len / 2048.0)


def _build_engine_args(
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_offloading_size: float,
) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=MODEL_ID,
        quantization="awq",
        kv_cache_dtype="turboquant_k8v4",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        kv_offloading_size=kv_offloading_size,
        kv_offloading_backend=os.environ.get("KV_OFFLOADING_BACKEND", "native"),
        max_num_seqs=_env_int("MAX_NUM_SEQS", "2"),
        trust_remote_code=True,
    )


def _create_engine() -> AsyncLLMEngine:
    requested = REQUESTED_MAX_MODEL_LEN
    gpu_mem = _auto_gpu_memory_utilization(requested)
    kv_offload = _auto_kv_offloading_size(requested)
    auto_fit = _env_bool("AUTO_FIT_MAX_MODEL_LEN", True)

    attempts: list[tuple[str, int, float]] = [
        ("requested", requested, gpu_mem),
    ]
    if auto_fit and requested != -1:
        attempts.append(("auto-fit", -1, gpu_mem))
        if not _env_explicit("GPU_MEMORY_UTILIZATION") and gpu_mem < 0.95:
            attempts.append(("auto-fit", -1, 0.95))

    last_error: Exception | None = None
    for label, max_len, mem_frac in attempts:
        logger.info(
            "Starting engine (%s): max_model_len=%s gpu_memory_utilization=%.2f "
            "kv_offloading_size=%.1f GiB",
            label,
            "auto" if max_len == -1 else max_len,
            mem_frac,
            kv_offload,
        )
        try:
            return AsyncLLMEngine.from_engine_args(
                _build_engine_args(
                    max_model_len=max_len,
                    gpu_memory_utilization=mem_frac,
                    kv_offloading_size=kv_offload,
                )
            )
        except RuntimeError as exc:
            last_error = exc
            logger.warning(
                "Engine startup failed (%s): %s",
                label,
                exc,
            )

    assert last_error is not None
    raise last_error


_THINKING_BLOCK = re.compile(
    r"<\s*(?:think|redacted_thinking)\s*>[\s\S]*?<\s*/\s*(?:think|redacted_thinking)\s*>",
    re.IGNORECASE,
)
_UNCLOSED_THINKING = re.compile(
    r"^[\s\S]*?<\s*(?:think|redacted_thinking)\s*>",
    re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    cleaned = _THINKING_BLOCK.sub("", text).lstrip()
    return _UNCLOSED_THINKING.sub("", cleaned).lstrip()


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _rename_element_index(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {key: _rename_element_index(item) for key, item in value.items()}
        if "element_index" in normalized and "index" not in normalized:
            normalized["index"] = normalized.pop("element_index")
        else:
            normalized.pop("element_index", None)
        return normalized
    if isinstance(value, list):
        return [_rename_element_index(item) for item in value]
    return value


def _normalize_completion(text: str) -> str:
    cleaned = _strip_thinking(text)
    candidate = _extract_json_text(cleaned)
    if not candidate.startswith(("{", "[")):
        return cleaned
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return cleaned
    return json.dumps(_rename_element_index(parsed), ensure_ascii=False)


engine: AsyncLLMEngine | None = None
tokenizer = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> str:
        return _normalize_message_content(value)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = Field(default=512, alias="max_tokens")
    stream: bool = False
    stop: str | list[str] | None = None

    model_config = {"populate_by_name": True}


def _build_sampling_params(body: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=body.temperature if body.temperature is not None else 0.7,
        top_p=body.top_p if body.top_p is not None else 1.0,
        max_tokens=body.max_tokens if body.max_tokens is not None else 512,
        stop=body.stop,
    )


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    payload = [{"role": m.role, "content": m.content} for m in messages]
    base = {"tokenize": False, "add_generation_prompt": True}
    if not _env_bool("ENABLE_THINKING", False):
        try:
            return tokenizer.apply_chat_template(
                payload,
                **base,
                enable_thinking=False,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(payload, **base)


def _estimate_prompt_tokens(prompt: str) -> int:
    return len(tokenizer.encode(prompt))


def _chat_completion_object(
    *,
    request_id: str,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_chunk(
    *,
    request_id: str,
    model: str,
    delta_content: str | None,
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, tokenizer, effective_max_model_len

    engine = _create_engine()
    effective_max_model_len = engine.model_config.max_model_len
    if effective_max_model_len < REQUESTED_MAX_MODEL_LEN:
        logger.warning(
            "Requested max context %d tokens exceeds GPU KV capacity; serving "
            "%d tokens with %.1f GiB CPU KV offload for spillover.",
            REQUESTED_MAX_MODEL_LEN,
            effective_max_model_len,
            _auto_kv_offloading_size(REQUESTED_MAX_MODEL_LEN),
        )

    maybe_tokenizer = engine.get_tokenizer()
    tokenizer = (
        await maybe_tokenizer if inspect.isawaitable(maybe_tokenizer) else maybe_tokenizer
    )
    yield
    engine = None
    tokenizer = None


app = FastAPI(title="TurboQuant vLLM OpenAI API", lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "max_model_len": effective_max_model_len,
                "context_length": effective_max_model_len,
                "requested_context_length": REQUESTED_MAX_MODEL_LEN,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if body.model != SERVED_MODEL_NAME and body.model != MODEL_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' not found. Use '{SERVED_MODEL_NAME}'.",
        )

    prompt = _messages_to_prompt(body.messages)
    sampling_params = _build_sampling_params(body)
    request_id = uuid.uuid4().hex
    prompt_tokens = _estimate_prompt_tokens(prompt)

    if body.stream:
        return StreamingResponse(
            _stream_chat_completion(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                model=body.model,
                prompt_tokens=prompt_tokens,
            ),
            media_type="text/event-stream",
        )

    final_text = ""
    completion_tokens = 0
    async for output in engine.generate(prompt, sampling_params, request_id):
        final_text = _normalize_completion(output.outputs[0].text)
        completion_tokens = len(output.outputs[0].token_ids)

    return JSONResponse(
        _chat_completion_object(
            request_id=request_id,
            model=body.model,
            content=final_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )


async def _stream_chat_completion(
    *,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model: str,
    prompt_tokens: int,
) -> AsyncGenerator[str, None]:
    assert engine is not None

    previous_text = ""
    async for output in engine.generate(prompt, sampling_params, request_id):
        current_text = _normalize_completion(output.outputs[0].text)
        delta = current_text[len(previous_text) :]
        previous_text = current_text
        if delta:
            yield _stream_chunk(
                request_id=request_id,
                model=model,
                delta_content=delta,
            )

    yield _stream_chunk(
        request_id=request_id,
        model=model,
        delta_content=None,
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"
