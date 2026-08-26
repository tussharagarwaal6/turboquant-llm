"""OpenAI-compatible FastAPI server backed by vLLM AsyncLLMEngine with TurboQuant KV cache."""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# WSL2 GPU passthrough does not expose UVA, which the V2 model runner requires.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

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

engine: AsyncLLMEngine | None = None
tokenizer = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


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
    return tokenizer.apply_chat_template(
        payload,
        tokenize=False,
        add_generation_prompt=True,
    )


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
    global engine, tokenizer

    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        quantization="awq",
        kv_cache_dtype="turboquant_k8v4",
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "16384")),
        trust_remote_code=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

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
        final_text = output.outputs[0].text
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
        current_text = output.outputs[0].text
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
