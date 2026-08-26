"""Introspect the vLLM async engine API surface for this installed version."""

import inspect

import vllm

print("vllm:", vllm.__version__)
print("top-level exports:", [n for n in dir(vllm) if "Async" in n or n in ("LLM", "SamplingParams")])

try:
    from vllm.v1.engine.async_llm import AsyncLLM

    print("\nAsyncLLM (v1) available")
    print("  from_engine_args:", hasattr(AsyncLLM, "from_engine_args"))
    print("  from_vllm_config:", hasattr(AsyncLLM, "from_vllm_config"))
    print("  generate sig:", inspect.signature(AsyncLLM.generate))
    print("  tokenizer attrs:", [a for a in dir(AsyncLLM) if "token" in a.lower()])
except Exception as exc:
    print("AsyncLLM (v1) import failed:", exc)

try:
    from vllm import AsyncLLMEngine

    print("\nAsyncLLMEngine available")
    print("  from_engine_args:", hasattr(AsyncLLMEngine, "from_engine_args"))
    print("  generate sig:", inspect.signature(AsyncLLMEngine.generate))
    print("  tokenizer attrs:", [a for a in dir(AsyncLLMEngine) if "token" in a.lower()])
except Exception as exc:
    print("AsyncLLMEngine import failed:", exc)

for path, name in [
    ("vllm.transformers_utils.tokenizer", "get_tokenizer"),
    ("vllm.inputs", "TokensPrompt"),
]:
    try:
        mod = __import__(path, fromlist=[name])
        print(f"OK {path}.{name}:", hasattr(mod, name))
    except Exception as exc:
        print(f"FAIL {path}.{name}:", exc)
