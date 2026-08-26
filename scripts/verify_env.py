"""Verify PyTorch/sm_120, vLLM version, and TurboQuant KV-cache dtype availability."""

import torch
import vllm

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("arch list:", torch.cuda.get_arch_list())
print("sm_120:", "sm_120" in torch.cuda.get_arch_list())
print("vllm:", vllm.__version__)

from vllm.engine.arg_utils import EngineArgs

parser = EngineArgs.add_cli_args(__import__("vllm.utils.argparse_utils", fromlist=["FlexibleArgumentParser"]).FlexibleArgumentParser())
for action in parser._actions:
    if "--kv-cache-dtype" in action.option_strings:
        print("kv-cache-dtype choices:", action.choices)
        break
else:
    print("kv-cache-dtype action not found")
