#!/usr/bin/env python3
"""Measure prompt-eval and decode throughput of a running llama.cpp server.

Sends one chat completion and prints the server-reported timings as a TSV row.
A synthetic filler prompt of --prompt-tokens length is prepended so decode is
measured at a realistic KV occupancy instead of a near-empty context.

Usage:
  python3 scripts/probe_llama_speed.py --prompt-tokens 16000 --n-predict 256
  python3 scripts/probe_llama_speed.py --model gemma4-26b-a4b --prompt-tokens 16000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request

# Deterministic filler vocabulary. Common English words tokenize to roughly one
# token each, so the requested token count lands within a few percent.
_WORDS = (
    "system module handler buffer request client server cache index value token "
    "record thread worker socket packet stream branch commit config schema table "
    "column filter mapper parser logger metric window session cursor kernel vector"
).split()

QUESTION = (
    "Ignore the log lines above. Write a Python class ConnectionPool that manages a "
    "fixed-size pool of database connections. Include acquire and release methods "
    "with a timeout, thread safety, and a context manager. Explain the design choices."
)


def filler(n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    rng = random.Random(1234)
    lines = []
    produced = 0
    line_no = 0
    while produced < n_tokens:
        words = " ".join(rng.choice(_WORDS) for _ in range(12))
        lines.append(f"log line {line_no}: {words}")
        produced += 18  # ~12 words + line prefix and newline
        line_no += 1
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="kat-coder-npu")
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    prefix = filler(args.prompt_tokens)
    content = f"{prefix}\n\n{QUESTION}" if prefix else QUESTION

    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": args.n_predict,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"{args.label}\tERROR\tERROR\tERROR\tERROR\t{exc}")
        return 1

    t = data.get("timings") or {}
    drafted = t.get("draft_n") or 0
    accepted = t.get("draft_n_accepted") or 0
    acceptance = f"{100.0 * accepted / drafted:.1f}%" if drafted else "n/a"

    def fmt(v: object) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "NA"

    print(
        "\t".join(
            [
                args.label,
                str(t.get("prompt_n", "NA")),
                fmt(t.get("prompt_per_second")),
                str(t.get("predicted_n", "NA")),
                fmt(t.get("predicted_per_second")),
                acceptance,
            ]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
