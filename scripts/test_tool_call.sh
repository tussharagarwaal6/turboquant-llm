#!/bin/bash
# Smoke test: server must return OpenAI tool_calls for native function calling.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-14B-AWQ}"

payload=$(MODEL="$MODEL" python3 - <<'PY'
import json
import os

model = os.environ["MODEL"]
print(json.dumps({
    "model": model,
    "messages": [
        {"role": "user", "content": "Search the web for todays weather in Delhi"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the internet",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "max_tokens": 256,
    "enable_thinking": False,
}))
PY
)

echo "POST ${BASE_URL}/chat/completions (model=${MODEL})"
response=$(curl -sS -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$payload")

echo "$response" | python3 -m json.tool

finish_reason=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['finish_reason'])")
tool_name=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); tc=d['choices'][0]['message'].get('tool_calls') or []; print(tc[0]['function']['name'] if tc else '')")

if [[ "$finish_reason" == "tool_calls" && "$tool_name" == "search_web" ]]; then
  echo "PASS: tool_calls returned (search_web)"
  exit 0
fi

echo "FAIL: expected finish_reason=tool_calls and search_web, got finish_reason=$finish_reason tool_name=$tool_name"
exit 1
