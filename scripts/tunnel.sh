#!/bin/bash
# Expose local turboquant-llm (:8000) via Cloudflare quick tunnel for Cursor.
# Cursor routes API calls through its cloud — localhost is blocked without a public URL.

set -euo pipefail

PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"
CLOUDFLARED="${CLOUDFLARED:-$HOME/bin/cloudflared}"

if ! curl -sf "${URL}/v1/models" >/dev/null 2>&1; then
	echo "Local API not reachable at ${URL}. Start the server first:" >&2
	echo "  bash scripts/serve.sh" >&2
	exit 1
fi

if [ ! -x "$CLOUDFLARED" ]; then
	echo "Installing cloudflared to ${CLOUDFLARED}..." >&2
	mkdir -p "$(dirname "$CLOUDFLARED")"
	curl -fsSL -o "$CLOUDFLARED" \
		https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
	chmod +x "$CLOUDFLARED"
fi

echo "Starting Cloudflare tunnel -> ${URL}" >&2
echo "Use in Cursor: Base URL = https://<tunnel-host>/v1" >&2
echo "Press Ctrl+C to stop." >&2
echo >&2

exec "$CLOUDFLARED" tunnel --url "$URL"
