#!/bin/bash
# Run the browser-use agent against local turboquant-llm (WSL2).

APP_DIR=/mnt/c/dev/turboquant-llm
VENV=""

for candidate in "$APP_DIR/.venv" "$HOME/turboquant-llm/.venv"; do
	if [ -x "$candidate/bin/python" ] && "$candidate/bin/python" -c "import browser_use" 2>/dev/null; then
		VENV="$candidate"
		break
	fi
done

if [ -z "$VENV" ]; then
	echo "No venv with browser-use found. Install into project venv:" >&2
	echo "  $APP_DIR/.venv/bin/pip install -r $APP_DIR/browser/requirements-browser.txt" >&2
	echo "  playwright install chromium" >&2
	exit 1
fi

cd "$APP_DIR" || exit 1
# Env vars loaded by browser/run_agent.py (avoids CRLF issues from shell source)
exec "$VENV/bin/python" browser/run_agent.py "$@"
