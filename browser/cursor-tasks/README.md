# Cursor browser tasks (no install)

Use these in **Cursor Agent mode**. The built-in browser MCP drives Chromium from the IDE.

## Before you start

1. Start turboquant-llm: `uvicorn app.server:app --host 0.0.0.0 --port 8000`
2. For Open WebUI tasks, also start Open WebUI on port 3000
3. Paste the contents of a task file below into Agent chat

## Tasks

| File | Goal |
|------|------|
| [01-api-docs.md](01-api-docs.md) | Verify FastAPI `/docs` and `/v1/models` |
| [02-openwebui-chat.md](02-openwebui-chat.md) | Send a chat via Open WebUI |
| [03-example-com.md](03-example-com.md) | Minimal public-site smoke test |

## Example prompt wrapper

```
Use the browser MCP tools to complete this task. Take screenshots at key steps.

<paste task file contents here>
```
