# Open WebUI end-to-end smoke test

1. Navigate to http://localhost:3000
2. If a login screen appears, sign in or create an account (first-run only)
3. Start a new chat
4. Select model Qwen/Qwen3-14B-AWQ if a model picker exists
5. Send: "Say hello in one sentence."
6. Wait until the assistant response finishes streaming
7. Return PASS if the response has at least 5 characters, else FAIL with reason
