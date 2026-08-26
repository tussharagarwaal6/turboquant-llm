#!/usr/bin/env python3
"""Run a Comet-like browser agent against the local turboquant-llm OpenAI API."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BROWSER_DIR = Path(__file__).resolve().parent
REPO_ROOT = BROWSER_DIR.parent


def _load_env() -> None:
    load_dotenv(BROWSER_DIR / ".env")
    load_dotenv(BROWSER_DIR / ".env.example", override=False)


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key, default)
    return raw.strip().strip("\r")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().strip("\r").lower() in {"1", "true", "yes", "on"}


def _read_task(*, task: str | None, task_file: Path | None) -> str:
    if task_file is not None:
        return task_file.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    if task:
        return task.strip()
    raise SystemExit("Provide a task string or --task-file.")


def _build_llm():
    from browser_use import ChatOpenAI

    return ChatOpenAI(
        model=_env_str("BROWSER_LLM_MODEL", "Qwen/Qwen3-14B-AWQ"),
        base_url=_env_str("OPENAI_API_BASE", "http://localhost:8000/v1"),
        api_key=_env_str("OPENAI_API_KEY", "local"),
        temperature=0.2,
        max_completion_tokens=int(_env_str("BROWSER_MAX_COMPLETION_TOKENS", "2048")),
    )


async def _run(task: str) -> str:
    from browser_use import Agent
    from browser_use.browser.profile import BrowserProfile

    headless = _env_bool("BROWSER_HEADLESS", False)
    use_vision = _env_bool("BROWSER_USE_VISION", False)
    max_steps = int(_env_str("BROWSER_MAX_STEPS", "25"))

    profile = BrowserProfile(headless=headless)
    agent = Agent(
        task=task,
        llm=_build_llm(),
        browser_profile=profile,
        use_vision=use_vision,
        use_thinking=False,
    )
    history = await agent.run(max_steps=max_steps)
    return history.final_result() or str(history)


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Browser-use agent for turboquant-llm")
    parser.add_argument("task", nargs="?", help="Natural-language browser task")
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Markdown file containing the task (e.g. browser/tasks/smoke_example.md)",
    )
    args = parser.parse_args()

    task = _read_task(task=args.task, task_file=args.task_file)
    print(f"Task ({len(task)} chars):\n{task[:500]}{'...' if len(task) > 500 else ''}\n", flush=True)

    try:
        result = asyncio.run(_run(task))
    except KeyboardInterrupt:
        raise SystemExit(130) from None

    print("\n--- Result ---")
    print(result)


if __name__ == "__main__":
    main()
