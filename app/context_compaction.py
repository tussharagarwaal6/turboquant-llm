"""Compact long chat histories by summarizing older turns before inference."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger("turboquant.context_compaction")

DEFAULT_SUMMARY_PROMPT = """Summarize the conversation history that will be removed from the active context.

Preserve:
- Key decisions, user preferences, and constraints
- Important tool results, code changes, and file references
- Current task state, unresolved questions, and next steps

Be factual and concise. Do not invent details.

Previous summary (if any):
{previous_summary}

Messages being summarized:
{compacted_messages}

Recent messages kept in full (for context only — do not repeat verbatim):
{recent_messages}

Write a single summary the assistant can use to continue the conversation."""


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().strip("\r").lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: str) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: str) -> int:
    return int(os.environ.get(key, default))


def compaction_enabled() -> bool:
    return _env_bool("ENABLE_CONTEXT_COMPACTION", True)


def compaction_threshold_tokens(max_model_len: int) -> int:
    ratio = _env_float("CONTEXT_COMPACTION_THRESHOLD_RATIO", "0.85")
    cap = _env_int("CONTEXT_COMPACTION_TOKEN_CAP", "0")
    threshold = int(max_model_len * ratio)
    if cap > 0:
        threshold = min(threshold, cap)
    reserve = _env_int("CONTEXT_COMPACTION_RESERVE_TOKENS", "512")
    return max(1024, threshold - reserve)


def retention_percentage() -> int:
    value = _env_int("CONTEXT_COMPACTION_RETENTION_PERCENT", "40")
    return min(50, max(10, value))


def summary_max_tokens() -> int:
    return _env_int("CONTEXT_COMPACTION_MAX_SUMMARY_TOKENS", "1024")


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    total = 4
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"image", "image_url"}:
                    total += _estimate_image_tokens(item)
                else:
                    total += estimate_tokens(item.get("text") or item.get("content"))
            else:
                total += estimate_tokens(item)
    else:
        total += estimate_tokens(content)
    total += estimate_tokens(message.get("tool_calls"))
    return total


def _estimate_image_tokens(item: dict[str, Any]) -> int:
    if item.get("type") == "image_url":
        url = (item.get("image_url") or {}).get("url") or ""
        if isinstance(url, str) and url.startswith("data:"):
            return max(1500, len(url) // 4)
        return 1500
    return 1200


def _messages_have_multimodal_content(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "image_url"}:
                return True
    return False


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def count_prompt_tokens(messages: list[dict[str, Any]], tokenizer: Any) -> int:
    heuristic = estimate_messages_tokens(messages)
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = len(tokenizer.encode(prompt))
        if _messages_have_multimodal_content(messages):
            return max(heuristic, encoded)
        return encoded
    except Exception:
        return heuristic


def _message_content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = _message_content_text(message)
        if message.get("tool_calls"):
            content = f"{content}\n[tool_calls present]" if content else "[tool_calls present]"
        if content.strip():
            lines.append(f"{role}: {content[:2000]}")
    return "\n".join(lines)


def _find_compaction_boundary(messages: list[dict[str, Any]], retention_pct: int) -> int:
    if len(messages) <= 3:
        return 0
    keep_count = max(2, len(messages) * retention_pct // 100)
    target = max(1, len(messages) - keep_count)
    user_boundaries = [idx for idx, message in enumerate(messages) if message.get("role") == "user"][1:]
    if not user_boundaries:
        return max(0, len(messages) - keep_count)
    for idx in reversed(user_boundaries):
        if idx <= target:
            return idx
    return 0


def _extract_previous_summary(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    for idx, message in enumerate(messages):
        content = _message_content_text(message)
        if content.startswith("[CONVERSATION SUMMARY]"):
            return messages[idx + 1 :], content.removeprefix("[CONVERSATION SUMMARY]").strip()
    return messages, None


def build_summary_prompt(
    *,
    compacted_messages: list[dict[str, Any]],
    recent_messages: list[dict[str, Any]],
    previous_summary: str | None,
) -> str:
    template = os.environ.get("CONTEXT_COMPACTION_PROMPT", DEFAULT_SUMMARY_PROMPT)
    return template.format(
        previous_summary=previous_summary or "(none)",
        compacted_messages=_format_messages_for_summary(compacted_messages) or "(empty)",
        recent_messages=_format_messages_for_summary(recent_messages) or "(empty)",
    )


def apply_summary(messages: list[dict[str, Any]], summary: str) -> list[dict[str, Any]]:
    summary_message = {
        "role": "user",
        "content": f"[CONVERSATION SUMMARY]\n{summary.strip()}",
    }
    return [summary_message, *messages]


def tool_pruning_enabled() -> bool:
    return _env_bool("ENABLE_TOOL_PRUNING", True)


def tool_prune_max_result_chars() -> int:
    return _env_int("TOOL_PRUNE_MAX_RESULT_CHARS", "2000")


def _tool_call_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _compact_tool_content(content: Any, *, max_chars: int) -> str:
    if content is None:
        return "(empty tool result)"
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    stripped = content.strip()
    if not stripped:
        return "(empty tool result)"

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped[:max_chars] + ("..." if len(stripped) > max_chars else "")

    if isinstance(parsed, dict):
        if isinstance(parsed.get("text"), str) and parsed["text"].strip():
            text = parsed["text"].strip()
            return text[:max_chars] + ("..." if len(text) > max_chars else "")
        if isinstance(parsed.get("error"), str) and parsed["error"].strip():
            return f"error: {parsed['error'].strip()[:max_chars]}"
        if isinstance(parsed.get("results"), list):
            parts: list[str] = []
            for item in parsed["results"]:
                if not isinstance(item, dict):
                    continue
                label = item.get("label") or item.get("source") or item.get("index")
                if isinstance(item.get("text"), str) and item["text"].strip():
                    parts.append(f"{label}: {item['text'].strip()}")
                elif isinstance(item.get("error"), str) and item["error"].strip():
                    parts.append(f"{label}: error: {item['error'].strip()}")
            if parts:
                joined = "\n".join(parts)
                return joined[:max_chars] + ("..." if len(joined) > max_chars else "")

    compact = json.dumps(parsed, ensure_ascii=False)
    return compact[:max_chars] + ("..." if len(compact) > max_chars else "")


def _ends_with_pending_tool_calls(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return last.get("role") == "assistant" and bool(last.get("tool_calls"))


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.add(str(tool_call["id"]))
    return ids


def _collapse_completed_tool_turn(
    messages: list[dict[str, Any]],
    start: int,
    *,
    max_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    """Replace assistant(tool_calls)+tool messages with one compact user message."""
    assistant = messages[start]
    names = _tool_call_names(assistant)
    expected_ids = _tool_call_ids(assistant)

    end = start + 1
    tool_messages: list[dict[str, Any]] = []
    while end < len(messages) and messages[end].get("role") == "tool":
        tool_message = messages[end]
        tool_call_id = tool_message.get("tool_call_id")
        if expected_ids and tool_call_id and str(tool_call_id) not in expected_ids:
            break
        tool_messages.append(tool_message)
        end += 1

    if not tool_messages:
        return messages, start + 1

    label = ", ".join(names) if names else "tool"
    result_lines = [
        _compact_tool_content(tool_message.get("content"), max_chars=max_chars)
        for tool_message in tool_messages
    ]
    collapsed = {
        "role": "user",
        "content": f"[TOOL RESULT: {label}]\n" + "\n".join(result_lines),
    }
    return [*messages[:start], collapsed, *messages[end:]], end


def prune_tool_messages(
    messages: list[dict[str, Any]],
    *,
    active_tool_request: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Collapse completed tool turns to compact result messages to save context."""
    if not tool_pruning_enabled() or not messages:
        return messages, False

    max_chars = tool_prune_max_result_chars()
    pruned = list(messages)
    changed = False
    idx = 0

    while idx < len(pruned):
        message = pruned[idx]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            idx += 1
            continue

        pending_at_end = _ends_with_pending_tool_calls(pruned) and idx == len(pruned) - 1
        if pending_at_end:
            break

        next_idx = idx + 1
        has_tool_results = next_idx < len(pruned) and pruned[next_idx].get("role") == "tool"
        if not has_tool_results:
            idx += 1
            continue

        # Keep the latest completed tool turn verbatim while a tool request is active.
        later_tool_turns = any(
            later.get("role") == "assistant" and later.get("tool_calls")
            for later in pruned[idx + 1 :]
        )
        if active_tool_request and not later_tool_turns:
            break

        before = json.dumps(pruned, ensure_ascii=False)
        pruned, idx = _collapse_completed_tool_turn(pruned, idx, max_chars=max_chars)
        if json.dumps(pruned, ensure_ascii=False) != before:
            changed = True
            logger.info("Pruned completed tool turn -> compact TOOL RESULT message")
        # Do not increment idx: collapsed message sits at idx now.

    return pruned, changed


def _messages_have_tool_context(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") == "tool":
            return True
        if message.get("tool_calls"):
            return True
    return False


def should_skip_compaction(body: dict[str, Any]) -> bool:
    """Skip full summarization while a tool turn is still in flight."""
    messages = body.get("messages") or []
    if isinstance(messages, list) and _ends_with_pending_tool_calls(messages):
        return True
    return False


async def maybe_compact_messages(
    messages: list[dict[str, Any]],
    *,
    max_model_len: int,
    tokenizer: Any,
    summarize: Callable[[str], Any],
    force: bool = False,
    aggressive: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Compact messages when near context limit; `summarize` generates summary text."""
    if not compaction_enabled() and not force:
        return messages, False

    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    non_system, previous_summary = _extract_previous_summary(non_system)

    threshold = compaction_threshold_tokens(max_model_len)
    estimated = count_prompt_tokens([*system_messages, *non_system], tokenizer)
    if not force and (estimated <= threshold or len(non_system) <= 3):
        return messages, False
    if len(non_system) <= 1:
        return messages, False

    retention_pct = retention_percentage()
    if aggressive:
        retention_pct = min(20, retention_pct)

    boundary = _find_compaction_boundary(non_system, retention_pct)
    compacted = non_system[:boundary]
    recent = non_system[boundary:]
    if not compacted or not recent:
        if aggressive and len(non_system) > 2:
            boundary = max(1, len(non_system) // 2)
            compacted = non_system[:boundary]
            recent = non_system[boundary:]
        if not compacted or not recent:
            return messages, False

    prompt = build_summary_prompt(
        compacted_messages=compacted,
        recent_messages=recent,
        previous_summary=previous_summary,
    )
    summary = (await summarize(prompt)).strip()
    if not summary:
        parts = [previous_summary] if previous_summary else []
        for message in compacted:
            text = _message_content_text(message)
            if text:
                parts.append(f"- {message.get('role', 'unknown')}: {text[:500]}")
        summary = "\n".join(parts)[:4000]

    logger.info(
        "Context compaction: %d -> %d messages (~%d tokens, threshold %d)",
        len(messages),
        len(system_messages) + 1 + len(recent),
        estimated,
        threshold,
    )
    return [*system_messages, *apply_summary(recent, summary)], True
