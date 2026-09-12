"""Fixed compaction and resume prompts.

The paper does not publish its literal templates. Keeping ours in one file makes
that experiment choice explicit and lets checkpoint preflight hash the templates.
"""

from __future__ import annotations

import copy
from typing import Any


SUMMARY_INSTRUCTION = """Summarize the current software-engineering task so another agent can continue it.
Preserve the original goal, repository and file paths, changes already made, commands and their
important outputs, failures and unresolved issues, tests and their results, the current working
state, and concrete next steps. Keep exact identifiers, error messages, and constraints when they
matter. Do not invent progress. Return only the actionable summary."""

RESUME_PREFIX = """The previous interaction history was compacted. Continue the same task from this summary.
Use the recent tool observations as the authoritative current state, and do not repeat completed work.

COMPACTED SUMMARY:
"""

TRUNCATION_MARKER = "\n\n[... tool observation truncated for context compaction ...]\n\n"


def summary_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*messages, {"role": "user", "content": SUMMARY_INSTRUCTION}]


def truncate_tool_observations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Halve the largest tool observation while retaining its beginning and end."""
    result = copy.deepcopy(messages)
    candidates = [
        (len(str(message.get("content") or "")), index)
        for index, message in enumerate(result)
        if message.get("role") == "tool" and isinstance(message.get("content"), str)
    ]
    if not candidates:
        return result
    length, index = max(candidates)
    if length <= len(TRUNCATION_MARKER) + 2:
        return result
    content = result[index]["content"]
    retained = max(2, length // 2 - len(TRUNCATION_MARKER))
    head = max(1, retained // 3)
    tail = max(1, retained - head)
    result[index]["content"] = content[:head] + TRUNCATION_MARKER + content[-tail:]
    return result


def resume_messages(
    messages: list[dict[str, Any]], summary: str, recent_steps: int
) -> list[dict[str, Any]]:
    """Build system + resume(summary) + the last complete atomic steps."""
    prefix, steps = split_atomic_steps(messages)
    kept: list[dict[str, Any]] = []
    if recent_steps:
        for step in steps[-recent_steps:]:
            kept.extend(step)

    # Preserve system messages and the original user request only through the
    # generated summary. The original user request is intentionally not copied
    # separately, matching the paper's reconstructed context.
    system = [m for m in prefix if m.get("role") == "system"]
    return [*system, {"role": "user", "content": f"{RESUME_PREFIX}\n{summary}"}, *kept]


def split_atomic_steps(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Split after the initial prompt into assistant action/observation atoms."""
    first_assistant = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
    prefix = list(messages[:first_assistant])
    steps: list[list[dict[str, Any]]] = []
    i = first_assistant
    while i < len(messages):
        if messages[i].get("role") != "assistant":
            # Unexpected user/tool material before an assistant action is kept
            # with the previous atom instead of being silently discarded.
            if steps:
                steps[-1].append(messages[i])
            else:
                prefix.append(messages[i])
            i += 1
            continue
        step = [messages[i]]
        i += 1
        while i < len(messages) and messages[i].get("role") != "assistant":
            step.append(messages[i])
            i += 1
        steps.append(step)
    return prefix, steps
