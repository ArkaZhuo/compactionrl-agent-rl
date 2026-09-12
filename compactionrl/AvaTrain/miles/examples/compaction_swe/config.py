"""Configuration for the isolated CompactionRL experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass

AUXILIARY_MODES = {"serve_untracked", "reject"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _required_positive_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"{name} must be provided by the model-aware launcher")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class CompactionConfig:
    """Runtime limits; the values are deliberately independent of normal SWE PPO."""

    context_budget: int
    model_sequence_limit: int
    trigger_tokens: int
    per_turn_tokens: int
    summary_tokens: int
    max_compactions: int
    recent_steps: int
    auxiliary_mode: str = "serve_untracked"

    @classmethod
    def from_env(cls) -> "CompactionConfig":
        context_budget = _positive_int("COMPACTION_CONTEXT_BUDGET", 65536)
        model_sequence_limit = _required_positive_int("COMPACTION_MODEL_SEQUENCE_LIMIT")
        trigger_tokens = _positive_int("COMPACTION_TRIGGER_TOKENS", 10240)
        per_turn_tokens = _positive_int("COMPACTION_MAX_TOKENS_PER_TURN", 2048)
        summary_tokens = _positive_int("COMPACTION_SUMMARY_MAX_TOKENS", 2048)
        max_compactions = int(os.environ.get("COMPACTION_MAX_COUNT", "3"))
        recent_steps = int(os.environ.get("COMPACTION_RECENT_STEPS", "2"))
        auxiliary_mode = os.environ.get("COMPACTION_AUXILIARY_MODE", "serve_untracked").strip().lower()

        if model_sequence_limit < context_budget:
            raise ValueError("COMPACTION_MODEL_SEQUENCE_LIMIT must be >= COMPACTION_CONTEXT_BUDGET")
        if trigger_tokens >= context_budget:
            raise ValueError("COMPACTION_TRIGGER_TOKENS must be smaller than the context budget")
        if trigger_tokens < max(per_turn_tokens, summary_tokens):
            raise ValueError(
                "COMPACTION_TRIGGER_TOKENS must reserve at least one execution turn and one summary"
            )
        if not 0 <= max_compactions <= 3:
            raise ValueError("COMPACTION_MAX_COUNT must be between 0 and 3")
        if recent_steps < 0:
            raise ValueError("COMPACTION_RECENT_STEPS must be non-negative")
        if auxiliary_mode not in AUXILIARY_MODES:
            raise ValueError(
                "COMPACTION_AUXILIARY_MODE must be one of "
                f"{sorted(AUXILIARY_MODES)}, got {auxiliary_mode!r}"
            )

        return cls(
            context_budget=context_budget,
            model_sequence_limit=model_sequence_limit,
            trigger_tokens=trigger_tokens,
            per_turn_tokens=per_turn_tokens,
            summary_tokens=summary_tokens,
            max_compactions=max_compactions,
            recent_steps=recent_steps,
            auxiliary_mode=auxiliary_mode,
        )

    def needs_compaction(self, prompt_tokens: int) -> bool:
        return self.max_compactions > 0 and prompt_tokens >= self.context_budget - self.trigger_tokens

    def available_generation_tokens(self, prompt_tokens: int, requested: int) -> int:
        # Every execution and summary call belongs to one C-token working
        # window. The model-native limit remains a final safety check, but it
        # must not silently turn the fourth window into a 262K request.
        sequence_limit = min(self.context_budget, self.model_sequence_limit)
        return max(0, min(int(requested), sequence_limit - prompt_tokens))

    def available_auxiliary_tokens(self, prompt_tokens: int, requested: int) -> int:
        """Bound an untracked helper response by the model context only.

        Auxiliary requests do not belong to the CompactionRL working window,
        so applying ``context_budget`` here could return an empty response for
        a valid helper call even though the model-native context still fits.
        """
        return max(0, min(int(requested), self.model_sequence_limit - prompt_tokens))
