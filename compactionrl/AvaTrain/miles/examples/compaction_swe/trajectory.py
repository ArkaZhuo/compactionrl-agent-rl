"""Token ledgers for independently optimized CompactionRL segments."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from miles.utils.types import Sample


def _decode_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        for tool_call in message.get("tool_calls") or []:
            arguments = tool_call["function"].get("arguments")
            if isinstance(arguments, str):
                tool_call["function"]["arguments"] = json.loads(arguments)
    return normalized


def render_prompt_ids(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
    text = tokenizer.apply_chat_template(
        _decode_tool_arguments(messages),
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=False,
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class TokenLedger:
    segment_type: str
    token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    response_text: str = ""
    finish_reason: str = ""

    def has_trainable_tokens(self) -> bool:
        return 1 in self.loss_mask

    def extends(self, prompt_ids: list[int]) -> bool:
        return prompt_ids[: len(self.token_ids)] == self.token_ids

    def pending_context_tokens(self, prompt_ids: list[int]) -> int:
        """Return new environment tokens since this segment's last completion."""
        if not self.extends(prompt_ids):
            raise RuntimeError("segment prompt no longer extends its token ledger")
        return len(prompt_ids) - len(self.token_ids)

    def append_context(self, prompt_ids: list[int]) -> int:
        """Append a tool/user observation tail without making it trainable."""
        tail_length = self.pending_context_tokens(prompt_ids)
        if tail_length:
            self.token_ids.extend(prompt_ids[-tail_length:])
            self.loss_mask.extend([0] * tail_length)
            self.log_probs.extend([0.0] * tail_length)
        return tail_length

    def append_turn(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        completion_log_probs: list[float],
        completion_text: str,
        finish_reason: str,
    ) -> None:
        if len(completion_ids) != len(completion_log_probs):
            raise RuntimeError("completion token/log-prob lengths differ")
        self.append_context(prompt_ids)
        self.token_ids.extend(completion_ids)
        self.loss_mask.extend([1] * len(completion_ids))
        self.log_probs.extend(completion_log_probs)
        self.response_text += completion_text
        self.finish_reason = finish_reason

    def to_sample(
        self,
        base_sample: Sample,
        *,
        reward: float,
        max_sequence_length: int,
        max_response_length: int,
        trajectory_id: str,
        segment_index: int,
        future_optimized_tokens: int,
        compaction_count: int,
    ) -> Sample:
        if not self.has_trainable_tokens():
            raise RuntimeError(f"{self.segment_type} segment has no trainable tokens")
        first_trainable = self.loss_mask.index(1)
        response_end = min(
            len(self.token_ids),
            first_trainable + max_response_length,
            max_sequence_length,
        )
        if response_end <= first_trainable:
            raise RuntimeError(
                f"{self.segment_type} segment has no room for a trainable token within "
                f"COMPACTION_MODEL_SEQUENCE_LIMIT={max_sequence_length}"
            )
        was_truncated = response_end < len(self.token_ids)
        sample = copy.deepcopy(base_sample)
        sample.tokens = list(self.token_ids[:response_end])
        sample.response = self.response_text
        sample.response_length = response_end - first_trainable
        sample.loss_mask = list(self.loss_mask[first_trainable:response_end])
        sample.rollout_log_probs = list(self.log_probs[first_trainable:response_end])
        sample.reward = float(reward)
        sample.status = Sample.Status.TRUNCATED if was_truncated else Sample.Status.COMPLETED
        sample.train_metadata = {
            "compaction": True,
            "trajectory_id": trajectory_id,
            "segment_index": segment_index,
            "segment_type": self.segment_type,
            "optimized_tokens": int(sum(sample.loss_mask)),
            "future_optimized_tokens": int(future_optimized_tokens),
            "compaction_count": int(compaction_count),
            "finish_reason": self.finish_reason,
        }
        return sample


@dataclass
class SegmentLedger:
    trajectory_id: str
    segments: list[TokenLedger] = field(default_factory=list)
    current: TokenLedger | None = None

    def has_trainable_tokens(self) -> bool:
        return any(segment.has_trainable_tokens() for segment in self.segments) or (
            self.current is not None and self.current.has_trainable_tokens()
        )

    def begin(self, segment_type: str) -> TokenLedger:
        if self.current is not None:
            raise RuntimeError("cannot begin a segment before finalizing the current one")
        self.current = TokenLedger(segment_type=segment_type)
        return self.current

    def finish_current(self, *, allow_empty: bool = False) -> TokenLedger | None:
        current = self.current
        if current is None:
            return None
        if not current.has_trainable_tokens():
            if allow_empty:
                self.current = None
                return None
            raise RuntimeError("cannot finalize an empty training segment")
        self.segments.append(current)
        self.current = None
        return current

    def to_samples(
        self,
        base_sample: Sample,
        reward: float,
        max_sequence_length: int,
        compaction_count: int,
    ) -> list[Sample]:
        # A rollout can end immediately after a compaction (for example when
        # the CLI exits after receiving a summary). Do not emit an empty
        # execution segment, but still retain all preceding trainable segments.
        self.finish_current(allow_empty=True)
        if not self.segments:
            raise RuntimeError("trajectory contains no trainable execution or summary segment")
        plans: list[tuple[int, TokenLedger, int, int]] = []
        for index, segment in enumerate(self.segments):
            first_trainable = segment.loss_mask.index(1)
            full_response_length = len(segment.token_ids) - first_trainable
            sequence_room = max_sequence_length - first_trainable
            response_length = min(full_response_length, sequence_room)
            if response_length <= 0:
                raise RuntimeError(
                    f"{segment.segment_type} segment has no trainable token within "
                    f"the {max_sequence_length}-token working window"
                )
            optimized = sum(segment.loss_mask[first_trainable : first_trainable + response_length])
            plans.append((index, segment, response_length, optimized))

        if not plans:
            raise RuntimeError("trajectory contains no trainable tokens within its configured limits")

        future_counts = [0] * len(plans)
        future = 0
        for plan_index in range(len(plans) - 1, -1, -1):
            future_counts[plan_index] = future
            future += plans[plan_index][3]

        return [
            segment.to_sample(
                base_sample,
                reward=reward,
                max_sequence_length=max_sequence_length,
                max_response_length=response_length,
                trajectory_id=self.trajectory_id,
                segment_index=segment_index,
                future_optimized_tokens=future_counts[plan_index],
                compaction_count=compaction_count,
            )
            for plan_index, (segment_index, segment, response_length, _) in enumerate(plans)
        ]
