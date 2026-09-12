"""OpenAI-compatible proxy with trainable context compaction."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from miles.utils.http_utils import post

from .config import CompactionConfig
from .prompts import resume_messages, summary_messages, truncate_tool_observations
from .trajectory import SegmentLedger, render_prompt_ids

MODEL_NAME = "default"
logger = logging.getLogger(__name__)


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


PROXY_HTTP_TIMEOUT_SECONDS = _positive_int_env("COMPACTION_PROXY_HTTP_TIMEOUT_SEC", 300)
# miles.utils.http_utils.post calls this max_retries, but its loop treats the
# value as the total number of attempts.  Keep it at one for generation calls.
PROXY_HTTP_MAX_ATTEMPTS = _positive_int_env("COMPACTION_PROXY_HTTP_MAX_ATTEMPTS", 1)


def _tool_call(name: str, arguments: Any, index: int) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "type": "function",
        "id": f"call_{uuid.uuid4().hex}_{index}",
        "function": {"name": name, "arguments": arguments},
    }


def parse_completion(
    text: str,
    *,
    tools: list[dict[str, Any]],
    tool_parser: str | None,
    reasoning_parser: str | None,
) -> dict[str, Any]:
    reasoning, body = "", text
    if reasoning_parser:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        reasoning, body = ReasoningParser(model_type=reasoning_parser, stream_reasoning=False).parse_non_stream(text)
        reasoning, body = reasoning or "", body or ""

    tool_calls: list[dict[str, Any]] = []
    if tool_parser and tools:
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        schemas = [Tool(**tool) for tool in tools if tool.get("type") == "function"]
        parser = FunctionCallParser(tools=schemas, tool_call_parser=tool_parser)
        if parser.has_tool_call(body):
            body, calls = parser.parse_non_stream(body)
            tool_calls = [_tool_call(call.name, call.parameters or "{}", i) for i, call in enumerate(calls)]

    message: dict[str, Any] = {"role": "assistant", "content": body}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    """Ignore transport-only tool-call ids when matching CLI history replays."""
    result = copy.deepcopy(message)
    result.pop("id", None)
    result.pop("tool_call_id", None)
    for call in result.get("tool_calls") or []:
        call.pop("id", None)
        function = call.get("function") or {}
        arguments = function.get("arguments")
        # qwen-code may replay a tool call with arguments as an object even
        # though the OpenAI response used a JSON string (or the reverse). Use
        # one canonical representation so a harmless transport conversion does
        # not desynchronize the raw-history prefix.
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                pass
    if result.get("content") is None:
        result["content"] = ""
    return result


def _messages_extend(prefix: list[dict[str, Any]], messages: list[dict[str, Any]]) -> bool:
    if len(messages) < len(prefix):
        return False
    return all(
        _canonical_message(a) == _canonical_message(b) for a, b in zip(prefix, messages[: len(prefix)], strict=True)
    )


def _common_token_prefix_length(left: list[int], right: list[int]) -> int:
    common = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        common += 1
    return common


def _common_message_prefix_length(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    common = 0
    for left_message, right_message in zip(left, right, strict=False):
        if _canonical_message(left_message) != _canonical_message(right_message):
            break
        common += 1
    return common


def _align_tool_call_ids(
    previous: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
    tail: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore proxy tool IDs on replayed tool observations.

    OpenAI clients are allowed to replace transport IDs while replaying a
    conversation. The IDs are ignored for prefix comparison, but the rendered
    conversation still needs the assistant call ID and its following tool
    message ID to agree. Pair calls by their stable order in the validated raw
    prefix and rewrite only the newly appended tail.
    """
    old_ids: list[str] = []
    replay_ids: list[str] = []
    for old_message, replay_message in zip(previous, replayed, strict=True):
        for old_call, replay_call in zip(
            old_message.get("tool_calls") or [], replay_message.get("tool_calls") or [], strict=False
        ):
            old_id = old_call.get("id")
            replay_id = replay_call.get("id")
            if old_id and replay_id:
                old_ids.append(str(old_id))
                replay_ids.append(str(replay_id))
    mapping = dict(zip(replay_ids, old_ids, strict=True))
    normalized = copy.deepcopy(tail)
    for message in normalized:
        tool_call_id = message.get("tool_call_id")
        if tool_call_id in mapping:
            message["tool_call_id"] = mapping[tool_call_id]
    return normalized


@dataclass
class CompactionModelProxy:
    tokenizer: Any
    loop: asyncio.AbstractEventLoop
    model_url: str
    sampling_params: dict[str, Any]
    config: CompactionConfig
    tool_parser: str | None = None
    reasoning_parser: str | None = None

    ledger: SegmentLedger = field(default_factory=lambda: SegmentLedger(trajectory_id=uuid.uuid4().hex), init=False)
    working_messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _raw_prefix: list[dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _compaction_count: int = field(default=0, init=False)
    _generated_tokens: int = field(default=0, init=False)
    _episode_tokens: int = field(default=0, init=False)
    _budget_exhausted: bool = field(default=False, init=False)
    _truncated_observations: int = field(default=0, init=False)
    _retained_recent_steps: list[int] = field(default_factory=list, init=False)
    _routing_token_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _rebase_count: int = field(default=0, init=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _lifecycle: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)
    _inflight: set[Future] = field(default_factory=set, init=False, repr=False)
    _active_completions: int = field(default=0, init=False, repr=False)
    _pending_generations: int = field(default=0, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _request_sequence: int = field(default=0, init=False, repr=False)
    _last_trajectory_request_id: str | None = field(default=None, init=False, repr=False)
    _route_request_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _route_prompt_tokens: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _route_generated_tokens: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _route_failure_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _max_active_requests: int = field(default=0, init=False, repr=False)
    _concurrent_request_failures: int = field(default=0, init=False, repr=False)

    def start(self) -> "CompactionModelProxy":
        with self._lifecycle:
            self._closing = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        server.proxy = self  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server, self._thread = server, thread
        return self

    def close(self, timeout_seconds: float = 30.0) -> None:
        if self._server is None:
            return

        server, thread = self._server, self._thread
        with self._lifecycle:
            self._closing = True
        server.shutdown()

        deadline = time.monotonic() + timeout_seconds
        with self._lifecycle:
            while self._active_completions or self._inflight or self._pending_generations:
                for future in tuple(self._inflight):
                    future.cancel()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    active = self._active_completions
                    inflight = len(self._inflight)
                    pending = self._pending_generations
                    break
                self._lifecycle.wait(timeout=remaining)
            else:
                active = inflight = pending = 0

        server.server_close()
        if thread is not None:
            thread.join(timeout=3)
        self._server = self._thread = None
        if active or inflight or pending:
            raise TimeoutError(
                "CompactionModelProxy did not drain before shutdown "
                f"(active_completions={active}, inflight_futures={inflight}, "
                f"pending_generations={pending})"
            )

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("proxy is not started")
        return int(self._server.server_address[1])

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    @property
    def episode_tokens(self) -> int:
        """Tokens after the first trainable token, excluding fixed/rebuilt prompts."""
        return self._episode_tokens

    @property
    def truncated_observations(self) -> int:
        return self._truncated_observations

    @property
    def retained_recent_steps(self) -> tuple[int, ...]:
        return tuple(self._retained_recent_steps)

    @property
    def rebase_count(self) -> int:
        return self._rebase_count

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def routing_metadata(self) -> dict[str, int]:
        main_requests = self._route_request_counts.get("main", 0)
        summary_requests = self._route_request_counts.get("summary", 0)
        auxiliary_requests = self._route_request_counts.get("auxiliary", 0)
        replay_rewrite_requests = self._route_request_counts.get("replay_rewrite", 0)
        main_tokens = self._route_generated_tokens.get("main", 0)
        summary_tokens = self._route_generated_tokens.get("summary", 0)
        auxiliary_tokens = self._route_generated_tokens.get("auxiliary", 0)
        replay_rewrite_tokens = self._route_generated_tokens.get("replay_rewrite", 0)
        accounted_tokens = main_tokens + summary_tokens + auxiliary_tokens + replay_rewrite_tokens
        trainable_tokens = main_tokens + summary_tokens + replay_rewrite_tokens
        return {
            "routing_requests": main_requests + summary_requests + auxiliary_requests + replay_rewrite_requests,
            "routing_main_requests": main_requests,
            "routing_summary_requests": summary_requests,
            "routing_auxiliary_requests": auxiliary_requests,
            "routing_replay_rewrite_requests": replay_rewrite_requests,
            "routing_main_prompt_tokens": self._route_prompt_tokens.get("main", 0),
            "routing_summary_prompt_tokens": self._route_prompt_tokens.get("summary", 0),
            "routing_auxiliary_prompt_tokens": self._route_prompt_tokens.get("auxiliary", 0),
            "routing_replay_rewrite_prompt_tokens": self._route_prompt_tokens.get("replay_rewrite", 0),
            "routing_main_generated_tokens": main_tokens,
            "routing_summary_generated_tokens": summary_tokens,
            "routing_auxiliary_generated_tokens": auxiliary_tokens,
            "routing_replay_rewrite_generated_tokens": replay_rewrite_tokens,
            "routing_trainable_generated_tokens": trainable_tokens,
            "routing_untracked_requests": auxiliary_requests,
            "routing_untracked_generated_tokens": auxiliary_tokens,
            "routing_failed_requests": sum(self._route_failure_counts.values()),
            "routing_concurrent_request_failures": self._concurrent_request_failures,
            "routing_max_active_requests": self._max_active_requests,
            "routing_accounted_generated_tokens": accounted_tokens,
            "routing_unexplained_generated_tokens": self._generated_tokens - accounted_tokens,
        }

    def lifecycle_metadata(self) -> dict[str, int]:
        """Return a synchronized snapshot used by the episode finalization gate."""
        with self._lifecycle:
            return {
                "active_completions": self._active_completions,
                "inflight_futures": len(self._inflight),
                "pending_generations": self._pending_generations,
            }

    def _routing_extends(self, prompt_ids: list[int]) -> bool:
        # A token-prefix continuation stays in the current execution segment.
        # Canonical replay rewrites are classified separately and trained in a
        # new fixed-context segment; unrelated auxiliary requests are served
        # without adding their tokens to the trainable ledger.
        return prompt_ids[: len(self._routing_token_ids)] == self._routing_token_ids

    def _append_routing_turn(self, prompt_ids: list[int], completion_ids: list[int]) -> None:
        self._routing_token_ids.extend(prompt_ids[len(self._routing_token_ids) :])
        self._routing_token_ids.extend(completion_ids)

    def _classify_request(self, messages: list[dict[str, Any]], prompt_ids: list[int]) -> tuple[str, int, int]:
        common_tokens = _common_token_prefix_length(self._routing_token_ids, prompt_ids)
        common_messages = _common_message_prefix_length(self._raw_prefix or [], messages)
        if self._routing_extends(prompt_ids):
            return "main", common_tokens, common_messages

        # A canonical message replay that extends the last main response but no
        # longer extends its token render is distinct from an unrelated request.
        route = (
            "replay_rewrite"
            if self._raw_prefix is not None and _messages_extend(self._raw_prefix, messages)
            else "auxiliary"
        )
        return route, common_tokens, common_messages

    def _active_request_count(self) -> int:
        with self._lifecycle:
            return self._active_completions

    def _record_request(
        self,
        *,
        route: str,
        prompt_tokens: int,
        generated_tokens: int,
        common_prefix_tokens: int,
        common_prefix_messages: int,
        failed: bool = False,
    ) -> str:
        self._request_sequence += 1
        request_id = f"{self.ledger.trajectory_id[:12]}:{self._request_sequence}"
        parent_request_id = self._last_trajectory_request_id
        active_requests = self._active_request_count()
        self._max_active_requests = max(self._max_active_requests, active_requests)
        self._route_request_counts[route] = self._route_request_counts.get(route, 0) + 1
        self._route_prompt_tokens[route] = self._route_prompt_tokens.get(route, 0) + prompt_tokens
        self._route_generated_tokens[route] = self._route_generated_tokens.get(route, 0) + generated_tokens
        if failed:
            self._route_failure_counts[route] = self._route_failure_counts.get(route, 0) + 1
        if route in {"main", "summary", "replay_rewrite"}:
            self._last_trajectory_request_id = request_id
        logger.info(
            "CompactionRL route request_id=%s parent_request_id=%s route=%s "
            "prompt_tokens=%d generated_tokens=%d routing_tokens=%d "
            "common_prefix_tokens=%d common_prefix_messages=%d active_requests=%d failed=%d",
            request_id,
            parent_request_id or "none",
            route,
            prompt_tokens,
            generated_tokens,
            len(self._routing_token_ids),
            common_prefix_tokens,
            common_prefix_messages,
            active_requests,
            int(failed),
        )
        return request_id

    def _completion_for_route(
        self,
        *,
        route: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        common_prefix_tokens: int,
        common_prefix_messages: int,
        terminal: bool = False,
    ) -> dict[str, Any]:
        try:
            if terminal:
                completion = self._terminal_completion()
            elif max_new_tokens == 0:
                completion = self._empty_completion()
            else:
                completion = self._run_generation(prompt_ids, max_new_tokens)
        except BaseException:
            self._record_request(
                route=route,
                prompt_tokens=len(prompt_ids),
                generated_tokens=0,
                common_prefix_tokens=common_prefix_tokens,
                common_prefix_messages=common_prefix_messages,
                failed=True,
            )
            raise
        self._generated_tokens += len(completion["token_ids"])
        self._record_request(
            route=route,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(completion["token_ids"]),
            common_prefix_tokens=common_prefix_tokens,
            common_prefix_messages=common_prefix_messages,
        )
        return completion

    def _pending_context_tokens(self, prompt_ids: list[int]) -> int:
        current = self.ledger.current
        if current is None or not current.has_trainable_tokens():
            # The initial prompt of the episode and each rebuilt post-compaction
            # prompt are before that segment's first action.
            return 0
        if current.extends(prompt_ids):
            return current.pending_context_tokens(prompt_ids)

        # The regular SWE-RL proxy simply stops recording a render that does
        # not extend its token ledger. Here routing has already established
        # that this is the main conversation, so preserve the old execution
        # segment and rebase the rewritten prompt as fixed context for a new
        # execution segment. No token is dropped or optimized twice.
        self.ledger.finish_current()
        self.ledger.begin("execution")
        self._rebase_count += 1
        logger.warning(
            "CompactionRL rebased a token-rewritten main request " "(routing_tokens=%d prompt_tokens=%d rebases=%d)",
            len(self._routing_token_ids),
            len(prompt_ids),
            self._rebase_count,
        )
        return 0

    def _begin_replay_rebase(self) -> None:
        """Start a sequential segment for a canonical history replay.

        qwen-code can replay the same semantic conversation with a different
        token rendering. The old segment must remain intact, while the replayed
        prompt becomes fixed context for a new segment. Only the new completion
        is optimized, so no policy token is dropped or trained twice.
        """
        if self.ledger.current is not None:
            self.ledger.finish_current(allow_empty=True)
        self.ledger.begin("execution_rebase")
        self._rebase_count += 1
        logger.warning(
            "CompactionRL started a trainable replay rebase " "(routing_tokens=%d rebases=%d)",
            len(self._routing_token_ids),
            self._rebase_count,
        )

    @staticmethod
    def _empty_completion() -> dict[str, Any]:
        return {"text": "", "token_ids": [], "log_probs": [], "finish_reason": "length"}

    @staticmethod
    def _terminal_completion() -> dict[str, Any]:
        return {"text": "", "token_ids": [], "log_probs": [], "finish_reason": "stop"}

    def _ingest_raw_messages(self, messages: list[dict[str, Any]]) -> None:
        if self._raw_prefix is None:
            self.working_messages = copy.deepcopy(messages)
            return

        # Before the first compaction, the proven SWE-RL behavior is to render
        # the CLI's current messages directly. Token-prefix routing, rather
        # than strict JSON equality, decides whether this is the main request.
        if self._compaction_count == 0:
            self.working_messages = copy.deepcopy(messages)
            return

        raw_prefix = self._raw_prefix
        if len(messages) < len(raw_prefix):
            raise RuntimeError(
                "recorded qwen-code replay has fewer messages than its previous prefix "
                f"after compaction ({len(messages)} < {len(raw_prefix)})"
            )
        tail = _align_tool_call_ids(raw_prefix, messages[: len(raw_prefix)], messages[len(raw_prefix) :])
        self.working_messages.extend(tail)

    def _set_raw_prefix_after_response(
        self, messages: list[dict[str, Any]], assistant_message: dict[str, Any]
    ) -> None:
        self._raw_prefix = [*copy.deepcopy(messages), copy.deepcopy(assistant_message)]

    def _run_generation(self, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
        with self._lifecycle:
            if self._closing:
                raise RuntimeError("compaction model proxy is closing")
            self._pending_generations += 1
            coroutine = self._tracked_generate(prompt_ids, max_new_tokens=max_new_tokens)
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            except BaseException:
                self._pending_generations -= 1
                coroutine.close()
                raise
            self._inflight.add(future)
        try:
            result = future.result()
            return result
        finally:
            with self._lifecycle:
                self._inflight.discard(future)
                self._lifecycle.notify_all()

    async def _generate(self, prompt_ids: list[int], *, max_new_tokens: int) -> dict[str, Any]:
        sampling_params = {
            **self.sampling_params,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": True,
        }
        # httpx's scalar timeout is phase-oriented, so also wrap the complete
        # operation in asyncio.timeout to enforce one wall-clock proxy deadline.
        async with asyncio.timeout(PROXY_HTTP_TIMEOUT_SECONDS):
            output = await post(
                self.model_url,
                {"input_ids": prompt_ids, "sampling_params": sampling_params, "return_logprob": True},
                max_retries=PROXY_HTTP_MAX_ATTEMPTS,
                timeout=PROXY_HTTP_TIMEOUT_SECONDS,
            )
        meta = output["meta_info"]
        log_probs: list[float] = []
        token_ids: list[int] = []
        for logprob, token_id, *_ in meta["output_token_logprobs"]:
            log_probs.append(float(logprob))
            token_ids.append(int(token_id))
        finish_reason = meta.get("finish_reason", "stop")
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type", "stop")
        return {
            "text": str(output.get("text", "")),
            "token_ids": token_ids,
            "log_probs": log_probs,
            "finish_reason": str(finish_reason),
        }

    async def _tracked_generate(self, prompt_ids: list[int], *, max_new_tokens: int) -> dict[str, Any]:
        try:
            return await self._generate(prompt_ids, max_new_tokens=max_new_tokens)
        finally:
            with self._lifecycle:
                self._pending_generations -= 1
                self._lifecycle.notify_all()

    def _summary_prompt(self, tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
        messages = summary_messages(self.working_messages)
        prompt_ids = render_prompt_ids(self.tokenizer, messages, tools)
        prompt_limit = min(self.config.context_budget, self.config.model_sequence_limit) - self.config.summary_tokens
        truncations = 0
        while len(prompt_ids) > prompt_limit:
            truncated = truncate_tool_observations(messages[:-1])
            if truncated == messages[:-1]:
                raise RuntimeError(
                    "CompactionRL summary prompt cannot fit after exhausting tool-observation truncation "
                    f"(prompt_tokens={len(prompt_ids)}, limit={prompt_limit})"
                )
            messages = [*truncated, messages[-1]]
            prompt_ids = render_prompt_ids(self.tokenizer, messages, tools)
            truncations += 1
        self._truncated_observations += truncations
        return messages, prompt_ids

    def _resume_context(self, summary: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt_limit = self.config.context_budget - self.config.trigger_tokens
        for recent_steps in range(self.config.recent_steps, -1, -1):
            messages = resume_messages(self.working_messages, summary, recent_steps)
            prompt_ids = render_prompt_ids(self.tokenizer, messages, tools)
            if len(prompt_ids) < prompt_limit:
                self._retained_recent_steps.append(recent_steps)
                return messages
        raise RuntimeError(
            "CompactionRL reconstructed context cannot fit even with zero recent steps " f"(limit={prompt_limit})"
        )

    def _compact(self, tools: list[dict[str, Any]]) -> None:
        if self.ledger.current is None or not self.ledger.current.has_trainable_tokens():
            raise RuntimeError("compaction requires a non-empty execution segment")
        if self._compaction_count >= self.config.max_compactions:
            return

        self.ledger.finish_current()
        summary_prompt, summary_ids = self._summary_prompt(tools)
        summary_max = self.config.available_generation_tokens(len(summary_ids), self.config.summary_tokens)
        if summary_max <= 0:
            raise RuntimeError(
                "CompactionRL cannot generate a summary within its working window "
                f"(context_budget={self.config.context_budget}, "
                f"model_sequence_limit={self.config.model_sequence_limit}, "
                f"prompt_tokens={len(summary_ids)})"
            )

        summary = self._completion_for_route(
            route="summary",
            prompt_ids=summary_ids,
            max_new_tokens=summary_max,
            common_prefix_tokens=0,
            common_prefix_messages=0,
        )
        self._episode_tokens += len(summary["token_ids"])
        summary_ledger = self.ledger.begin("summary")
        summary_ledger.append_turn(
            summary_ids,
            summary["token_ids"],
            summary["log_probs"],
            summary["text"],
            summary["finish_reason"],
        )
        self.ledger.finish_current()
        self._compaction_count += 1
        self.working_messages = self._resume_context(summary["text"], tools)
        self.ledger.begin("execution")

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lifecycle:
            if self._closing:
                raise RuntimeError("compaction model proxy is closing")
            if self._failure is not None:
                raise RuntimeError("compaction model proxy episode has already failed") from self._failure
            self._active_completions += 1
            concurrent = self._active_completions > 1
            if concurrent:
                self._max_active_requests = max(self._max_active_requests, self._active_completions)
                self._concurrent_request_failures += 1
        try:
            if concurrent:
                raise RuntimeError(
                    "CompactionRL received overlapping chat-completion requests; "
                    "their causal order is undefined, so this episode cannot be trained"
                )
            return self._complete(body)
        except BaseException as exc:
            if self._failure is None:
                self._failure = exc
            raise
        finally:
            with self._lifecycle:
                self._active_completions -= 1
                self._lifecycle.notify_all()

    def _complete(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body["messages"]
        tools = body.get("tools") or []
        raw_prompt_ids = render_prompt_ids(self.tokenizer, messages, tools)
        with self._lock:
            route, common_prefix_tokens, common_prefix_messages = self._classify_request(messages, raw_prompt_ids)
            if self._budget_exhausted:
                completion = self._completion_for_route(
                    route=route,
                    prompt_ids=raw_prompt_ids,
                    max_new_tokens=0,
                    common_prefix_tokens=common_prefix_tokens,
                    common_prefix_messages=common_prefix_messages,
                    terminal=True,
                )
                message = parse_completion(
                    completion["text"],
                    tools=tools,
                    tool_parser=self.tool_parser,
                    reasoning_parser=self.reasoning_parser,
                )
                return self._response_payload(body, message, completion)

            if route == "auxiliary":
                # qwen-code may issue an unrelated helper request while the
                # canonical agent conversation is still in flight.  Match the
                # proven SWE-RL behavior: serve it so the CLI can continue, but
                # keep it out of the ordered trajectory and therefore out of
                # PPO/GAE.  _completion_for_route accounts its tokens as
                # explicitly untracked, so the final token reconciliation can
                # still prove that no generated token was silently dropped.
                max_new_tokens = self.config.available_auxiliary_tokens(
                    len(raw_prompt_ids), self.config.per_turn_tokens
                )
                completion = self._completion_for_route(
                    route=route,
                    prompt_ids=raw_prompt_ids,
                    max_new_tokens=max_new_tokens,
                    common_prefix_tokens=common_prefix_tokens,
                    common_prefix_messages=common_prefix_messages,
                )
                message = parse_completion(
                    completion["text"],
                    tools=tools,
                    tool_parser=self.tool_parser,
                    reasoning_parser=self.reasoning_parser,
                )
                return self._response_payload(body, message, completion)

            is_replay_rewrite = route == "replay_rewrite"
            if is_replay_rewrite:
                self._begin_replay_rebase()
            generation_route = "replay_rewrite" if is_replay_rewrite else "main"

            self._ingest_raw_messages(messages)
            if self.ledger.current is None:
                self.ledger.begin("execution")

            # Before compaction working_messages is the CLI request itself.
            # Reuse the exact render used by the proven SWE-RL prefix router;
            # rendering the same semantic history twice is unnecessary and can
            # expose nondeterministic template normalization.
            prompt_ids = (
                raw_prompt_ids
                if self._compaction_count == 0
                else render_prompt_ids(self.tokenizer, self.working_messages, tools)
            )
            pending_context_tokens = self._pending_context_tokens(prompt_ids)

            should_compact = (
                self.config.needs_compaction(len(prompt_ids)) and self._compaction_count < self.config.max_compactions
            )
            if should_compact:
                # The first request, and the first request after a compaction,
                # start with an empty execution segment. There is nothing to
                # summarize yet, so let the model generate the first bounded
                # turn before considering another compaction.
                if self.ledger.current.has_trainable_tokens():
                    if pending_context_tokens:
                        self.ledger.current.append_context(prompt_ids)
                        self._episode_tokens += pending_context_tokens
                    self._compact(tools)
                    prompt_ids = render_prompt_ids(self.tokenizer, self.working_messages, tools)
                elif len(prompt_ids) >= min(self.config.context_budget, self.config.model_sequence_limit):
                    raise RuntimeError(
                        "CompactionRL cannot start an execution segment: "
                        f"prompt_tokens={len(prompt_ids)} meets or exceeds "
                        f"COMPACTION_CONTEXT_BUDGET={self.config.context_budget}, "
                        "and there are no generated tokens to compact"
                    )

            max_new_tokens = self.config.available_generation_tokens(len(prompt_ids), self.config.per_turn_tokens)
            pending_context_tokens = self._pending_context_tokens(prompt_ids)
            if max_new_tokens == 0:
                # Before the fourth window an overlong indivisible observation
                # is summarized on the next request. Once all three compactions
                # are used, reaching C ends the rollout instead of falling
                # through to the model's much larger native context.
                self._budget_exhausted = self._compaction_count >= self.config.max_compactions
                completion = self._completion_for_route(
                    route=generation_route,
                    prompt_ids=prompt_ids,
                    max_new_tokens=0,
                    common_prefix_tokens=common_prefix_tokens,
                    common_prefix_messages=common_prefix_messages,
                    terminal=self._budget_exhausted,
                )
            else:
                completion = self._completion_for_route(
                    route=generation_route,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    common_prefix_tokens=common_prefix_tokens,
                    common_prefix_messages=common_prefix_messages,
                )

            message = parse_completion(
                completion["text"],
                tools=tools,
                tool_parser=self.tool_parser,
                reasoning_parser=self.reasoning_parser,
            )
            current = self.ledger.current
            if completion["token_ids"]:
                current.append_turn(
                    prompt_ids,
                    completion["token_ids"],
                    completion["log_probs"],
                    completion["text"],
                    completion["finish_reason"],
                )
            elif pending_context_tokens:
                current.append_context(prompt_ids)
            self._episode_tokens += pending_context_tokens + len(completion["token_ids"])
            self.working_messages.append(message)
            if is_replay_rewrite:
                self._routing_token_ids = [*raw_prompt_ids, *completion["token_ids"]]
            else:
                self._append_routing_turn(raw_prompt_ids, completion["token_ids"])
            self._set_raw_prefix_after_response(messages, message)

        return self._response_payload(body, message, completion)

    @staticmethod
    def _response_payload(body: dict[str, Any], message: dict[str, Any], completion: dict[str, Any]) -> dict[str, Any]:
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else completion["finish_reason"],
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body["model"],
            "choices": [choice],
        }

    def to_samples(self, base_sample: Any, reward: float) -> list[Any]:
        return self.ledger.to_samples(
            base_sample,
            reward,
            min(self.config.context_budget, self.config.model_sequence_limit),
            self._compaction_count,
        )


def _stream_frames(payload: dict[str, Any]) -> list[Any]:
    choice = payload["choices"][0]

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            **payload,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    return [chunk(choice["message"]), chunk({}, choice["finish_reason"]), "[DONE]"]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    @property
    def _proxy(self) -> CompactionModelProxy:
        return self.server.proxy  # type: ignore[attr-defined,return-value]

    def do_GET(self) -> None:
        if urlsplit(self.path).path.rstrip("/") == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
            return
        self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if urlsplit(self.path).path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})
                return
            payload = self._proxy.complete(body)
            if body.get("stream"):
                self._send_stream(payload)
            else:
                self._send_json(200, payload)
        except Exception as exc:  # noqa: BLE001 - return an actionable agent error
            traceback.print_exc()
            self._send_json(500, {"error": {"message": str(exc), "type": exc.__class__.__name__}})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for frame in _stream_frames(payload):
            data = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        self.close_connection = True
