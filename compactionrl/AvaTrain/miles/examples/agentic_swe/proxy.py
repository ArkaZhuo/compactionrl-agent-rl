"""The OpenAI endpoint the agent CLI talks to, one proxy per rollout.

Each turn: render prompt tokens -> SGLang /generate -> parse into an assistant
message -> record -> reply. The CLI's turn limit and the sandbox TTL bound the
episode, so the proxy never cuts a turn short.

Tool schemas go to both the renderer and the parser: untyped arguments render
differently from what the model emitted and break the ledger prefix.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from miles.utils.http_utils import post

from trajectory import Trajectory, render_prompt_ids

MODEL_NAME = "default"


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


PROXY_HTTP_TIMEOUT_SECONDS = _positive_int_env("AGENTIC_PROXY_HTTP_TIMEOUT_SEC", 300)
# http_utils calls this max_retries, but it is the total attempt count.
PROXY_HTTP_MAX_ATTEMPTS = _positive_int_env("AGENTIC_PROXY_HTTP_MAX_ATTEMPTS", 1)


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
    """Raw model text -> OpenAI assistant message.

    ``reasoning_content`` stays separate because the CLI echoes it back verbatim.
    """
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


@dataclass
class ModelProxy:
    """Serves one sandbox's agent and records its main trajectory."""

    tokenizer: Any
    loop: asyncio.AbstractEventLoop
    model_url: str  # SGLang /generate endpoint, from miles' router table
    sampling_params: dict[str, Any]
    max_tokens_per_turn: int
    tool_parser: str | None = None
    reasoning_parser: str | None = None

    trajectory: Trajectory = field(default_factory=Trajectory, init=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _lifecycle: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)
    _inflight: set[concurrent.futures.Future] = field(default_factory=set, init=False, repr=False)
    _active_completions: int = field(default=0, init=False, repr=False)
    _pending_generations: int = field(default=0, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> ModelProxy:
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
            # This flag and future registration share one condition, so a
            # request cannot slip into SGLang after the close snapshot.
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
                "ModelProxy did not drain before shutdown "
                f"(active_completions={active}, inflight_futures={inflight}, "
                f"pending_generations={pending})"
            )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def lifecycle_metadata(self) -> dict[str, int]:
        with self._lifecycle:
            return {
                "active_completions": self._active_completions,
                "inflight_futures": len(self._inflight),
                "pending_generations": self._pending_generations,
            }

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        """Complete one OpenAI request and record it when it extends the main trajectory."""
        with self._lifecycle:
            if self._closing:
                raise RuntimeError("model proxy is closing")
            self._active_completions += 1
        try:
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
        messages, tools = body["messages"], body.get("tools") or []
        prompt_ids = render_prompt_ids(self.tokenizer, messages, tools)
        with self._lock:
            record = self.trajectory.extends(prompt_ids)
            max_new_tokens = self._max_new_tokens(prompt_ids) if record else self.max_tokens_per_turn
            if max_new_tokens == 0:
                completion = {"text": "", "token_ids": [], "log_probs": [], "finish_reason": "length"}
            else:
                with self._lifecycle:
                    if self._closing:
                        raise RuntimeError("model proxy is closing")
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
                    completion = future.result()
                finally:
                    with self._lifecycle:
                        self._inflight.discard(future)
                        self._lifecycle.notify_all()
            message = parse_completion(
                completion["text"],
                tools=tools,
                tool_parser=self.tool_parser,
                reasoning_parser=self.reasoning_parser,
            )
            if record:
                self.trajectory.append_turn(prompt_ids, completion["token_ids"], completion["log_probs"])

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

    def _max_new_tokens(self, prompt_ids: list[int]) -> int:
        """Apply both the trajectory-wide and per-turn response limits."""
        total_limit = int(self.sampling_params["max_new_tokens"])
        if 1 in self.trajectory.loss_mask:
            first_trainable = self.trajectory.loss_mask.index(1)
            consumed = len(self.trajectory.token_ids) - first_trainable
            consumed += max(0, len(prompt_ids) - len(self.trajectory.token_ids))
        else:
            consumed = 0
        remaining = max(0, total_limit - consumed)
        return min(remaining, self.max_tokens_per_turn)

    async def _generate(self, prompt_ids: list[int], *, max_new_tokens: int) -> dict[str, Any]:
        """One SGLang generation, by token ids so the ledger owns the prompt."""
        sampling_params = {
            **self.sampling_params,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": True,
        }
        payload = {"input_ids": prompt_ids, "sampling_params": sampling_params, "return_logprob": True}
        # A scalar httpx timeout is phase-oriented, so the outer timeout is the
        # wall-clock bound for this entire generation request.
        async with asyncio.timeout(PROXY_HTTP_TIMEOUT_SECONDS):
            output = await post(
                self.model_url,
                payload,
                max_retries=PROXY_HTTP_MAX_ATTEMPTS,
                timeout=PROXY_HTTP_TIMEOUT_SECONDS,
            )
        # Malformed responses must raise: a dropped token misaligns the sequence.
        meta = output["meta_info"]
        log_probs, token_ids = [], []
        for logprob, token_id, *_ in meta["output_token_logprobs"]:
            log_probs.append(float(logprob))
            token_ids.append(int(token_id))
        return {
            "text": str(output["text"]),
            "token_ids": token_ids,
            "log_probs": log_probs,
            "finish_reason": meta["finish_reason"]["type"],
        }

    async def _tracked_generate(self, prompt_ids: list[int], *, max_new_tokens: int) -> dict[str, Any]:
        try:
            return await self._generate(prompt_ids, max_new_tokens=max_new_tokens)
        finally:
            with self._lifecycle:
                self._pending_generations -= 1
                self._lifecycle.notify_all()


def _stream_frames(payload: dict[str, Any]) -> list[Any]:
    """Replay a finished completion as chunks; agent CLIs ask for streams by default."""
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

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    @property
    def _proxy(self) -> ModelProxy:
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
        except Exception as exc:  # noqa: BLE001 - the agent must see an error, not a hang
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
