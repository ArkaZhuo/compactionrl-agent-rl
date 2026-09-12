"""CPU-only smoke test for the exact Agentic SWE reverse-tunnel path."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import sandbox


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        if urlsplit(self.path).path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "default", "object": "model"}]})
            return
        body = b"agentic-swe-tunnel-ok\n"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("content-length") or "0")
        body = json.loads(self.rfile.read(length).decode() or "{}")
        if urlsplit(self.path).path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "unknown endpoint"}})
            return
        self.server.model_requests += 1  # type: ignore[attr-defined]
        if body.get("stream"):
            frames = [
                {"id": "chatcmpl-smoke", "object": "chat.completion.chunk", "model": "default", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "PROTOCOL_SMOKE_OK"}, "finish_reason": None}]},
                {"id": "chatcmpl-smoke", "object": "chat.completion.chunk", "model": "default", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("connection", "close")
            self.end_headers()
            for frame in frames:
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        self._json(
            200,
            {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": "default",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "PROTOCOL_SMOKE_OK"}, "finish_reason": "stop"}],
            },
        )

    def _json(self, status: int, payload: dict[str, object]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


async def main(template: str) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    server.model_requests = 0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = await sandbox.create_sandbox(template=template, envs={})
    try:
        async with sandbox.reverse_tunnel(
            runtime,
            proxy_port=server.server_address[1],
            sandbox_port=30001,
            server_port=19090,
            wstunnel_bin="/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel",
            user="root",
        ):
            command = (
                "/opt/miniconda3/bin/python -c "
                "'import urllib.request; "
                "print(urllib.request.urlopen(\"http://127.0.0.1:30001/health\", timeout=10).read().decode().strip())'"
            )
            result = await sandbox.run(runtime, command, timeout=60, user="root", cwd="/testbed")
            if result.exit_code != 0 or "agentic-swe-tunnel-ok" not in result.stdout:
                raise RuntimeError(f"reverse tunnel failed: {result.output[-2000:]}")
            print("READY Inspire Sandbox reverse tunnel reached the trainer-side HTTP endpoint")
            project_hash = hashlib.sha256(b"/testbed").hexdigest()
            command = "\n".join(
                [
                    "set -euo pipefail",
                    f"mkdir -p \"${{HOME}}/.qwen/tmp/{project_hash}\"",
                    "/__avaeval_agentic_protocol_v1__/frameworks/qwen_code/bin/qwen "
                    "--approval-mode yolo --max-session-turns 2 --auth-type openai "
                    "--openai-base-url http://127.0.0.1:30001/v1 "
                    "--openai-api-key agentic-swe --model default "
                    "'Reply exactly PROTOCOL_SMOKE_OK and do not use tools.'",
                ]
            )
            result = await sandbox.run(runtime, command, timeout=120, user="root", cwd="/testbed")
            if result.exit_code != 0 or server.model_requests < 1:  # type: ignore[attr-defined]
                raise RuntimeError(f"qwen-code protocol smoke failed: {result.output[-4000:]}")
            print(
                "READY qwen-code reached the trainer-side OpenAI endpoint "
                f"requests={server.model_requests}"  # type: ignore[attr-defined]
            )
    finally:
        with contextlib.suppress(Exception):
            await runtime.kill()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.template))
