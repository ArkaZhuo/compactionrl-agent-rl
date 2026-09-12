import asyncio
import concurrent.futures
import sys
import threading
from pathlib import Path

import pytest

AGENTIC_SWE_DIR = Path(__file__).resolve().parents[4] / "examples" / "agentic_swe"
sys.path.insert(0, str(AGENTIC_SWE_DIR))

import proxy as proxy_module  # noqa: E402


def test_proxy_close_cancels_and_drains_generation(monkeypatch):
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()

    generation_started = threading.Event()
    generation_cancelled = threading.Event()

    async def blocked_generate(prompt_ids, *, max_new_tokens):
        generation_started.set()
        try:
            await asyncio.Future()
        finally:
            generation_cancelled.set()

    model_proxy = proxy_module.ModelProxy(
        tokenizer=object(),
        loop=loop,
        model_url="http://unused/generate",
        sampling_params={"max_new_tokens": 16},
        max_tokens_per_turn=8,
    ).start()
    monkeypatch.setattr(proxy_module, "render_prompt_ids", lambda *_: [1, 2, 3])
    monkeypatch.setattr(model_proxy, "_generate", blocked_generate)

    errors = []

    def complete():
        try:
            model_proxy.complete({"messages": [], "model": "default"})
        except BaseException as exc:  # cancellation is the expected result
            errors.append(exc)

    request_thread = threading.Thread(target=complete)
    request_thread.start()
    assert generation_started.wait(timeout=2)

    try:
        model_proxy.close(timeout_seconds=2)
        request_thread.join(timeout=2)

        assert not request_thread.is_alive()
        assert generation_cancelled.wait(timeout=2)
        assert len(errors) == 1
        assert isinstance(errors[0], concurrent.futures.CancelledError)
        assert not model_proxy._inflight
        assert model_proxy._active_completions == 0
        assert model_proxy._pending_generations == 0
        with pytest.raises(RuntimeError, match="closing"):
            model_proxy.complete({"messages": [], "model": "default"})
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()
