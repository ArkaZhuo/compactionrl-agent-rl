from unittest.mock import MagicMock

from miles.backends.sglang_utils.sglang_engine import SGLangEngine


def test_release_memory_pauses_before_flush_and_offload():
    engine = object.__new__(SGLangEngine)
    calls = []
    engine.pause_generation = MagicMock(side_effect=lambda **kwargs: calls.append(("pause", kwargs)))
    engine.flush_cache = MagicMock(side_effect=lambda: calls.append(("flush", {})))
    engine._make_request = MagicMock(
        side_effect=lambda endpoint, payload: calls.append((endpoint, payload)) or {"ok": True}
    )

    result = SGLangEngine.release_memory_occupation(engine, tags=["kv_cache"])

    assert result == {"ok": True}
    assert calls == [
        ("pause", {"mode": "abort", "timeout_seconds": 120}),
        ("flush", {}),
        ("release_memory_occupation", {"tags": ["kv_cache"]}),
    ]
