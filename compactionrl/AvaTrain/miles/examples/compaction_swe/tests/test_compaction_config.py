from __future__ import annotations

from compaction_swe.config import CompactionConfig


def test_paper_window_defaults(monkeypatch):
    for name in (
        "COMPACTION_CONTEXT_BUDGET",
        "COMPACTION_TRIGGER_TOKENS",
        "COMPACTION_MAX_TOKENS_PER_TURN",
        "COMPACTION_SUMMARY_MAX_TOKENS",
        "COMPACTION_MAX_COUNT",
        "COMPACTION_RECENT_STEPS",
        "COMPACTION_TRAJECTORY_TOKEN_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COMPACTION_MODEL_SEQUENCE_LIMIT", "262144")

    config = CompactionConfig.from_env()

    assert config.context_budget == 65536
    assert config.trigger_tokens == 10240
    assert config.max_compactions == 3
    assert config.recent_steps == 2
    assert config.auxiliary_mode == "serve_untracked"
    assert config.needs_compaction(55295) is False
    assert config.needs_compaction(55296) is True
    assert not hasattr(config, "trajectory_token_limit")


def test_generation_is_bounded_by_working_window_not_native_context():
    config = CompactionConfig(
        context_budget=65536,
        model_sequence_limit=262144,
        trigger_tokens=10240,
        per_turn_tokens=2048,
        summary_tokens=2048,
        max_compactions=3,
        recent_steps=2,
    )

    assert config.available_generation_tokens(60000, 2048) == 2048
    assert config.available_generation_tokens(65000, 2048) == 536
    assert config.available_generation_tokens(65536, 2048) == 0


def test_auxiliary_generation_uses_native_context_not_working_window():
    config = CompactionConfig(
        context_budget=65536,
        model_sequence_limit=262144,
        trigger_tokens=10240,
        per_turn_tokens=2048,
        summary_tokens=2048,
        max_compactions=3,
        recent_steps=2,
    )

    assert config.available_auxiliary_tokens(65536, 2048) == 2048
    assert config.available_auxiliary_tokens(262144, 2048) == 0


def test_auxiliary_mode_can_require_clean_compaction_trajectories(monkeypatch):
    monkeypatch.setenv("COMPACTION_MODEL_SEQUENCE_LIMIT", "262144")
    monkeypatch.setenv("COMPACTION_AUXILIARY_MODE", "reject")

    config = CompactionConfig.from_env()

    assert config.auxiliary_mode == "reject"


def test_auxiliary_mode_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("COMPACTION_MODEL_SEQUENCE_LIMIT", "262144")
    monkeypatch.setenv("COMPACTION_AUXILIARY_MODE", "train")

    try:
        CompactionConfig.from_env()
    except ValueError as exc:
        assert "COMPACTION_AUXILIARY_MODE" in str(exc)
    else:
        raise AssertionError("unknown auxiliary mode was accepted")
