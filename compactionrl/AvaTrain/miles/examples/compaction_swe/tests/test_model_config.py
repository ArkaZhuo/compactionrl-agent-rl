import json

import pytest

from compaction_swe.model_config import native_sequence_limit


def test_native_sequence_limit_reads_nested_text_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"text_config": {"max_position_embeddings": 262144}}),
        encoding="utf-8",
    )
    assert native_sequence_limit(path) == 262144


def test_native_sequence_limit_reads_flat_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"max_position_embeddings": 65536}), encoding="utf-8")
    assert native_sequence_limit(path) == 65536


@pytest.mark.parametrize("value", [None, 0, -1, True, "262144"])
def test_native_sequence_limit_rejects_invalid_values(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"max_position_embeddings": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="max_position_embeddings"):
        native_sequence_limit(path)
