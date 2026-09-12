"""Read model-native limits used by the CompactionRL proxy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def native_sequence_limit(config_path: str | Path) -> int:
    """Return the HF text model's positive maximum position count."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    text_config = config.get("text_config", config)
    value = text_config.get("max_position_embeddings")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"HF config has no positive integer max_position_embeddings: {config_path}"
        )
    return value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    print(native_sequence_limit(args.config_path))
