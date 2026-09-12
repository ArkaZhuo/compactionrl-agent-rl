"""Load the original rollout while allowing short, configurable smoke tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_ORIGINAL_PATH = (
    Path(__file__).resolve().parents[2]
    / "miles"
    / "examples"
    / "agentic_swe"
    / "generate.py"
)
_SPEC = importlib.util.spec_from_file_location("_avatrain_original_agentic_generate", _ORIGINAL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load original Miles generate module: {_ORIGINAL_PATH}")
_ORIGINAL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ORIGINAL
_SPEC.loader.exec_module(_ORIGINAL)

_ORIGINAL.MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", str(_ORIGINAL.MAX_TURNS)))
_ORIGINAL.AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT_SEC", str(_ORIGINAL.AGENT_TIMEOUT)))

generate = _ORIGINAL.generate
