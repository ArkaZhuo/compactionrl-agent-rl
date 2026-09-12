"""SWE-Dev scoring adapter layered over the unmodified Miles SWE example."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SWEBENCH_RUNTIME_DIR = os.environ.get("SWEBENCH_RUNTIME_DIR", "").strip()
if _SWEBENCH_RUNTIME_DIR and _SWEBENCH_RUNTIME_DIR not in sys.path:
    # Append so the image's matched NumPy and other compiled packages win over
    # dependency copies stored beside the standalone SWE-bench runtime.
    sys.path.append(_SWEBENCH_RUNTIME_DIR)

from swebench.harness.log_parsers import MAP_REPO_TO_PARSER_PY
from swebench.harness.log_parsers.python import parse_log_pytest

_ORIGINAL_PATH = (
    Path(__file__).resolve().parents[2]
    / "miles"
    / "examples"
    / "agentic_swe"
    / "swe.py"
)
_SPEC = importlib.util.spec_from_file_location("_avatrain_original_agentic_swe", _ORIGINAL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load original Miles SWE module: {_ORIGINAL_PATH}")
_ORIGINAL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ORIGINAL
_SPEC.loader.exec_module(_ORIGINAL)


@dataclass(frozen=True)
class SweTask(_ORIGINAL.SweTask):
    """Original task implementation plus the parser declared by SWE-Dev."""

    log_parser: str | None = None

    @classmethod
    def from_sample(cls, sample: Any) -> "SweTask":
        metadata = sample.metadata
        install = metadata["install_config"]
        return cls(
            repo=metadata["repo"],
            workdir=metadata["repo_workdir"],
            template=metadata["inspire_template"],
            env=dict(metadata.get("docker_image_env") or {}),
            sandbox_user=metadata.get("docker_image_default_user") or None,
            base_commit=metadata["base_commit"],
            test_patch=metadata.get("test_patch", ""),
            test_command=install["test_cmd"],
            fail_to_pass=tuple(metadata["FAIL_TO_PASS"]),
            pass_to_pass=tuple(metadata["PASS_TO_PASS"]),
            log_parser=metadata.get("swebench_log_parser"),
        )

    def score(self, test_output: str) -> float:
        if self.log_parser is None:
            parser = MAP_REPO_TO_PARSER_PY[self.repo]
        elif self.log_parser == "pytest":
            parser = parse_log_pytest
        else:
            raise ValueError(f"unsupported SWE-bench log parser: {self.log_parser!r}")

        parsed = parser(test_output or "", None)
        parsed = {name.strip(): status for name, status in parsed.items()}
        passed = {name for name, status in parsed.items() if status in {"PASSED", "XFAIL"}}
        skipped = {name for name, status in parsed.items() if status == "SKIPPED"}

        f2p_expected = {name.strip() for name in self.fail_to_pass}
        p2p_expected = {name.strip() for name in self.pass_to_pass}
        f2p_fixed = passed & f2p_expected
        f2p_skipped = skipped & f2p_expected
        p2p_skipped = skipped & p2p_expected
        p2p_broken = p2p_expected - passed - p2p_skipped

        f2p_ratio = len(f2p_fixed) / max(len(f2p_expected - f2p_skipped), 1)
        p2p_ratio = 1.0 - len(p2p_broken) / max(len(p2p_expected - p2p_skipped), 1)
        return f2p_ratio * p2p_ratio
