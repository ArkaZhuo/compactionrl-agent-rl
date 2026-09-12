"""Prepare and grade one SWE-bench Verified task."""

from __future__ import annotations

import os
import logging
import shlex
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

_SWEBENCH_RUNTIME_DIR = os.environ.get("SWEBENCH_RUNTIME_DIR", "").strip()
if _SWEBENCH_RUNTIME_DIR and _SWEBENCH_RUNTIME_DIR not in sys.path:
    # Keep the host runtime ahead of dependencies bundled in the target dir.
    sys.path.append(_SWEBENCH_RUNTIME_DIR)

from swebench.harness.log_parsers import MAP_REPO_TO_PARSER_PY
from swebench.harness.log_parsers.python import parse_log_pytest

# swebench 4.1 ships get_modified_files but not get_new_files, so the test patch
# is split here. unidiff comes with swebench, which parses patches with it too.
from unidiff import PatchSet

import sandbox

if TYPE_CHECKING:
    from miles.utils.types import Sample

CANDIDATE_PATCH_PATH = "/tmp/candidate.patch"
TEST_PATCH_PATH = "/tmp/test.patch"
EVAL_TIMEOUT = int(os.environ.get("SWE_GRADER_TIMEOUT_SEC", "600"))
if EVAL_TIMEOUT <= 0:
    raise ValueError(f"SWE_GRADER_TIMEOUT_SEC must be positive, got {EVAL_TIMEOUT}")

logger = logging.getLogger(__name__)


def _apply_patch(path: str) -> str:
    return shlex.join(
        [
            "git",
            "apply",
            "-v",
            "--3way",
            "--recount",
            "--ignore-space-change",
            "--whitespace=nowarn",
            path,
        ]
    )


@dataclass(frozen=True)
class SweTask:
    instance_id: str
    sandbox_project: str
    repo: str
    workdir: str
    template: str
    env: dict[str, str]
    sandbox_user: str | None
    base_commit: str
    test_patch: str
    test_command: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    log_parser: str | None = None

    @classmethod
    def from_sample(cls, sample: Sample) -> SweTask:
        metadata = sample.metadata
        install = metadata["install_config"]
        return cls(
            instance_id=metadata["instance_id"],
            sandbox_project=metadata.get("sandbox_project", "primary"),
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

    async def setup(self, runtime: Any) -> None:
        """Pin the repo at the base commit, so the later diff is the agent's work."""
        script = "\n".join(
            [
                "set -euo pipefail",
                f"git reset --hard {shlex.quote(self.base_commit)}",
                "git clean -fd",
            ]
        )
        result = await sandbox.run(runtime, script, timeout=600, user=self.sandbox_user, cwd=self.workdir)
        if result.exit_code != 0:
            raise RuntimeError(f"workspace preparation failed: {result.output[-2000:]}")

    async def reward(self, runtime: Any) -> float:
        """Take the agent's diff, then grade it in a sandbox it never touched."""
        patch = await self.collect_patch(runtime)
        return await self.grade_patch(patch)

    async def grade_patch(self, patch: str) -> float:
        """Grade an already captured patch in a fresh, isolated sandbox."""
        clean = await sandbox.create_sandbox(
            template=self.template,
            envs=self.env,
            project=self.sandbox_project,
        )
        try:
            for path, content in ((CANDIDATE_PATCH_PATH, patch), (TEST_PATCH_PATH, self.test_patch)):
                if content.strip():
                    await clean.files.write(path, content, user=self.sandbox_user)
            try:
                result = await sandbox.run(
                    clean,
                    self._eval_script(patch),
                    timeout=EVAL_TIMEOUT,
                    user=self.sandbox_user,
                    cwd=self.workdir,
                )
            except sandbox.TimeoutException:
                # A hanging repository test is a failed candidate, not a
                # reason to block the full rollout or discard its GRPO group.
                logger.warning(
                    "SWE grader timed out instance=%s repo=%s timeout_sec=%d reward=0",
                    self.instance_id,
                    self.repo,
                    EVAL_TIMEOUT,
                )
                return 0.0
            infrastructure_errors = (
                "CondaError: Run 'conda init' before 'conda activate'",
                "/opt/miniconda3/bin/activate: No such file or directory",
                "conda: command not found",
            )
            matched_error = next(
                (marker for marker in infrastructure_errors if marker in result.output),
                None,
            )
            if matched_error is not None:
                raise RuntimeError(
                    f"SWE grader infrastructure failure for {self.instance_id}: "
                    f"{matched_error}; output tail={result.output[-2000:]!r}"
                )
            reward = self.score(result.output)
            logger.info(
                "SWE grader instance=%s repo=%s base_commit=%s patch_bytes=%d test_exit_code=%d "
                "test_output_bytes=%d reward=%.8f",
                self.instance_id,
                self.repo,
                self.base_commit,
                len(patch.encode("utf-8", errors="replace")),
                result.exit_code,
                len(result.output.encode("utf-8", errors="replace")),
                reward,
            )
            if reward == 0.0:
                logger.warning(
                    "SWE grader zero reward instance=%s repo=%s test_output_tail=%r",
                    self.instance_id,
                    self.repo,
                    result.output[-2000:],
                )
            return reward
        finally:
            await clean.kill()

    async def collect_patch(self, runtime: Any) -> str:
        """Capture the candidate patch before the mutable agent sandbox is released."""
        script = "\n".join(
            [
                "set -euo pipefail",
                "git add -N .",
                "git diff",
            ]
        )
        result = await sandbox.run(runtime, script, timeout=600, user=self.sandbox_user, cwd=self.workdir)
        if result.exit_code != 0:
            # Swallowing this would grade an empty patch and record a false zero.
            raise RuntimeError(f"could not read the candidate diff: {result.output[-2000:]}")
        return result.stdout

    def _eval_script(self, patch: str) -> str:
        lines = [
            "set -e",
            f"git reset --hard {shlex.quote(self.base_commit)}",
        ]
        if patch.strip():
            lines.append(_apply_patch(CANDIDATE_PATCH_PATH))
        if self.test_patch.strip():
            # The agent may have edited the tests it is graded on; reset only
            # the files the official patch owns, not the candidate patch.
            files = list(PatchSet(self.test_patch))
            modified = [file.source_file[2:] for file in files if file.source_file.startswith("a/")]
            new = [file.target_file[2:] for file in files if file.source_file == "/dev/null"]
            if modified:
                lines.append(shlex.join(["git", "checkout", self.base_commit, "--", *modified]))
            if new:
                lines.append(shlex.join(["rm", "-f", "--", *new]))
            lines.append(_apply_patch(TEST_PATCH_PATH))
        return "\n".join([*lines, "set +e", self.test_command])

    def score(self, test_output: str) -> float:
        """Reward = (fraction of FAIL_TO_PASS fixed) x (fraction of PASS_TO_PASS kept).

        Dense because a binary flag leaves hard instances with no gradient.
        """
        # Python map only: no parser in it reads test_spec, so None is safe. The
        # combined map has one (immutable-js) that does, and a repo outside
        # SWE-bench Verified should fail loudly here rather than mis-parse.
        if self.log_parser is None:
            parser = MAP_REPO_TO_PARSER_PY[self.repo]
        elif self.log_parser == "pytest":
            parser = parse_log_pytest
        else:
            raise ValueError(f"unsupported SWE-bench log parser: {self.log_parser!r}")
        parsed = parser(test_output or "", None)  # type: ignore[arg-type]
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
