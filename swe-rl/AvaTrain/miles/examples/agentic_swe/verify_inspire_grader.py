"""Run one real SWE-bench grading pass in clean Inspire sandboxes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from pathlib import Path

import sandbox
from swe import SweTask


async def main(data_path: Path) -> None:
    with data_path.open(encoding="utf-8") as source:
        row = json.loads(next(line for line in source if line.strip()))
    metadata = row["metadata"]
    task = SweTask(
        instance_id=metadata["instance_id"],
        sandbox_project=metadata.get("sandbox_project", "primary"),
        repo=metadata["repo"],
        workdir=metadata["repo_workdir"],
        template=metadata["inspire_template"],
        env=dict(metadata.get("docker_image_env") or {}),
        sandbox_user=metadata.get("docker_image_default_user") or None,
        base_commit=metadata["base_commit"],
        test_patch=metadata.get("test_patch", ""),
        test_command=metadata["install_config"]["test_cmd"],
        fail_to_pass=tuple(metadata["FAIL_TO_PASS"]),
        pass_to_pass=tuple(metadata["PASS_TO_PASS"]),
        log_parser=metadata.get("swebench_log_parser"),
    )
    runtime = await sandbox.create_sandbox(
        template=task.template,
        envs=task.env,
        project=task.sandbox_project,
    )
    try:
        await task.setup(runtime)
        reward = await task.reward(runtime)
    finally:
        with contextlib.suppress(Exception):
            await runtime.kill()
    if not 0.0 <= reward <= 1.0:
        raise RuntimeError(f"invalid reward: {reward}")
    print(f"READY official SWE-bench grader instance={row['label']} empty_patch_reward={reward:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.data))
