"""The agentic rollout: one call == one episode.

    boot a sandbox -> tunnel the proxy into it -> let the CLI work
        -> ask the task for a reward -> hand the trajectory to miles

Nothing here knows the task; swapping the import below trains something else.
Wired in with ``--custom-generate-function-path generate.generate``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shlex

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.sglang_rollout import get_model_url

from proxy import MODEL_NAME, ModelProxy
import sandbox
from swe import SweTask

SANDBOX_MODEL_PORT = 30001  # where the tunnel surfaces the proxy inside the sandbox
WSTUNNEL_SERVER_PORT = 19090  # sandbox-side ingress port the host client dials
# Paths inside the prebuilt sandbox template; host-side wstunnel comes from PATH.
AGENT_BIN = "/__avaeval_agentic_protocol_v1__/frameworks/qwen_code/bin/qwen"
WSTUNNEL_BIN = "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "80"))
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT_SEC", str(90 * 60)))
MAX_TOKENS_PER_TURN = int(os.environ.get("AGENT_MAX_TOKENS_PER_TURN", "8192"))
QWEN_API_TIMEOUT_MS = int(os.environ.get("QWEN_CODE_API_TIMEOUT_MS", "360000"))
QWEN_MAX_RETRIES = int(os.environ.get("QWEN_CODE_MAX_RETRIES", "0"))

logger = logging.getLogger(__name__)
_PROJECT_SEMAPHORES: dict[tuple[str, int], asyncio.Semaphore] = {}


def project_semaphore(project: str) -> asyncio.Semaphore | None:
    """Return the optional whole-episode concurrency gate for one project."""
    variable = f"SANDBOX_CONCURRENCY_{project.upper()}"
    limit = int(os.environ.get(variable, "0"))
    if limit < 0:
        raise ValueError(f"{variable} must be non-negative, got {limit}")
    if limit == 0:
        return None
    key = (project, limit)
    semaphore = _PROJECT_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _PROJECT_SEMAPHORES[key] = semaphore
        logger.info("Sandbox project concurrency gate project=%s limit=%d", project, limit)
    return semaphore


def agent_command(workdir: str, prompt: str) -> str:
    """Non-interactive qwen-code; swapping the agent CLI is this function alone."""
    base_url = f"http://127.0.0.1:{SANDBOX_MODEL_PORT}/v1"
    # qwen-code needs ~/.qwen/tmp/<sha256(cwd)>, so workdir must not be a symlink.
    project_hash = hashlib.sha256(workdir.encode("utf-8")).hexdigest()
    settings_path = "/tmp/avaeval-qwen-settings.json"
    settings = '{"model":{"generationConfig":{"maxRetries":' + str(QWEN_MAX_RETRIES) + "}}}"
    cli = [
        AGENT_BIN,
        "--approval-mode",
        "yolo",
        "--max-session-turns",
        str(MAX_TURNS),
        "--auth-type",
        "openai",
        "--openai-base-url",
        base_url,
        "--openai-api-key",
        "agentic-swe",
        "--model",
        MODEL_NAME,
    ]
    return "\n".join(
        [
            "set -euo pipefail",
            'mkdir -p "${HOME}/.qwen/tmp/' + project_hash + '"',
            f"printf '%s\\n' {shlex.quote(settings)} > {shlex.quote(settings_path)}",
            f"export QWEN_CODE_SYSTEM_SETTINGS_PATH={shlex.quote(settings_path)}",
            f"export QWEN_CODE_API_TIMEOUT_MS={QWEN_API_TIMEOUT_MS}",
            f"{shlex.join(cli)} {shlex.quote(prompt)}",
        ]
    )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    task = SweTask.from_sample(input.sample)
    semaphore = project_semaphore(task.sandbox_project)
    if semaphore is None:
        return await generate_episode(input, task)
    async with semaphore:
        return await generate_episode(input, task)


async def generate_episode(input: GenerateFnInput, task: SweTask) -> GenerateFnOutput:
    args, sample, state = input.args, input.sample, input.state
    prompt = sample.prompt[0]["content"]

    proxy = ModelProxy(
        tokenizer=state.tokenizer,
        loop=asyncio.get_running_loop(),
        model_url=get_model_url(args, MODEL_NAME),
        sampling_params=input.sampling_params,
        max_tokens_per_turn=MAX_TOKENS_PER_TURN,
        tool_parser=getattr(args, "sglang_tool_call_parser", None),
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None),
    )
    async with contextlib.AsyncExitStack() as stack:
        proxy.start()
        stack.push_async_callback(asyncio.to_thread, proxy.close)
        runtime = await sandbox.create_sandbox(
            template=task.template,
            envs=task.env,
            project=task.sandbox_project,
        )
        try:
            async with sandbox.reverse_tunnel(
                runtime,
                proxy_port=proxy.port,
                sandbox_port=SANDBOX_MODEL_PORT,
                server_port=WSTUNNEL_SERVER_PORT,
                wstunnel_bin=WSTUNNEL_BIN,
                user=task.sandbox_user,
            ):
                await task.setup(runtime)
                agent_result = await sandbox.run(
                    runtime,
                    agent_command(task.workdir, prompt),
                    timeout=AGENT_TIMEOUT,
                    user=task.sandbox_user,
                    cwd=task.workdir,
                )
                logger.info(
                    "Agent finished label=%s exit_code=%d stdout_bytes=%d stderr_bytes=%d "
                    "stdout_tail=%r stderr_tail=%r",
                    sample.label,
                    agent_result.exit_code,
                    len(agent_result.stdout.encode("utf-8", errors="replace")),
                    len(agent_result.stderr.encode("utf-8", errors="replace")),
                    agent_result.stdout[-2000:],
                    agent_result.stderr[-1000:],
                )
                patch = await task.collect_patch(runtime)
        finally:
            # Release the mutable Agent sandbox before requesting the clean
            # grader. This keeps grading isolated while avoiding two live
            # Sandboxes per rollout at the capacity peak.
            await runtime.kill()
        if not proxy.trajectory.has_trainable_tokens():
            # A timed-out first model request cannot become a training sample.
            # Do not consume another scarce sandbox just to grade an empty patch.
            raise RuntimeError("agent produced no trainable trajectory")
        reward = await task.grade_patch(patch)
    result = proxy.trajectory.to_sample(
        sample,
        reward=reward,
        max_response_length=int(input.sampling_params["max_new_tokens"]),
    )
    if result is None:
        raise RuntimeError("agent produced no trainable trajectory")
    logger.info(
        "Trajectory finalized label=%s response_tokens=%d status=%s reward=%.6f",
        sample.label,
        result.response_length,
        result.status.value,
        float(reward),
    )
    return GenerateFnOutput(samples=result)
