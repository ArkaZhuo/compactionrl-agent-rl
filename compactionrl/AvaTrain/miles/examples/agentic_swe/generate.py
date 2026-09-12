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
import json
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
TUNNEL_READY_TIMEOUT = int(os.environ.get("TUNNEL_READY_TIMEOUT_SEC", "60"))
TUNNEL_READY_PROBE_TIMEOUT = int(os.environ.get("TUNNEL_READY_PROBE_TIMEOUT_SEC", "5"))
MAX_SESSION_TURNS_EXIT_CODE = 53
MAX_SESSION_TURNS_MESSAGE = "Reached max session turns"

logger = logging.getLogger(__name__)
_PROJECT_SEMAPHORES: dict[tuple[str, int], asyncio.Semaphore] = {}


def reached_max_session_turns(result) -> bool:
    """Return whether qwen-code stopped cleanly at its configured turn limit."""
    return (
        result.exit_code == MAX_SESSION_TURNS_EXIT_CODE
        and MAX_SESSION_TURNS_MESSAGE in (result.stderr or "")
    )


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
    settings = json.dumps(
        {
            "model": {
                "generationConfig": {
                    "maxRetries": QWEN_MAX_RETRIES,
                    # Keep qwen-code's own compressor above this rollout's 16K
                    # limit; the proxy owns the actual trajectory budget.
                    "contextWindowSize": 1_000_000,
                },
                "skipNextSpeakerCheck": True,
            },
            # These features issue unrelated background model requests after
            # the main answer and must not be part of a PPO trajectory.
            "memory": {
                "enableManagedAutoMemory": False,
                "enableManagedAutoDream": False,
                "enableAutoSkill": False,
            },
        },
        separators=(",", ":"),
    )
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


async def wait_for_reverse_tunnel(runtime, *, user: str | None) -> None:
    """Verify the proxy is reachable from the sandbox before starting qwen-code."""
    if TUNNEL_READY_TIMEOUT <= 0 or TUNNEL_READY_PROBE_TIMEOUT <= 0:
        raise ValueError("reverse-tunnel readiness timeouts must be positive")

    probe_source = """\
import json
import urllib.request

with urllib.request.urlopen(\"http://127.0.0.1:30001/v1/models\", timeout=%d) as response:
    payload = json.load(response)
assert response.status == 200
assert any(item.get(\"id\") == \"default\" for item in payload.get(\"data\", []))
print(\"agentic-swe-tunnel-ready\")
""" % TUNNEL_READY_PROBE_TIMEOUT
    command = f"/opt/miniconda3/bin/python -c {shlex.quote(probe_source)}"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TUNNEL_READY_TIMEOUT
    attempts = 0
    last_detail = "probe did not run"

    while True:
        attempts += 1
        try:
            result = await sandbox.run(
                runtime,
                command,
                timeout=TUNNEL_READY_PROBE_TIMEOUT + 5,
                user=user,
            )
            last_detail = result.output[-1000:] or f"exit_code={result.exit_code}"
            if result.exit_code == 0 and "agentic-swe-tunnel-ready" in result.stdout:
                logger.info("Agentic SWE reverse tunnel ready attempts=%d", attempts)
                return
        except Exception as exc:
            last_detail = f"{exc.__class__.__name__}: {exc}"

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RuntimeError(
                "Agentic SWE reverse tunnel did not become reachable from the sandbox "
                f"within {TUNNEL_READY_TIMEOUT}s after {attempts} probes; last={last_detail!r}"
            )
        await asyncio.sleep(min(1.0, remaining))


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
        runtime = None
        try:
            runtime = await sandbox.create_sandbox(
                template=task.template,
                envs=task.env,
                project=task.sandbox_project,
            )
            async with sandbox.reverse_tunnel(
                runtime,
                proxy_port=proxy.port,
                sandbox_port=SANDBOX_MODEL_PORT,
                server_port=WSTUNNEL_SERVER_PORT,
                wstunnel_bin=WSTUNNEL_BIN,
                user=task.sandbox_user,
            ):
                await wait_for_reverse_tunnel(runtime, user=task.sandbox_user)
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
                max_session_turns_reached = reached_max_session_turns(agent_result)
                if agent_result.exit_code != 0 and not max_session_turns_reached:
                    raise RuntimeError(
                        "qwen-code agent exited unsuccessfully; refusing to grade or train "
                        f"an incomplete episode (exit_code={agent_result.exit_code})"
                    )
                if max_session_turns_reached:
                    logger.info(
                        "Agent reached configured max session turns; preserving the bounded "
                        "trajectory after integrity checks label=%s max_turns=%d",
                        sample.label,
                        MAX_TURNS,
                    )

                # Stop accepting model requests while the tunnel is still
                # alive, then prove that no partial request survived shutdown.
                await asyncio.to_thread(proxy.close)
                lifecycle = proxy.lifecycle_metadata()
                if any(lifecycle.values()):
                    raise RuntimeError(
                        "Agentic SWE proxy did not drain completely; refusing to train "
                        f"an incomplete episode: {lifecycle}"
                    )
                if proxy.failure is not None:
                    raise RuntimeError(
                        "Agentic SWE proxy failed during the episode; refusing to grade or train "
                        "a partial trajectory"
                    ) from proxy.failure
                patch = await task.collect_patch(runtime)
        finally:
            # Release the mutable Agent sandbox before requesting the clean
            # grader. This keeps grading isolated while avoiding two live
            # Sandboxes per rollout at the capacity peak.
            if runtime is not None:
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
