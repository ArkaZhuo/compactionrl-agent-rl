"""One SWE rollout with trainable context compaction."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shlex

# The platform helpers remain unchanged from the isolated copy's regular SWE
# example. The launcher puts that directory after this package on PYTHONPATH.
import sandbox
from swe import SweTask

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.sglang_rollout import get_model_url

from .config import CompactionConfig
from .proxy import MODEL_NAME, CompactionModelProxy

SANDBOX_MODEL_PORT = 30001
WSTUNNEL_SERVER_PORT = 19090
AGENT_BIN = "/__avaeval_agentic_protocol_v1__/frameworks/qwen_code/bin/qwen"
WSTUNNEL_BIN = "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "250"))
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT_SEC", str(90 * 60)))
QWEN_API_TIMEOUT_MS = int(os.environ.get("QWEN_CODE_API_TIMEOUT_MS", "360000"))
QWEN_MAX_RETRIES = int(os.environ.get("QWEN_CODE_MAX_RETRIES", "0"))
TUNNEL_READY_TIMEOUT = int(os.environ.get("TUNNEL_READY_TIMEOUT_SEC", "60"))
TUNNEL_READY_PROBE_TIMEOUT = int(os.environ.get("TUNNEL_READY_PROBE_TIMEOUT_SEC", "5"))

logger = logging.getLogger(__name__)
_PROJECT_SEMAPHORES: dict[tuple[str, int], asyncio.Semaphore] = {}


def project_semaphore(project: str) -> asyncio.Semaphore | None:
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
    base_url = f"http://127.0.0.1:{SANDBOX_MODEL_PORT}/v1"
    project_hash = hashlib.sha256(workdir.encode("utf-8")).hexdigest()
    settings_path = "/tmp/avaeval-qwen-settings.json"
    # Qwen Code 0.21.0 enables managed memory extraction and dreaming by
    # default.  Those background model calls have an unrelated system prompt,
    # so they must not enter (or invalidate) the ordered PPO trajectory.  The
    # system settings file has higher precedence than user/project settings in
    # the sandbox, making these controls deterministic for every episode.
    settings = json.dumps(
        {
            "model": {
                # The proxy, not Qwen Code, owns the real 64K working-window
                # limit. Keep the client's own ~85%-of-context compressor
                # above the maximum four-window raw history so it cannot issue
                # a second, unrelated summarization request mid-episode.
                "generationConfig": {
                    "maxRetries": QWEN_MAX_RETRIES,
                    "contextWindowSize": 1_000_000,
                },
                "skipNextSpeakerCheck": True,
            },
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
        "agentic-swe-compactionrl",
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
    if TUNNEL_READY_TIMEOUT <= 0 or TUNNEL_READY_PROBE_TIMEOUT <= 0:
        raise ValueError("reverse-tunnel readiness timeouts must be positive")

    probe_source = """\
import json
import urllib.request

with urllib.request.urlopen(\"http://127.0.0.1:30001/v1/models\", timeout=%d) as response:
    payload = json.load(response)
assert response.status == 200
assert any(item.get(\"id\") == \"default\" for item in payload.get(\"data\", []))
print(\"compactionrl-tunnel-ready\")
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
            if result.exit_code == 0 and "compactionrl-tunnel-ready" in result.stdout:
                logger.info("CompactionRL reverse tunnel ready attempts=%d", attempts)
                return
        except Exception as exc:  # Preserve cancellation while retrying transport failures.
            last_detail = f"{exc.__class__.__name__}: {exc}"

        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RuntimeError(
                "CompactionRL reverse tunnel did not become reachable from the sandbox "
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
    config = CompactionConfig.from_env()
    prompt = sample.prompt[0]["content"]
    proxy = CompactionModelProxy(
        tokenizer=state.tokenizer,
        loop=asyncio.get_running_loop(),
        model_url=get_model_url(args, MODEL_NAME),
        sampling_params=input.sampling_params,
        config=config,
        tool_parser=getattr(args, "sglang_tool_call_parser", None),
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None),
    )

    patch = ""
    reward = 0.0
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
                if agent_result.exit_code != 0:
                    raise RuntimeError(
                        "qwen-code agent exited unsuccessfully; refusing to grade or train "
                        f"an incomplete episode (exit_code={agent_result.exit_code})"
                    )

                # Finalize the proxy while the reverse tunnel is still alive.
                # close() stops new requests, cancels/drains any request left by
                # qwen-code, and is idempotent when the AsyncExitStack calls it
                # again during cleanup.
                await asyncio.to_thread(proxy.close)
                lifecycle = proxy.lifecycle_metadata()
                if any(lifecycle.values()):
                    raise RuntimeError(
                        "CompactionRL proxy did not drain completely; refusing to train "
                        f"an incomplete episode: {lifecycle}"
                    )
                if proxy.failure is not None:
                    raise RuntimeError(
                        "CompactionRL proxy failed during the episode; refusing to grade or train "
                        "a partial trajectory"
                    ) from proxy.failure
                routing = proxy.routing_metadata()
                if routing["routing_failed_requests"] != 0:
                    raise RuntimeError(
                        "CompactionRL observed failed model requests; refusing to grade or train "
                        f"a partial trajectory: failed_requests={routing['routing_failed_requests']}"
                    )
                patch = await task.collect_patch(runtime)
        finally:
            if runtime is not None:
                await runtime.kill()
        if not proxy.ledger.has_trainable_tokens():
            raise RuntimeError("agent produced no trainable CompactionRL segment")
        if config.auxiliary_mode == "reject" and routing["routing_auxiliary_requests"]:
            raise RuntimeError(
                "strict CompactionRL rejected an episode with auxiliary requests: "
                f"requests={routing['routing_auxiliary_requests']} "
                f"tokens={routing['routing_auxiliary_generated_tokens']}"
            )
        reward = await task.grade_patch(patch)

    segments = proxy.to_samples(sample, reward)
    if not segments:
        raise RuntimeError("agent produced no trainable CompactionRL segment")
    if routing["routing_unexplained_generated_tokens"] != 0:
        raise RuntimeError("CompactionRL request accounting did not reconcile with total generation: " f"{routing}")
    # Auxiliary qwen-code requests are intentionally served but excluded from
    # the ordered PPO trajectory. They remain explicitly accounted for in
    # routing metadata; only unexplained tokens are invalid.
    optimized_tokens = sum(int(s.train_metadata["optimized_tokens"]) for s in segments)
    if optimized_tokens != routing["routing_trainable_generated_tokens"]:
        raise RuntimeError(
            "CompactionRL trainable token accounting does not match emitted segment loss masks: "
            f"optimized={optimized_tokens}, routing={routing}"
        )
    for segment in segments:
        segment.train_metadata.update(routing)
    logger.info(
        "CompactionRL finalized label=%s segments=%d compactions=%d "
        "optimized_tokens=%d generated_tokens=%d episode_tokens=%d "
        "main_requests=%d main_tokens=%d summary_requests=%d summary_tokens=%d "
        "auxiliary_requests=%d auxiliary_tokens=%d replay_rewrite_requests=%d "
        "replay_rewrite_tokens=%d untracked_requests=%d untracked_tokens=%d "
        "unexplained_tokens=%d max_active_requests=%d truncated_observations=%d "
        "retained_recent_steps=%s rebases=%d reward=%.6f",
        sample.label,
        len(segments),
        proxy.compaction_count,
        optimized_tokens,
        proxy.generated_tokens,
        proxy.episode_tokens,
        routing["routing_main_requests"],
        routing["routing_main_generated_tokens"],
        routing["routing_summary_requests"],
        routing["routing_summary_generated_tokens"],
        routing["routing_auxiliary_requests"],
        routing["routing_auxiliary_generated_tokens"],
        routing["routing_replay_rewrite_requests"],
        routing["routing_replay_rewrite_generated_tokens"],
        routing["routing_untracked_requests"],
        routing["routing_untracked_generated_tokens"],
        routing["routing_unexplained_generated_tokens"],
        routing["routing_max_active_requests"],
        proxy.truncated_observations,
        proxy.retained_recent_steps,
        proxy.rebase_count,
        float(reward),
    )
    return GenerateFnOutput(samples=segments)
