"""Rebuild registry-missing SWE-Dev images as layered Inspire templates.

The local Docker build contexts are already validated by the 1000-image
prewarm gate.  This tool builds one shared Ubuntu/Conda/Python 3.9/protocol
template, then applies only each instance's latest ``setup_repo.sh`` context.
All state is recorded in the same manifest used by public-image builds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from inspire_sandbox import AsyncSandbox, AsyncTemplate, SandboxSpecCode, Template, default_build_logger


PROTOCOL_ROOT = "/__avaeval_agentic_protocol_v1__"
QWEN_BIN = f"{PROTOCOL_ROOT}/frameworks/qwen_code/bin/qwen"
WSTUNNEL_BIN = f"{PROTOCOL_ROOT}/linux/bin/wstunnel"
MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return dict(metadata.get("remote_env_info") or metadata)


def template_alias(instance_id: str, prefix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    digest = hashlib.sha256(instance_id.encode()).hexdigest()[:10]
    room = 63 - len(prefix) - len(digest) - 2
    if room < 8:
        raise ValueError(f"template prefix is too long: {prefix!r}")
    return f"{prefix}-{slug[:room].rstrip('-')}-{digest}"


def replace_image_namespace(image: str, namespace: str) -> str:
    _, separator, remainder = image.partition("/")
    if not separator or not remainder:
        raise ValueError(f"image has no namespace: {image!r}")
    return f"{namespace}/{remainder}"


def safe_context_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "__")


def latest_context(build_root: Path, image: str, base_commit: str) -> Path:
    name = safe_context_name(image)
    candidates = [
        path
        for path in build_root.glob(f"*/contexts/{name}")
        if (path / "setup_repo.sh").is_file()
    ]
    compatible: list[Path] = []
    for path in candidates:
        script = (path / "setup_repo.sh").read_text(encoding="utf-8", errors="replace")
        if base_commit in script:
            compatible.append(path)
    if not compatible:
        raise FileNotFoundError(f"no context containing base commit {base_commit} for {image}")
    return max(compatible, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def load_protocol(bundle: Path) -> dict[str, Any]:
    required = [
        bundle / "manifest.json",
        bundle / "frameworks/qwen_code/bin/qwen",
        bundle / "frameworks/qwen_code/node/bin/node",
        bundle / "linux/bin/wstunnel",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol bundle is incomplete: {missing}")
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def load_manifest(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("protocol") != protocol:
            raise ValueError("template manifest schema or protocol mismatch")
        manifest.setdefault("templates", {})
        return manifest
    return {"schema_version": 1, "protocol": protocol, "templates": {}}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{path} is outside shared root {root}") from exc


def make_shared_template(args: argparse.Namespace):
    setup_env = args.env_context / "setup_env.sh"
    if not setup_env.is_file():
        raise FileNotFoundError(setup_env)
    context = Template(file_context_path=str(args.shared_root), file_ignore_patterns=[])
    builder = (
        context.from_image("ubuntu:22.04")
        .run_cmd(
            "export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y "
            "wget git build-essential libffi-dev libtiff-dev python3 python3-pip "
            "python-is-python3 jq curl locales locales-all tzdata pkg-config && "
            "rm -rf /var/lib/apt/lists/*",
            user="root",
        )
        .run_cmd(
            f"wget -q {MINICONDA_URL} -O /tmp/miniconda.sh && "
            "bash /tmp/miniconda.sh -b -p /opt/miniconda3 && rm -f /tmp/miniconda.sh && "
            "/opt/miniconda3/bin/conda config --append channels conda-forge",
            user="root",
        )
        .copy(relative(setup_env, args.shared_root), "/root/setup_env.sh", user="root", mode=0o755)
        .run_cmd("/bin/bash /root/setup_env.sh", user="root")
        .copy(relative(args.protocol_bundle / "frameworks", args.shared_root), f"{PROTOCOL_ROOT}/frameworks", user="root")
        .copy(relative(args.protocol_bundle / "linux", args.shared_root), f"{PROTOCOL_ROOT}/linux", user="root")
        .copy(
            relative(args.protocol_bundle / "manifest.json", args.shared_root),
            f"{PROTOCOL_ROOT}/manifest.json",
            user="root",
            mode=0o644,
        )
        .run_cmd(
            f"chmod 0755 {QWEN_BIN} {PROTOCOL_ROOT}/frameworks/qwen_code/node/bin/node {WSTUNNEL_BIN}",
            user="root",
        )
        .run_cmd(
            "source /opt/miniconda3/bin/activate && conda activate testbed && "
            f"python --version && pytest --version && {QWEN_BIN} --version && {WSTUNNEL_BIN} --version",
            user="root",
        )
        .set_envs({"PATH": f"{PROTOCOL_ROOT}/linux/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .set_user("root")
        .set_workdir("/")
    )
    return builder


def make_instance_template(
    args: argparse.Namespace, instance: dict[str, Any], context_path: Path
):
    template = Template(file_context_path=str(context_path), file_ignore_patterns=[])
    return (
        template.from_template(args.base_template)
        .copy("setup_repo.sh", "/root/setup_repo.sh", user="root", mode=0o755)
        .run_cmd("sed -i -e 's/\\r$//' /root/setup_repo.sh && /bin/bash /root/setup_repo.sh", user="root")
        .run_cmd(
            f"git -C /testbed reset --hard {instance['base_commit']} && "
            "git -C /testbed clean -fd && "
            f"test \"$(git -C /testbed rev-parse HEAD)\" = \"{instance['base_commit']}\"",
            user="root",
        )
        .run_cmd(
            "source /opt/miniconda3/bin/activate && conda activate testbed && python --version && "
            f"{QWEN_BIN} --version && {WSTUNNEL_BIN} --version",
            user="root",
        )
        .set_user("root")
        .set_workdir("/testbed")
    )


async def verify_template(alias: str, instance: dict[str, Any] | None = None) -> None:
    sandbox = await AsyncSandbox.create(template=alias, timeout=600, network={"allow_public_traffic": True})
    try:
        commands = [
            "test \"$(whoami)\" = root",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "test \"$(python -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')\" = 3.9",
            f"{QWEN_BIN} --version",
            f"{WSTUNNEL_BIN} --version",
        ]
        cwd = "/"
        if instance is not None:
            cwd = "/testbed"
            commands.extend(
                [
                    "test \"$(pwd)\" = /testbed",
                    f"git reset --hard {instance['base_commit']}",
                    "git clean -fd",
                    f"test \"$(git rev-parse HEAD)\" = \"{instance['base_commit']}\"",
                ]
            )
        result = await sandbox.commands.run(
            " && ".join(commands), timeout=300, request_timeout=360, user="root", cwd=cwd
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
    finally:
        await sandbox.kill()


async def run(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol_bundle)
    manifest = load_manifest(args.manifest, protocol)
    rows = read_jsonl(args.dataset)
    instances = [source_metadata(row) for row in rows]
    if args.instance_id:
        wanted = set(args.instance_id)
        instances = [item for item in instances if item.get("instance_id") in wanted]
        found = {item.get("instance_id") for item in instances}
        if wanted != found:
            raise ValueError(f"unknown instance ids: {sorted(wanted - found)}")
    if args.limit:
        instances = instances[: args.limit]

    selected: list[tuple[dict[str, Any], Path]] = []
    for instance in instances:
        for key in ("instance_id", "image", "base_commit"):
            if not instance.get(key):
                raise ValueError(f"row is missing {key}")
        selected.append((instance, latest_context(args.build_root, instance["image"], instance["base_commit"])))

    if args.action == "validate":
        make_shared_template(args)
        for instance, context in selected:
            alias = template_alias(instance["instance_id"], args.name_prefix)
            make_instance_template(args, instance, context)
            print(f"VALID {instance['instance_id']} context={context} alias={alias}")
        print(f"SUMMARY valid={len(selected)} base_template={args.base_template}")
        return

    if args.action == "build-base":
        info = await AsyncTemplate.build(
            make_shared_template(args),
            args.base_template,
            spec_code=SandboxSpecCode.G_C2,
            skip_cache=args.rebuild,
            on_build_logs=default_build_logger(min_level="info"),
        )
        manifest["shared_environment"] = {
            "alias": args.base_template,
            "status": "ready",
            "template_id": info.template_id,
            "build_id": info.build_id,
            "python": "3.9",
            "protocol": protocol,
        }
        write_manifest(args.manifest, manifest)
        print(f"READY shared_template={args.base_template} template_id={info.template_id}")
        return

    if args.action == "verify-base":
        await verify_template(args.base_template)
        print(f"READY verified_shared_template={args.base_template}")
        return

    shared = manifest.get("shared_environment") or {}
    if shared.get("status") != "ready" or shared.get("alias") != args.base_template:
        raise ValueError("shared environment is not ready; run build-base and verify-base first")

    if args.action == "verify":
        if len(selected) != 1:
            raise ValueError("verify requires exactly one selected instance")
        instance, _ = selected[0]
        entry = manifest["templates"].get(instance["instance_id"]) or {}
        if entry.get("status") != "ready":
            raise ValueError(f"template is not ready: {entry}")
        await verify_template(entry["alias"], instance)
        print(f"READY verified_template={entry['alias']} instance={instance['instance_id']}")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def build_one(index: int, instance: dict[str, Any], context: Path) -> None:
        instance_id = instance["instance_id"]
        alias = template_alias(instance_id, args.name_prefix)
        existing = manifest["templates"].get(instance_id) or {}
        if existing.get("status") == "ready" and not args.rebuild:
            print(f"PRESENT [{index}/{len(selected)}] {instance_id} -> {existing['alias']}")
            return
        async with semaphore:
            entry = {
                "alias": alias,
                "image": replace_image_namespace(instance["image"], args.image_namespace),
                "base_commit": instance["base_commit"],
                "status": "building",
                "build_source": "local_context_rebuild",
                "context": str(context),
                "shared_template": args.base_template,
            }
            async with lock:
                manifest["templates"][instance_id] = entry
                write_manifest(args.manifest, manifest)
            print(f"BUILD [{index}/{len(selected)}] {instance_id} context={context} alias={alias}")
            try:
                info = await AsyncTemplate.build(
                    make_instance_template(args, instance, context),
                    alias,
                    spec_code=SandboxSpecCode.G_C2,
                    skip_cache=args.rebuild,
                    on_build_logs=default_build_logger(min_level="info"),
                )
                entry.update(status="ready", template_id=info.template_id, build_id=info.build_id)
                print(f"READY [{index}/{len(selected)}] {instance_id} template_id={info.template_id}")
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"FAILED [{index}/{len(selected)}] {instance_id}: {entry['error']}")
                if args.fail_fast:
                    raise
            finally:
                async with lock:
                    manifest["templates"][instance_id] = entry
                    write_manifest(args.manifest, manifest)

    await asyncio.gather(
        *(build_one(index, instance, context) for index, (instance, context) in enumerate(selected, start=1))
    )
    ready = sum(entry.get("status") == "ready" for entry in manifest["templates"].values())
    failed = sum(entry.get("status") == "failed" for entry in manifest["templates"].values())
    print(f"SUMMARY selected={len(selected)} ready_total={ready} failed_total={failed} manifest={args.manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "build-base", "verify-base", "build", "verify"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--env-context", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--base-template", default="sywang-fzk-swedev-py39-qc0210-v1")
    parser.add_argument("--name-prefix", default="sywang-fzk-swedev-qc0210")
    parser.add_argument("--image-namespace", default="swerebench")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
