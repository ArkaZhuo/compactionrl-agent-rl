"""Build resumable Inspire templates for SWE-bench Verified instance images."""

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


def load_protocol_manifest(bundle: Path) -> dict[str, Any]:
    required = [
        bundle / "manifest.json",
        bundle / "frameworks/qwen_code/bin/qwen",
        bundle / "frameworks/qwen_code/node/bin/node",
        bundle / "frameworks/qwen_code/lib/qwen-code/cli-entry.js",
        bundle / "linux/bin/wstunnel",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol bundle is incomplete: {missing}")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported protocol manifest")
    return manifest


def load_template_manifest(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported template manifest")
        if manifest.get("protocol") != protocol:
            raise ValueError("template manifest was created with a different protocol bundle")
        manifest.setdefault("templates", {})
        return manifest
    return {"schema_version": 1, "protocol": protocol, "templates": {}}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_template(bundle: Path, instance: dict[str, Any]):
    # copy() paths are relative to file_context_path. Keeping the two shared
    # protocol directories as separate layers maximizes cross-instance cache use.
    template = Template(file_context_path=str(bundle), file_ignore_patterns=[])
    builder = (
        template.from_image(instance["image"])
        .copy("frameworks", f"{PROTOCOL_ROOT}/frameworks", user="root")
        .copy("linux", f"{PROTOCOL_ROOT}/linux", user="root")
        .copy("manifest.json", f"{PROTOCOL_ROOT}/manifest.json", user="root", mode=0o644)
        .run_cmd(
            f"chmod 0755 {QWEN_BIN} {PROTOCOL_ROOT}/frameworks/qwen_code/node/bin/node {WSTUNNEL_BIN}",
            user="root",
        )
        .run_cmd("test -d /testbed/.git", user="root")
        .run_cmd(
            f"actual=$(git -C /testbed rev-parse HEAD) && echo image_head=$actual && "
            f"git -C /testbed cat-file -e \"{instance['base_commit']}^{{commit}}\"",
            user="root",
        )
        .run_cmd(f"{QWEN_BIN} --version", user="root")
        .run_cmd(f"{WSTUNNEL_BIN} --version", user="root")
        .set_user("root")
        .set_workdir("/testbed")
    )
    return builder


async def verify_runtime(alias: str, instance: dict[str, Any]) -> None:
    sandbox = await AsyncSandbox.create(
        template=alias,
        timeout=600,
        network={"allow_public_traffic": True},
    )
    try:
        command = " && ".join(
            [
                "test \"$(pwd)\" = /testbed",
                "test \"$(whoami)\" = root",
                f"git cat-file -e \"{instance['base_commit']}^{{commit}}\"",
                f"git reset --hard \"{instance['base_commit']}\"",
                "git clean -fd",
                f"test \"$(git rev-parse HEAD)\" = \"{instance['base_commit']}\"",
                f"{QWEN_BIN} --version",
                f"{WSTUNNEL_BIN} --version",
                "git status --porcelain=v1",
            ]
        )
        result = await sandbox.commands.run(command, timeout=300, request_timeout=360, user="root", cwd="/testbed")
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
    finally:
        await sandbox.kill()


async def run(args: argparse.Namespace) -> None:
    protocol = load_protocol_manifest(args.protocol_bundle)
    manifest = load_template_manifest(args.manifest, protocol)
    rows = read_jsonl(args.dataset)
    instances = [source_metadata(row) for row in rows]
    if args.image_namespace:
        for instance in instances:
            image = instance.get("image")
            if image:
                instance["image"] = replace_image_namespace(image, args.image_namespace)

    if args.instance_id:
        wanted = set(args.instance_id)
        instances = [instance for instance in instances if instance.get("instance_id") in wanted]
        missing = wanted - {instance.get("instance_id") for instance in instances}
        if missing:
            raise ValueError(f"unknown instance ids: {sorted(missing)}")
    if args.limit is not None:
        instances = instances[: args.limit]

    required = {"instance_id", "image", "base_commit"}
    for instance in instances:
        missing = required - instance.keys()
        if missing:
            raise ValueError(f"row is missing fields: {sorted(missing)}")

    if args.action == "validate":
        aliases: set[str] = set()
        for instance in instances:
            alias = template_alias(instance["instance_id"], args.name_prefix)
            if alias in aliases:
                raise ValueError(f"template alias collision: {alias}")
            aliases.add(alias)
            print(f"VALID {instance['instance_id']} image={instance['image']} alias={alias}")
        # Construct one complete SDK definition. The protocol copy layers are
        # identical for all instances, so hashing the ~200 MB bundle 500 times
        # would add no coverage and is prohibitively slow on the shared HDD.
        if instances:
            make_template(args.protocol_bundle, instances[0])
        print(f"SUMMARY valid={len(instances)}")
        return

    if args.action == "verify":
        if len(instances) != 1:
            raise ValueError("verify requires exactly one selected instance")
        instance = instances[0]
        entry = manifest["templates"].get(instance["instance_id"]) or {}
        if entry.get("status") != "ready":
            raise ValueError(f"template is not ready: {entry}")
        await verify_runtime(entry["alias"], instance)
        print(f"READY verified_template={entry['alias']} instance={instance['instance_id']}")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    manifest_lock = asyncio.Lock()

    async def build_one(index: int, instance: dict[str, Any]) -> None:
        instance_id = instance["instance_id"]
        alias = template_alias(instance_id, args.name_prefix)
        existing = manifest["templates"].get(instance_id) or {}
        if existing.get("status") == "ready" and not args.rebuild:
            print(f"PRESENT [{index}/{len(instances)}] {instance_id} -> {existing['alias']}")
            return
        async with semaphore:
            print(f"BUILD [{index}/{len(instances)}] {instance_id} image={instance['image']} alias={alias}")
            entry = {
                "alias": alias,
                "image": instance["image"],
                "base_commit": instance["base_commit"],
                "status": "building",
            }
            async with manifest_lock:
                manifest["templates"][instance_id] = entry
                write_manifest(args.manifest, manifest)
            try:
                info = await AsyncTemplate.build(
                    make_template(args.protocol_bundle, instance),
                    alias,
                    spec_code=SandboxSpecCode.G_C1,
                    skip_cache=args.rebuild,
                    on_build_logs=default_build_logger(min_level="info"),
                )
                entry.update(
                    {
                        "status": "ready",
                        "template_id": info.template_id,
                        "build_id": info.build_id,
                    }
                )
                print(f"READY [{index}/{len(instances)}] {instance_id} template_id={info.template_id}")
            except Exception as exc:
                entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED [{index}/{len(instances)}] {instance_id}: {entry['error']}")
                if args.fail_fast:
                    raise
            finally:
                async with manifest_lock:
                    manifest["templates"][instance_id] = entry
                    write_manifest(args.manifest, manifest)

    await asyncio.gather(*(build_one(i, instance) for i, instance in enumerate(instances, start=1)))
    ready = sum(entry.get("status") == "ready" for entry in manifest["templates"].values())
    failed = sum(entry.get("status") == "failed" for entry in manifest["templates"].values())
    print(f"SUMMARY selected={len(instances)} ready_total={ready} failed_total={failed} manifest={args.manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "build", "verify"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name-prefix", default="sywang-fzk-swev-qc0210")
    parser.add_argument(
        "--image-namespace",
        help="Replace the first component of every source image (for example, swebench -> swerebench).",
    )
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
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
