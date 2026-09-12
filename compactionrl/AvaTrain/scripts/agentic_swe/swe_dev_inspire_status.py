"""Read-only preflight/status report for SWE-Dev Inspire template construction."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return dict(metadata.get("remote_env_info") or metadata)


def safe_context_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry-manifest", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--prewarm-progress", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--public-concurrency", type=int, default=4)
    parser.add_argument("--rebuild-concurrency", type=int, default=2)
    parser.add_argument("--sample-public-seconds", type=float, default=149.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source)
    instances = {source_metadata(row)["instance_id"]: source_metadata(row) for row in rows}
    if len(instances) != len(rows):
        raise ValueError("source contains duplicate instance ids")

    registry: dict[str, Any] = {}
    if args.registry_manifest.is_file():
        registry = json.loads(args.registry_manifest.read_text(encoding="utf-8"))
    registry_images = registry.get("images") or {}
    registry_states = Counter(
        "public" if entry.get("available") is True else "missing" if entry.get("available") is False else "error"
        for instance_id, entry in registry_images.items()
        if instance_id in instances
    )
    registry_unseen = len(instances) - sum(registry_states.values())

    templates: dict[str, Any] = {}
    shared_environment: dict[str, Any] = {}
    if args.template_manifest.is_file():
        template_manifest = json.loads(args.template_manifest.read_text(encoding="utf-8"))
        templates = template_manifest.get("templates") or {}
        shared_environment = template_manifest.get("shared_environment") or {}
    template_states = Counter(
        (templates.get(instance_id) or {}).get("status", "unseen") for instance_id in instances
    )
    ready_sources = Counter(
        (entry.get("build_source") or "public_image")
        for instance_id, entry in templates.items()
        if instance_id in instances and entry.get("status") == "ready"
    )

    context_names = {
        path.parent.name
        for path in args.build_root.glob("*/contexts/*/setup_repo.sh")
        if path.is_file()
    }
    context_covered = sum(
        safe_context_name(instance["image"]) in context_names for instance in instances.values()
    )

    prewarm: dict[str, Any] = {}
    if args.prewarm_progress.is_file():
        prewarm = json.loads(args.prewarm_progress.read_text(encoding="utf-8"))

    confirmed_public_todo = sum(
        registry_images.get(instance_id, {}).get("available") is True
        and (templates.get(instance_id) or {}).get("status") != "ready"
        for instance_id in instances
    )
    confirmed_missing_todo = sum(
        registry_images.get(instance_id, {}).get("available") is False
        and (templates.get(instance_id) or {}).get("status") != "ready"
        for instance_id in instances
    )
    public_minutes = math.ceil(
        confirmed_public_todo * args.sample_public_seconds / max(args.public_concurrency, 1) / 60
    )

    print("SWE-Dev Inspire construction status")
    print(f"source_total                 = {len(instances)}")
    print(
        "registry                    = "
        f"public:{registry_states['public']} missing:{registry_states['missing']} "
        f"errors:{registry_states['error']} unseen:{registry_unseen}"
    )
    print(
        "templates                   = "
        f"ready:{template_states['ready']} building:{template_states['building']} "
        f"failed:{template_states['failed']} unseen:{template_states['unseen']}"
    )
    print(f"ready_sources               = {dict(sorted(ready_sources.items()))}")
    print(f"local_context_coverage       = {context_covered}/{len(instances)}")
    print(
        "local_prewarm               = "
        f"requested:{prewarm.get('requested')} present:{prewarm.get('present')} "
        f"succeeded:{prewarm.get('succeeded')} failed:{prewarm.get('failed')}"
    )
    print(f"shared_rebuild_environment   = {shared_environment.get('status', 'not-built')}")
    print(f"confirmed_public_todo        = {confirmed_public_todo}")
    print(f"confirmed_missing_todo       = {confirmed_missing_todo}")
    print(
        "public_batch_estimate       = "
        f"~{public_minutes} min at concurrency={args.public_concurrency} "
        f"using conservative first-build sample {args.sample_public_seconds:.0f}s/image"
    )

    problems: list[str] = []
    if len(instances) != 1000:
        problems.append(f"expected 1000 source rows, got {len(instances)}")
    if context_covered != len(instances):
        problems.append(f"missing local contexts for {len(instances) - context_covered} rows")
    if not (
        prewarm.get("requested") == 1000
        and prewarm.get("present") == 1000
        and prewarm.get("succeeded") == 1000
        and prewarm.get("failed") == 0
    ):
        problems.append("local 1000-image prewarm gate is not green")
    unresolved_registry = registry_states["error"] + registry_unseen
    if unresolved_registry:
        problems.append(f"registry routing unresolved for {unresolved_registry} rows")
    if registry_states["missing"] and shared_environment.get("status") != "ready":
        problems.append("shared Python 3.9 rebuild environment is not built and verified")

    if problems:
        print("FULL_BATCH_READY             = NO")
        for problem in problems:
            print(f"BLOCKER                     = {problem}")
    else:
        print("FULL_BATCH_READY             = YES")


if __name__ == "__main__":
    main()
