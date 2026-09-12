"""Convert official SWE-bench Verified rows to the Agentic SWE JSONL contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER_PY
from swebench.harness.test_spec.python import get_test_directives


REQUIRED_SOURCE_FIELDS = {
    "instance_id",
    "repo",
    "version",
    "base_commit",
    "problem_statement",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "image",
}


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


def load_ready_templates(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported template manifest schema: {manifest.get('schema_version')!r}")
    templates = manifest.get("templates") or {}
    return {
        instance_id: entry
        for instance_id, entry in templates.items()
        if entry.get("status") == "ready" and entry.get("alias")
    }


def official_test_command(instance: dict[str, Any]) -> str:
    repo = instance["repo"]
    version = instance["version"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    base_command = specs["test_cmd"]
    if not isinstance(base_command, str):
        raise ValueError(f"{instance['instance_id']}: expected one Python test command, got {base_command!r}")

    directives = get_test_directives(instance)
    test_command = " ".join([base_command, *directives])

    # Official SWE-bench instance images contain the testbed conda environment,
    # but SDK command execution starts a fresh non-login shell. Reproduce the
    # environment-sensitive portion of the official eval script explicitly.
    commands = [
        *(specs.get("eval_commands") or []),
    ]
    if specs.get("install"):
        commands.append(specs["install"])
    commands.append(test_command)
    # Activate testbed in the parent shell so every subsequent command inherits
    # the selected Python environment.  Parenthesizing `source`/`conda activate`
    # separately discards their shell state and makes every grade return zero.
    return "source /opt/miniconda3/bin/activate testbed && " + " && ".join(
        f"({command})" for command in commands
    )


def convert_row(row: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    instance = source_metadata(row)
    instance_id = instance["instance_id"]
    missing = REQUIRED_SOURCE_FIELDS - instance.keys()
    if missing:
        raise ValueError(f"{instance_id}: missing source fields {sorted(missing)}")
    if instance["repo"] not in MAP_REPO_TO_PARSER_PY:
        raise ValueError(f"{instance_id}: unsupported Verified parser for repo {instance['repo']!r}")
    if template.get("image") != instance["image"]:
        raise ValueError(
            f"{instance_id}: template image mismatch: {template.get('image')!r} != {instance['image']!r}"
        )

    metadata = {
        "instance_id": instance_id,
        "repo": instance["repo"],
        "repo_workdir": "/testbed",
        "base_commit": instance["base_commit"],
        "version": instance["version"],
        "image_name": instance["image"],
        "inspire_template": template["alias"],
        "docker_image_default_user": "root",
        "docker_image_env": {},
        "test_patch": instance["test_patch"],
        "FAIL_TO_PASS": list(instance["FAIL_TO_PASS"]),
        "PASS_TO_PASS": list(instance["PASS_TO_PASS"]),
        "install_config": {"test_cmd": official_test_command(instance)},
    }
    return {
        "prompt": [{"role": "user", "content": instance["problem_statement"]}],
        "label": instance_id,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source)
    if args.limit is not None:
        rows = rows[: args.limit]
    templates = load_ready_templates(args.template_manifest)

    instance_ids = [source_metadata(row).get("instance_id") for row in rows]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("source contains duplicate instance_id values")
    missing_templates = sorted(set(instance_ids) - templates.keys())
    if missing_templates:
        raise ValueError(
            f"missing ready templates for {len(missing_templates)} instance(s): {missing_templates[:10]}"
        )

    converted = [convert_row(row, templates[row_id]) for row, row_id in zip(rows, instance_ids, strict=True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in converted:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    repos = {row["metadata"]["repo"] for row in converted}
    print(
        f"READY Agentic SWE data rows={len(converted)} repos={len(repos)} "
        f"templates={len(converted)} output={args.output}"
    )


if __name__ == "__main__":
    main()
