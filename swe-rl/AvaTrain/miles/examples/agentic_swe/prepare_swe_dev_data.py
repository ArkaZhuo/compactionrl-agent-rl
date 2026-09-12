"""Convert SWE-Dev/SWE-bench Extra rows to the Agentic SWE JSONL contract."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


SPEC_SOURCE = "nebius-swe-bench-extra@11dcbfb30e19552df2a2f8030bd764adc95c92a5"
TEST_COMMAND = (
    "pytest --no-header -rA --tb=no -p no:cacheprovider -p no:randomly "
    "-W ignore::DeprecationWarning"
)
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


def replace_image_namespace(image: str, namespace: str) -> str:
    _, separator, remainder = image.partition("/")
    if not separator or not remainder:
        raise ValueError(f"image has no namespace: {image!r}")
    return f"{namespace}/{remainder}"


def load_ready_templates(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported template manifest schema: {manifest.get('schema_version')!r}")
    return {
        instance_id: entry
        for instance_id, entry in (manifest.get("templates") or {}).items()
        if entry.get("status") == "ready" and entry.get("alias")
    }


def string_list(instance: dict[str, Any], key: str) -> list[str]:
    value = instance.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{instance.get('instance_id')}: {key} must be a list of strings")
    return value


def is_scored_test_identifier(identifier: str) -> bool:
    """Return whether an identifier can name a scored pytest test.

    SWE-bench's pytest log parser keeps only the first whitespace-delimited
    token from a result line.  Consequently, parameterized identifiers that
    appear to have an unclosed ``[`` can still be valid published identifiers.
    Validate only the stable contract required by the grader: a Python test
    path, ``::``, and a non-empty test selector.
    """
    path, separator, selector = identifier.strip().partition("::")
    return bool(separator and path.endswith(".py") and selector)


def validated_scored_tests(
    instance_id: str, fail_to_pass: list[str], pass_to_pass: list[str]
) -> tuple[list[str], list[str]]:
    invalid_fail_to_pass = [
        identifier for identifier in fail_to_pass if not is_scored_test_identifier(identifier)
    ]
    if invalid_fail_to_pass:
        raise ValueError(
            f"{instance_id}: malformed FAIL_TO_PASS identifier(s): {invalid_fail_to_pass!r}"
        )

    # SWE-Dev contains one malformed published P2P entry ("[").  It cannot
    # identify a test and therefore must not participate in reward scoring.
    valid_pass_to_pass = [
        identifier for identifier in pass_to_pass if is_scored_test_identifier(identifier)
    ]
    return fail_to_pass, valid_pass_to_pass


def grading_test_command(fail_to_pass: list[str], pass_to_pass: list[str]) -> str:
    # Run only files containing scored tests.  Whole-repository collection can
    # fail on unrelated optional test dependencies before any F2P/P2P test is
    # executed.  File-level selectors avoid oversized parameterized node IDs
    # while keeping the selectors used for reward scoring in metadata.
    test_paths = sorted(
        {
            identifier.strip().split("::", 1)[0]
            for identifier in [*fail_to_pass, *pass_to_pass]
            if identifier.strip().split("::", 1)[0].endswith(".py")
        }
    )
    if not test_paths:
        raise ValueError("SWE-Dev row has no valid Python test paths")

    # Activate testbed in the same shell that launches pytest.  Do not wrap the
    # activation in a subshell: environment changes would be discarded before
    # the test command runs, causing Conda's "run conda init" error.
    command = shlex.join([*shlex.split(TEST_COMMAND), *test_paths])
    return f"source /opt/miniconda3/bin/activate testbed && {command}"


def convert_row(
    row: dict[str, Any], template: dict[str, Any], *, image_namespace: str
) -> dict[str, Any]:
    instance = source_metadata(row)
    instance_id = instance.get("instance_id") or "<unknown>"
    missing = REQUIRED_SOURCE_FIELDS - instance.keys()
    if missing:
        raise ValueError(f"{instance_id}: missing source fields {sorted(missing)}")
    if instance["version"] != "0.0":
        raise ValueError(f"{instance_id}: unsupported SWE-Dev version {instance['version']!r}")
    if instance.get("swebench_spec_source") != SPEC_SOURCE:
        raise ValueError(f"{instance_id}: source is not the pinned SWE-bench Extra revision")
    expected_image = replace_image_namespace(instance["image"], image_namespace)
    if template.get("image") != expected_image:
        raise ValueError(
            f"{instance_id}: template image mismatch: {template.get('image')!r} != {expected_image!r}"
        )

    fail_to_pass = string_list(instance, "FAIL_TO_PASS")
    pass_to_pass = string_list(instance, "PASS_TO_PASS")
    if not fail_to_pass:
        raise ValueError(f"{instance_id}: FAIL_TO_PASS must not be empty")
    fail_to_pass, pass_to_pass = validated_scored_tests(
        instance_id, fail_to_pass, pass_to_pass
    )
    metadata = {
        "instance_id": instance_id,
        "repo": instance["repo"],
        "repo_workdir": "/testbed",
        "base_commit": instance["base_commit"],
        "version": instance["version"],
        "image_name": expected_image,
        "inspire_template": template["alias"],
        "docker_image_default_user": "root",
        "docker_image_env": {},
        "test_patch": instance["test_patch"],
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "install_config": {"test_cmd": grading_test_command(fail_to_pass, pass_to_pass)},
        "swebench_log_parser": "pytest",
        "swebench_spec_source": SPEC_SOURCE,
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
    parser.add_argument("--image-namespace", default="swerebench")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--allow-missing-templates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.source)
    if args.instance_id:
        wanted = set(args.instance_id)
        rows = [row for row in rows if source_metadata(row).get("instance_id") in wanted]
        found = {source_metadata(row).get("instance_id") for row in rows}
        if wanted != found:
            raise ValueError(f"unknown instance ids: {sorted(wanted - found)}")

    instance_ids = [source_metadata(row).get("instance_id") for row in rows]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("source contains duplicate instance_id values")
    templates = load_ready_templates(args.template_manifest)
    missing_templates = sorted(set(instance_ids) - templates.keys())
    if missing_templates and not args.allow_missing_templates:
        raise ValueError(
            f"missing ready templates for {len(missing_templates)} instance(s): {missing_templates[:10]}"
        )

    converted = [
        convert_row(row, templates[instance_id], image_namespace=args.image_namespace)
        for row, instance_id in zip(rows, instance_ids, strict=True)
        if instance_id in templates
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in converted:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    repos = {row["metadata"]["repo"] for row in converted}
    print(
        f"READY Agentic SWE-Dev rows={len(converted)} repos={len(repos)} "
        f"missing_templates={len(missing_templates)} output={args.output}"
    )


if __name__ == "__main__":
    main()
