"""Split SWE-Dev deterministically across two Inspire Sandbox projects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return dict(metadata.get("remote_env_info") or metadata)


def instance_id(row: dict[str, Any]) -> str:
    value = source_metadata(row).get("instance_id") or row.get("label")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no instance_id")
    return value


def selected_ids(rows: list[dict[str, Any]], count: int) -> set[str]:
    if not 1 <= count < len(rows):
        raise ValueError(f"secondary count must be between 1 and {len(rows) - 1}")
    ranked = sorted(
        (hashlib.sha256(instance_id(row).encode()).hexdigest(), instance_id(row))
        for row in rows
    )
    return {value for _, value in ranked[:count]}


def partition(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.source)
    chosen = selected_ids(rows, args.secondary_count)
    subset = [row for row in rows if instance_id(row) in chosen]
    write_jsonl(args.secondary_source, subset)
    print(
        f"READY secondary_source rows={len(subset)} total={len(rows)} "
        f"output={args.secondary_source}"
    )


def merge(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.source)
    chosen = selected_ids(source_rows, args.secondary_count)
    primary_rows = read_jsonl(args.primary_data)
    manifest = json.loads(args.secondary_manifest.read_text(encoding="utf-8"))
    templates = manifest.get("templates") or {}
    ready = {
        key: value
        for key, value in templates.items()
        if value.get("status") == "ready" and value.get("alias")
    }
    missing = sorted(chosen - ready.keys())
    if missing:
        raise ValueError(f"secondary project is missing {len(missing)} ready templates: {missing[:10]}")

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {"primary": 0, "secondary": 0}
    for row in primary_rows:
        label = instance_id(row)
        if label in seen:
            raise ValueError(f"duplicate instance_id: {label}")
        seen.add(label)
        metadata = dict(row["metadata"])
        if label in chosen:
            metadata["inspire_template"] = ready[label]["alias"]
            metadata["sandbox_project"] = "secondary"
            counts["secondary"] += 1
        else:
            metadata["sandbox_project"] = "primary"
            counts["primary"] += 1
        output.append({**row, "metadata": metadata})

    expected = {instance_id(row) for row in source_rows}
    if seen != expected:
        raise ValueError(f"primary data/source mismatch: missing={len(expected-seen)} extra={len(seen-expected)}")
    write_jsonl(args.output, output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"READY dual_project_data rows={len(output)} primary={counts['primary']} "
        f"secondary={counts['secondary']} sha256={digest} output={args.output}"
    )


def status(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.source)
    chosen = selected_ids(source_rows, args.secondary_count)
    if not args.secondary_manifest.exists():
        print(f"STATUS selected={len(chosen)} ready=0 failed=0 building=0 missing={len(chosen)}")
        return
    manifest = json.loads(args.secondary_manifest.read_text(encoding="utf-8"))
    templates = manifest.get("templates") or {}
    states: dict[str, int] = {}
    for label in chosen:
        state = (templates.get(label) or {}).get("status", "missing")
        states[state] = states.get(state, 0) + 1
    print("STATUS selected=" + str(len(chosen)) + " " + " ".join(f"{k}={v}" for k, v in sorted(states.items())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["partition", "merge", "status"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--secondary-source", type=Path, required=True)
    parser.add_argument("--secondary-manifest", type=Path, required=True)
    parser.add_argument("--primary-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secondary-count", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {"partition": partition, "merge": merge, "status": status}[arguments.action](arguments)
