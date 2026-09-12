#!/usr/bin/env bash
set -euo pipefail

# Derive the runnable 500-row secondary-only dataset from the audited dual
# manifest.  Keeping the original rows unchanged preserves their prepared
# template aliases and sandbox environment metadata.
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk}"
SOURCE="${SOURCE:-${SHARED_ROOT}/swe-rl/data/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl}"
OUTPUT="${OUTPUT:-${SHARED_ROOT}/swe-rl/data/swe_dev_500_secondary_project_avatrain_qwen_code_0.21.0.jsonl}"

SOURCE="${SOURCE}" OUTPUT="${OUTPUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE"])
output = Path(os.environ["OUTPUT"])
rows = []
labels = set()
with source.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        if metadata.get("sandbox_project") != "secondary":
            continue
        label = row.get("label")
        template = metadata.get("inspire_template")
        if not isinstance(label, str) or not label:
            raise SystemExit(f"invalid label at source line {line_number}")
        if label in labels:
            raise SystemExit(f"duplicate secondary label: {label}")
        if not isinstance(template, str) or not template:
            raise SystemExit(f"secondary row has no template at source line {line_number}")
        labels.add(label)
        rows.append(row)

if len(rows) != 500:
    raise SystemExit(f"expected exactly 500 secondary rows, found {len(rows)}")

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

print(f"READY secondary_only rows={len(rows)} output={output}")
PY
