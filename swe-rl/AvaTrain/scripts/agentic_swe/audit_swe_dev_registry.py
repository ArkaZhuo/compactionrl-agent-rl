"""Audit public SWE-rebench images and split SWE-Dev data for Inspire builds.

The local SWE-Dev dataset names images in the private ``swebench`` namespace
used by the isolated CPU Docker daemon.  Most corresponding images are also
published under ``swerebench``.  Inspire can pull the public images directly;
the remaining rows need a separate remote-rebuild path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--missing-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public-namespace", default="swerebench")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--proxy", default="http://127.0.0.1:7892")
    parser.add_argument(
        "--check-method",
        choices=("hub-catalog", "hub-api", "docker-manifest"),
        default="hub-catalog",
        help="Use Docker Hub's paginated repository catalog to minimize anonymous API requests.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached per-instance results in an existing manifest.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not make network requests; materialize only cached conclusive results.",
    )
    parser.add_argument(
        "--allow-incomplete-output",
        action="store_true",
        help="Write confirmed public/missing subsets even when transient errors remain.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return dict(metadata.get("remote_env_info") or metadata)


def public_image(image: str, namespace: str) -> str:
    _, separator, remainder = image.partition("/")
    if not separator or not remainder:
        raise ValueError(f"image has no namespace: {image!r}")
    return f"{namespace}/{remainder}"


def inspect_image(
    image: str,
    *,
    timeout: int,
    environment: dict[str, str],
    method: str,
    proxy: str,
) -> tuple[bool | None, str]:
    if method == "hub-api":
        namespace, separator, repository_tag = image.partition("/")
        repository, tag_separator, tag = repository_tag.rpartition(":")
        if not separator or not tag_separator:
            return None, f"unsupported Docker Hub image reference: {image}"
        url = f"https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repository}/tags/{tag}"
        handlers: list[urllib.request.BaseHandler] = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(url, headers={"User-Agent": "AvaTrain-SWE-Dev-audit/1"})
        try:
            with opener.open(request, timeout=timeout) as response:
                response.read(1)
                return response.status == 200, f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False, "HTTP 404"
            return None, f"HTTP {exc.code}: {exc.reason}"
        except (TimeoutError, urllib.error.URLError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image],
            env=environment,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    detail = (result.stderr or "").strip().splitlines()
    return result.returncode == 0, (detail[-1] if detail else "")[-500:]


def fetch_hub_catalog(
    namespace: str, *, timeout: int, proxy: str
) -> tuple[set[str], int]:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    url: str | None = (
        f"https://hub.docker.com/v2/namespaces/{namespace}/repositories?page_size=100&page=1"
    )
    repositories: set[str] = set()
    pages = 0
    while url:
        pages += 1
        if pages > 10_000:
            raise RuntimeError("Docker Hub catalog pagination did not terminate")
        request = urllib.request.Request(url, headers={"User-Agent": "AvaTrain-SWE-Dev-audit/1"})
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Docker Hub catalog HTTP {exc.code}: {exc.reason}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Docker Hub catalog {type(exc).__name__}: {exc}") from exc
        for result in payload.get("results") or []:
            name = result.get("name")
            if isinstance(name, str) and name:
                repositories.add(name)
        url = payload.get("next")
        print(f"CATALOG page={pages} repositories={len(repositories)}", flush=True)
    return repositories, pages


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        temporary = Path(output.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(output.name)
    temporary.replace(path)


def routed_row(row: dict[str, Any], image: str) -> dict[str, Any]:
    routed = json.loads(json.dumps(row))
    metadata = routed.setdefault("metadata", {})
    if "remote_env_info" in metadata:
        metadata["remote_env_info"]["image"] = image
    else:
        metadata["image"] = image
    metadata.setdefault("swe_dev_provenance", {})["inspire_source_image"] = image
    return routed


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.timeout_sec < 1:
        raise SystemExit("workers and timeout-sec must be positive")
    rows = read_jsonl(args.input)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance = source_metadata(row)
        instance_id = instance.get("instance_id")
        image = instance.get("image")
        if not instance_id or not image:
            raise ValueError("input row is missing instance_id or image")
        if instance_id in by_id:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        by_id[instance_id] = row

    previous: dict[str, Any] = {}
    if args.manifest.is_file() and not args.refresh:
        loaded = json.loads(args.manifest.read_text(encoding="utf-8"))
        if (
            loaded.get("public_namespace") == args.public_namespace
            and loaded.get("check_method") == args.check_method
        ):
            previous = loaded.get("images") or {}

    environment = os.environ.copy()
    if args.proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            environment[key] = args.proxy
        for key in ("NO_PROXY", "no_proxy"):
            environment[key] = "127.0.0.1,localhost"

    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, str, str]] = []
    for instance_id, row in by_id.items():
        original = source_metadata(row)["image"]
        candidate = public_image(original, args.public_namespace)
        cached = previous.get(instance_id)
        if cached and cached.get("candidate") == candidate and isinstance(cached.get("available"), bool):
            results[instance_id] = cached
        elif args.cache_only:
            results[instance_id] = cached or {
                "original": original,
                "candidate": candidate,
                "available": None,
                "detail": "not cached",
            }
        else:
            pending.append((instance_id, original, candidate))

    started = time.monotonic()
    if args.check_method == "hub-catalog" and pending:
        try:
            catalog, pages = fetch_hub_catalog(
                args.public_namespace, timeout=args.timeout_sec, proxy=args.proxy
            )
        except RuntimeError as exc:
            for instance_id, original, candidate in pending:
                results[instance_id] = {
                    "original": original,
                    "candidate": candidate,
                    "available": None,
                    "detail": str(exc),
                }
        else:
            for instance_id, original, candidate in pending:
                repository_tag = candidate.partition("/")[2]
                repository = repository_tag.rpartition(":")[0]
                available = repository in catalog
                results[instance_id] = {
                    "original": original,
                    "candidate": candidate,
                    "available": available,
                    "detail": f"catalog pages={pages}",
                }
        pending = []

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="swe-dev-registry") as pool:
        futures = {
            pool.submit(
                inspect_image,
                candidate,
                timeout=args.timeout_sec,
                environment=environment,
                method=args.check_method,
                proxy=args.proxy,
            ): (
                instance_id,
                original,
                candidate,
            )
            for instance_id, original, candidate in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            instance_id, original, candidate = futures[future]
            available, detail = future.result()
            results[instance_id] = {
                "original": original,
                "candidate": candidate,
                "available": available,
                "detail": detail,
            }
            state = "PUBLIC" if available is True else "MISSING" if available is False else "ERROR"
            print(
                f"{state} "
                f"[{completed}/{len(pending)}] {instance_id} {candidate}",
                flush=True,
            )

    public_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for instance_id, row in by_id.items():
        entry = results[instance_id]
        if entry["available"] is True:
            public_rows.append(routed_row(row, entry["candidate"]))
        elif entry["available"] is False:
            missing_rows.append(row)

    errors = [instance_id for instance_id, entry in results.items() if entry["available"] is None]

    manifest = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "public_namespace": args.public_namespace,
        "total": len(rows),
        "public": len(public_rows),
        "missing": len(missing_rows),
        "errors": len(errors),
        "check_method": args.check_method,
        "cache_only": args.cache_only,
        "elapsed_sec": round(time.monotonic() - started, 1),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images": dict(sorted(results.items())),
    }
    atomic_json(args.manifest, manifest)
    if errors:
        if args.allow_incomplete_output:
            atomic_jsonl(args.public_output, public_rows)
            atomic_jsonl(args.missing_output, missing_rows)
            print(
                f"PARTIAL total={len(rows)} public={len(public_rows)} missing={len(missing_rows)} "
                f"errors={len(errors)} manifest={args.manifest}"
            )
            return
        raise SystemExit(
            f"audit incomplete: {len(errors)} transient errors; rerun to retry cached failures; "
            f"manifest={args.manifest}"
        )
    atomic_jsonl(args.public_output, public_rows)
    atomic_jsonl(args.missing_output, missing_rows)
    print(
        f"READY total={len(rows)} public={len(public_rows)} missing={len(missing_rows)} "
        f"manifest={args.manifest}"
    )


if __name__ == "__main__":
    main()
