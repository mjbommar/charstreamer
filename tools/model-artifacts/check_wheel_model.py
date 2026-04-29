#!/usr/bin/env python3
"""Validate that a built CharStreamer wheel contains a model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import PurePosixPath
from typing import Any

MODEL_FORMAT = "charstreamer.model-bundle.v1"
MODEL_NAME = "charstreamer-default"
MODEL_PREFIX = "charstreamer/models/default/"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", help="Wheel file to validate")
    parser.add_argument(
        "--require-burn",
        action="store_true",
        help="Require a Burn-backed model engine",
    )
    args = parser.parse_args()

    with zipfile.ZipFile(args.wheel) as wheel:
        manifest = read_manifest(wheel)
        validate_manifest(wheel, manifest, require_burn=args.require_burn)

    print(
        "validated wheel model:",
        manifest["name"],
        manifest.get("version", "unknown-version"),
        manifest["engine"],
    )
    return 0


def read_manifest(wheel: zipfile.ZipFile) -> dict[str, Any]:
    manifest_path = f"{MODEL_PREFIX}manifest.json"
    try:
        return json.loads(wheel.read(manifest_path).decode("utf-8"))
    except KeyError as error:
        raise SystemExit(f"wheel is missing {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"wheel model manifest is invalid JSON: {error}") from error


def validate_manifest(
    wheel: zipfile.ZipFile,
    manifest: dict[str, Any],
    *,
    require_burn: bool,
) -> None:
    if manifest.get("format") != MODEL_FORMAT:
        raise SystemExit(f"manifest format must be {MODEL_FORMAT!r}")
    if manifest.get("name") != MODEL_NAME:
        raise SystemExit(f"manifest name must be {MODEL_NAME!r}")

    engine = manifest.get("engine")
    if not isinstance(engine, str) or not engine:
        raise SystemExit("manifest engine must be a non-empty string")
    if require_burn and not engine.startswith("burn"):
        raise SystemExit(f"manifest engine must be Burn-backed, got {engine!r}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SystemExit("manifest files must be a non-empty list")

    names = set(wheel.namelist())
    for file_info in files:
        if not isinstance(file_info, dict):
            raise SystemExit("manifest file entries must be objects")
        rel_path = file_info.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise SystemExit("manifest file entry missing path")
        path = PurePosixPath(rel_path)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe model file path: {rel_path}")

        wheel_path = MODEL_PREFIX + path.as_posix()
        if wheel_path not in names:
            raise SystemExit(f"wheel is missing model file: {wheel_path}")

        data = wheel.read(wheel_path)
        expected_bytes = file_info.get("bytes")
        if expected_bytes is not None and len(data) != expected_bytes:
            raise SystemExit(f"model file byte count mismatch: {rel_path}")

        expected_sha = file_info.get("sha256")
        if expected_sha is not None and hashlib.sha256(data).hexdigest() != expected_sha:
            raise SystemExit(f"model file sha256 mismatch: {rel_path}")


if __name__ == "__main__":
    raise SystemExit(main())
