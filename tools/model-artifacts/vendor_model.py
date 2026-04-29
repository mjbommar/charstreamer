#!/usr/bin/env python3
"""Validate and vendor a CharStreamer model bundle for wheel builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

MODEL_FORMAT = "charstreamer.model-bundle.v1"
MODEL_NAME = "charstreamer-default"
DEFAULT_DEST = Path("crates/charstreamer-python/python/charstreamer/models/default")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Model bundle directory or .zip archive")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination model directory, default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--archive-out",
        type=Path,
        help="Optional normalized .zip archive path to create after validation",
    )
    parser.add_argument(
        "--require-burn",
        action="store_true",
        help="Require a Burn engine in the manifest",
    )
    args = parser.parse_args()

    root, temp_dir = prepare_input(args.bundle)
    try:
        manifest = load_manifest(root)
        validate_manifest(root, manifest, require_burn=args.require_burn)
        vendor(root, args.dest)
        normalize_manifest(args.dest)
        if args.archive_out:
            write_archive(args.dest, args.archive_out)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    return 0


def prepare_input(bundle: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    bundle = bundle.expanduser().resolve()
    if bundle.is_dir():
        return bundle, None
    if not bundle.is_file():
        raise SystemExit(f"model bundle does not exist: {bundle}")
    if bundle.suffix != ".zip":
        raise SystemExit(f"model bundle must be a directory or .zip archive: {bundle}")

    temp_dir = tempfile.TemporaryDirectory(prefix="charstreamer-model-")
    root = Path(temp_dir.name)
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"unsafe archive member path: {member.filename}")
            archive.extract(member, root)

    if (root / "manifest.json").is_file():
        return root, temp_dir

    children = [path for path in root.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / "manifest.json").is_file():
        return children[0], temp_dir

    raise SystemExit("model archive must contain manifest.json at root or one top-level directory")


def load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest.json in {root}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid manifest JSON: {error}") from error


def validate_manifest(root: Path, manifest: dict[str, Any], *, require_burn: bool) -> None:
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

    for file_info in files:
        if not isinstance(file_info, dict):
            raise SystemExit("manifest file entries must be objects")
        rel_path = file_info.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise SystemExit("manifest file entry missing path")
        path = Path(rel_path)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe manifest file path: {rel_path}")
        file_path = root / path
        if not file_path.is_file():
            raise SystemExit(f"manifest file missing: {rel_path}")

        expected_bytes = file_info.get("bytes")
        if expected_bytes is not None and file_path.stat().st_size != expected_bytes:
            raise SystemExit(f"manifest file byte count mismatch: {rel_path}")

        expected_sha = file_info.get("sha256")
        if expected_sha is not None and sha256(file_path) != expected_sha:
            raise SystemExit(f"manifest file sha256 mismatch: {rel_path}")


def vendor(root: Path, dest: Path) -> None:
    dest = dest.resolve()
    if dest.exists():
        for child in dest.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        dest.mkdir(parents=True)

    for child in root.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def normalize_manifest(root: Path) -> None:
    manifest = load_manifest(root)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SystemExit("manifest files must be a list")

    for file_info in files:
        if not isinstance(file_info, dict):
            raise SystemExit("manifest file entries must be objects")
        rel_path = file_info.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise SystemExit("manifest file entry missing path")
        path = Path(rel_path)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe manifest file path: {rel_path}")
        file_path = root / path
        if not file_path.is_file():
            raise SystemExit(f"manifest file missing after vendoring: {rel_path}")
        file_info["bytes"] = file_path.stat().st_size
        file_info["sha256"] = sha256(file_path)

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_archive(root: Path, archive_out: Path) -> None:
    archive_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                archive.write(path, path.relative_to(root).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
