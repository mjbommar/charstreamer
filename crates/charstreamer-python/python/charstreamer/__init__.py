"""Python interface for CharStreamer.

The Python layer is intentionally thin: the hot path lives in the Rust/PyO3
extension, while this wrapper resolves model artifacts and exposes a stable
``charstreamer.Segmenter`` entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from . import _native
from .types import (
    Annotation,
    AnnotationDict,
    BenchmarkResult,
    BenchmarkResultDict,
    ModelInfo,
    ModelInfoDict,
    Runtime,
    Span,
    SpanDict,
)

SegmenterConfig = _native.SegmenterConfig

__version__ = _native.__version__

_MODEL_FORMAT = "charstreamer.model-bundle.v1"
_MODEL_NAME = "charstreamer-default"
_SUPPORTED_MODEL_ENGINES = {"burn_shallow_mlp_sentence_v1"}
_SUPPORTED_STRUCTURE_ENGINES = {"burn_multilabel_mlp_structure_v1"}
_MODEL_ARCHIVE = f"{_MODEL_NAME}-{__version__}.zip"
_GITHUB_RELEASE_BASE = "https://github.com/mjbommar/charstreamer/releases/download"
_DEFAULT_MODEL_URL = f"{_GITHUB_RELEASE_BASE}/v{__version__}/{_MODEL_ARCHIVE}"


@dataclass(frozen=True)
class ModelResolution:
    """Resolved default model metadata."""

    resolved: bool
    source: str
    path: str | None
    manifest: dict[str, Any] | None
    error: str | None = None

    def to_model_info(self) -> ModelInfo:
        return ModelInfo(
            resolved=self.resolved,
            source=self.source,
            path=self.path,
            manifest=self.manifest,
            error=self.error,
            runtime=_runtime_name(self),
            model_inference=_model_runtime_available(self),
        )

    def to_dict(self) -> ModelInfoDict:
        return self.to_model_info().to_dict()


def render(text: str, spans: Iterable[Any]) -> str:
    """Render character-offset spans into tagged text.

    ``spans`` may contain ``Span`` objects, previous-style span dictionaries, or
    ``(label, start, end[, score])`` tuples using character offsets.
    """

    return _native.render(
        text,
        [_coerce_render_span(span, byte_offsets=False) for span in spans],
    )


def render_bytes(text: str, spans: Iterable[Any]) -> str:
    """Render byte-offset spans into tagged text.

    ``spans`` may contain ``Span`` objects, previous-style span dictionaries, or
    ``(label, start_byte, end_byte[, score])`` tuples using UTF-8 byte offsets.
    """

    return _native.render_bytes(
        text,
        [_coerce_render_span(span, byte_offsets=True) for span in spans],
    )


def _coerce_render_span(span: Any, *, byte_offsets: bool) -> tuple[str, int, int, float | None]:
    if isinstance(span, Span):
        span_label, span_start, span_end, span_score = (
            span.as_byte_tuple() if byte_offsets else span.as_char_tuple()
        )
        return (span_label, span_start, span_end, span_score)

    if isinstance(span, Mapping):
        label = str(span["label"])
        if byte_offsets:
            start = int(span.get("start_byte", span["start"]))
            end = int(span.get("end_byte", span["end"]))
        else:
            start = int(span["start"])
            end = int(span["end"])
        score_value = span.get("score")
        return (
            label,
            start,
            end,
            None if score_value is None else float(score_value),
        )

    values = tuple(span)
    render_score: Any
    if len(values) == 3:
        label, start, end = values
        render_score = None
    elif len(values) == 4:
        label, start, end, render_score = values
    else:
        raise ValueError("render spans must be Span objects, dicts, or 3/4-tuples")

    return (
        str(label),
        int(start),
        int(end),
        None if render_score is None else float(render_score),
    )


class Segmenter:
    """High-level segmenter facade.

    ``default()`` resolves the configured model artifact before constructing the
    Rust segmenter. A supported Burn bundle is required; the package does not
    synthesize annotations from rule code when no model is available.
    """

    def __init__(
        self,
        config: SegmenterConfig | None = None,
        *,
        model: ModelResolution | None = None,
        require_model: bool = True,
    ) -> None:
        self._model = model or _resolve_default_model(allow_download=False)
        if _model_runtime_available(self._model):
            if self._model.path is None:
                raise RuntimeError("resolved model is missing a local path")
            self._inner = _native.Segmenter.from_model_dir(self._model.path, config)
            return
        raise RuntimeError(_missing_model_runtime_message(self._model))

    @classmethod
    def default(
        cls,
        config: SegmenterConfig | None = None,
        *,
        allow_download: bool | None = None,
        require_model: bool = True,
    ) -> "Segmenter":
        model = _resolve_default_model(allow_download=allow_download)
        return cls(config, model=model, require_model=require_model)

    def model_info(self) -> ModelInfo:
        return self._model.to_model_info()

    def model_info_dict(self) -> ModelInfoDict:
        return self.model_info().to_dict()

    def spans(self, text: str) -> tuple[Span, ...]:
        return tuple(Span.from_dict(span) for span in self._inner.spans(text))

    def spans_dict(self, text: str) -> list[SpanDict]:
        return [span.to_dict() for span in self.spans(text)]

    def annotate(self, text: str) -> Annotation:
        return Annotation.from_native(self._inner.annotate(text), self.model_info())

    def annotate_dict(self, text: str) -> AnnotationDict:
        return self.annotate(text).to_dict()

    def tagged(self, text: str) -> str:
        return self._inner.tagged(text)

    def benchmark(self, text: str, iterations: int = 10) -> BenchmarkResult:
        return BenchmarkResult.from_native(
            self._inner.benchmark(text, iterations),
            self.model_info(),
        )

    def benchmark_dict(self, text: str, iterations: int = 10) -> BenchmarkResultDict:
        return self.benchmark(text, iterations).to_dict()


def annotate(text: str) -> Annotation:
    return Segmenter.default().annotate(text)


def annotate_dict(text: str) -> AnnotationDict:
    return Segmenter.default().annotate_dict(text)


def spans(text: str) -> tuple[Span, ...]:
    return Segmenter.default().spans(text)


def spans_dict(text: str) -> list[SpanDict]:
    return Segmenter.default().spans_dict(text)


def tagged(text: str) -> str:
    return Segmenter.default().tagged(text)


def model_info(
    *,
    allow_download: bool | None = None,
    require_model: bool = False,
) -> ModelInfo:
    resolution = _resolve_default_model(allow_download=allow_download)
    if require_model and not _model_runtime_available(resolution):
        raise RuntimeError(_missing_model_runtime_message(resolution))
    return resolution.to_model_info()


def model_info_dict(
    *,
    allow_download: bool | None = None,
    require_model: bool = False,
) -> ModelInfoDict:
    return model_info(allow_download=allow_download, require_model=require_model).to_dict()


def _model_runtime_available(resolution: ModelResolution) -> bool:
    if not resolution.resolved or not resolution.manifest:
        return False
    if resolution.manifest.get("engine") not in _SUPPORTED_MODEL_ENGINES:
        return False
    structure = resolution.manifest.get("structure")
    if structure is None:
        return True
    return structure.get("engine") in _SUPPORTED_STRUCTURE_ENGINES


def _runtime_name(resolution: ModelResolution) -> Runtime:
    if _model_runtime_available(resolution):
        if resolution.manifest and resolution.manifest.get("structure"):
            return "burn_combined_segmentation"
        return "burn_sentence_boundary"
    return "unavailable"


def _missing_model_runtime_message(resolution: ModelResolution) -> str:
    if not resolution.resolved:
        return (
            "no CharStreamer model artifact was resolved; set "
            "CHARSTREAMER_MODEL_PATH, vendor a model into the wheel, or set "
            "CHARSTREAMER_AUTO_DOWNLOAD=1 / CHARSTREAMER_MODEL_URL"
        )
    engine = resolution.manifest.get("engine") if resolution.manifest else None
    return (
        f"resolved model artifact uses engine {engine!r}, but this build does "
        "not expose model-backed inference for that engine"
    )


def _resolve_default_model(*, allow_download: bool | None) -> ModelResolution:
    local_path = os.environ.get("CHARSTREAMER_MODEL_PATH")
    if local_path:
        resolved = _load_model_dir(Path(local_path), source="env")
        if resolved.resolved:
            return resolved

    bundled = _bundled_model_dir()
    if bundled is not None:
        resolved = _load_model_dir(bundled, source="bundled")
        if resolved.resolved:
            return resolved

    cache_dir = _cache_model_dir()
    resolved = _load_model_dir(cache_dir, source="cache")
    if resolved.resolved:
        return resolved

    if _should_download(allow_download):
        downloaded = _download_and_extract_default_model(cache_dir)
        if downloaded.resolved:
            return downloaded
        return downloaded

    return ModelResolution(
        resolved=False,
        source="unresolved",
        path=None,
        manifest=None,
        error="no model artifact resolved",
    )


def _should_download(allow_download: bool | None) -> bool:
    if allow_download is not None:
        return allow_download
    value = os.environ.get("CHARSTREAMER_AUTO_DOWNLOAD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _bundled_model_dir() -> Path | None:
    try:
        model_dir = resources.files(__package__).joinpath("models").joinpath("default")
    except (AttributeError, FileNotFoundError):
        return None
    if model_dir.joinpath("manifest.json").is_file():
        return Path(str(model_dir))
    return None


def _cache_model_dir() -> Path:
    cache_root = os.environ.get("CHARSTREAMER_MODEL_CACHE")
    if cache_root:
        return Path(cache_root).expanduser().resolve() / "default"
    return Path.home() / ".cache" / "charstreamer" / "models" / "default"


def _load_model_dir(path: Path, *, source: str) -> ModelResolution:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return ModelResolution(False, source, str(path), None, "missing manifest.json")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(path, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return ModelResolution(False, source, str(path), None, str(error))

    return ModelResolution(True, source, str(path), manifest)


def _validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("format") != _MODEL_FORMAT:
        raise ValueError(
            f"invalid model format {manifest.get('format')!r}; expected {_MODEL_FORMAT!r}"
        )
    if manifest.get("name") != _MODEL_NAME:
        raise ValueError(
            f"invalid model name {manifest.get('name')!r}; expected {_MODEL_NAME!r}"
        )
    if not manifest.get("engine"):
        raise ValueError("model manifest must include an engine")

    for file_info in manifest.get("files", []):
        rel_path = file_info.get("path")
        if not rel_path or Path(rel_path).is_absolute() or ".." in Path(rel_path).parts:
            raise ValueError(f"invalid model file path {rel_path!r}")
        file_path = path / rel_path
        if not file_path.is_file():
            raise ValueError(f"model file is missing: {rel_path}")
        expected_size = file_info.get("bytes")
        if expected_size is not None and file_path.stat().st_size != expected_size:
            raise ValueError(f"model file has wrong size: {rel_path}")
        expected_sha = file_info.get("sha256")
        if expected_sha is not None and _sha256(file_path) != expected_sha:
            raise ValueError(f"model file has wrong sha256: {rel_path}")


def _download_and_extract_default_model(cache_dir: Path) -> ModelResolution:
    url = os.environ.get("CHARSTREAMER_MODEL_URL", _DEFAULT_MODEL_URL)
    timeout = float(os.environ.get("CHARSTREAMER_MODEL_TIMEOUT", "10"))
    download_dir = cache_dir.parent / "_downloads"
    archive_path = download_dir / Path(url).name

    try:
        download_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with archive_path.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        _extract_model_archive(archive_path, cache_dir)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, ValueError) as error:
        return ModelResolution(False, "download", str(cache_dir), None, str(error))

    return _load_model_dir(cache_dir, source="download")


def _extract_model_archive(archive_path: Path, cache_dir: Path) -> None:
    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe model archive path: {member.filename}")
            archive.extract(member, tmp_dir)

    root = _archive_root(tmp_dir)
    _validate_manifest(root, json.loads((root / "manifest.json").read_text(encoding="utf-8")))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), cache_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def _archive_root(tmp_dir: Path) -> Path:
    if (tmp_dir / "manifest.json").is_file():
        return tmp_dir
    children = [path for path in tmp_dir.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / "manifest.json").is_file():
        return children[0]
    raise ValueError("model archive must contain manifest.json at root or one top-level directory")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "Annotation",
    "AnnotationDict",
    "BenchmarkResult",
    "BenchmarkResultDict",
    "ModelInfo",
    "ModelInfoDict",
    "Segmenter",
    "SegmenterConfig",
    "Span",
    "SpanDict",
    "annotate",
    "annotate_dict",
    "model_info",
    "model_info_dict",
    "render",
    "render_bytes",
    "spans",
    "spans_dict",
    "tagged",
    "__version__",
]
