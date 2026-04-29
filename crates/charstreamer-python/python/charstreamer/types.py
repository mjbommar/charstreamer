"""Typed public result models for CharStreamer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

Label = Literal["sentence", "paragraph", "metadata", "section", "list_item", "dialogue"]
Runtime = Literal["burn_combined_segmentation", "burn_sentence_boundary", "unavailable"]

_LABELS = frozenset({"sentence", "paragraph", "metadata", "section", "list_item", "dialogue"})


class SpanDict(TypedDict):
    label: Label
    start: int
    end: int
    start_byte: int
    end_byte: int
    score: float


class ModelInfoDict(TypedDict):
    resolved: bool
    source: str
    path: str | None
    manifest: dict[str, Any] | None
    error: str | None
    runtime: Runtime
    model_inference: bool


class AnnotationDict(TypedDict):
    tagged: str
    spans: list[SpanDict]
    model: ModelInfoDict


class BenchmarkResultDict(TypedDict):
    iterations: int
    input_bytes: int
    input_chars: int
    processed_bytes: int
    processed_chars: int
    seconds: float
    bytes_per_second: float
    chars_per_second: float
    mib_per_second: float
    span_count: int
    tagged_bytes: int
    model: ModelInfoDict


class _DictCompatible(Mapping[str, Any]):
    """Small mapping shim for compatibility with the previous dict API."""

    def to_dict(self) -> Any:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class Span(_DictCompatible):
    """A scored annotation span.

    Character offsets are exposed as ``start``/``end``. Raw UTF-8 byte offsets
    are exposed separately as ``start_byte``/``end_byte`` for streaming and
    zero-copy integrations.
    """

    label: Label
    start: int
    end: int
    start_byte: int
    end_byte: int
    score: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Span":
        return cls(
            label=_parse_label(raw["label"]),
            start=int(raw["start"]),
            end=int(raw["end"]),
            start_byte=int(raw["start_byte"]),
            end_byte=int(raw["end_byte"]),
            score=float(raw["score"]),
        )

    def to_dict(self) -> SpanDict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "score": self.score,
        }

    def as_char_tuple(self) -> tuple[str, int, int, float]:
        return (self.label, self.start, self.end, self.score)

    def as_byte_tuple(self) -> tuple[str, int, int, float]:
        return (self.label, self.start_byte, self.end_byte, self.score)


@dataclass(frozen=True)
class ModelInfo(_DictCompatible):
    """Resolved model metadata and runtime status."""

    resolved: bool
    source: str
    path: str | None
    manifest: dict[str, Any] | None
    error: str | None
    runtime: Runtime
    model_inference: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelInfo":
        return cls(
            resolved=bool(raw["resolved"]),
            source=str(raw["source"]),
            path=_optional_str(raw.get("path")),
            manifest=_optional_dict(raw.get("manifest")),
            error=_optional_str(raw.get("error")),
            runtime=_parse_runtime(raw["runtime"]),
            model_inference=bool(raw["model_inference"]),
        )

    def to_dict(self) -> ModelInfoDict:
        return {
            "resolved": self.resolved,
            "source": self.source,
            "path": self.path,
            "manifest": self.manifest,
            "error": self.error,
            "runtime": self.runtime,
            "model_inference": self.model_inference,
        }


@dataclass(frozen=True)
class Annotation(_DictCompatible):
    """Rendered annotation plus typed spans and model metadata."""

    tagged: str
    spans: tuple[Span, ...]
    model: ModelInfo

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Annotation":
        return cls(
            tagged=str(raw["tagged"]),
            spans=tuple(Span.from_dict(span) for span in raw["spans"]),
            model=ModelInfo.from_dict(raw["model"]),
        )

    @classmethod
    def from_native(cls, raw: Mapping[str, Any], model: ModelInfo) -> "Annotation":
        return cls(
            tagged=str(raw["tagged"]),
            spans=tuple(Span.from_dict(span) for span in raw["spans"]),
            model=model,
        )

    def to_dict(self) -> AnnotationDict:
        return {
            "tagged": self.tagged,
            "spans": [span.to_dict() for span in self.spans],
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkResult(_DictCompatible):
    """Throughput report returned by ``Segmenter.benchmark``."""

    iterations: int
    input_bytes: int
    input_chars: int
    processed_bytes: int
    processed_chars: int
    seconds: float
    bytes_per_second: float
    chars_per_second: float
    mib_per_second: float
    span_count: int
    tagged_bytes: int
    model: ModelInfo

    @classmethod
    def from_native(cls, raw: Mapping[str, Any], model: ModelInfo) -> "BenchmarkResult":
        return cls(
            iterations=int(raw["iterations"]),
            input_bytes=int(raw["input_bytes"]),
            input_chars=int(raw["input_chars"]),
            processed_bytes=int(raw["processed_bytes"]),
            processed_chars=int(raw["processed_chars"]),
            seconds=float(raw["seconds"]),
            bytes_per_second=float(raw["bytes_per_second"]),
            chars_per_second=float(raw["chars_per_second"]),
            mib_per_second=float(raw["mib_per_second"]),
            span_count=int(raw["span_count"]),
            tagged_bytes=int(raw["tagged_bytes"]),
            model=model,
        )

    def to_dict(self) -> BenchmarkResultDict:
        return {
            "iterations": self.iterations,
            "input_bytes": self.input_bytes,
            "input_chars": self.input_chars,
            "processed_bytes": self.processed_bytes,
            "processed_chars": self.processed_chars,
            "seconds": self.seconds,
            "bytes_per_second": self.bytes_per_second,
            "chars_per_second": self.chars_per_second,
            "mib_per_second": self.mib_per_second,
            "span_count": self.span_count,
            "tagged_bytes": self.tagged_bytes,
            "model": self.model.to_dict(),
        }


def _parse_label(value: Any) -> Label:
    if not isinstance(value, str) or value not in _LABELS:
        raise ValueError(f"unknown CharStreamer label: {value!r}")
    return cast(Label, value)


def _parse_runtime(value: Any) -> Runtime:
    if value in {"burn_combined_segmentation", "burn_sentence_boundary", "unavailable"}:
        return cast(Runtime, value)
    raise ValueError(f"unknown CharStreamer runtime: {value!r}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"expected dict or None, got {type(value).__name__}")
    return value


__all__ = [
    "Annotation",
    "AnnotationDict",
    "BenchmarkResult",
    "BenchmarkResultDict",
    "Label",
    "ModelInfo",
    "ModelInfoDict",
    "Runtime",
    "Span",
    "SpanDict",
]
