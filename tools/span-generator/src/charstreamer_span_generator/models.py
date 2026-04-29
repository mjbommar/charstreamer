from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DEFAULT_LABEL_DEFINITIONS: dict[str, str] = {
    "section_heading": (
        "A heading, title, caption, or section label that introduces a following "
        "block of content."
    ),
    "paragraph": (
        "A contiguous prose block or block-level text region that should be "
        "treated as one paragraph."
    ),
    "sentence": (
        "A sentence-level unit. In streaming windows this can be a complete "
        "sentence or the visible portion of a sentence that started before "
        "or continues after the current target."
    ),
    "dialogue": (
        "A direct-speech or speaker-turn region, typically quoted speech or a "
        "transcript-style speaker block."
    ),
    "list_item": (
        "A bullet, numbered clause, or enumerated list item, including its full "
        "content."
    ),
    "metadata": (
        "Front matter, boilerplate, headers, page furniture, signature blocks, "
        "or other non-body metadata-like text."
    ),
}


class TaggedAnnotation(BaseModel):
    tagged_text: str
    notes: list[str] = Field(default_factory=list)


class PerLabelTaggedAnnotation(BaseModel):
    sentence: str
    paragraph: str
    section: str
    dialogue: str
    list_item: str
    metadata: str
    notes: list[str]


class TextUnit(BaseModel):
    unit_id: int
    kind: Literal["line", "block"]
    text: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int


class UnitLabelAssignment(BaseModel):
    unit_id: int
    label: str


class UnitLabelAnnotation(BaseModel):
    units: list[UnitLabelAssignment]
    notes: list[str] = Field(default_factory=list)


class SentenceBoundaryCandidate(BaseModel):
    candidate_id: int
    parent_span_id: int
    char_end: int
    byte_end: int
    marker: str


class SentenceBoundaryAnnotation(BaseModel):
    break_after_candidate_ids: list[int]
    notes: list[str] = Field(default_factory=list)


class SpanAnnotation(BaseModel):
    id: int
    label: str
    start: int
    end: int
    char_start: int
    char_end: int
    parent_id: int | None = None
    left_open: bool = False
    right_open: bool = False


class ValidationReport(BaseModel):
    protocol: Literal[
        "inline_tags",
        "per_label_inline_tags",
        "unit_labels",
        "unit_labels_sentence_candidates",
    ]
    exact_roundtrip: bool | None = None
    stripped_text_matches: bool | None = None
    well_nested: bool | None = None
    allowed_labels_only: bool
    unit_coverage_complete: bool | None = None
    unit_order_preserved: bool | None = None
    sentence_candidate_validation: bool | None = None
    span_count: int


class AnnotationMetadata(BaseModel):
    provider: Literal["openai", "mock"]
    model: str
    attempt: int
    response_id: str | None = None
    prompt_cache_key: str | None = None
    notes: list[str] = Field(default_factory=list)


class ChunkAnnotationRecord(BaseModel):
    protocol: Literal["inline_tags", "unit_labels", "unit_labels_sentence_candidates"]
    dataset: str
    split: str
    source_identifier: str
    mime_type: str
    doc_index: int
    chunk_index: int
    requested_labels: list[str]
    doc_char_start: int
    doc_char_end: int
    doc_byte_start: int
    doc_byte_end: int
    text: str
    tagged_text: str = ""
    units: list[TextUnit] = Field(default_factory=list)
    unit_annotations: list[UnitLabelAssignment] = Field(default_factory=list)
    sentence_candidates: list[SentenceBoundaryCandidate] = Field(default_factory=list)
    spans: list[SpanAnnotation]
    validation: ValidationReport
    annotation: AnnotationMetadata
