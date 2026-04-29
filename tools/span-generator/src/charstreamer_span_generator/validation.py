from __future__ import annotations

import re
from dataclasses import dataclass

from .models import (
    SentenceBoundaryCandidate,
    SpanAnnotation,
    TextUnit,
    UnitLabelAssignment,
    ValidationReport,
)


TAG_PATTERN = re.compile(r"<\|(/?)([a-z_]+)\|>")


class TaggedTextValidationError(ValueError):
    pass


class UnitAnnotationValidationError(ValueError):
    pass


class SentenceBoundaryValidationError(ValueError):
    pass


@dataclass(slots=True)
class _OpenSpan:
    label: str
    char_start: int
    byte_start: int
    parent_id: int | None
    span_id: int


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def strip_tags(tagged_text: str) -> str:
    return TAG_PATTERN.sub("", tagged_text)


def parse_tagged_text(
    tagged_text: str,
    original_text: str,
    allowed_labels: set[str],
) -> tuple[list[SpanAnnotation], ValidationReport]:
    char_to_byte = _char_to_byte_offsets(original_text)
    original_char_index = 0
    last_end = 0
    stack: list[_OpenSpan] = []
    spans: list[SpanAnnotation] = []
    next_span_id = 0

    for match in TAG_PATTERN.finditer(tagged_text):
        plain = tagged_text[last_end : match.start()]
        if plain:
            next_char_index = original_char_index + len(plain)
            if original_text[original_char_index:next_char_index] != plain:
                raise TaggedTextValidationError(
                    "tagged_text does not round-trip to the original text"
                )
            original_char_index = next_char_index

        closing, label = match.groups()
        if label not in allowed_labels:
            raise TaggedTextValidationError(f"disallowed label tag: {label}")

        if closing:
            if not stack:
                raise TaggedTextValidationError(f"closing unopened tag: {label}")
            open_span = stack.pop()
            if open_span.label != label:
                raise TaggedTextValidationError(
                    f"crossing or mismatched tag close: expected {open_span.label}, got {label}"
                )
            spans.append(
                SpanAnnotation(
                    id=open_span.span_id,
                    label=label,
                    start=open_span.byte_start,
                    end=char_to_byte[original_char_index],
                    char_start=open_span.char_start,
                    char_end=original_char_index,
                    parent_id=open_span.parent_id,
                )
            )
        else:
            stack.append(
                _OpenSpan(
                    label=label,
                    char_start=original_char_index,
                    byte_start=char_to_byte[original_char_index],
                    parent_id=stack[-1].span_id if stack else None,
                    span_id=next_span_id,
                )
            )
            next_span_id += 1

        last_end = match.end()

    tail = tagged_text[last_end:]
    if tail:
        next_char_index = original_char_index + len(tail)
        if original_text[original_char_index:next_char_index] != tail:
            raise TaggedTextValidationError(
                "tagged_text tail does not round-trip to the original text"
            )
        original_char_index = next_char_index

    if original_char_index != len(original_text):
        raise TaggedTextValidationError("tagged_text did not cover the full original text")
    if stack:
        raise TaggedTextValidationError("unclosed tags remain in tagged_text")

    spans.sort(key=lambda span: (span.start, span.end, span.label, span.id))
    return spans, ValidationReport(
        protocol="inline_tags",
        exact_roundtrip=True,
        stripped_text_matches=strip_tags(tagged_text) == original_text,
        well_nested=True,
        allowed_labels_only=True,
        span_count=len(spans),
    )


def render_tagged_text(text: str, spans: list[SpanAnnotation]) -> str:
    for span in spans:
        if span.char_start >= span.char_end:
            raise TaggedTextValidationError("cannot render empty spans")
    ordered = sorted(spans, key=lambda span: (span.char_start, -span.char_end, span.label))
    for left, right in zip(ordered, ordered[1:]):
        crossing = (
            left.char_start < right.char_start < left.char_end < right.char_end
        )
        if crossing:
            raise TaggedTextValidationError("cannot render crossing spans")

    opens: dict[int, list[SpanAnnotation]] = {}
    closes: dict[int, list[SpanAnnotation]] = {}
    for span in ordered:
        opens.setdefault(span.char_start, []).append(span)
        closes.setdefault(span.char_end, []).append(span)

    for start_list in opens.values():
        start_list.sort(key=lambda span: (-span.char_end, span.label))
    for close_list in closes.values():
        close_list.sort(key=lambda span: (span.char_start, span.label), reverse=True)

    parts: list[str] = []
    for char_index in range(len(text) + 1):
        for span in closes.get(char_index, []):
            parts.append(f"<|/{span.label}|>")
        for span in opens.get(char_index, []):
            parts.append(f"<|{span.label}|>")
        if char_index < len(text):
            parts.append(text[char_index])
    return "".join(parts)


def build_line_units(text: str) -> list[TextUnit]:
    if not text:
        return []

    char_to_byte = _char_to_byte_offsets(text)
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [text]

    units: list[TextUnit] = []
    char_start = 0
    for unit_id, line in enumerate(lines):
        char_end = char_start + len(line)
        units.append(
            TextUnit(
                unit_id=unit_id,
                kind="line",
                text=line,
                char_start=char_start,
                char_end=char_end,
                byte_start=char_to_byte[char_start],
                byte_end=char_to_byte[char_end],
            )
        )
        char_start = char_end
    return units


def validate_unit_annotations(
    unit_annotations: list[UnitLabelAssignment],
    units: list[TextUnit],
    allowed_labels: set[str],
    *,
    none_label: str = "none",
) -> tuple[list[SpanAnnotation], ValidationReport]:
    expected_ids = [unit.unit_id for unit in units]
    expected_id_set = set(expected_ids)
    seen: dict[int, UnitLabelAssignment] = {}
    order: list[int] = []

    for assignment in unit_annotations:
        if assignment.unit_id not in expected_id_set:
            raise UnitAnnotationValidationError(f"unknown unit_id: {assignment.unit_id}")
        if assignment.unit_id in seen:
            raise UnitAnnotationValidationError(f"duplicate unit_id: {assignment.unit_id}")
        if assignment.label != none_label and assignment.label not in allowed_labels:
            raise UnitAnnotationValidationError(
                f"disallowed label for unit {assignment.unit_id}: {assignment.label}"
            )
        seen[assignment.unit_id] = assignment
        order.append(assignment.unit_id)

    missing = [unit_id for unit_id in expected_ids if unit_id not in seen]
    if missing:
        preview = ", ".join(str(unit_id) for unit_id in missing[:8])
        raise UnitAnnotationValidationError(f"missing unit labels for ids: {preview}")

    spans: list[SpanAnnotation] = []
    current_label: str | None = None
    current_units: list[TextUnit] = []
    next_span_id = 0

    def flush_current() -> None:
        nonlocal current_label, current_units, next_span_id
        if current_label is None or not current_units:
            current_label = None
            current_units = []
            return
        start_unit = current_units[0]
        end_unit = current_units[-1]
        spans.append(
            SpanAnnotation(
                id=next_span_id,
                label=current_label,
                start=start_unit.byte_start,
                end=end_unit.byte_end,
                char_start=start_unit.char_start,
                char_end=end_unit.char_end,
                parent_id=None,
            )
        )
        next_span_id += 1
        current_label = None
        current_units = []

    ordered_units = sorted(units, key=lambda unit: unit.unit_id)
    for unit in ordered_units:
        label = seen[unit.unit_id].label
        normalized = None if label == none_label else label
        if normalized != current_label:
            flush_current()
            current_label = normalized
        if normalized is not None:
            current_units.append(unit)
    flush_current()

    return spans, ValidationReport(
        protocol="unit_labels",
        allowed_labels_only=True,
        unit_coverage_complete=True,
        unit_order_preserved=order == expected_ids,
        span_count=len(spans),
    )


def build_sentence_candidates(
    text: str,
    *,
    parent_span_id: int,
    char_start: int,
    byte_start: int,
) -> list[SentenceBoundaryCandidate]:
    char_to_byte = _char_to_byte_offsets(text)
    candidates: list[SentenceBoundaryCandidate] = []
    index = 0
    next_candidate_id = 0
    while index < len(text):
        char = text[index]
        if char in ".!?":
            candidate_end = index + 1
            while candidate_end < len(text) and text[candidate_end] in "\"'”’)]}":
                candidate_end += 1
            if candidate_end < len(text) and not text[candidate_end].isspace():
                index = candidate_end
                continue
            if candidate_end >= len(text):
                index = candidate_end
                continue
            if char == "." and _looks_like_abbreviation(text, index):
                index = candidate_end
                continue
            left = text[max(0, index - 36) : index + 1]
            right = text[candidate_end : min(len(text), candidate_end + 36)]
            marker = f"{left}<<<BREAK>>>{right}"
            candidates.append(
                SentenceBoundaryCandidate(
                    candidate_id=next_candidate_id,
                    parent_span_id=parent_span_id,
                    char_end=char_start + candidate_end,
                    byte_end=byte_start + char_to_byte[candidate_end],
                    marker=marker,
                )
            )
            next_candidate_id += 1
            index = candidate_end
            continue
        index += 1
    return candidates


def _looks_like_abbreviation(text: str, period_index: int) -> bool:
    prefix = text[: period_index + 1]
    if re.search(r"\b(?:[A-Z]\.){2,}$", prefix):
        return True
    if re.search(r"\b[0-9]+\.$", prefix):
        return True
    if re.search(
        r"\b(?:No|Nos|Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Co|Corp|Dept|Sec|Art|Fig|al|e\.g|i\.e|vs|v|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    return False


def validate_sentence_breaks(
    *,
    parent_span: SpanAnnotation,
    text: str,
    candidate_break_ids: list[int],
    candidates: list[SentenceBoundaryCandidate],
    next_span_id: int,
) -> tuple[list[SpanAnnotation], int]:
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    seen_ids: set[int] = set()
    ordered_breaks: list[SentenceBoundaryCandidate] = []
    for candidate_id in candidate_break_ids:
        if candidate_id not in candidate_map:
            raise SentenceBoundaryValidationError(f"unknown sentence candidate_id: {candidate_id}")
        if candidate_id in seen_ids:
            raise SentenceBoundaryValidationError(
                f"duplicate sentence candidate_id: {candidate_id}"
            )
        seen_ids.add(candidate_id)
        ordered_breaks.append(candidate_map[candidate_id])

    if ordered_breaks != sorted(ordered_breaks, key=lambda candidate: candidate.char_end):
        raise SentenceBoundaryValidationError("sentence candidate ids are not in increasing order")

    boundary_char_positions = [candidate.char_end for candidate in ordered_breaks]
    boundary_byte_positions = [candidate.byte_end for candidate in ordered_breaks]
    boundary_char_positions.append(parent_span.char_end)
    boundary_byte_positions.append(parent_span.end)

    spans: list[SpanAnnotation] = []
    current_char_start = parent_span.char_start
    current_byte_start = parent_span.start

    for char_end, byte_end in zip(boundary_char_positions, boundary_byte_positions):
        trimmed_char_start, trimmed_byte_start, trimmed_char_end, trimmed_byte_end = (
            trim_span_to_content(
                text,
                absolute_char_start=current_char_start,
                absolute_char_end=char_end,
                absolute_byte_start=current_byte_start,
                absolute_byte_end=byte_end,
            )
        )
        if trimmed_char_start < trimmed_char_end:
            spans.append(
                SpanAnnotation(
                    id=next_span_id,
                    label="sentence",
                    start=trimmed_byte_start,
                    end=trimmed_byte_end,
                    char_start=trimmed_char_start,
                    char_end=trimmed_char_end,
                    parent_id=parent_span.id,
                )
            )
            next_span_id += 1
        current_char_start = char_end
        current_byte_start = byte_end

    return spans, next_span_id


def trim_span_to_content(
    text: str,
    *,
    absolute_char_start: int,
    absolute_char_end: int,
    absolute_byte_start: int,
    absolute_byte_end: int,
) -> tuple[int, int, int, int]:
    segment = text[absolute_char_start:absolute_char_end]
    leading = len(segment) - len(segment.lstrip())
    trailing = len(segment.rstrip())
    trimmed = segment.strip()
    if not trimmed:
        return absolute_char_start, absolute_byte_start, absolute_char_start, absolute_byte_start

    trimmed_char_start = absolute_char_start + leading
    trimmed_char_end = absolute_char_start + trailing

    prefix = segment[:leading]
    trimmed_segment = segment[leading:trailing]
    trimmed_byte_start = absolute_byte_start + len(prefix.encode("utf-8"))
    trimmed_byte_end = trimmed_byte_start + len(trimmed_segment.encode("utf-8"))
    if trimmed_byte_end > absolute_byte_end:
        raise SentenceBoundaryValidationError("trimmed sentence span exceeded parent byte range")
    return trimmed_char_start, trimmed_byte_start, trimmed_char_end, trimmed_byte_end
