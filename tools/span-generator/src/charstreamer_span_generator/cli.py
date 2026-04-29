from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import orjson
from datasets import load_dataset
from openai import OpenAI

from .models import (
    AnnotationMetadata,
    ChunkAnnotationRecord,
    DEFAULT_LABEL_DEFINITIONS,
    SentenceBoundaryAnnotation,
    SentenceBoundaryCandidate,
    SpanAnnotation,
    TextUnit,
    UnitLabelAnnotation,
    UnitLabelAssignment,
    ValidationReport,
)
from .validation import (
    SentenceBoundaryValidationError,
    UnitAnnotationValidationError,
    build_sentence_candidates,
    validate_sentence_breaks,
    validate_unit_annotations,
)


OPENAI_SYSTEM_PROMPT = """You are a document segmentation annotator.

You label stable text units. You do NOT rewrite source text. You do NOT emit
byte offsets. You return one primary label per unit.

Hard rules:
- Return a JSON object matching the schema.
- Every unit_id must appear exactly once.
- Use exactly one label per unit.
- Allowed labels are the requested labels plus `none`.
- Use `none` for blank lines, separators, or lines that do not clearly match a
  requested label.
- Do not emit byte offsets or copy the full document back.

Label guidance:
- `metadata`: attorney blocks, captions, page furniture, docket headers,
  signature lines, court names, case numbers, filing dates, or other boilerplate.
- `section_heading`: standalone title, heading, caption, subject line, article
  title, or section label. Use this for lines like `Subject: ...`, `Re: ...`,
  `Относно: ...`, article titles, and similar short display text.
- `paragraph`: body prose or other main-content paragraph block.
- `dialogue`: quoted speech or transcript-style speaker turn.
- `list_item`: bullet, numbered item, clause, or enumerated list entry.
- `none`: empty line, separator, or no requested target label.

Example output:
{"units":[{"unit_id":0,"label":"metadata"},{"unit_id":1,"label":"section_heading"},{"unit_id":2,"label":"paragraph"}],"notes":["short optional note"]}
"""


OPENAI_SENTENCE_SYSTEM_PROMPT = """You are a sentence boundary annotator.

You do NOT rewrite source text. You do NOT emit byte offsets. You only choose
which precomputed candidate boundaries end a sentence.

Hard rules:
- Return a JSON object matching the schema.
- `break_after_candidate_ids` must contain only candidate ids from the prompt.
- Keep ids in increasing order.
- Do not include the final end-of-block boundary; it is added automatically.
- If there are no internal sentence breaks, return an empty list.

Choose a candidate id only when the sentence should end immediately after the
candidate marker.
"""


BLOCK_LABELS = {"section_heading", "paragraph", "dialogue", "list_item", "metadata"}
SENTENCE_PARENT_LABELS = {"paragraph", "dialogue", "list_item"}
NONE_LABEL = "none"


@dataclass(slots=True)
class TextChunk:
    source_identifier: str
    mime_type: str
    doc_index: int
    chunk_index: int
    text: str
    doc_char_start: int
    doc_char_end: int
    doc_byte_start: int
    doc_byte_end: int


@dataclass(slots=True)
class AnnotationResult:
    protocol: str
    spans: list[SpanAnnotation]
    validation: ValidationReport
    metadata: AnnotationMetadata
    tagged_text: str = ""
    units: list[TextUnit] = field(default_factory=list)
    unit_annotations: list[UnitLabelAssignment] = field(default_factory=list)
    sentence_candidates: list[SentenceBoundaryCandidate] = field(default_factory=list)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a Hugging Face dataset, annotate text spans, and emit JSONL."
    )
    parser.add_argument(
        "--dataset",
        default="alea-institute/kl3m-data-sample-005-shuffled",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--provider",
        choices=("openai",),
        default="openai",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-docs", type=_positive_int, default=1)
    parser.add_argument("--limit-chunks", type=_positive_int)
    parser.add_argument("--max-chars-per-chunk", type=_positive_int, default=4000)
    parser.add_argument("--shuffle-buffer-size", type=_positive_int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts", type=_positive_int, default=3)
    parser.add_argument("--max-output-tokens", type=_positive_int, default=16000)
    parser.add_argument("--sentence-max-candidates", type=_positive_int, default=12)
    parser.add_argument("--sentence-min-block-chars", type=_positive_int, default=40)
    parser.add_argument("--sentence-min-alpha-chars", type=_positive_int, default=20)
    parser.add_argument(
        "--sentence-parent-labels",
        nargs="+",
        default=["paragraph", "dialogue", "list_item"],
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["metadata", "section_heading", "paragraph", "sentence", "dialogue", "list_item"],
    )
    parser.add_argument("--service-tier", default="auto")
    parser.add_argument("--prompt-cache-prefix", default="charstreamer-span-generator-v1")
    parser.add_argument("--env-json")
    parser.add_argument("--debug-dir")
    parser.add_argument("--error-output")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--progress-every", type=_positive_int, default=100)
    parser.add_argument(
        "--include-tagged-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    maybe_load_env_json(args.env_json)
    unsupported = sorted(set(args.labels) - set(DEFAULT_LABEL_DEFINITIONS))
    if unsupported:
        raise SystemExit(f"unsupported labels: {', '.join(unsupported)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_output_path = Path(args.error_output) if args.error_output else None
    if error_output_path is not None:
        error_output_path.parent.mkdir(parents=True, exist_ok=True)

    annotator = build_annotator(args)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    written = 0
    failures = 0
    with output_path.open("wb") as handle:
        error_handle = (
            error_output_path.open("wb") if error_output_path is not None else None
        )
        for doc_index, row in enumerate(dataset):
            if doc_index >= args.limit_docs:
                break
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            source_identifier = str(row.get("identifier", f"row-{doc_index}"))
            mime_type = str(row.get("mime_type", "text/plain"))
            for chunk in iter_text_chunks(
                source_identifier=source_identifier,
                mime_type=mime_type,
                doc_index=doc_index,
                text=text,
                max_chars=args.max_chars_per_chunk,
            ):
                try:
                    result = annotator.annotate(chunk, args.labels)
                except Exception as exc:
                    failures += 1
                    if error_handle is not None:
                        error_record = {
                            "dataset": args.dataset,
                            "split": args.split,
                            "source_identifier": chunk.source_identifier,
                            "mime_type": chunk.mime_type,
                            "doc_index": chunk.doc_index,
                            "chunk_index": chunk.chunk_index,
                            "doc_char_start": chunk.doc_char_start,
                            "doc_char_end": chunk.doc_char_end,
                            "doc_byte_start": chunk.doc_byte_start,
                            "doc_byte_end": chunk.doc_byte_end,
                            "error": repr(exc),
                            "text_preview": chunk.text[:500],
                        }
                        error_handle.write(orjson.dumps(error_record))
                        error_handle.write(b"\n")
                        error_handle.flush()
                    if not args.continue_on_error:
                        raise
                    if failures % max(args.progress_every, 1) == 0:
                        print(
                            f"progress: wrote={written} failures={failures} "
                            f"last_failed_chunk={chunk.doc_index}:{chunk.chunk_index}",
                            flush=True,
                        )
                    continue
                record = ChunkAnnotationRecord(
                    protocol=result.protocol,
                    dataset=args.dataset,
                    split=args.split,
                    source_identifier=chunk.source_identifier,
                    mime_type=chunk.mime_type,
                    doc_index=chunk.doc_index,
                    chunk_index=chunk.chunk_index,
                    requested_labels=list(args.labels),
                    doc_char_start=chunk.doc_char_start,
                    doc_char_end=chunk.doc_char_end,
                    doc_byte_start=chunk.doc_byte_start,
                    doc_byte_end=chunk.doc_byte_end,
                    text=chunk.text,
                    tagged_text=result.tagged_text if args.include_tagged_text else "",
                    units=result.units,
                    unit_annotations=result.unit_annotations,
                    sentence_candidates=result.sentence_candidates,
                    spans=result.spans,
                    validation=result.validation,
                    annotation=result.metadata,
                )
                handle.write(orjson.dumps(record.model_dump(mode="json")))
                handle.write(b"\n")
                handle.flush()
                written += 1
                if written % args.progress_every == 0:
                    print(f"progress: wrote={written} failures={failures}", flush=True)
                if args.limit_chunks is not None and written >= args.limit_chunks:
                    print(
                        f"wrote {written} annotated chunks to {output_path} "
                        f"(failures={failures})",
                        flush=True,
                    )
                    if error_handle is not None:
                        error_handle.close()
                    return

        if error_handle is not None:
            error_handle.close()

    print(f"wrote {written} annotated chunks to {output_path} (failures={failures})", flush=True)


def maybe_load_env_json(path: str | None) -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    if not path:
        return
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit("env json must be an object mapping env var names to values")
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str) and key not in os.environ:
            os.environ[key] = value


def build_annotator(args: argparse.Namespace) -> "BaseAnnotator":
    return OpenAIAnnotator(
        model_name=args.model,
        max_attempts=args.max_attempts,
        max_output_tokens=args.max_output_tokens,
        service_tier=args.service_tier,
        prompt_cache_prefix=args.prompt_cache_prefix,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        sentence_max_candidates=args.sentence_max_candidates,
        sentence_min_block_chars=args.sentence_min_block_chars,
        sentence_min_alpha_chars=args.sentence_min_alpha_chars,
        sentence_parent_labels=set(args.sentence_parent_labels),
    )


def iter_text_chunks(
    *,
    source_identifier: str,
    mime_type: str,
    doc_index: int,
    text: str,
    max_chars: int,
) -> Iterator[TextChunk]:
    char_to_byte = _char_to_byte_offsets(text)
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [text]

    chunk_lines: list[str] = []
    chunk_char_start = 0
    running_char_index = 0
    chunk_index = 0

    for line in lines:
        if len(line) > max_chars and not chunk_lines:
            start = running_char_index
            for piece_start in range(0, len(line), max_chars):
                piece = line[piece_start : piece_start + max_chars]
                piece_char_start = start + piece_start
                piece_char_end = piece_char_start + len(piece)
                yield TextChunk(
                    source_identifier=source_identifier,
                    mime_type=mime_type,
                    doc_index=doc_index,
                    chunk_index=chunk_index,
                    text=piece,
                    doc_char_start=piece_char_start,
                    doc_char_end=piece_char_end,
                    doc_byte_start=char_to_byte[piece_char_start],
                    doc_byte_end=char_to_byte[piece_char_end],
                )
                chunk_index += 1
            running_char_index += len(line)
            continue

        if chunk_lines and sum(len(part) for part in chunk_lines) + len(line) > max_chars:
            chunk_text = "".join(chunk_lines)
            chunk_char_end = chunk_char_start + len(chunk_text)
            yield TextChunk(
                source_identifier=source_identifier,
                mime_type=mime_type,
                doc_index=doc_index,
                chunk_index=chunk_index,
                text=chunk_text,
                doc_char_start=chunk_char_start,
                doc_char_end=chunk_char_end,
                doc_byte_start=char_to_byte[chunk_char_start],
                doc_byte_end=char_to_byte[chunk_char_end],
            )
            chunk_index += 1
            chunk_lines = [line]
            chunk_char_start = running_char_index
        else:
            if not chunk_lines:
                chunk_char_start = running_char_index
            chunk_lines.append(line)
        running_char_index += len(line)

    if chunk_lines:
        chunk_text = "".join(chunk_lines)
        chunk_char_end = chunk_char_start + len(chunk_text)
        yield TextChunk(
            source_identifier=source_identifier,
            mime_type=mime_type,
            doc_index=doc_index,
            chunk_index=chunk_index,
            text=chunk_text,
            doc_char_start=chunk_char_start,
            doc_char_end=chunk_char_end,
            doc_byte_start=char_to_byte[chunk_char_start],
            doc_byte_end=char_to_byte[chunk_char_end],
        )


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def build_logical_block_units(text: str) -> list[TextUnit]:
    if not text:
        return []

    char_to_byte = _char_to_byte_offsets(text)
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [text]

    units: list[TextUnit] = []
    current_parts: list[str] = []
    current_char_start = 0
    running_char_index = 0

    def flush_current(unit_id: int) -> int:
        nonlocal current_parts, current_char_start
        if not current_parts:
            return unit_id
        block_text = "".join(current_parts)
        stripped = block_text.strip()
        if stripped:
            char_end = current_char_start + len(block_text)
            units.append(
                TextUnit(
                    unit_id=unit_id,
                    kind="block",
                    text=block_text,
                    char_start=current_char_start,
                    char_end=char_end,
                    byte_start=char_to_byte[current_char_start],
                    byte_end=char_to_byte[char_end],
                )
            )
            unit_id += 1
        current_parts = []
        return unit_id

    next_unit_id = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            next_unit_id = flush_current(next_unit_id)
            running_char_index += len(line)
            continue
        if not current_parts:
            current_char_start = running_char_index
            current_parts = [line]
            running_char_index += len(line)
            continue
        current_text = "".join(current_parts)
        if should_merge_into_current_block(current_text, line):
            current_parts.append(line)
        else:
            next_unit_id = flush_current(next_unit_id)
            current_char_start = running_char_index
            current_parts = [line]
        running_char_index += len(line)

    flush_current(next_unit_id)
    return units


class BaseAnnotator:
    def annotate(
        self,
        chunk: TextChunk,
        labels: list[str],
    ) -> AnnotationResult:
        raise NotImplementedError


class OpenAIAnnotator(BaseAnnotator):
    def __init__(
        self,
        *,
        model_name: str,
        max_attempts: int,
        max_output_tokens: int,
        service_tier: str,
        prompt_cache_prefix: str,
        debug_dir: Path | None,
        sentence_max_candidates: int,
        sentence_min_block_chars: int,
        sentence_min_alpha_chars: int,
        sentence_parent_labels: set[str],
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY is not set; provide credentials before running annotation"
            )
        self._client = OpenAI()
        self._model_name = model_name
        self._max_attempts = max_attempts
        self._max_output_tokens = max_output_tokens
        self._service_tier = service_tier
        self._prompt_cache_prefix = prompt_cache_prefix
        self._debug_dir = debug_dir
        self._sentence_max_candidates = sentence_max_candidates
        self._sentence_min_block_chars = sentence_min_block_chars
        self._sentence_min_alpha_chars = sentence_min_alpha_chars
        self._sentence_parent_labels = sentence_parent_labels
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def annotate(
        self,
        chunk: TextChunk,
        labels: list[str],
    ) -> AnnotationResult:
        block_labels = [label for label in labels if label in BLOCK_LABELS]
        want_sentences = "sentence" in labels
        if not block_labels:
            if want_sentences:
                block_labels = ["paragraph"]
            else:
                raise RuntimeError("no block labels requested for OpenAI annotation")

        units = build_logical_block_units(chunk.text)
        if not units:
            return AnnotationResult(
                protocol="unit_labels",
                spans=[],
                validation=ValidationReport(
                    protocol="unit_labels",
                    allowed_labels_only=True,
                    unit_coverage_complete=True,
                    unit_order_preserved=True,
                    span_count=0,
                ),
                metadata=AnnotationMetadata(
                    provider="openai",
                    model=self._model_name,
                    attempt=1,
                    notes=["empty chunk"],
                ),
                units=[],
                unit_annotations=[],
            )

        label_guide = "\n".join(
            f"- {label}: {DEFAULT_LABEL_DEFINITIONS[label]}" for label in block_labels
        )
        serialized_units = [
            {
                "unit_id": unit.unit_id,
                "text": unit.text,
            }
            for unit in units
        ]
        base_messages = [
            {
                "role": "system",
                "content": OPENAI_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Requested labels:\n{label_guide}\n\n"
                    f"Allowed labels for this run: {', '.join(block_labels)}, {NONE_LABEL}\n\n"
                    "Return a JSON object with a `units` array. Each array item must have "
                    "a `unit_id` and a `label`. Every unit_id must appear exactly once.\n\n"
                    f"Source identifier: {chunk.source_identifier}\n"
                    f"MIME type: {chunk.mime_type}\n"
                    f"Chunk index: {chunk.chunk_index}\n\n"
                    "Classify the following line units:\n"
                    f"{json.dumps(serialized_units, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

        prompt_cache_key = self._prompt_cache_key(block_labels)
        error_note = ""
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            messages = list(base_messages)
            if error_note:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous attempt failed deterministic validation.\n"
                            f"Validation error: {error_note}\n"
                            "Retry and return a complete, valid unit-label assignment."
                        ),
                    }
                )

            response = self._client.responses.parse(
                model=self._model_name,
                input=messages,
                text_format=UnitLabelAnnotation,
                max_output_tokens=self._max_output_tokens,
                prompt_cache_key=prompt_cache_key,
                service_tier=self._service_tier,
                temperature=0,
            )
            parsed = response.output_parsed
            if parsed is None:
                error_note = "response.output_parsed was empty"
                continue
            try:
                spans, validation = validate_unit_annotations(
                    parsed.units,
                    units,
                    set(block_labels),
                    none_label=NONE_LABEL,
                )
            except UnitAnnotationValidationError as exc:
                last_error = exc
                error_note = str(exc)
                self._write_debug_artifact(
                    chunk=chunk,
                    parsed=parsed,
                    units=units,
                    attempt=attempt,
                    response_id=getattr(response, "id", None),
                    error_message=error_note,
                )
                continue
            sentence_candidates: list[SentenceBoundaryCandidate] = []
            notes = list(parsed.notes)
            protocol = "unit_labels"
            if want_sentences:
                try:
                    sentence_spans, sentence_candidates, sentence_notes = (
                        self._annotate_sentences_for_block_spans(chunk, spans)
                    )
                except SentenceBoundaryValidationError as exc:
                    last_error = exc
                    error_note = str(exc)
                    self._write_sentence_debug_artifact(
                        chunk=chunk,
                        block_spans=spans,
                        sentence_candidates=sentence_candidates,
                        attempt=attempt,
                        response_id=getattr(response, "id", None),
                        error_message=error_note,
                    )
                    continue
                spans = sorted(
                    spans + sentence_spans,
                    key=lambda span: (span.char_start, span.char_end, span.label, span.id),
                )
                validation = validation.model_copy(
                    update={
                        "protocol": "unit_labels_sentence_candidates",
                        "sentence_candidate_validation": True,
                        "span_count": len(spans),
                    }
                )
                notes.extend(sentence_notes)
                protocol = "unit_labels_sentence_candidates"
            else:
                validation = validation.model_copy(update={"span_count": len(spans)})
            return AnnotationResult(
                protocol=protocol,
                spans=spans,
                validation=validation,
                metadata=AnnotationMetadata(
                    provider="openai",
                    model=self._model_name,
                    attempt=attempt,
                    response_id=getattr(response, "id", None),
                    prompt_cache_key=prompt_cache_key,
                    notes=notes,
                ),
                units=units,
                unit_annotations=parsed.units,
                sentence_candidates=sentence_candidates,
            )

        message = "annotation failed after maximum attempts"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise RuntimeError(message)

    def _annotate_sentences_for_block_spans(
        self,
        chunk: TextChunk,
        block_spans: list[SpanAnnotation],
    ) -> tuple[list[SpanAnnotation], list[SentenceBoundaryCandidate], list[str]]:
        sentence_spans: list[SpanAnnotation] = []
        sentence_candidates: list[SentenceBoundaryCandidate] = []
        notes: list[str] = []
        next_span_id = max((span.id for span in block_spans), default=-1) + 1
        for block_span in block_spans:
            if block_span.label not in self._sentence_parent_labels:
                continue
            block_text = chunk.text[block_span.char_start : block_span.char_end]
            if not is_sentence_eligible_block(
                block_text,
                label=block_span.label,
                min_block_chars=self._sentence_min_block_chars,
                min_alpha_chars=self._sentence_min_alpha_chars,
            ):
                notes.append(f"skip_sentence_block:{block_span.id}:ineligible")
                continue
            candidates = build_sentence_candidates(
                block_text,
                parent_span_id=block_span.id,
                char_start=block_span.char_start,
                byte_start=block_span.start,
            )
            if len(candidates) > self._sentence_max_candidates:
                notes.append(
                    f"skip_sentence_block:{block_span.id}:too_many_candidates:{len(candidates)}"
                )
                continue
            sentence_candidates.extend(candidates)
            if not candidates:
                if block_has_sentence_terminal(block_text):
                    sentence_spans.append(
                        SpanAnnotation(
                            id=next_span_id,
                            label="sentence",
                            start=block_span.start,
                            end=block_span.end,
                            char_start=block_span.char_start,
                            char_end=block_span.char_end,
                            parent_id=block_span.id,
                        )
                    )
                    next_span_id += 1
                else:
                    notes.append(f"skip_sentence_block:{block_span.id}:no_terminal")
                continue

            annotation = self._request_sentence_breaks(
                chunk=chunk,
                block_span=block_span,
                block_text=block_text,
                candidates=candidates,
            )
            sentence_spans_block, next_span_id = validate_sentence_breaks(
                parent_span=block_span,
                text=chunk.text,
                candidate_break_ids=annotation.break_after_candidate_ids,
                candidates=candidates,
                next_span_id=next_span_id,
            )
            filtered_sentence_spans: list[SpanAnnotation] = []
            for sentence_span in sentence_spans_block:
                sentence_text = chunk.text[sentence_span.char_start : sentence_span.char_end]
                if block_has_sentence_terminal(sentence_text):
                    filtered_sentence_spans.append(sentence_span)
                else:
                    notes.append(
                        f"skip_sentence_span:{sentence_span.id}:no_terminal"
                    )
            sentence_spans.extend(filtered_sentence_spans)
            notes.extend(annotation.notes)
        return sentence_spans, sentence_candidates, notes

    def _request_sentence_breaks(
        self,
        *,
        chunk: TextChunk,
        block_span: SpanAnnotation,
        block_text: str,
        candidates: list[SentenceBoundaryCandidate],
    ) -> SentenceBoundaryAnnotation:
        serialized_candidates = [
            {
                "candidate_id": candidate.candidate_id,
                "marker": candidate.marker,
            }
            for candidate in candidates
        ]
        prompt_cache_key = self._prompt_cache_key(
            [f"sentence:{block_span.label}:{len(candidates)}"]
        )
        error_note = ""
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            messages = [
                {"role": "system", "content": OPENAI_SENTENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Source identifier: {chunk.source_identifier}\n"
                        f"Chunk index: {chunk.chunk_index}\n"
                        f"Parent block label: {block_span.label}\n"
                        f"Parent block char range: {block_span.char_start}..{block_span.char_end}\n\n"
                        "Parent block text:\n"
                        f"{block_text}\n\n"
                        "Internal candidate boundaries:\n"
                        f"{json.dumps(serialized_candidates, ensure_ascii=False, indent=2)}"
                    ),
                },
            ]
            if error_note:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous attempt failed deterministic validation.\n"
                            f"Validation error: {error_note}\n"
                            "Retry with valid increasing candidate ids only."
                        ),
                    }
                )
            response = self._client.responses.parse(
                model=self._model_name,
                input=messages,
                text_format=SentenceBoundaryAnnotation,
                max_output_tokens=self._max_output_tokens,
                prompt_cache_key=prompt_cache_key,
                service_tier=self._service_tier,
                temperature=0,
            )
            parsed = response.output_parsed
            if parsed is None:
                error_note = "response.output_parsed was empty"
                continue
            try:
                validate_sentence_breaks(
                    parent_span=block_span,
                    text=chunk.text,
                    candidate_break_ids=parsed.break_after_candidate_ids,
                    candidates=candidates,
                    next_span_id=0,
                )
            except SentenceBoundaryValidationError as exc:
                last_error = exc
                error_note = str(exc)
                self._write_sentence_debug_artifact(
                    chunk=chunk,
                    block_spans=[block_span],
                    sentence_candidates=candidates,
                    attempt=attempt,
                    response_id=getattr(response, "id", None),
                    error_message=error_note,
                    parsed=parsed,
                )
                continue
            return parsed
        message = "sentence annotation failed after maximum attempts"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise SentenceBoundaryValidationError(message)

    def _prompt_cache_key(self, labels: list[str]) -> str:
        label_hash = hashlib.sha256(",".join(labels).encode("utf-8")).hexdigest()[:16]
        return f"{self._prompt_cache_prefix}:{self._model_name}:{label_hash}"

    def _write_debug_artifact(
        self,
        *,
        chunk: TextChunk,
        parsed: UnitLabelAnnotation,
        units: list[TextUnit],
        attempt: int,
        response_id: str | None,
        error_message: str,
    ) -> None:
        if self._debug_dir is None:
            return
        payload = {
            "source_identifier": chunk.source_identifier,
            "chunk_index": chunk.chunk_index,
            "attempt": attempt,
            "response_id": response_id,
            "error": error_message,
            "source_text": chunk.text,
            "units": [unit.model_dump(mode="json") for unit in units],
            "unit_annotations": [unit.model_dump(mode="json") for unit in parsed.units],
            "notes": parsed.notes,
        }
        name = f"doc{chunk.doc_index:04d}-chunk{chunk.chunk_index:04d}-attempt{attempt}.json"
        (self._debug_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _write_sentence_debug_artifact(
        self,
        *,
        chunk: TextChunk,
        block_spans: list[SpanAnnotation],
        sentence_candidates: list[SentenceBoundaryCandidate],
        attempt: int,
        response_id: str | None,
        error_message: str,
        parsed: SentenceBoundaryAnnotation | None = None,
    ) -> None:
        if self._debug_dir is None:
            return
        payload = {
            "source_identifier": chunk.source_identifier,
            "chunk_index": chunk.chunk_index,
            "attempt": attempt,
            "response_id": response_id,
            "error": error_message,
            "source_text": chunk.text,
            "block_spans": [span.model_dump(mode="json") for span in block_spans],
            "sentence_candidates": [
                candidate.model_dump(mode="json") for candidate in sentence_candidates
            ],
            "sentence_break_after_candidate_ids": (
                parsed.break_after_candidate_ids if parsed is not None else None
            ),
            "notes": parsed.notes if parsed is not None else [],
        }
        name = (
            f"doc{chunk.doc_index:04d}-chunk{chunk.chunk_index:04d}-"
            f"sentence-attempt{attempt}.json"
        )
        (self._debug_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def should_merge_into_current_block(current_text: str, next_line: str) -> bool:
    current_type = classify_display_type(current_text)
    next_type = classify_display_type(next_line)
    if current_type == "metadata" or next_type == "metadata":
        return False
    if current_type == "heading" or next_type == "heading":
        return False
    if current_type == "list_item":
        return next_type in {"prose", "dialogue"}
    if current_type in {"prose", "dialogue"}:
        if next_type in {"prose", "dialogue"}:
            return True
        if next_type == "list_item":
            return False
        return starts_continuation_line(next_line)
    return False


def classify_display_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if looks_like_metadata_line(stripped):
        return "metadata"
    if looks_like_subject_or_caption_line(stripped) or looks_like_heading(stripped):
        return "heading"
    if looks_like_list_item(stripped):
        return "list_item"
    if looks_like_dialogue_line(stripped):
        return "dialogue"
    return "prose"


def starts_continuation_line(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    return bool(re.match(r"^(?:[a-z0-9(\[\"'“‘]|[ivxlcdm]+\.)", stripped))


def looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if len(stripped) <= 120 and "\n" not in stripped:
        alpha = [char for char in stripped if char.isalpha()]
        if alpha:
            upper_ratio = sum(char.isupper() for char in alpha) / len(alpha)
            return upper_ratio >= 0.75
    return False


def looks_like_subject_or_caption_line(text: str) -> bool:
    normalized = normalized_block_text(text)
    if not normalized:
        return False
    if len(normalized) <= 220 and re.match(
        r"^(?:subject|re|regarding|about|caption|otnosno|относно|objet|betreff|oggetto|asunto|tema)\s*:",
        normalized,
        re.IGNORECASE,
    ):
        return True
    if len(normalized) <= 160 and normalized.endswith(":"):
        return True
    if len(normalized) <= 220 and re.match(r"^article\s+\d+", normalized, re.IGNORECASE):
        return True
    return False


def looks_like_headingish_paragraph(text: str) -> bool:
    normalized = normalized_block_text(text)
    if len(normalized) <= 220 and looks_like_subject_or_caption_line(normalized):
        return True
    if len(normalized) <= 140 and looks_like_heading(normalized):
        return True
    return False


def looks_like_metadataish_paragraph(text: str) -> bool:
    normalized = normalized_block_text(text)
    if len(normalized) <= 220 and looks_like_metadata_line(normalized):
        return True
    if re.match(
        r"^(?:answer|reply|response|отговор)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True
    return False


def looks_like_list_item(text: str) -> bool:
    if re.match(r"^(?:[\-\*\u2022]|\(?[0-9A-Za-z]{1,4}[.)]|¶\d+\.)\s*", text):
        return True
    if re.match(r"^[\"'“‘][^\"'”’]{1,80}[\"'”’]\s+means\b", text, re.IGNORECASE):
        return True
    return False


def looks_like_dialogue_line(text: str) -> bool:
    stripped = text.strip()
    if re.match(r'^[\"“].+[\"”]$', stripped, re.DOTALL):
        return True
    return bool(re.match(r"^[A-Z][A-Z0-9 .'-]{1,40}:\s", stripped))


def looks_like_metadata_line(text: str) -> bool:
    metadata_markers = (
        "UNITED STATES DISTRICT COURT",
        "SUPREME COURT",
        "Case ",
        "Document ",
        "Filed ",
        "Page ",
        "Page ID",
        "Attorney",
        "Attorneys",
        "Bar No.",
        "Telephone:",
        "Facsimile:",
        "E-Mail:",
        "/s/",
        "JUDGE",
    )
    if any(marker in text for marker in metadata_markers):
        return True
    if re.match(r"^[0-9 ]+$", text):
        return True
    alpha = [char for char in text if char.isalpha()]
    if alpha:
        upper_ratio = sum(char.isupper() for char in alpha) / len(alpha)
        if upper_ratio >= 0.92 and len(text) <= 160:
            return True
    return False


def is_sentence_eligible_block(
    text: str,
    *,
    label: str,
    min_block_chars: int,
    min_alpha_chars: int,
) -> bool:
    stripped = normalized_block_text(text)
    if label not in SENTENCE_PARENT_LABELS:
        return False
    if len(stripped) < min_block_chars:
        return False
    alpha_chars = [char for char in stripped if char.isalpha()]
    if len(alpha_chars) < min_alpha_chars:
        return False
    if not any(char.islower() for char in stripped):
        return False
    if looks_like_heading(stripped):
        return False
    if looks_like_subject_or_caption_line(stripped):
        return False
    if looks_like_metadata_line(stripped):
        return False
    if looks_like_metadataish_paragraph(stripped):
        return False
    return True


def normalized_block_text(text: str) -> str:
    return " ".join(text.replace("&#xD;", "").split())


def block_has_sentence_terminal(text: str) -> bool:
    stripped = normalized_block_text(text).rstrip()
    return bool(re.search(r'[.!?](?:["\'”’)\]]+)?$', stripped))
