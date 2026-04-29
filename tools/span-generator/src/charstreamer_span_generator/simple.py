from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import orjson
from datasets import load_dataset
from openai import OpenAI

from .models import PerLabelTaggedAnnotation, SpanAnnotation, TaggedAnnotation, ValidationReport
from .validation import TaggedTextValidationError, parse_tagged_text


SYSTEM_PROMPT = """You annotate arbitrary streaming text windows by inserting control tokens.

Return exactly the original segment text with only annotation control tokens added.
Do not rewrite, normalize, delete, reorder, summarize, or add source text.

Allowed paired tags use this exact syntax:
- <|sentence|>...<|/sentence|>
- <|paragraph|>...<|/paragraph|>
- <|section|>...<|/section|>
- <|dialogue|>...<|/dialogue|>
- <|list_item|>...<|/list_item|>
- <|metadata|>...<|/metadata|>

Rules:
- Use only requested labels.
- When a focus label is provided, prioritize that label only. If no complete
  or partial visible focus-label span exists inside the target, leave the
  target untagged.
- Tags must be balanced and properly nested.
- Context before/after the target is only for deciding whether target-edge text
  is complete. Never include context text in the output.
- If the target begins or ends in the middle of a structure, tag the visible
  portion that is inside TARGET_SEGMENT. The validator will mark the span edge
  as open.
- For sentence labels, tag visible sentence material even when the target cuts
  off the beginning or end of the sentence.
- Labels can overlap across separate focused passes. Do not avoid a label just
  because the same text may also be a paragraph, sentence, list item, etc.
- Prefer high precision over broad coverage.
- If unsure, leave text untagged.
- Return a JSON object matching the schema with `tagged_text` and optional `notes`.
"""


PER_LABEL_SYSTEM_PROMPT = """You annotate arbitrary streaming text windows with independent per-label tag streams.

Return a JSON object with:
- sentence, paragraph, section, dialogue, list_item, metadata: each value is one
  independently tagged copy of TARGET_SEGMENT for that label.
- notes: optional short notes.

For each label field, return exactly the original TARGET_SEGMENT text with only that label's
control tokens inserted. Do not rewrite, normalize, delete, reorder, summarize, or add source text.

Allowed paired tags use this exact syntax:
- <|sentence|>...<|/sentence|>
- <|paragraph|>...<|/paragraph|>
- <|section|>...<|/section|>
- <|dialogue|>...<|/dialogue|>
- <|list_item|>...<|/list_item|>
- <|metadata|>...<|/metadata|>

Rules:
- Annotate every requested label independently.
- It is correct and expected for the same characters to be tagged under multiple labels
  in different per-label values, e.g. sentence text inside paragraph text.
- If a label has no visible span in the target, return TARGET_SEGMENT unchanged for that label.
- Context before/after the target is only for deciding whether target-edge text is complete.
  Never include context text in any output value.
- If the target begins or ends in the middle of a structure, tag the visible portion that is
  inside TARGET_SEGMENT. The validator will mark the span edge as open.
- Prefer high precision over broad coverage. It is better to leave a field unchanged than to
  tag an ambiguous or merely keyword-matching span.
- Do not create a tag just because the field exists. Fields with no clear visible span must
  be exact untagged copies of TARGET_SEGMENT.
- Labels are semantic structural roles, not keyword searches: a line mentioning a court, date,
  patent office, quotation mark, number, or document title is not automatically metadata,
  dialogue, list_item, or section.
- Do not omit obvious sentence/paragraph spans when those labels are requested.
- Return only the structured JSON object matching the schema.
"""


class SpanQualityError(ValueError):
    pass


LABEL_GUIDANCE: dict[str, str] = {
    "sentence": (
        "Visible sentence material. Tag complete sentences and target-edge "
        "sentence continuations. Do not tag headings, table rows, or pure citations "
        "unless they function as sentence text."
    ),
    "paragraph": (
        "Visible body-prose paragraph material. A paragraph can start before the "
        "target or continue after it; wrap the visible prose portion inside the "
        "target. Do not tag standalone headings, docket/case furniture, signature "
        "lines, tables, or list markers as paragraph unless they are clearly part "
        "of a main-content prose block."
    ),
    "section": (
        "Visible standalone section heading/title/caption/article-label text. Tag "
        "only the heading line or visible heading fragment. Do not tag ordinary "
        "sentences, clauses, patent prose, citations, or body paragraphs merely "
        "because they mention a topic."
    ),
    "dialogue": (
        "Visible dialogue material: direct speech, Q/A transcript turns, or "
        "speaker-labeled utterances. Do not tag quoted legal terms, quoted titles, "
        "tables, citations, or ordinary prose containing quotation marks."
    ),
    "list_item": (
        "Visible bullet, numbered, lettered, or enumerated list entry material. "
        "A list item can start before or continue after the target. Do not tag "
        "page numbers, numbered citations, table rows, or section numbers unless "
        "the text is clearly an item in a list."
    ),
    "metadata": (
        "Visible document furniture such as captions, docket/case metadata, filing "
        "dates, parties, signature blocks, page headers/footers, addresses, or "
        "boilerplate identifiers. Do not tag ordinary body prose merely because it "
        "mentions a court, date, patent office, statute, or document title."
    ),
}

FOCUS_INSTRUCTIONS: dict[str, str] = {
    "sentence": "Wrap visible sentence text, including left/right target-edge continuations.",
    "paragraph": "Wrap visible paragraph text, including partial paragraphs at target edges.",
    "section": "Wrap visible heading/title lines, not following body text.",
    "dialogue": "Wrap visible quoted speech, Q/A text, or speaker turns, including partial edge dialogue.",
    "list_item": "Wrap visible enumerated/bulleted item material, including partial edge items.",
    "metadata": "Wrap visible metadata/furniture spans, but leave ordinary body prose untagged.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream random HF records, annotate short random segments with inline control tokens, and write JSONL."
    )
    parser.add_argument(
        "--dataset",
        default="alea-institute/kl3m-data-sample-005-shuffled",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--error-output")
    parser.add_argument(
        "--annotation-protocol",
        choices=["per-label", "inline"],
        default="per-label",
        help="per-label validates one tagged text per label and supports overlapping spans.",
    )
    parser.add_argument("--limit", type=positive_int, default=100)
    parser.add_argument("--max-records", type=positive_int, default=100000)
    parser.add_argument("--min-chars", type=positive_int, default=50)
    parser.add_argument("--max-chars", type=positive_int, default=1000)
    parser.add_argument("--context-chars", type=non_negative_int, default=120)
    parser.add_argument(
        "--min-alpha-ratio",
        type=probability,
        default=0.18,
        help="Skip sampled segments with too little alphabetic text before annotation.",
    )
    parser.add_argument(
        "--max-symbol-ratio",
        type=probability,
        default=0.40,
        help="Skip sampled segments dominated by non-space symbols before annotation.",
    )
    parser.add_argument(
        "--max-control-ratio",
        type=probability,
        default=0.02,
        help="Skip sampled segments with too many control characters before annotation.",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["mixed", "natural", "random"],
        default="mixed",
        help="mixed keeps random negatives while preferring natural paragraph/sentence-like targets.",
    )
    parser.add_argument(
        "--natural-probability",
        type=probability,
        default=0.5,
        help="Probability of trying a natural segment first when --segment-mode=mixed.",
    )
    parser.add_argument(
        "--edge-jitter-probability",
        type=probability,
        default=0.5,
        help="Probability of cutting a focused candidate into a partial streaming window.",
    )
    parser.add_argument(
        "--label-strategy",
        choices=["all", "round-robin", "random"],
        default="all",
        help="Request all labels per row, or request one focus label per row for legacy focused generation.",
    )
    parser.add_argument(
        "--sample-focus-strategy",
        choices=["none", "round-robin", "random"],
        default="round-robin",
        help="Choose a label only for target-window sampling while still annotating requested labels.",
    )
    parser.add_argument(
        "--target-labels",
        nargs="+",
        help="Labels to cycle/sample when --label-strategy is round-robin or random; defaults to --labels.",
    )
    parser.add_argument("--shuffle-buffer-size", type=positive_int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--temperature",
        type=temperature_value,
        help="Optional sampling temperature. Omitted by default because some reasoning models reject it.",
    )
    parser.add_argument("--top-p", type=probability)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="low",
        help="Reasoning effort for models that support it; use low for fast deterministic annotation.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["low", "medium", "high"],
        default="low",
        help="Output verbosity for supported GPT-5-family models.",
    )
    parser.add_argument("--max-attempts", type=positive_int, default=3)
    parser.add_argument("--max-output-tokens", type=positive_int, default=12000)
    parser.add_argument(
        "--service-tier",
        choices=["auto", "default", "flex", "scale", "priority"],
        default="auto",
    )
    parser.add_argument("--prompt-cache-key")
    parser.add_argument("--store-response", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--env-json")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-spans",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only write rows with at least one validated span; useful for positive-only QA, not default training data.",
    )
    parser.add_argument(
        "--strict-span-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject likely partial semantic spans after exact round-trip validation.",
    )
    parser.add_argument("--progress-every", type=positive_int, default=25)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["sentence", "paragraph", "section", "dialogue", "list_item", "metadata"],
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def temperature_value(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 2.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 2")
    return parsed


def main() -> None:
    args = parse_args()
    if args.min_chars > args.max_chars:
        raise SystemExit("--min-chars must be <= --max-chars")
    maybe_load_env_json(args.env_json)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    labels = validate_labels(args.labels)
    target_labels = validate_labels(args.target_labels or labels)
    if not set(target_labels).issubset(set(labels)):
        raise SystemExit("--target-labels must be a subset of --labels")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = Path(args.error_output) if args.error_output else None
    if error_path is not None:
        error_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    rng = random.Random(args.seed)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    written = 0
    failures = 0
    with output_path.open("wb") as out_handle:
        err_handle = error_path.open("wb") if error_path is not None else None
        try:
            for record_index, row in enumerate(dataset):
                if record_index >= args.max_records or written >= args.limit:
                    break
                text = row.get("text")
                if not isinstance(text, str):
                    continue
                requested_labels, annotation_focus_label = choose_labels(
                    labels=labels,
                    target_labels=target_labels,
                    label_strategy=args.label_strategy,
                    sample_index=written,
                    rng=rng,
                )
                sample_focus_label = choose_sample_focus_label(
                    target_labels=target_labels,
                    sample_focus_strategy=args.sample_focus_strategy,
                    sample_index=written,
                    rng=rng,
                )
                segment = sample_segment(
                    text,
                    rng,
                    args.min_chars,
                    args.max_chars,
                    args.context_chars,
                    args.segment_mode,
                    args.natural_probability,
                    sample_focus_label,
                    args.edge_jitter_probability,
                )
                if segment is None:
                    continue
                if not segment_quality_ok(
                    segment["text"],
                    min_alpha_ratio=args.min_alpha_ratio,
                    max_symbol_ratio=args.max_symbol_ratio,
                    max_control_ratio=args.max_control_ratio,
                ):
                    continue
                try:
                    annotation = annotate_segment(
                        client=client,
                        model=args.model,
                        service_tier=args.service_tier,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        reasoning_effort=args.reasoning_effort,
                        verbosity=args.verbosity,
                        prompt_cache_key=args.prompt_cache_key,
                        store_response=args.store_response,
                        max_output_tokens=args.max_output_tokens,
                        max_attempts=args.max_attempts,
                        annotation_protocol=args.annotation_protocol,
                        labels=requested_labels,
                        focus_label=annotation_focus_label,
                        text=segment["text"],
                        prefix_context=segment["prefix_context"],
                        suffix_context=segment["suffix_context"],
                        segment=segment,
                        strict_span_quality=args.strict_span_quality,
                    )
                    tagged = annotation["tagged_text"]
                    tagged_text_by_label = annotation["tagged_text_by_label"]
                    spans = annotation["spans"]
                    validation = annotation["validation"]
                    if args.require_spans and not spans:
                        continue
                except Exception as exc:
                    failures += 1
                    write_error(
                        err_handle,
                        args=args,
                        row=row,
                        record_index=record_index,
                        segment=segment,
                        error=exc,
                    )
                    if not args.continue_on_error:
                        raise
                    continue

                payload = {
                    "dataset": args.dataset,
                    "split": args.split,
                    "source_identifier": str(row.get("identifier", f"row-{record_index}")),
                    "mime_type": str(row.get("mime_type", "")),
                    "record_index": record_index,
                    "sample_index": written,
                    "segment_char_start": segment["char_start"],
                    "segment_char_end": segment["char_end"],
                    "segment_byte_start": segment["byte_start"],
                    "segment_byte_end": segment["byte_end"],
                    "segment_mode": segment["mode"],
                    "source_span": segment.get("source_span"),
                    "annotation_protocol": args.annotation_protocol,
                    "label_strategy": args.label_strategy,
                    "sample_focus_strategy": args.sample_focus_strategy,
                    "sample_focus_label": sample_focus_label,
                    "focus_label": annotation_focus_label,
                    "context": {
                        "prefix": segment["prefix_context"],
                        "suffix": segment["suffix_context"],
                    },
                    "text": segment["text"],
                    "tagged_text": tagged,
                    "tagged_text_by_label": tagged_text_by_label,
                    "labels": requested_labels,
                    "spans": [span.model_dump(mode="json") for span in spans],
                    "validation": validation.model_dump(mode="json"),
                    "annotation": {
                        "provider": "openai",
                        "model": args.model,
                        "response_id": annotation["response_id"],
                        "attempt": annotation["attempt"],
                        "parameters": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "reasoning_effort": args.reasoning_effort,
                            "verbosity": args.verbosity,
                            "max_output_tokens": args.max_output_tokens,
                            "service_tier": args.service_tier,
                            "prompt_cache_key": args.prompt_cache_key,
                            "store_response": args.store_response,
                            "edge_jitter_probability": args.edge_jitter_probability,
                            "annotation_protocol": args.annotation_protocol,
                            "sample_focus_strategy": args.sample_focus_strategy,
                        },
                        "notes": annotation["notes"],
                    },
                }
                out_handle.write(orjson.dumps(payload))
                out_handle.write(b"\n")
                out_handle.flush()
                written += 1
                if written % args.progress_every == 0:
                    print(f"progress: wrote={written} failures={failures}", flush=True)
        finally:
            if err_handle is not None:
                err_handle.close()

    print(f"wrote {written} annotated segments to {output_path} (failures={failures})", flush=True)


def validate_labels(labels: list[str]) -> list[str]:
    allowed = {"sentence", "paragraph", "section", "dialogue", "list_item", "metadata"}
    unique = list(dict.fromkeys(labels))
    unsupported = sorted(set(unique) - allowed)
    if unsupported:
        raise SystemExit(f"unsupported labels: {', '.join(unsupported)}")
    return unique


def choose_labels(
    *,
    labels: list[str],
    target_labels: list[str],
    label_strategy: str,
    sample_index: int,
    rng: random.Random,
) -> tuple[list[str], str | None]:
    if label_strategy == "all":
        return labels, None
    if label_strategy == "round-robin":
        focus_label = target_labels[sample_index % len(target_labels)]
        return [focus_label], focus_label
    if label_strategy == "random":
        focus_label = rng.choice(target_labels)
        return [focus_label], focus_label
    raise ValueError(f"unsupported label strategy: {label_strategy}")


def choose_sample_focus_label(
    *,
    target_labels: list[str],
    sample_focus_strategy: str,
    sample_index: int,
    rng: random.Random,
) -> str | None:
    if sample_focus_strategy == "none":
        return None
    if sample_focus_strategy == "round-robin":
        return target_labels[sample_index % len(target_labels)]
    if sample_focus_strategy == "random":
        return rng.choice(target_labels)
    raise ValueError(f"unsupported sample focus strategy: {sample_focus_strategy}")


def maybe_load_env_json(path: str | None) -> None:
    if os.getenv("OPENAI_API_KEY") or not path:
        return
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit("env json must be an object mapping env var names to values")
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str) and key not in os.environ:
            os.environ[key] = value


def sample_segment(
    text: str,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
    context_chars: int = 120,
    segment_mode: str = "mixed",
    natural_probability: float = 0.5,
    focus_label: str | None = None,
    edge_jitter_probability: float = 0.5,
) -> dict[str, Any] | None:
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return None

    if focus_label is not None:
        focus_span = sample_label_focused_span(cleaned, rng, min_chars, max_chars, focus_label)
        if focus_span is not None:
            target_span = jitter_focused_span(
                cleaned,
                focus_span[0],
                focus_span[1],
                rng,
                min_chars,
                max_chars,
                edge_jitter_probability,
            )
            return build_segment(
                cleaned,
                target_span[0],
                target_span[1],
                context_chars,
                f"focus:{focus_label}",
                source_span=focus_span,
            )

    natural_first = segment_mode == "natural" or (
        segment_mode == "mixed" and rng.random() < natural_probability
    )
    if natural_first:
        natural_span = sample_natural_span(cleaned, rng, min_chars, max_chars)
        if natural_span is not None:
            return build_segment(cleaned, natural_span[0], natural_span[1], context_chars, "natural")
        if segment_mode == "natural":
            return None

    random_span = sample_random_span(cleaned, rng, min_chars, max_chars)
    if random_span is None:
        return None
    return build_segment(cleaned, random_span[0], random_span[1], context_chars, "random")


def sample_random_span(
    text: str,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
) -> tuple[int, int] | None:
    length = rng.randint(min_chars, min(max_chars, len(text)))
    start = rng.randint(0, len(text) - length)
    end = start + length
    start, end = snap_to_whitespace(text, start, end, min_chars, max_chars)
    if end - start < min_chars:
        return None
    return start, end


def build_segment(
    text: str,
    start: int,
    end: int,
    context_chars: int,
    mode: str,
    source_span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    segment = text[start:end]
    char_to_byte = char_to_byte_offsets(text)
    return {
        "char_start": start,
        "char_end": end,
        "byte_start": char_to_byte[start],
        "byte_end": char_to_byte[end],
        "source_char_len": len(text),
        "prefix_context": text[max(0, start - context_chars) : start],
        "suffix_context": text[end : min(len(text), end + context_chars)],
        "mode": mode,
        "source_span": (
            {"char_start": source_span[0], "char_end": source_span[1]}
            if source_span is not None
            else None
        ),
        "text": segment,
    }


def jitter_focused_span(
    text: str,
    start: int,
    end: int,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
    edge_jitter_probability: float,
) -> tuple[int, int]:
    span_len = end - start
    if span_len <= min_chars or rng.random() >= edge_jitter_probability:
        return start, min(end, start + min(max_chars, span_len))

    target_len = rng.randint(min_chars, min(max_chars, span_len))
    max_start = end - target_len
    if max_start <= start:
        return start, start + target_len

    # Bias toward cutting at least one side so production data includes
    # left_open/right_open continuation examples.
    if rng.random() < 0.5:
        target_start = rng.randint(start + 1, max_start)
    else:
        target_start = rng.randint(start, max_start - 1)
    target_end = target_start + target_len
    return target_start, target_end


def sample_natural_span(
    text: str,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for start, end in paragraph_spans(text):
        length = end - start
        if min_chars <= length <= max_chars:
            candidates.append((start, end))
        if length > max_chars:
            candidates.extend(sentence_window_spans(text, start, end, min_chars, max_chars))
    if not candidates:
        candidates.extend(sentence_window_spans(text, 0, len(text), min_chars, max_chars))
    return rng.choice(candidates) if candidates else None


def sample_label_focused_span(
    text: str,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
    focus_label: str,
) -> tuple[int, int] | None:
    candidate_builders = {
        "sentence": lambda: sentence_window_spans(text, 0, len(text), min_chars, max_chars),
        "paragraph": lambda: paragraph_candidate_spans(text, min_chars, max_chars),
        "section": lambda: line_window_spans(
            text, candidate_line_spans(text, is_section_like_line), min_chars, max_chars
        ),
        "dialogue": lambda: line_window_spans(
            text, candidate_line_spans(text, is_dialogue_like_line), min_chars, max_chars
        ),
        "list_item": lambda: list_item_candidate_spans(text, min_chars, max_chars),
        "metadata": lambda: line_window_spans(
            text, candidate_line_spans(text, is_metadata_like_line), min_chars, max_chars
        ),
    }
    build_candidates = candidate_builders.get(focus_label)
    if build_candidates is None:
        return None
    candidates = build_candidates()
    return rng.choice(candidates) if candidates else None


def bounded_spans(
    spans: list[tuple[int, int]],
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    return [(start, end) for start, end in spans if min_chars <= end - start <= max_chars]


def paragraph_candidate_spans(
    text: str,
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for start, end in paragraph_spans(text):
        paragraph_text = text[start:end].strip()
        if not is_paragraph_like_text(paragraph_text):
            continue
        length = end - start
        if min_chars <= length <= max_chars:
            candidates.append((start, end))
        elif length > max_chars:
            candidates.extend(sentence_window_spans(text, start, end, min_chars, max_chars))
    return candidates


def is_paragraph_like_text(text: str) -> bool:
    if not text or "\n\n" in text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    if is_section_like_line(first) or is_list_item_like_line(first) or is_metadata_like_line(first):
        return False
    alpha_count = sum(char.isalpha() for char in text)
    lower_count = sum(char.islower() for char in text)
    if alpha_count < 40 or lower_count < 20:
        return False
    return sentence_ends_with_terminal(text) or bool(SENTENCE_END_RE.search(text))


def list_item_candidate_spans(
    text: str,
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    lines = line_spans(text)
    candidates: list[tuple[int, int]] = []
    anchors: list[tuple[int, int]] = []
    for index, (line_start, line_end, line) in enumerate(lines):
        if not is_list_item_like_line(line):
            continue
        start, _anchor_end = trim_offsets(text, line_start, line_end)
        end = line_end
        next_index = index + 1
        while next_index < len(lines):
            next_start, next_end, next_line = lines[next_index]
            stripped = next_line.strip()
            if not stripped:
                break
            if is_list_item_like_line(next_line) or is_section_like_line(next_line):
                break
            if next_line[:1].isspace():
                end = next_end
                next_index += 1
                continue
            break
        start, end = trim_offsets(text, start, end)
        anchors.append((start, end))
        if min_chars <= end - start <= max_chars:
            candidates.append((start, end))
    if candidates:
        return list(dict.fromkeys(candidates))
    return line_window_spans(text, anchors, min_chars, max_chars)


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    block_start = 0
    for match in re.finditer(r"\n\s*\n", text):
        spans.extend(trimmed_span(text, block_start, match.start()))
        block_start = match.end()
    spans.extend(trimmed_span(text, block_start, len(text)))
    return spans


def trimmed_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return [(start, end)] if start < end else []


def line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        spans.append((start, end, line))
        start = end
    if start < len(text):
        spans.append((start, len(text), text[start:]))
    return spans


def candidate_line_spans(
    text: str,
    predicate: Any,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for start, end, line in line_spans(text):
        if predicate(line):
            stripped_start, stripped_end = trim_offsets(text, start, end)
            if stripped_start < stripped_end:
                candidates.append((stripped_start, stripped_end))
    return candidates


def line_window_spans(
    text: str,
    anchors: list[tuple[int, int]],
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    if not anchors:
        return []
    lines = line_spans(text)
    candidates: list[tuple[int, int]] = []
    for anchor_start, anchor_end in anchors:
        line_index = next(
            index for index, (line_start, line_end, _line) in enumerate(lines)
            if line_start <= anchor_start < line_end
        )
        for left in range(line_index, max(-1, line_index - 4), -1):
            for right in range(line_index, min(len(lines), line_index + 5)):
                start = lines[left][0]
                end = lines[right][1]
                start, end = trim_offsets(text, start, end)
                length = end - start
                if min_chars <= length <= max_chars and start <= anchor_start < anchor_end <= end:
                    candidates.append((start, end))
        if anchor_end - anchor_start >= min_chars and anchor_end - anchor_start <= max_chars:
            candidates.append((anchor_start, anchor_end))
    return list(dict.fromkeys(candidates))


def trim_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


SECTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:section|article|chapter|part|title|appendix|schedule|exhibit)\b"
    r"|(?:[IVXLCDM]+|[0-9]+(?:\.[0-9]+)*|[A-Z])[\).]\s+[A-Z]"
    r")",
    re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•]|(?:\(?[0-9A-Za-z]{1,4}\)?[\).])|(?:[ivxlcdm]{1,8}[\).]))\s+\S",
    re.IGNORECASE,
)
SPEAKER_RE = re.compile(r"^\s*[A-Z][A-Z .'\-]{1,40}:\s+\S")
METADATA_RE = re.compile(
    r"\b(?:case\s+no\.?|civil\s+action|filed|docket|court|plaintiff|defendant|"
    r"attorneys?|counsel|address|telephone|email|page\s+\d+|signature|signed|"
    r"date:|by order|commission file|irs employer|registrant|sec\.?)\b",
    re.IGNORECASE,
)


def is_section_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 140:
        return False
    if SECTION_RE.search(stripped):
        return True
    letters = [char for char in stripped if char.isalpha()]
    return len(letters) >= 6 and sum(char.isupper() for char in letters) / len(letters) > 0.85


def is_dialogue_like_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 10:
        return False
    return SPEAKER_RE.search(stripped) is not None or (
        ("“" in stripped and "”" in stripped) or stripped.count('"') >= 2
    )


def is_list_item_like_line(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 8 and LIST_ITEM_RE.search(stripped) is not None


def is_metadata_like_line(line: str) -> bool:
    stripped = line.strip()
    return 5 <= len(stripped) <= 180 and METADATA_RE.search(stripped) is not None


SENTENCE_END_RE = re.compile(r"[.!?][\"')\]}”’»›]*(?=\s|$)")


def sentence_window_spans(
    text: str,
    start: int,
    end: int,
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    units: list[tuple[int, int]] = []
    unit_start = start
    for match in SENTENCE_END_RE.finditer(text, start, end):
        unit_end = match.end()
        while unit_start < unit_end and text[unit_start].isspace():
            unit_start += 1
        if unit_start < unit_end:
            units.append((unit_start, unit_end))
        unit_start = unit_end

    candidates: list[tuple[int, int]] = []
    for left_index, (candidate_start, _first_end) in enumerate(units):
        candidate_end = candidate_start
        for unit_start, unit_end in units[left_index:]:
            if unit_start - candidate_start > max_chars:
                break
            candidate_end = unit_end
            length = candidate_end - candidate_start
            if length > max_chars:
                break
            if length >= min_chars:
                candidates.append((candidate_start, candidate_end))
    return candidates


def snap_to_whitespace(
    text: str,
    start: int,
    end: int,
    min_chars: int,
    max_chars: int,
) -> tuple[int, int]:
    left = start
    while left > 0 and not text[left - 1].isspace() and start - left < 40:
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace() and right - end < 40:
        right += 1
    if right - left <= max_chars:
        start, end = left, right
    if end - start < min_chars:
        end = min(len(text), start + min_chars)
    return start, end


def segment_quality_ok(
    text: str,
    *,
    min_alpha_ratio: float,
    max_symbol_ratio: float,
    max_control_ratio: float,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    chars = list(stripped)
    total = len(chars)
    alpha = sum(char.isalpha() for char in chars)
    whitespace = sum(char.isspace() for char in chars)
    controls = sum((ord(char) < 32 and char not in "\n\r\t") for char in chars)
    symbols = sum(
        (not char.isalnum() and not char.isspace() and char not in ".:,;!?()[]{}'\"-/")
        for char in chars
    )
    if alpha / total < min_alpha_ratio:
        return False
    if controls / total > max_control_ratio:
        return False
    if symbols / total > max_symbol_ratio:
        return False
    if total >= 40 and whitespace / total < 0.02:
        return False
    if longest_nonspace_run(stripped) > 160:
        return False
    return True


def longest_nonspace_run(text: str) -> int:
    longest = 0
    current = 0
    for char in text:
        if char.isspace():
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


SENTENCE_TERMINALS = frozenset(".?!。！？")
SENTENCE_TRAILING_CLOSERS = frozenset("\"'”’»›)]}")
BLOCK_LABELS = frozenset({"paragraph", "section", "dialogue", "list_item", "metadata"})


def add_span_edge_flags(
    segment: dict[str, Any],
    spans: list[SpanAnnotation],
) -> list[SpanAnnotation]:
    return [
        span.model_copy(update=span_edge_flags(segment, span))
        for span in spans
    ]


def span_edge_flags(segment: dict[str, Any], span: SpanAnnotation) -> dict[str, bool]:
    text = segment["text"]
    span_text = text[span.char_start : span.char_end]
    prefix_context = segment.get("prefix_context", "")
    suffix_context = segment.get("suffix_context", "")
    left_open = False
    right_open = False

    if span.char_start == 0:
        left_open = edge_continues_left(span.label, prefix_context)
    if span.char_end == len(text):
        right_open = edge_continues_right(span.label, span_text, suffix_context)

    return {"left_open": left_open, "right_open": right_open}


def edge_continues_left(label: str, prefix_context: str) -> bool:
    if not prefix_context or not prefix_context.strip():
        return False
    if label == "sentence":
        return not has_sentence_left_boundary(prefix_context)
    if label == "paragraph":
        return not has_blank_boundary_before(prefix_context)
    return not has_line_boundary_before(prefix_context)


def edge_continues_right(label: str, span_text: str, suffix_context: str) -> bool:
    if not suffix_context or not suffix_context.strip():
        return False
    if label == "sentence":
        return not sentence_ends_with_terminal(span_text)
    if label == "paragraph":
        return (not sentence_ends_with_terminal(span_text)) or not has_blank_boundary_after(suffix_context)
    if label in {"dialogue", "list_item"}:
        return (not sentence_ends_with_terminal(span_text)) or not has_line_boundary_after(suffix_context)
    return not has_line_boundary_after(suffix_context)


def validate_span_quality(segment: dict[str, Any], spans: list[SpanAnnotation]) -> None:
    text = segment["text"]
    for span in spans:
        span_text = text[span.char_start : span.char_end]
        if span.label == "sentence":
            validate_sentence_quality(segment, span, span_text)
        elif span.label in BLOCK_LABELS:
            validate_block_quality(segment, span)


def validate_sentence_quality(
    segment: dict[str, Any],
    span: SpanAnnotation,
    span_text: str,
) -> None:
    if not span.right_open and has_unbalanced_directional_quotes(span_text):
        raise SpanQualityError("sentence span contains unbalanced directional quotes")
    if not span.right_open and not sentence_ends_with_terminal(span_text):
        raise SpanQualityError("sentence span does not end with sentence-terminal punctuation")

    context_before = (
        segment["text"][: span.char_start]
        if span.char_start > 0
        else segment["prefix_context"]
    )
    if not span.left_open and not has_sentence_left_boundary(context_before):
        raise SpanQualityError("sentence span begins after non-boundary context")

    if span.char_end < len(segment["text"]) and not sentence_ends_with_terminal(span_text):
        raise SpanQualityError("sentence span ends before non-boundary target text")


def validate_block_quality(segment: dict[str, Any], span: SpanAnnotation) -> None:
    context_before = (
        segment["text"][: span.char_start]
        if span.char_start > 0
        else segment["prefix_context"]
    )
    context_after = (
        segment["text"][span.char_end :]
        if span.char_end < len(segment["text"])
        else segment["suffix_context"]
    )

    if span.label == "paragraph":
        if not span.left_open and not has_blank_boundary_before(context_before):
            raise SpanQualityError("paragraph span begins after non-paragraph-boundary context")
        if not span.right_open and not has_blank_boundary_after(context_after):
            raise SpanQualityError("paragraph span ends before non-paragraph-boundary context")
        return

    if not span.left_open and not has_line_boundary_before(context_before):
        raise SpanQualityError(f"{span.label} span begins after non-line-boundary context")
    if not span.right_open and not has_line_boundary_after(context_after):
        raise SpanQualityError(f"{span.label} span ends before non-line-boundary context")


def sentence_ends_with_terminal(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in SENTENCE_TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    if re.search(r"(\.\s*){3,}$|…$", stripped):
        return False
    return bool(stripped) and stripped[-1] in SENTENCE_TERMINALS


def has_unbalanced_directional_quotes(text: str) -> bool:
    return text.count("“") != text.count("”")


def has_sentence_left_boundary(context: str) -> bool:
    if not context or not context.strip():
        return True
    stripped = context.rstrip()
    return (
        sentence_ends_with_terminal(stripped)
        or bool(re.search(r"\n\s*\n\s*$", context))
    )


def has_blank_boundary_before(context: str) -> bool:
    return not context or not context.strip() or bool(re.search(r"\n\s*\n\s*$", context))


def has_blank_boundary_after(context: str) -> bool:
    return not context or not context.strip() or bool(re.match(r"^\s*\n\s*\n", context))


def has_line_boundary_before(context: str) -> bool:
    return not context or not context.strip() or bool(re.search(r"\n\s*$", context))


def has_line_boundary_after(context: str) -> bool:
    return not context or not context.strip() or bool(re.match(r"^\s*\n", context))


def annotate_segment(
    *,
    client: OpenAI,
    model: str,
    service_tier: str,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: str | None,
    verbosity: str | None,
    prompt_cache_key: str | None,
    store_response: bool,
    max_output_tokens: int,
    max_attempts: int,
    annotation_protocol: str,
    labels: list[str],
    focus_label: str | None,
    text: str,
    prefix_context: str,
    suffix_context: str,
    segment: dict[str, Any],
    strict_span_quality: bool,
) -> dict[str, Any]:
    if annotation_protocol == "per-label":
        return annotate_segment_per_label(
            client=client,
            model=model,
            service_tier=service_tier,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            prompt_cache_key=prompt_cache_key,
            store_response=store_response,
            max_output_tokens=max_output_tokens,
            max_attempts=max_attempts,
            labels=labels,
            text=text,
            prefix_context=prefix_context,
            suffix_context=suffix_context,
            segment=segment,
            strict_span_quality=strict_span_quality,
        )
    if annotation_protocol == "inline":
        return annotate_segment_inline(
            client=client,
            model=model,
            service_tier=service_tier,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            prompt_cache_key=prompt_cache_key,
            store_response=store_response,
            max_output_tokens=max_output_tokens,
            max_attempts=max_attempts,
            labels=labels,
            focus_label=focus_label,
            text=text,
            prefix_context=prefix_context,
            suffix_context=suffix_context,
            segment=segment,
            strict_span_quality=strict_span_quality,
        )
    raise ValueError(f"unsupported annotation protocol: {annotation_protocol}")


def annotate_segment_inline(
    *,
    client: OpenAI,
    model: str,
    service_tier: str,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: str | None,
    verbosity: str | None,
    prompt_cache_key: str | None,
    store_response: bool,
    max_output_tokens: int,
    max_attempts: int,
    labels: list[str],
    focus_label: str | None,
    text: str,
    prefix_context: str,
    suffix_context: str,
    segment: dict[str, Any],
    strict_span_quality: bool,
) -> dict[str, Any]:
    label_text = ", ".join(labels)
    guidance_text = "\n".join(f"- {label}: {LABEL_GUIDANCE[label]}" for label in labels)
    focus_text = (
        f"\nFocus label: {focus_label}\n"
        f"Focus instruction: {FOCUS_INSTRUCTIONS[focus_label]}\n"
        "Only tag this focus label. If no complete focus-label span exists in TARGET_SEGMENT, return TARGET_SEGMENT unchanged.\n"
        if focus_label is not None
        else ""
    )
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        retry_note = (
            ""
            if not last_error
            else f"\nPrevious attempt failed validation: {last_error}\nTry again with exact text preservation."
        )
        prefix_block = (
            "<<<PREFIX_CONTEXT>>>\n"
            f"{prefix_context}\n"
            "<<<END_PREFIX_CONTEXT>>>\n\n"
            if prefix_context
            else ""
        )
        suffix_block = (
            "\n\n<<<SUFFIX_CONTEXT>>>\n"
            f"{suffix_context}\n"
            "<<<END_SUFFIX_CONTEXT>>>"
            if suffix_context
            else ""
        )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Requested labels: {label_text}\n"
                        f"Label definitions:\n{guidance_text}\n"
                        f"{focus_text}"
                        "Annotate only TARGET_SEGMENT by inserting requested control-token tags. "
                        "Use context only to avoid tagging incomplete edge spans."
                        f"{retry_note}\n\n"
                        f"{prefix_block}"
                        "<<<TARGET_SEGMENT>>>\n"
                        f"{text}\n"
                        "<<<END_TARGET_SEGMENT>>>"
                        f"{suffix_block}"
                    ),
                },
            ],
            "text_format": TaggedAnnotation,
            "max_output_tokens": max_output_tokens,
            "service_tier": service_tier,
            "store": store_response,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        if verbosity is not None:
            request_kwargs["text"] = {"verbosity": verbosity}
        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = prompt_cache_key

        response = client.responses.parse(**request_kwargs)
        parsed = response.output_parsed
        if parsed is None:
            last_error = "empty structured response"
            continue
        try:
            parsed_spans, _validation = parse_tagged_text(parsed.tagged_text, text, set(labels))
            spans = add_span_edge_flags(segment, parsed_spans)
            if strict_span_quality:
                validate_span_quality(segment, spans)
        except (TaggedTextValidationError, SpanQualityError) as exc:
            last_error = str(exc)
            continue
        return {
            "tagged_text": parsed.tagged_text,
            "tagged_text_by_label": {},
            "spans": spans,
            "validation": _validation,
            "response_id": getattr(response, "id", None),
            "attempt": attempt,
            "notes": parsed.notes,
        }
    raise RuntimeError(f"annotation failed after {max_attempts} attempts: {last_error}")


def annotate_segment_per_label(
    *,
    client: OpenAI,
    model: str,
    service_tier: str,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: str | None,
    verbosity: str | None,
    prompt_cache_key: str | None,
    store_response: bool,
    max_output_tokens: int,
    max_attempts: int,
    labels: list[str],
    text: str,
    prefix_context: str,
    suffix_context: str,
    segment: dict[str, Any],
    strict_span_quality: bool,
) -> dict[str, Any]:
    label_text = ", ".join(labels)
    guidance_text = "\n".join(f"- {label}: {LABEL_GUIDANCE[label]}" for label in labels)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        retry_note = (
            ""
            if not last_error
            else f"\nPrevious attempt failed validation: {last_error}\nTry again with exact text preservation for every label."
        )
        prefix_block = (
            "<<<PREFIX_CONTEXT>>>\n"
            f"{prefix_context}\n"
            "<<<END_PREFIX_CONTEXT>>>\n\n"
            if prefix_context
            else ""
        )
        suffix_block = (
            "\n\n<<<SUFFIX_CONTEXT>>>\n"
            f"{suffix_context}\n"
            "<<<END_SUFFIX_CONTEXT>>>"
            if suffix_context
            else ""
        )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": PER_LABEL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Requested labels: {label_text}\n"
                        f"Label definitions:\n{guidance_text}\n"
                        "For each label field, return the exact TARGET_SEGMENT with only that label's tags inserted. "
                        "For labels with no visible span, return the exact TARGET_SEGMENT unchanged. "
                        "Do not suppress paragraph labels just because sentence labels also apply, and do not suppress "
                        "sentence labels just because paragraph labels also apply."
                        f"{retry_note}\n\n"
                        f"{prefix_block}"
                        "<<<TARGET_SEGMENT>>>\n"
                        f"{text}\n"
                        "<<<END_TARGET_SEGMENT>>>"
                        f"{suffix_block}"
                    ),
                },
            ],
            "text_format": PerLabelTaggedAnnotation,
            "max_output_tokens": max_output_tokens,
            "service_tier": service_tier,
            "store": store_response,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        if verbosity is not None:
            request_kwargs["text"] = {"verbosity": verbosity}
        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = prompt_cache_key

        response = client.responses.parse(**request_kwargs)
        parsed = response.output_parsed
        if parsed is None:
            last_error = "empty structured response"
            continue
        try:
            tagged_text_by_label = normalize_per_label_tagged_texts(
                {
                    "sentence": parsed.sentence,
                    "paragraph": parsed.paragraph,
                    "section": parsed.section,
                    "dialogue": parsed.dialogue,
                    "list_item": parsed.list_item,
                    "metadata": parsed.metadata,
                },
                labels,
            )
            spans, validation = parse_per_label_tagged_texts(
                tagged_text_by_label,
                text,
                labels,
                segment,
                strict_span_quality=strict_span_quality,
            )
        except (TaggedTextValidationError, SpanQualityError) as exc:
            last_error = str(exc)
            continue
        return {
            "tagged_text": "",
            "tagged_text_by_label": tagged_text_by_label,
            "spans": spans,
            "validation": validation,
            "response_id": getattr(response, "id", None),
            "attempt": attempt,
            "notes": parsed.notes,
        }
    raise RuntimeError(f"annotation failed after {max_attempts} attempts: {last_error}")


def normalize_per_label_tagged_texts(
    tagged_text_by_label: dict[str, str],
    labels: list[str],
) -> dict[str, str]:
    expected = set(labels)
    actual = set(tagged_text_by_label)
    missing = sorted(expected - actual)
    if missing:
        raise TaggedTextValidationError(f"missing per-label tagged_text keys: {', '.join(missing)}")
    normalized = {}
    for label in labels:
        value = tagged_text_by_label[label]
        if not isinstance(value, str):
            raise TaggedTextValidationError(f"per-label tagged_text for {label} is not a string")
        normalized[label] = value
    return normalized


def parse_per_label_tagged_texts(
    tagged_text_by_label: dict[str, str],
    text: str,
    labels: list[str],
    segment: dict[str, Any],
    *,
    strict_span_quality: bool,
) -> tuple[list[SpanAnnotation], ValidationReport]:
    merged: list[SpanAnnotation] = []
    for label in labels:
        parsed_spans, validation = parse_tagged_text(
            tagged_text_by_label[label],
            text,
            {label},
        )
        if not validation.exact_roundtrip or not validation.stripped_text_matches:
            raise TaggedTextValidationError(f"{label} per-label text failed round-trip validation")
        spans = add_span_edge_flags(segment, parsed_spans)
        if strict_span_quality:
            validate_span_quality(segment, spans)
        merged.extend(spans)

    merged.sort(key=lambda span: (span.char_start, -span.char_end, span.label, span.id))
    renumbered = [
        span.model_copy(update={"id": index, "parent_id": None})
        for index, span in enumerate(merged)
    ]
    return renumbered, ValidationReport(
        protocol="per_label_inline_tags",
        exact_roundtrip=True,
        stripped_text_matches=True,
        well_nested=True,
        allowed_labels_only=True,
        span_count=len(renumbered),
    )


def write_error(
    handle: Any,
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    record_index: int,
    segment: dict[str, Any],
    error: Exception,
) -> None:
    if handle is None:
        return
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "source_identifier": str(row.get("identifier", f"row-{record_index}")),
        "record_index": record_index,
        "segment_char_start": segment["char_start"],
        "segment_char_end": segment["char_end"],
        "text": segment["text"],
        "error": repr(error),
    }
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")
    handle.flush()


if __name__ == "__main__":
    main()
