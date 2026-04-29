import pytest
from random import Random

from charstreamer_span_generator.models import SpanAnnotation
from charstreamer_span_generator.simple import parse_args
from charstreamer_span_generator.simple import (
    SpanQualityError,
    add_span_edge_flags,
    choose_labels,
    choose_sample_focus_label,
    parse_per_label_tagged_texts,
    sample_segment,
    validate_span_quality,
)


def test_simple_generator_keeps_empty_span_rows_by_default() -> None:
    args = parse_args(["--output", "/tmp/annotations.jsonl"])

    assert args.require_spans is False
    assert args.min_chars == 50
    assert args.max_chars == 1000
    assert args.annotation_protocol == "per-label"
    assert args.label_strategy == "all"
    assert args.sample_focus_strategy == "round-robin"
    assert args.max_output_tokens == 12000


def test_simple_generator_can_require_positive_spans_for_qa() -> None:
    args = parse_args(["--output", "/tmp/annotations.jsonl", "--require-spans"])

    assert args.require_spans is True


def test_simple_generator_exposes_llm_parameters() -> None:
    args = parse_args(
        [
            "--output",
            "/tmp/annotations.jsonl",
            "--temperature",
            "0.2",
            "--top-p",
            "0.95",
            "--reasoning-effort",
            "medium",
            "--verbosity",
            "medium",
            "--service-tier",
            "priority",
            "--prompt-cache-key",
            "charstreamer-span-v1",
            "--edge-jitter-probability",
            "0.25",
            "--store-response",
        ]
    )

    assert args.temperature == 0.2
    assert args.top_p == 0.95
    assert args.reasoning_effort == "medium"
    assert args.verbosity == "medium"
    assert args.service_tier == "priority"
    assert args.prompt_cache_key == "charstreamer-span-v1"
    assert args.edge_jitter_probability == 0.25
    assert args.store_response is True


def test_sample_segment_can_select_natural_targets() -> None:
    text = "Short.\n\nThis is a complete sentence. This is another complete sentence.\n\nTail."
    segment = sample_segment(
        text,
        Random(7),
        min_chars=40,
        max_chars=90,
        context_chars=12,
        segment_mode="natural",
    )

    assert segment is not None
    assert segment["mode"] == "natural"
    assert segment["text"] == "This is a complete sentence. This is another complete sentence."


def test_choose_labels_round_robins_focus_labels() -> None:
    labels = ["sentence", "paragraph", "metadata"]
    first_labels, first_focus = choose_labels(
        labels=labels,
        target_labels=labels,
        label_strategy="round-robin",
        sample_index=0,
        rng=Random(7),
    )
    second_labels, second_focus = choose_labels(
        labels=labels,
        target_labels=labels,
        label_strategy="round-robin",
        sample_index=1,
        rng=Random(7),
    )

    assert first_labels == ["sentence"]
    assert first_focus == "sentence"
    assert second_labels == ["paragraph"]
    assert second_focus == "paragraph"


def test_choose_sample_focus_round_robins_independently() -> None:
    labels = ["sentence", "paragraph", "metadata"]

    assert (
        choose_sample_focus_label(
            target_labels=labels,
            sample_focus_strategy="round-robin",
            sample_index=4,
            rng=Random(7),
        )
        == "paragraph"
    )
    assert (
        choose_sample_focus_label(
            target_labels=labels,
            sample_focus_strategy="none",
            sample_index=4,
            rng=Random(7),
        )
        is None
    )


def test_per_label_tagged_texts_allow_overlapping_sentence_paragraph() -> None:
    text = "This is one sentence. This is another sentence."
    segment = {"text": text, "prefix_context": "", "suffix_context": ""}
    spans, validation = parse_per_label_tagged_texts(
        {
            "paragraph": f"<|paragraph|>{text}<|/paragraph|>",
            "sentence": (
                "<|sentence|>This is one sentence.<|/sentence|> "
                "<|sentence|>This is another sentence.<|/sentence|>"
            ),
        },
        text,
        ["paragraph", "sentence"],
        segment,
        strict_span_quality=True,
    )

    assert validation.protocol == "per_label_inline_tags"
    assert [span.label for span in spans] == ["paragraph", "sentence", "sentence"]
    assert spans[0].char_start == 0
    assert spans[0].char_end == len(text)


def test_sample_segment_can_select_metadata_focused_targets() -> None:
    text = (
        "Normal body text that should not be selected first.\n"
        "Case No. 1:24-cv-12345\n"
        "Filed April 28, 2026\n"
        "More body text follows with complete sentences."
    )
    segment = sample_segment(
        text,
        Random(7),
        min_chars=40,
        max_chars=120,
        context_chars=12,
        segment_mode="mixed",
        focus_label="metadata",
        edge_jitter_probability=0.0,
    )

    assert segment is not None
    assert segment["mode"] == "focus:metadata"
    assert "Case No." in segment["text"] or "Filed" in segment["text"]


def test_sample_segment_can_select_paragraph_focused_targets() -> None:
    text = (
        "CASE NO. 123\n\n"
        "This is a complete body paragraph with enough ordinary prose to be useful. "
        "It has two complete sentences.\n\n"
        "1. This is an enumerated item that should not be chosen as a paragraph."
    )
    segment = sample_segment(
        text,
        Random(7),
        min_chars=80,
        max_chars=140,
        context_chars=12,
        segment_mode="mixed",
        focus_label="paragraph",
        edge_jitter_probability=0.0,
    )

    assert segment is not None
    assert segment["mode"] == "focus:paragraph"
    assert segment["text"].startswith("This is a complete body paragraph")


def test_sample_segment_can_select_list_item_focused_targets() -> None:
    text = (
        "Intro paragraph before the list.\n\n"
        "1. This is a complete enumerated list item with enough text to satisfy the sampling minimum.\n"
        "2. This is another complete enumerated list item with enough text to satisfy the sampling minimum."
    )
    segment = sample_segment(
        text,
        Random(7),
        min_chars=60,
        max_chars=120,
        context_chars=12,
        segment_mode="mixed",
        focus_label="list_item",
        edge_jitter_probability=0.0,
    )

    assert segment is not None
    assert segment["mode"] == "focus:list_item"
    assert segment["text"].startswith(("1.", "2."))


def test_span_quality_rejects_sentence_without_terminal_punctuation() -> None:
    segment = {
        "text": "This sentence is not complete",
        "prefix_context": "",
        "suffix_context": " because more text follows",
    }
    span = SpanAnnotation(
        id=0,
        label="sentence",
        start=0,
        end=len(segment["text"]),
        char_start=0,
        char_end=len(segment["text"]),
        parent_id=None,
    )

    with pytest.raises(SpanQualityError, match="terminal"):
        validate_span_quality(segment, [span])


def test_span_quality_allows_right_open_sentence_continuation() -> None:
    segment = {
        "text": "This sentence is not complete",
        "prefix_context": "",
        "suffix_context": " because more text follows",
    }
    span = SpanAnnotation(
        id=0,
        label="sentence",
        start=0,
        end=len(segment["text"]),
        char_start=0,
        char_end=len(segment["text"]),
        parent_id=None,
    )

    [span] = add_span_edge_flags(segment, [span])

    assert span.left_open is False
    assert span.right_open is True
    validate_span_quality(segment, [span])


def test_span_quality_allows_left_open_paragraph_continuation() -> None:
    text = "continued paragraph text inside an arbitrary streaming window."
    segment = {
        "text": text,
        "prefix_context": "This paragraph started before the target ",
        "suffix_context": "\n\nNext paragraph.",
    }
    span = SpanAnnotation(
        id=0,
        label="paragraph",
        start=0,
        end=len(text),
        char_start=0,
        char_end=len(text),
        parent_id=None,
    )

    [span] = add_span_edge_flags(segment, [span])

    assert span.left_open is True
    assert span.right_open is False
    validate_span_quality(segment, [span])


def test_span_quality_rejects_sentence_after_non_boundary_context() -> None:
    text = "continued text. A complete sentence."
    segment = {
        "text": text,
        "prefix_context": "This is",
        "suffix_context": "",
    }
    span = SpanAnnotation(
        id=0,
        label="sentence",
        start=0,
        end=len("continued text."),
        char_start=0,
        char_end=len("continued text."),
        parent_id=None,
    )

    with pytest.raises(SpanQualityError, match="non-boundary"):
        validate_span_quality(segment, [span])


def test_span_quality_rejects_unbalanced_quote_sentence() -> None:
    text = "The witness said “this sentence is not complete..."
    segment = {
        "text": text,
        "prefix_context": "",
        "suffix_context": " and the quote continues",
    }
    span = SpanAnnotation(
        id=0,
        label="sentence",
        start=0,
        end=len(text),
        char_start=0,
        char_end=len(text),
        parent_id=None,
    )

    with pytest.raises(SpanQualityError, match="quotes"):
        validate_span_quality(segment, [span])


def test_span_quality_rejects_partial_paragraph_at_right_edge() -> None:
    text = "A full-looking paragraph that actually continues"
    segment = {
        "text": text,
        "prefix_context": "\n\n",
        "suffix_context": " after the sampled target",
    }
    span = SpanAnnotation(
        id=0,
        label="paragraph",
        start=0,
        end=len(text),
        char_start=0,
        char_end=len(text),
        parent_id=None,
    )

    with pytest.raises(SpanQualityError, match="paragraph"):
        validate_span_quality(segment, [span])
