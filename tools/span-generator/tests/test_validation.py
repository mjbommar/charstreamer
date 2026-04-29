from charstreamer_span_generator.cli import (
    block_has_sentence_terminal,
    build_logical_block_units,
    is_sentence_eligible_block,
    looks_like_dialogue_line,
    looks_like_list_item,
    parse_args,
)
from charstreamer_span_generator.models import UnitLabelAssignment
from charstreamer_span_generator.validation import (
    SentenceBoundaryValidationError,
    TaggedTextValidationError,
    build_sentence_candidates,
    UnitAnnotationValidationError,
    build_line_units,
    parse_tagged_text,
    render_tagged_text,
    validate_sentence_breaks,
    validate_unit_annotations,
)


def test_render_and_parse_roundtrip_nested_spans() -> None:
    text = "Title\n\nHello world. \"Quoted line.\""
    spans = [
        {
            "id": 0,
            "label": "paragraph",
            "char_start": 0,
            "char_end": len(text),
            "parent_id": None,
        },
        {
            "id": 1,
            "label": "sentence",
            "char_start": 7,
            "char_end": 19,
            "parent_id": 0,
        },
        {
            "id": 2,
            "label": "dialogue",
            "char_start": 20,
            "char_end": len(text),
            "parent_id": 0,
        },
    ]
    from charstreamer_span_generator.models import SpanAnnotation

    rendered = render_tagged_text(
        text,
        [
            SpanAnnotation(start=0, end=0, **span)  # type: ignore[arg-type]
            for span in spans
        ],
    )
    parsed, validation = parse_tagged_text(
        rendered,
        text,
        {"paragraph", "sentence", "dialogue"},
    )
    assert validation.exact_roundtrip
    assert [span.label for span in parsed] == ["paragraph", "sentence", "dialogue"]


def test_parse_rejects_text_changes() -> None:
    text = "Original text."
    tagged_text = "<|sentence|>Changed text.<|/sentence|>"
    try:
        parse_tagged_text(tagged_text, text, {"sentence"})
    except TaggedTextValidationError as exc:
        assert "round-trip" in str(exc)
    else:
        raise AssertionError("expected deterministic validation failure")


def test_validate_unit_annotations_merges_adjacent_units() -> None:
    text = "HEADING\nFirst body line\nSecond body line\n\nSIGNED\n"
    units = build_line_units(text)
    assignments = [
        UnitLabelAssignment(unit_id=0, label="section_heading"),
        UnitLabelAssignment(unit_id=1, label="paragraph"),
        UnitLabelAssignment(unit_id=2, label="paragraph"),
        UnitLabelAssignment(unit_id=3, label="none"),
        UnitLabelAssignment(unit_id=4, label="metadata"),
    ]
    spans, validation = validate_unit_annotations(
        assignments,
        units,
        {"section_heading", "paragraph", "metadata"},
    )
    assert validation.protocol == "unit_labels"
    assert validation.unit_coverage_complete
    assert [span.label for span in spans] == ["section_heading", "paragraph", "metadata"]
    assert spans[1].char_start == units[1].char_start
    assert spans[1].char_end == units[2].char_end


def test_validate_unit_annotations_rejects_missing_unit() -> None:
    units = build_line_units("A\nB\n")
    assignments = [UnitLabelAssignment(unit_id=0, label="paragraph")]
    try:
        validate_unit_annotations(assignments, units, {"paragraph"})
    except UnitAnnotationValidationError as exc:
        assert "missing unit labels" in str(exc)
    else:
        raise AssertionError("expected missing-unit validation failure")


def test_validate_sentence_breaks_builds_sentences_from_candidates() -> None:
    from charstreamer_span_generator.models import SpanAnnotation

    text = "First sentence. Second sentence. Final sentence."
    parent = SpanAnnotation(
        id=0,
        label="paragraph",
        start=0,
        end=len(text.encode("utf-8")),
        char_start=0,
        char_end=len(text),
        parent_id=None,
    )
    candidates = build_sentence_candidates(
        text,
        parent_span_id=parent.id,
        char_start=parent.char_start,
        byte_start=parent.start,
    )
    spans, next_span_id = validate_sentence_breaks(
        parent_span=parent,
        text=text,
        candidate_break_ids=[candidates[0].candidate_id, candidates[1].candidate_id],
        candidates=candidates,
        next_span_id=1,
    )
    assert next_span_id == 4
    assert [span.label for span in spans] == ["sentence", "sentence", "sentence"]
    assert [text[span.char_start:span.char_end] for span in spans] == [
        "First sentence.",
        "Second sentence.",
        "Final sentence.",
    ]
    assert all(span.parent_id == 0 for span in spans)


def test_validate_sentence_breaks_rejects_unknown_candidate() -> None:
    from charstreamer_span_generator.models import SpanAnnotation

    text = "A short sentence. Another one."
    parent = SpanAnnotation(
        id=0,
        label="paragraph",
        start=0,
        end=len(text.encode("utf-8")),
        char_start=0,
        char_end=len(text),
        parent_id=None,
    )
    candidates = build_sentence_candidates(
        text,
        parent_span_id=parent.id,
        char_start=parent.char_start,
        byte_start=parent.start,
    )
    try:
        validate_sentence_breaks(
            parent_span=parent,
            text=text,
            candidate_break_ids=[999],
            candidates=candidates,
            next_span_id=1,
        )
    except SentenceBoundaryValidationError as exc:
        assert "unknown sentence candidate_id" in str(exc)
    else:
        raise AssertionError("expected sentence-boundary validation failure")


def test_sentence_candidates_skip_legal_abbreviations() -> None:
    text = "Under 42 U.S.C. § 405(g) the matter is remanded. Another sentence follows."
    candidates = build_sentence_candidates(
        text,
        parent_span_id=0,
        char_start=0,
        byte_start=0,
    )
    markers = [candidate.marker for candidate in candidates]
    assert len(candidates) == 1
    assert "<<<BREAK>>>" in markers[0]
    assert "remanded." in markers[0]


def test_sentence_eligible_block_rejects_subject_lines() -> None:
    assert not is_sentence_eligible_block(
        "Относно: Взаимоотношения между ЕС и Туркменистан",
        label="paragraph",
        min_block_chars=20,
        min_alpha_chars=10,
    )
    assert is_sentence_eligible_block(
        "This is a normal body paragraph with enough lowercase text to count as a sentence-bearing block.",
        label="paragraph",
        min_block_chars=20,
        min_alpha_chars=10,
    )


def test_block_has_sentence_terminal_rejects_colon_and_entity_suffix() -> None:
    assert not block_has_sentence_terminal(
        "For the purpose of this Decision the following definitions shall apply:&#xD;\n"
    )
    assert block_has_sentence_terminal("This is a complete sentence.&#xD;\n")


def test_build_logical_block_units_merges_wrapped_paragraph_lines() -> None:
    text = (
        "IN THE COURT OF APPEALS\n"
        "NO. 2018-CA-00436-COA\n"
        "\n"
        "¶1. Devonta Pipkin pleaded guilty to deliberate-design murder and was sentenced to life\n"
        "in the custody of the Mississippi Department of Corrections. He filed a petition.\n"
        "\n"
        "Subject: Request for production\n"
    )
    units = build_logical_block_units(text)
    assert [unit.kind for unit in units] == ["block", "block", "block", "block"]
    assert units[0].text == "IN THE COURT OF APPEALS\n"
    assert units[1].text == "NO. 2018-CA-00436-COA\n"
    assert "in the custody of the Mississippi Department of Corrections." in units[2].text
    assert units[3].text == "Subject: Request for production\n"


def test_build_logical_block_units_keeps_numbered_items_separate() -> None:
    text = "(2)First item.\n(3)Second item.\n(4)Third item.\n"
    units = build_logical_block_units(text)
    assert len(units) == 3
    assert units[0].text == "(2)First item.\n"
    assert units[1].text == "(3)Second item.\n"
    assert units[2].text == "(4)Third item.\n"


def test_looks_like_list_item_handles_no_space_after_marker() -> None:
    assert looks_like_list_item("(3)The Commission is committed to enhance transparency.")
    assert looks_like_list_item("1.The information shall be published.")
    assert looks_like_list_item("¶2. The trial court denied relief.")
    assert looks_like_list_item("‘Director-General’ means Director-General or Head of Service.")


def test_looks_like_dialogue_line_requires_actual_dialogue_shape() -> None:
    assert looks_like_dialogue_line('"Hello there."')
    assert looks_like_dialogue_line("SPEAKER: Hello there.")
    assert not looks_like_dialogue_line(
        'EyePoint Pharmaceuticals, Inc. (“EyePoint”) and ImprimisRx, LLC (“Imprimis”) entered into an agreement.'
    )


def test_legacy_generator_exposes_max_chunks_per_doc() -> None:
    args = parse_args(["--output", "/tmp/out.jsonl", "--max-chunks-per-doc", "2"])

    assert args.max_chunks_per_doc == 2
