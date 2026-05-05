"""Smoke tests for the synthetic generator."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from charstreamer_abbrev_augment import build_record, TEMPLATES


def test_template_returns_valid_record() -> None:
    rng = random.Random(0)
    for template in TEMPLATES:
        text, sentences = template(rng)
        assert isinstance(text, str)
        assert sentences
        for sent in sentences:
            assert sent in text, f"{template.__name__}: sentence {sent!r} not in {text!r}"


def test_build_record_spans_are_byte_aligned() -> None:
    rng = random.Random(1)
    for _ in range(200):
        rec = build_record(rng)
        text_bytes = rec["text"].encode("utf-8")
        for sp in rec["spans"]:
            slice_bytes = text_bytes[sp["start"]:sp["end"]]
            slice_str = slice_bytes.decode("utf-8")
            assert slice_str.strip() == slice_str
            assert slice_str in rec["text"]


def test_build_record_is_deterministic_for_seed() -> None:
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    a = build_record(rng_a)
    b = build_record(rng_b)
    assert a == b


def test_jsonl_record_serializes_round_trip() -> None:
    rng = random.Random(7)
    rec = build_record(rng)
    line = json.dumps(rec)
    parsed = json.loads(line)
    assert parsed == rec
    assert parsed["text"]
    assert parsed["spans"]
    for sp in parsed["spans"]:
        assert sp["label"] == "sentence"
        assert sp["start"] < sp["end"]
        assert sp["right_open"] is False
