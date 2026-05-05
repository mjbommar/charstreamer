"""Unit tests for the eval helpers (don't require the charstreamer wheel)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from charstreamer_abbrev_eval import evaluate_segmenter, gold_breaks_from_marker


def test_gold_breaks_handles_none() -> None:
    assert gold_breaks_from_marker("Mr. Jones arrived.", None) == []
    assert gold_breaks_from_marker("Mr. Jones arrived.", "") == []


def test_gold_breaks_resolves_single_marker() -> None:
    text = "Mr. Smith arrived. He was tired."
    breaks = gold_breaks_from_marker(text, "Mr. Smith arrived.")
    assert breaks == [18]


def test_gold_breaks_resolves_multiple_markers() -> None:
    text = "He arrived. She left. They met."
    breaks = gold_breaks_from_marker(text, "He arrived.;She left.")
    assert breaks == [11, 21]


class _FakeSegmenter:
    def __init__(self, predictions: dict[str, list[int]]):
        self._preds = predictions

    def annotate(self, text: str) -> dict:
        ends = list(self._preds.get(text, [])) + [len(text.encode("utf-8"))]
        spans = []
        cursor = 0
        for end in ends:
            if end <= cursor:
                continue
            spans.append({"label": "sentence", "start": cursor, "end": end, "score": 1.0})
            cursor = end + 1
        return {"spans": spans, "tagged": text}


def test_perfect_segmenter_scores_one() -> None:
    suite = [
        {"id": "a", "text": "Hi. There.", "gold_marker": "Hi."},
    ]
    seg = _FakeSegmenter({"Hi. There.": [3]})
    res = evaluate_segmenter(seg, suite)
    assert res["tp"] == 1
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["f1"] == 1.0


def test_oversplitter_has_low_precision() -> None:
    suite = [
        {"id": "a", "text": "Mr. Jones is here.", "gold_marker": None},
    ]
    seg = _FakeSegmenter({"Mr. Jones is here.": [3]})
    res = evaluate_segmenter(seg, suite)
    assert res["tp"] == 0
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["precision"] == 0.0


def test_undersplitter_has_low_recall() -> None:
    suite = [
        {"id": "a", "text": "He arrived. She left.", "gold_marker": "He arrived."},
    ]
    seg = _FakeSegmenter({"He arrived. She left.": []})
    res = evaluate_segmenter(seg, suite)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 1
    assert res["recall"] == 0.0
