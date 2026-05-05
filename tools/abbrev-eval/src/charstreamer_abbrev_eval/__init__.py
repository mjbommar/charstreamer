"""F1 regression evaluator for charstreamer's sentence-segmentation model.

Loads a charstreamer ``Segmenter``, runs it against ``data/eval/abbrev/eval.jsonl``
(or any compatible JSONL with the gold-marker schema), and reports
precision/recall/F1 against the implied sentence-end byte offsets.

Usage::

    python -m charstreamer_abbrev_eval --suite data/eval/abbrev/eval.jsonl
    python -m charstreamer_abbrev_eval --min-f1 0.90        # exit non-zero if below

The evaluator is deterministic and fast (~1 second on 100 cases). Suitable as
a CI gate after the charstreamer wheel has been installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def gold_breaks_from_marker(text: str, marker: str | None) -> list[int]:
    """Return byte offsets of internal sentence ends, derived from ``marker``.

    ``marker`` is a ``;``-separated list of substrings; for each, the sentence-end
    byte is ``text.index(m) + len(m.encode("utf-8"))``. ``None`` or an empty
    marker means the input is a single sentence (no internal breaks).
    """
    if not marker:
        return []
    breaks: list[int] = []
    for m in marker.split(";"):
        m = m.strip()
        if not m:
            continue
        idx = text.index(m)
        breaks.append(idx + len(m.encode("utf-8")))
    return sorted(set(breaks))


def predicted_breaks(text: str, segmenter: Any) -> list[int]:
    """Internal sentence-end byte offsets from a charstreamer Segmenter result.

    The last span's end is the natural end of text, which is *not* a "break"
    in our sense — only internal ends count.
    """
    annotation = segmenter.annotate(text)
    if hasattr(annotation, "spans"):
        spans = annotation.spans
    elif isinstance(annotation, Mapping):
        spans = annotation["spans"]
    else:
        spans = list(annotation)

    sentence_spans = [
        sp for sp in spans
        if (getattr(sp, "label", None) or sp["label"]) == "sentence"
    ]
    sentence_spans.sort(
        key=lambda sp: getattr(sp, "start", None) if hasattr(sp, "start") else sp["start"]
    )
    breaks: list[int] = []
    for sp in sentence_spans[:-1]:  # last span: natural end of text
        end = getattr(sp, "end", None) if hasattr(sp, "end") else sp["end"]
        breaks.append(int(end))
    return breaks


def load_suite(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def evaluate_segmenter(segmenter: Any, suite: Iterable[dict]) -> dict:
    """Evaluate a Segmenter against the gold-marker JSONL schema."""
    tp = fp = fn = 0
    failures: list[dict] = []
    for case in suite:
        text = case["text"]
        gold = set(gold_breaks_from_marker(text, case.get("gold_marker")))
        pred = set(predicted_breaks(text, segmenter))
        case_tp = len(gold & pred)
        case_fp = len(pred - gold)
        case_fn = len(gold - pred)
        tp += case_tp
        fp += case_fp
        fn += case_fn
        if case_fp or case_fn:
            failures.append({
                "id": case.get("id"),
                "fp": sorted(pred - gold),
                "fn": sorted(gold - pred),
            })
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="charstreamer-abbrev-eval")
    parser.add_argument(
        "--suite",
        default=str(Path("data/eval/abbrev/eval.jsonl")),
        help="Path to gold-marker JSONL eval suite. Default: data/eval/abbrev/eval.jsonl.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("CHARSTREAMER_MODEL_PATH"),
        help="Optional model bundle directory. Default: bundled wheel model.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="Exit with non-zero status if F1 < min-f1. Default: report only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (suitable for CI).",
    )
    args = parser.parse_args(argv)

    if args.model_dir:
        os.environ["CHARSTREAMER_MODEL_PATH"] = args.model_dir

    import charstreamer
    segmenter = charstreamer.Segmenter()

    suite = load_suite(args.suite)
    result = evaluate_segmenter(segmenter, suite)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"suite={args.suite} cases={len(suite)} "
            f"f1={result['f1']:.4f} p={result['precision']:.4f} r={result['recall']:.4f} "
            f"tp={result['tp']} fp={result['fp']} fn={result['fn']}"
        )
        if result["failures"]:
            print(f"\nfailures ({len(result['failures'])} cases):")
            for f in result["failures"]:
                print(f"  [{f['id']}] fp={f['fp']} fn={f['fn']}")

    if args.min_f1 is not None and result["f1"] < args.min_f1:
        print(f"\nFAIL: f1 {result['f1']:.4f} < min-f1 {args.min_f1:.4f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
