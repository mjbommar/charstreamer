"""Split eval_suite_v2 into a threshold-tuning half and a measurement half.

Strategy: alternate IDs (sorted) so both halves cover the same diversity.
Output:
  eval_suite_v2.thresh.train.jsonl — for trainer --threshold-eval
  eval_suite_v2.measure.jsonl     — for held-out measurement (run_eval --suite ...)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "eval_suite_v2.jsonl"

cases = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
cases.sort(key=lambda c: c["id"])
thresh = cases[::2]
measure = cases[1::2]


def to_train_record(rec):
    text = rec["text"]
    marker = rec.get("gold_marker")
    breaks = []
    if marker:
        for m in marker.split(";"):
            m = m.strip()
            if not m:
                continue
            idx = text.index(m)
            breaks.append(idx + len(m.encode("utf-8")))
    breaks = sorted(set(breaks))
    boundaries = [0] + breaks + [len(text.encode("utf-8"))]
    spans = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        sub = text[s:e]
        lead = len(sub) - len(sub.lstrip())
        trail = len(sub) - len(sub.rstrip())
        s2 = s + lead
        e2 = e - trail
        if e2 <= s2:
            continue
        spans.append({"label": "sentence", "start": s2, "end": e2,
                      "char_start": s2, "char_end": e2, "right_open": False})
    return {"text": text, "spans": spans}


# threshold half — train form (with sentence spans)
(HERE / "eval_suite_v2.thresh.train.jsonl").write_text(
    "\n".join(json.dumps(to_train_record(r)) for r in thresh) + "\n"
)
# measurement half — eval form (gold_marker)
(HERE / "eval_suite_v2.measure.jsonl").write_text(
    "\n".join(json.dumps(r) for r in measure) + "\n"
)
print(f"thresh: {len(thresh)}  measure: {len(measure)}")
