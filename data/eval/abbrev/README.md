# Abbreviation eval suite

Held-out evaluation set for the sentence-boundary model focused on
abbreviation contexts (titles, citations, decimals, place suffixes,
times, etc.). Used to measure the abbreviation-handling quality of
charstreamer's default sentence model and to tune its decision threshold.

## Files

- `eval.jsonl` (94 cases) — full held-out abbreviation suite. Used for headline
  F1 reporting. **Never used for training**; for threshold tuning, use
  `threshold_tuning.jsonl` (which is one half of this set in trainer-form).
- `threshold_tuning.jsonl` (47 cases, trainer schema) — alternating IDs sorted
  from `eval.jsonl`, materialized as a JSONL with explicit `sentence` spans.
  Pass to `train_sentence_burn` via `--threshold-eval` so the manifest
  threshold is tuned on a representative slice.
- `measure.jsonl` (47 cases, gold-marker schema) — the other alternating half
  of `eval.jsonl`. Use this for **release-time held-out** measurement so the
  reported F1 is on data the threshold-tuner never saw.
- `_split.py` — reproduces the split. Re-run if you ever change `eval.jsonl`.

## Schema

Cases use a "gold marker" schema where the gold sentence-end positions are
implied by substring markers within the text (avoids hand-counting byte
offsets):

```jsonl
{"id": "title-dr-1", "text": "Dr. Smith met with the patient. He was concerned.", "gold_marker": "the patient."}
{"id": "abbrev-end-1", "text": "He bought apples, oranges, pears, etc.", "gold_marker": null}
{"id": "clean-1", "text": "He arrived early. She arrived late. They left.", "gold_marker": "He arrived early.;She arrived late."}
```

- `text` — input text.
- `gold_marker` — `null` if there is **no internal break** (text is one
  sentence). Otherwise a `;`-separated list of substrings; the gold sentence
  end byte for each is `text.index(marker) + len(marker.encode("utf-8"))`.

The `threshold_tuning.jsonl` file is the same content materialized into the
trainer's standard schema (`{"text": ..., "spans": [{"label": "sentence",
...}, ...]}`).

## Usage

Eval at the command line:

```bash
CHARSTREAMER_EVAL_SUITE=data/eval/abbrev/eval.jsonl \
  python logs/abbrev-research/run_eval.py --model-dir <model-dir>
```

Tune threshold at training time:

```bash
target/release/examples/train_sentence_burn \
    --threshold-eval data/eval/abbrev/threshold_tuning.jsonl \
    ...
```

Held-out measurement:

```bash
CHARSTREAMER_EVAL_SUITE=data/eval/abbrev/measure.jsonl \
  python logs/abbrev-research/run_eval.py --model-dir <model-dir>
```

## Coverage

The 94 cases are organized roughly:

- Title abbreviations (Dr., Mr., Mrs., Ms., Prof., Rev., Jr., Sr.) — both
  abbrev-mid-sentence and abbrev-end-of-sentence forms.
- Corporate (Inc., Ltd., Co., Corp.).
- Place (St. Louis, addresses ending in St./Ave./Blvd.).
- Acronyms (U.S., U.K., E.U., M.D., Ph.D., M.S., B.A., LL.M., D.D.S.).
- Citations (`v.`, `Cf.`, `id.`, `e.g.`, `i.e.`).
- Decimals and version numbers (`1.2.3`, `4.5.6`, prices like `$19.99`).
- Time (`a.m.`, `p.m.`).
- URLs (`www.example.com`).
- Month and day abbreviations (`Jan.`, `Wed.`).
- Quoted dialogue, parenthesized sentences, exclamations, questions.
- Plain-prose controls (no abbreviations).

The lexicon used here is **disjoint from** `tools/abbrev-augment/`'s synthetic
generator; gains on this set therefore reflect generalization rather than
memorization of the training distribution.
