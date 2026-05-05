# charstreamer-abbrev-augment

Template-based synthetic training data generator for charstreamer's sentence
boundary model. Used to augment the corpus with abbreviation-aware contexts
(titles, citations, decimals, etc.) that the model would otherwise see too
rarely to learn from.

Lexicons are intentionally disjoint from any eval suite shipped under
`data/eval/` so that improvements measured on the held-out eval reflect
generalization rather than memorization.

## Run

```bash
python -m charstreamer_abbrev_augment \
    --n 8000 --seed 12345 \
    --out data/synthetic/abbrev_augment/abbrev_v4_8k.jsonl
```

The output JSONL schema matches `data/synthetic/kl3m_streaming_spans_*.jsonl`
and feeds directly into `train_sentence_burn`:

```bash
target/release/examples/train_sentence_burn \
    --input data/synthetic/kl3m_streaming_spans_20260428_10k.jsonl \
    --input data/synthetic/abbrev_augment/abbrev_v4_8k.jsonl \
    --token-shape-features --terminal-keep-rate 0.10 \
    --threshold-eval data/eval/abbrev/threshold_tuning.jsonl \
    --out dist-models/charstreamer-default-0.1.5
```

## Reproducing the 0.1.5 default-model augment data

```bash
mkdir -p data/synthetic/abbrev_augment
python -m charstreamer_abbrev_augment --n 5000 --seed 12345 \
    --out data/synthetic/abbrev_augment/abbrev_v1_5k.jsonl
python -m charstreamer_abbrev_augment --n 6000 --seed 12345 \
    --out data/synthetic/abbrev_augment/abbrev_v2_6k.jsonl
# (etc. — see CHANGELOG for full list)
```

For the canonical recipe and seed list see `CHANGELOG.md` and
`docs/quality/release-gates.md`.

## Extending

Add abbreviations or contexts to the lexicon constants (`TITLE_DR`, `CORP`,
`PLACE_ST_NAMED`, etc.) near the top of `__init__.py`. Add new templates as
plain functions returning `(text, [sentence_text, ...])`; register them in the
`TEMPLATES` list. Pure stdlib; no other dependencies.

The generator is deterministic given `--seed`. Reuse seed 12345 for the
canonical augment files; pick a different seed (e.g. 999) for held-out
threshold-tuning slices.
