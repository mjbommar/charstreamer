# charstreamer-silver-corpus

Silver-label real-world text (Federal Register, PubMed Central, Project
Gutenberg) for charstreamer's sentence-boundary trainer. Uses nupunkt with a
small sanity filter that suppresses its known false-positive patterns.

## Run

```bash
python -m charstreamer_silver_corpus \
    --n-fedreg 40 --n-pmc 30 \
    --out data/synthetic/realtext_silver/silver_v1.jsonl
```

Default: 40 Federal Register documents (latest) + 30 PMC open-access articles
(latest) + 4 Project Gutenberg books (1342, 98, 84, 1661). Output schema
matches the trainer's existing JSONL format (`text` + sentence `spans`).

The corpus is mildly nondeterministic: the Federal Register and PMC APIs
return latest documents, so the corpus changes day to day.

## Sanity filter

The filter strips nupunkt's specific systematic FPs:

- Roman numeral section markers (`I.`, `II.`, `IV.`)
- `Sec.` followed by section numbers (in glossary tables)
- `U.S.`, `U.K.`, `U.S.C.`, `C.F.R.` mid-name acronyms
- Decimal contexts (digit-period-digit)
- Single-digit enumerated items (`1.`, `2.`)
- Common title abbreviations (Mr./Mrs./Dr./Ph.D./...)
- Typesetting markers (`[`, `_`, `*`) immediately after the period

What's left is high-quality silver labels matching real-world distribution.

## Used by

The 0.1.6 default model was trained on the union of:

- `data/synthetic/kl3m_streaming_spans_20260428_10k.jsonl` (existing)
- `data/synthetic/abbrev_augment/abbrev_aug_25k.jsonl` (templated, see
  `tools/abbrev-augment/`)
- `data/synthetic/realtext_silver/silver_v1.jsonl` (this tool, 202 records,
  2.8 MB)

The silver corpus closed the real-text F1 gap to nupunkt — see `CHANGELOG.md`
0.1.6 for measurements.
