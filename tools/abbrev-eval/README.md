# charstreamer-abbrev-eval

F1 regression evaluator for charstreamer's sentence-segmentation model on the
canonical abbreviation eval suite (`data/eval/abbrev/eval.jsonl`).

## Usage

```bash
# Default suite, default bundled model, report only
python -m charstreamer_abbrev_eval

# Custom suite, custom model, fail if F1 below floor
python -m charstreamer_abbrev_eval \
    --suite data/eval/abbrev/measure.jsonl \
    --model-dir dist-models/charstreamer-default-0.1.5 \
    --min-f1 0.90 \
    --json
```

Exit code is non-zero if `--min-f1` is set and the actual F1 is below it.

`charstreamer` is imported lazily inside the entry point; the unit tests in
`tests/` use a fake segmenter and do not require the wheel installed.

## CI gate

Add to `.github/workflows/release.yml` (after the wheel is installed):

```yaml
- name: Abbreviation regression
  run: |
    uv pip install ./tools/abbrev-eval
    python -m charstreamer_abbrev_eval --min-f1 0.90 --json
```

The eval suite is intentionally **disjoint** from the abbreviation augment
generator (`tools/abbrev-augment/`), so this F1 measures generalization rather
than memorization.

## Schema

See `data/eval/abbrev/README.md` for the gold-marker schema.
