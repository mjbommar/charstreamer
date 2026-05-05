# Changelog

## 0.1.5 - Abbreviation-Aware Sentence Model

This release retrains the default sentence-boundary model with structural
token-shape features and abbreviation-rich synthetic data. The primary fix is
sharply reduced over-splitting on common abbreviations
(`Dr.`, `Mr.`, `Mrs.`, `U.S.`, `Cf.`, decimals like `1.2.3`, `a.m.`/`p.m.`,
month and day abbreviations, `Inc.`/`Ltd.`/`Co.`, `St. Louis`, etc.) while
preserving recall on plain prose.

Included:

- new `BurnSentenceFeatureConfig.token_shape_features` boolean config (default
  `false`); when enabled, the kernel adds a 12-dimensional `TokenShapeAppender`
  block emitting purely positional structural signals (decimal context,
  preceding-token alpha length and capitalization, internal-period count,
  next-token shape, paragraph-break crossing). Bundles trained without the flag
  remain valid and load unchanged.
- new trainer flags in
  `crates/charstreamer-segmentation/examples/train_sentence_burn.rs`:
  - `--token-shape-features` enables the new appender
  - `--terminal-keep-rate R` controls negative-sample keep rate at `.`/`!`/`?`
    positions independently from `--negative-keep-rate`
  - `--threshold-eval PATH` tunes the manifest threshold on a held-out JSONL
    slice instead of the random validation split
- new `tools/abbrev-augment/` package: pure-stdlib template-based generator for
  abbreviation-rich synthetic training data, deterministic given `--seed`,
  invokable via `python -m charstreamer_abbrev_augment`
- new `tools/abbrev-eval/` package: F1 regression evaluator for the canonical
  eval suite, invokable via `python -m charstreamer_abbrev_eval --min-f1 0.90`
- new `data/eval/abbrev/` canonical evaluation suite: 94 cases covering titles,
  citations, decimals, addresses, time, acronyms, and plain-prose controls,
  plus a 47/47 train/measure split for held-out F1 reporting
- new `data/synthetic/abbrev_augment/abbrev_aug_25k.jsonl` (25k records)
  generated from the new generator and used in the default-model training mix
- structural Rust unit tests for `TokenShapeAppender` in
  `crates/charstreamer-kernels`
- pytest smoke tests for the two new tool packages

Headline metrics on the held-out abbreviation suite
(`data/eval/abbrev/measure.jsonl`, 47 cases never used in training or
threshold tuning):

| | precision | recall | F1 |
|---|---|---|---|
| previous default (0.1.4) | 0.467 | 0.969 | 0.625 |
| new default (0.1.5) | 0.857 | 0.960 | 0.906 |

Plain-prose regression suite (9 cases, no abbreviations): F1 0.976 → 1.000.

Known status:

- the default semantic structure model (`paragraph`, `metadata`, `section`,
  `list_item`) is unchanged from 0.1.2
- `dialogue` remains reserved until a balanced training set exists
- a small number of dense long-abbreviation lists ("Dr. Adams, Mr. Brown,
  Mrs. Cook, and Ms. Diaz attended...") still produce occasional internal
  splits; these are tracked as known long-tail cases on
  `data/eval/abbrev/eval.jsonl`

## 0.1.4 - Typed Python API

This release adds a typed Python wrapper surface while preserving explicit
dictionary conversion for JSON and existing integrations.

Included:

- immutable typed result dataclasses: `Span`, `Annotation`, `ModelInfo`, and
  `BenchmarkResult`
- `TypedDict` result shapes for dictionary/JSON output
- `py.typed` marker for PEP 561 type-checker support
- `_native.pyi` stubs for the PyO3 extension boundary
- typed `Segmenter.annotate()`, `spans()`, `model_info()`, and `benchmark()`
- explicit compatibility methods such as `.to_dict()`,
  `segmenter.annotate_dict(text)`, and `charstreamer.model_info_dict()`

Known status:

- the default semantic model remains the `paragraph`, `metadata`, `section`,
  and `list_item` checkpoint from `0.1.2`
- this release changes high-level Python return types from plain dictionaries
  to typed mapping-compatible objects

## 0.1.3 - Cross-Platform Wheel Release

This release republishes the current model-backed CharStreamer package with a
full platform wheel matrix and corrected release automation.

Included:

- Linux x86_64 and aarch64 wheels
- macOS x86_64 and arm64 wheels
- Windows x86_64 and arm64 wheels
- source distribution
- one validated default Burn model bundle reused across all wheels
- offline model smoke tests for every platform wheel
- fixed macOS OpenBLAS/GCC runtime library bundling

Known status:

- the default semantic model remains the `paragraph`, `metadata`, `section`,
  and `list_item` checkpoint from `0.1.2`
- `dialogue` remains reserved until a balanced dialogue training set exists

## 0.1.2 - Combined Semantic Model Wheel

This release corrects the Python package metadata and makes the combined
model-backed segmentation bundle the latest public wheel.

Included:

- vendored Burn sentence-boundary model
- vendored Burn semantic structure model for `paragraph`, `metadata`,
  `section`, and `list_item`
- reusable line byte n-gram and line-context feature primitives
- offline default model loading from the Python wheel
- GitHub release model archive matching the wheel version

Known status:

- `dialogue` is intentionally not emitted until a balanced dialogue dataset is
  available
- the model is an early semantic segmentation checkpoint and should be validated
  on task-specific corpora before production use

## 0.1.1 - Model-Backed Wheel

This release is the first CharStreamer checkpoint with a vendored, loadable
Burn model in the Python wheel.

Included:

- default `burn_shallow_mlp_sentence_v1` sentence-boundary model bundle
- Burn named-msgpack save/load support for the shallow MLP backend
- Rust `BurnSentenceSegmenter` for model-backed sentence spans only
- PyO3/Python `Segmenter.default(require_model=True)` model-backed runtime
- Rust training/export example for the default sentence-boundary bundle
- wheel/model validation gates and offline smoke tests
- release workflow fixes for OpenBLAS-linked manylinux wheels

Known status:

- the default model is a narrow production slice, not the final multi-label
  semantic segmentation model
- structural semantic labels are not emitted until they have trained model
  support

## 0.1.0 - Initial Public Checkpoint

This is the first public development checkpoint for CharStreamer.

Included:

- byte-first core text and candidate abstractions
- reusable scanning and feature kernels
- native CPU linear/logistic model primitives
- Burn neural backend experiments for CPU training and inference
- PyO3/maturin extension crate scaffold
- GitHub Actions CI and tag-driven PyPI/GitHub release workflow
- semantic segmentation utilities and long-document benchmarks
- experiment manifests and documentation for prior benchmark work
- streaming span-generator tool for OpenAI-assisted weak labeling

Known status:

- APIs are not yet stable
- generated training data and model artifacts are intentionally not checked in
- Burn is the only active external Rust model backend
