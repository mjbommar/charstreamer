# Changelog

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
