# Changelog

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
