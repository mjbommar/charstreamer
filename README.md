# CharStreamer

CharStreamer is a Rust-first toolkit for high-throughput text stream
classification and segmentation. It is designed around reusable byte/character
window primitives, feature kernels, and CPU model backends that can be composed
for tasks such as sentence boundary detection, semantic span annotation,
IOB-style tagging, and format change-point detection.

The repository currently contains an early public development release. APIs are
expected to evolve, but the core direction is stable: byte-first data layout,
explicit Unicode-aware feature paths, reusable hot-loop primitives, and a small
external model-backend surface centered on Burn.

## What Is Included

- `charstreamer-core`: text views, candidate buffers, feature matrices,
  pipelines, metrics, corpus helpers, and decoders.
- `charstreamer-kernels`: byte scanners, Unicode category features, rolling
  window features, and reusable feature appenders.
- `charstreamer-models-native`: native CPU linear/logistic model primitives.
- `charstreamer-backend-burn`: Burn-based neural model experiments for CPU
  training and inference.
- `charstreamer-segmentation`: rule-based semantic segmentation and merged
  annotation rendering.
- `charstreamer-python`: PyO3/maturin extension module for Python access.
- `charstreamer-experiments`: local experiment runners and reproducibility
  manifests. This crate is not intended for crates.io publication.
- `tools/span-generator`: streaming weak-label data generation utility for
  OpenAI-assisted semantic span annotation.

## Repository Status

This is a first public checkpoint, not a polished stable API release. The main
release goals for this checkpoint are:

- keep the source tree buildable and testable from a clean clone
- keep generated data, model artifacts, virtual environments, and logs out of
  git
- preserve enough docs and experiment manifests to reproduce current design
  decisions
- keep the active external Rust model dependency path focused on Burn

## Build And Test

Prerequisites:

- Rust 1.89 or newer
- a system BLAS/OpenBLAS installation for the current Burn `ndarray` backend
- Python 3.9+ and `maturin` only if building the Python extension
- `uv` only if using `tools/span-generator`

Run the Rust test suite:

```bash
cargo test --workspace
```

Run focused benchmarks:

```bash
cargo bench -p charstreamer-kernels
cargo bench -p charstreamer-core
cargo bench -p charstreamer-models-native
cargo bench -p charstreamer-segmentation
```

Build the Python extension locally:

```bash
cd crates/charstreamer-python
maturin develop
```

Build the release wheel locally:

```bash
uvx maturin build --release --manifest-path crates/charstreamer-python/Cargo.toml --out dist
uvx twine check dist/*
```

## Release

The public distribution target is one PyPI package: `charstreamer`.

GitHub Actions handles normal releases from tags:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow builds a single `cp39-abi3` manylinux wheel, checks the
wheel metadata, smoke-tests an isolated install, publishes to PyPI, and attaches
the wheel to a GitHub release.

PyPI publishing uses GitHub Actions secrets `PYPI_USERNAME` and
`PYPI_API_TOKEN`, which can be populated from a local `.pypirc`. The workflow
also uses the GitHub environment named `pypi`, so it can be switched to PyPI
Trusted Publishing later without changing the release trigger.

## Quick Examples

Run the narrow boundary pipeline example:

```bash
cargo run -p charstreamer-core --example narrow_slice
```

Run the mixed XML/CSV format-switch example:

```bash
cargo run -p charstreamer-core --example format_switch
```

Run the long-document segmentation timing example. If
`data/bench/war_and_peace.txt` is not present, it uses a synthetic fallback:

```bash
cargo run --release -p charstreamer-segmentation --example time_once
```

## Data Policy

Generated datasets, OpenAI annotation outputs, local logs, Python virtual
environments, model artifacts, and benchmark texts are intentionally ignored.
The checked-in repository should contain source, docs, tests, manifests, and
small reproducibility scaffolding only.

## Documentation

Start with:

- [docs/README.md](docs/README.md)
- [docs/reference/architecture.md](docs/reference/architecture.md)
- [docs/reference/primitives.md](docs/reference/primitives.md)
- [docs/reference/model-families.md](docs/reference/model-families.md)
- [docs/quality/release-gates.md](docs/quality/release-gates.md)
- [docs/results.md](docs/results.md)

Experiment manifests are documented in [specs/README.md](specs/README.md).

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
