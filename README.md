# CharStreamer

CharStreamer is a Rust/Python toolkit for high-throughput text stream
classification and segmentation. The default Python package ships model-backed
sentence and semantic-structure annotation; the Rust crates expose the reusable
byte/character windows, feature kernels, decoders, and Burn model backend used
to build that runtime.

Current default labels:

- `sentence`
- `paragraph`
- `metadata`
- `section`
- `list_item`

The default runtime must load a supported model bundle. It does not synthesize
semantic annotations from hard-coded rules when a model is unavailable.

## Install

```bash
pip install charstreamer
```

The Python package exposes a `cp39-abi3` extension, so one wheel works across
Python 3.9+ for a given OS/architecture. If no wheel is available for your
platform, source builds require Rust 1.89+ and a working BLAS/OpenBLAS setup.

## Quick Start

```python
import charstreamer

text = """Background
The court reviewed the invoice. The shipment was late.

Notice was timely.
"""

segmenter = charstreamer.Segmenter.default()
annotation = segmenter.annotate(text)

print(segmenter.model_info()["runtime"])
print(annotation["spans"][:5])
print(annotation["tagged"])
```

Output is a dictionary with scored spans and a rendered tagged string. Exact
scores and semantic labels depend on the model version. Abridged `v0.1.3`
output for the text above looks like:

```text
burn_combined_segmentation
[
  {"label": "section", "start": 0, "end": 10, "start_byte": 0, "end_byte": 10, "score": ...},
  {"label": "paragraph", "start": 11, "end": 65, "start_byte": 11, "end_byte": 65, "score": ...},
  {"label": "sentence", "start": 11, "end": 42, "start_byte": 11, "end_byte": 42, "score": ...},
  {"label": "sentence", "start": 43, "end": 65, "start_byte": 43, "end_byte": 65, "score": ...},
  {"label": "metadata", "start": 67, "end": 85, "start_byte": 67, "end_byte": 85, "score": ...}
]
<|section|>Background</|section|>
<|paragraph|><|sentence|>The court reviewed the invoice.</|sentence|> <|sentence|>The shipment was late.</|sentence|></|paragraph|>

<|metadata|><|sentence|>Notice was timely.</|sentence|></|metadata|>
```

Convenience functions are also available:

```python
import charstreamer

print(charstreamer.spans("The court reviewed the invoice. Notice was timely."))
print(charstreamer.tagged("The court reviewed the invoice. Notice was timely."))
```

## Model Loading

`Segmenter.default()` resolves models in this order:

- `CHARSTREAMER_MODEL_PATH`
- bundled wheel data under `charstreamer/models/default/`
- local cache under `~/.cache/charstreamer/models/default`
- GitHub release artifact, unless downloads are disabled

Production startup should assert that a real model is available:

```python
import charstreamer

charstreamer.model_info(allow_download=False, require_model=True)
segmenter = charstreamer.Segmenter.default(allow_download=False, require_model=True)
```

Useful environment variables:

- `CHARSTREAMER_AUTO_DOWNLOAD=0`: disable release-artifact downloads.
- `CHARSTREAMER_MODEL_PATH=/path/to/model`: use a specific model directory.
- `CHARSTREAMER_MODEL_CACHE=/path/to/cache`: override the model cache root.
- `CHARSTREAMER_MODEL_URL=https://.../charstreamer-default-<version>.zip`: use a specific download URL.

## Current Release

`v0.1.3` is the current model-backed release. It vendors a Burn
sentence-boundary model and a Burn semantic-structure model.

Current default bundle metrics:

```text
runtime: burn_combined_segmentation
sentence validation f1: 0.977
semantic fixed-validation macro f1: 0.746
semantic labels: paragraph, metadata, section, list_item
```

Notes:

- `v0.1.0` on PyPI did not contain a trained Burn model.
- `v0.1.1` contained the first model-backed wheel, but its PyPI long description
  predated the combined semantic model.
- `dialogue` is reserved until a balanced dialogue training set exists.

## Platform Builds

The release workflow is configured to build:

- Linux x86_64
- Linux aarch64
- macOS x86_64
- macOS arm64
- Windows x86_64
- Windows arm64
- sdist

Native dependency bundling is platform-specific:

- Linux wheels are repaired by `maturin`/`auditwheel`.
- macOS wheels are repaired with `delocate`.
- Windows wheels install OpenBLAS through `vcpkg` and are repaired with
  `delvewheel`.

## Rust Development

Prerequisites:

- Rust 1.89 or newer
- system BLAS/OpenBLAS for the current Burn `ndarray` backend
- Python 3.9+ and `maturin` only if building the Python extension
- `uv` only if using Python tools

Run tests:

```bash
cargo test --workspace
```

Run focused benchmarks:

```bash
cargo bench -p charstreamer-kernels
cargo bench -p charstreamer-core
cargo bench -p charstreamer-models-native
```

Build the Python extension locally:

```bash
cd crates/charstreamer-python
maturin develop
```

Build a local wheel from the checked-in vendored model:

```bash
python3 tools/model-artifacts/vendor_model.py \
  --require-burn \
  --archive-out dist-models/charstreamer-default-0.1.3.zip \
  crates/charstreamer-python/python/charstreamer/models/default

uvx --with 'maturin[patchelf]' maturin build \
  --release \
  --manifest-path crates/charstreamer-python/Cargo.toml \
  --out dist

uvx twine check dist/*.whl
python3 tools/model-artifacts/check_wheel_model.py --require-burn dist/charstreamer-*.whl
```

Run example pipelines:

```bash
cargo run -p charstreamer-core --example narrow_slice
cargo run -p charstreamer-core --example format_switch
```

## Repository Layout

- `charstreamer-core`: text views, candidate buffers, feature matrices,
  pipelines, metrics, corpus helpers, and decoders.
- `charstreamer-kernels`: byte scanners, Unicode category features, rolling
  window features, and reusable feature appenders.
- `charstreamer-models-native`: native CPU linear/logistic model primitives.
- `charstreamer-backend-burn`: Burn-based CPU neural model backend.
- `charstreamer-segmentation`: model-backed segmentation and merged annotation
  rendering.
- `charstreamer-python`: PyO3/maturin extension module for Python access.
- `charstreamer-experiments`: local experiment runners and reproducibility
  manifests. This crate is not intended for crates.io publication.
- `tools/span-generator`: streaming weak-label data generation utility for
  OpenAI-assisted semantic span annotation.

## Release Process

The public distribution target is one PyPI package: `charstreamer`.

Run the manual GitHub Actions release workflow:

```bash
gh workflow run Release \
  -f tag=v0.1.3
```

By default, the workflow creates the normalized
`charstreamer-default-<version>.zip` bundle from the checked-in vendored model
and uses that exact bundle for every wheel. To release from an externally staged
bundle:

```bash
gh workflow run Release \
  -f tag=v0.1.3 \
  -f model_artifact_url=https://.../charstreamer-default-0.1.3.zip
```

The workflow fails if any wheel lacks a supported Burn model bundle or if the
offline smoke test cannot load and run the default model.

PyPI publishing currently uses GitHub Actions secrets `PYPI_USERNAME` and
`PYPI_API_TOKEN`. The workflow also uses the GitHub environment named `pypi`, so
it can be switched to PyPI Trusted Publishing later.

## Data Policy

Generated datasets, OpenAI annotation outputs, local logs, Python virtual
environments, and benchmark texts are intentionally ignored. The checked-in
repository should contain source, docs, tests, manifests, small reproducibility
scaffolding, and the currently vendored default model bundle used by the Python
wheel.

## Documentation

Start with:

- [docs/README.md](docs/README.md)
- [docs/reference/architecture.md](docs/reference/architecture.md)
- [docs/reference/primitives.md](docs/reference/primitives.md)
- [docs/reference/model-families.md](docs/reference/model-families.md)
- [docs/reference/model-artifacts.md](docs/reference/model-artifacts.md)
- [docs/reference/build-and-packaging.md](docs/reference/build-and-packaging.md)
- [docs/quality/release-gates.md](docs/quality/release-gates.md)
- [docs/results.md](docs/results.md)

Experiment manifests are documented in [specs/README.md](specs/README.md).

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
