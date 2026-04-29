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

Important model status:

- `v0.1.0` on PyPI does **not** contain a trained Burn model.
- Current source includes model-artifact validation and Python model-resolution
  plumbing, but production Burn model inference still has to be connected before
  the next model-backed public release.
- Until that work is complete, `charstreamer.Segmenter.default()` reports its
  model status and uses the native heuristic segmenter as the fallback runtime.

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

Validate that a wheel contains a usable vendored model:

```bash
python3 tools/model-artifacts/check_wheel_model.py --require-burn dist/charstreamer-*.whl
```

## Release

The public distribution target is one PyPI package: `charstreamer`.

GitHub Actions handles releases through the manual `Release` workflow. A
model-backed release must provide `model_artifact_url` pointing at a validated
`charstreamer-default-<version>.zip` bundle before the wheel is built.

The model bundle is validated and copied into:

```text
charstreamer/models/default/
```

inside the wheel. The release also attaches the normalized model zip to the
GitHub release. The release workflow fails if the wheel does not contain a
Burn-backed model bundle or if the Python default path cannot load a usable
model offline.

Run a release with the GitHub CLI:

```bash
gh workflow run Release \
  -f tag=v0.1.1 \
  -f model_artifact_url=https://example.com/charstreamer-default-0.1.1.zip
```

The release workflow builds a single `cp39-abi3` manylinux wheel, checks the
wheel metadata, validates the model artifact, smoke-tests an isolated install,
publishes to PyPI, and attaches the wheel and model zip to a GitHub release.

PyPI publishing uses GitHub Actions secrets `PYPI_USERNAME` and
`PYPI_API_TOKEN`, which can be populated from a local `.pypirc`. The workflow
also uses the GitHub environment named `pypi`, so it can be switched to PyPI
Trusted Publishing later without changing the release trigger.

## Quick Examples

Install from PyPI:

```bash
pip install charstreamer
```

Use from Python:

```python
import charstreamer

text = """# Background
The court reviewed the invoice. The shipment was late.

- Notice was timely.
"""

segmenter = charstreamer.Segmenter.default()
print(segmenter.model_info())
annotation = segmenter.annotate(text)
print(annotation["spans"][:3])
print(annotation["tagged"])
```

Offline fallback output without a vendored model
(`CHARSTREAMER_AUTO_DOWNLOAD=0`) looks like:

```python
{
    "resolved": False,
    "source": "heuristic",
    "path": None,
    "manifest": None,
    "error": None,
    "runtime": "native_heuristic",
    "model_inference": False,
}
```

Example spans and tagged text:

```python
[
    {"label": "section", "start": 0, "end": 12, "start_byte": 0, "end_byte": 12, "score": 0.98},
    {"label": "paragraph", "start": 0, "end": 67, "start_byte": 0, "end_byte": 67, "score": 1.0},
    {"label": "sentence", "start": 13, "end": 44, "start_byte": 13, "end_byte": 44, "score": 1.0},
]

<|paragraph|><|section|># Background</|section|>
<|sentence|>The court reviewed the invoice.</|sentence|>
<|sentence|>The shipment was late.</|sentence|></|paragraph|>
<|paragraph|><|list_item|>- Notice was timely.</|list_item|></|paragraph|>
```

Force model-backed execution in tests or production startup:

```python
import charstreamer

charstreamer.model_info(allow_download=False, require_model=True)
segmenter = charstreamer.Segmenter.default(require_model=True)
```

That call must fail if the wheel is heuristic-only. This is intentional: it
prevents silently shipping a package that looks model-backed but is not.

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
- [docs/reference/model-artifacts.md](docs/reference/model-artifacts.md)
- [docs/quality/release-gates.md](docs/quality/release-gates.md)
- [docs/results.md](docs/results.md)

Experiment manifests are documented in [specs/README.md](specs/README.md).

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
