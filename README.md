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
- `charstreamer-segmentation`: model-backed segmentation and merged annotation
  rendering.
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
- `v0.1.1` contains the first model-backed wheel, but its PyPI long description
  predates the combined semantic model.
- `v0.1.2` is the current model-backed release. The wheel vendors a Burn
  sentence-boundary model and a Burn semantic structure model for `paragraph`,
  `metadata`, `section`, and `list_item`.
- The default runtime must load a supported model bundle. It must not synthesize
  semantic annotations from hard-coded rules when a model is unavailable.

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
```

Build the Python extension locally:

```bash
cd crates/charstreamer-python
maturin develop
```

Build the release wheel locally:

```bash
python3 tools/model-artifacts/vendor_model.py \
  --require-burn \
  --archive-out dist-models/charstreamer-default-0.1.2.zip \
  target/model/charstreamer-default-0.1.1-release

uvx --with 'maturin[patchelf]' maturin build \
  --release \
  --manifest-path crates/charstreamer-python/Cargo.toml \
  --out dist

uvx twine check dist/*.whl
```

Validate that a wheel contains a usable vendored model:

```bash
python3 tools/model-artifacts/check_wheel_model.py --require-burn dist/charstreamer-*.whl
```

## Release

The public distribution target is one PyPI package: `charstreamer`.

GitHub Actions handles releases through the manual `Release` workflow. A
model-backed release must provide or pre-upload a validated
`charstreamer-default-<version>.zip` bundle before the wheel is built.

The model bundle is validated and copied into:

```text
charstreamer/models/default/
```

inside the wheel. The release also attaches the normalized model zip to the
GitHub release. The release workflow fails if the wheel does not contain a
Burn-backed model bundle or if the Python default path cannot load a usable
model offline.

The usual sequence is:

```bash
gh release create v0.1.2 \
  dist-models/charstreamer-default-0.1.2.zip \
  --target main \
  --title "CharStreamer v0.1.2" \
  --notes-file CHANGELOG.md

gh workflow run Release \
  -f tag=v0.1.2
```

Alternatively, pass `-f model_artifact_url=https://.../charstreamer-default-0.1.2.zip`
to the workflow and it will download that bundle directly.

The release workflow builds a single `cp39-abi3` manylinux wheel, checks the
wheel metadata, validates the model artifact, smoke-tests an isolated install,
publishes to PyPI, and attaches the wheel and model zip to a GitHub release.

PyPI publishing uses GitHub Actions secrets `PYPI_USERNAME` and
`PYPI_API_TOKEN`, which can be populated from a local `.pypirc`. The workflow
also uses the GitHub environment named `pypi`, so it can be switched to PyPI
Trusted Publishing later without changing the release trigger.

Current default bundle metrics:

```text
runtime: burn_combined_segmentation
sentence validation f1: 0.977
semantic fixed-validation macro f1: 0.746
semantic labels: paragraph, metadata, section, list_item
```

## Quick Examples

Install from PyPI:

```bash
pip install charstreamer
```

Use from Python:

```python
import charstreamer

text = "The court reviewed the invoice. The shipment was late. Notice was timely."

segmenter = charstreamer.Segmenter.default()
print(segmenter.model_info())
annotation = segmenter.annotate(text)
print(annotation["spans"][:3])
print(annotation["tagged"])
```

Model-backed output with the current wheel looks like:

```python
{
    "resolved": True,
    "source": "bundled",
    "path": ".../site-packages/charstreamer/models/default",
    "manifest": {"engine": "burn_shallow_mlp_sentence_v1", "structure": {"engine": "burn_multilabel_mlp_structure_v1", "...": "..."}, "...": "..."},
    "error": None,
    "runtime": "burn_combined_segmentation",
    "model_inference": True,
}
```

Example spans and tagged text:

```python
[
    {"label": "sentence", "start": 0, "end": 31, "start_byte": 0, "end_byte": 31, "score": 0.99},
    {"label": "sentence", "start": 32, "end": 54, "start_byte": 32, "end_byte": 54, "score": 0.99},
    {"label": "sentence", "start": 55, "end": 73, "start_byte": 55, "end_byte": 73, "score": 0.96},
]

<|sentence|>The court reviewed the invoice.</|sentence|> <|sentence|>The shipment was late.</|sentence|> <|sentence|>Notice was timely.</|sentence|>
```

Force model-backed execution in tests or production startup:

```python
import charstreamer

charstreamer.model_info(allow_download=False, require_model=True)
segmenter = charstreamer.Segmenter.default(require_model=True)
```

That call must fail if no supported model is available. This is intentional: it
prevents silently shipping a package that looks model-backed but is not.

The current default model is an early combined sentence and structure model.
`dialogue` is intentionally excluded until a balanced dialogue dataset exists.

Run the narrow boundary pipeline example:

```bash
cargo run -p charstreamer-core --example narrow_slice
```

Run the mixed XML/CSV format-switch example:

```bash
cargo run -p charstreamer-core --example format_switch
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
