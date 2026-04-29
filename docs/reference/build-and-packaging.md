# Build And Packaging

## Goal

Define the intended repository layout, crate boundaries, packaging story, and
tooling expectations for Rust and Python deliverables.

## Repository layout

Target structure:

```text
charstreamer/
  Cargo.toml
  rust-toolchain.toml
  crates/
    charstreamer-core/
    charstreamer-kernels/
    charstreamer-models-native/
    charstreamer-backend-burn/
    charstreamer-py/
  benches/
  examples/
  python/
    pyproject.toml
    charstreamer/
  docs/
```

## Crate roles

### `charstreamer-core`

- types
- traits
- workspaces
- chunking
- decoders
- pipeline orchestration

### `charstreamer-kernels`

- byte scanning
- ASCII class tables
- UTF-8 helper kernels
- SIMD dispatch

### `charstreamer-models-native`

- native linear/logistic training and inference
- native shallow tree training and inference
- native serialization format

### `charstreamer-backend-burn`

- optional neural backend
- tiny MLP/CNN/LSTM experiments

### `charstreamer-py`

- PyO3 bindings
- maturin packaging
- NumPy adapters

## Cargo workspace policy

- root workspace should pin shared dependency versions where practical
- crates should compile independently
- optional backends must be feature-gated
- default features should stay minimal

## Feature flags

Suggested root-level features:

- `python`
- `burn-backend`
- `simd-nightly` if ever needed
- `serde`

Suggested principles:

- no neural or heavy backend in default build
- no Python dependency in pure Rust build
- no nightly requirement for the main supported path

## Build targets

Primary support targets:

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `aarch64-apple-darwin`
- `x86_64-apple-darwin`
- `x86_64-pc-windows-msvc`

Python wheel targets should follow the subset that maturin can support reliably.

## CPU optimization policy

- support runtime feature detection on x64 and arm64
- keep portable fallbacks for all kernels
- avoid hard-wiring AVX2-only assumptions into public behavior
- benchmarks should report which execution path was used

## Python packaging policy

Use PyO3 + maturin.

Reasons:

- direct Rust extension build
- mature wheel story
- good NumPy interop
- straightforward manylinux/macOS packaging

Package split:

- Rust crate in `crates/charstreamer-py/`
- Python packaging metadata in `python/`

## Artifact policy

Ship:

- Rust crates through crates.io eventually
- Python wheels through PyPI eventually
- native model artifacts as regular files

Do not require:

- ONNX runtime for default first-party models
- Python for Rust-only inference

## CI expectations

Minimum CI jobs:

- format and lint
- unit tests
- property/fuzz smoke tests where available
- docs build
- benchmark smoke build
- Python extension build

Recommended matrix:

- Linux x64
- Linux arm64
- macOS arm64
- Windows x64
