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
- `aarch64-pc-windows-msvc`

Python wheel targets should follow the subset that maturin can support reliably.
The public wheel matrix is:

- `manylinux` Linux x86_64 on `ubuntu-24.04`
- `manylinux` Linux aarch64 on `ubuntu-24.04-arm`
- macOS x86_64 on `macos-15-intel`
- macOS arm64 on `macos-15`
- Windows x86_64 on `windows-2025`
- Windows arm64 on `windows-11-arm`

The Python extension uses `abi3-py39`, so each supported OS/architecture only
needs one wheel rather than one wheel per Python minor version.

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

Native dependency policy:

- Linux wheels should be repaired by `maturin`/`auditwheel`; OpenBLAS and
  Fortran runtime libraries are vendored when dynamically linked.
- macOS wheels should be repaired with `delocate` so Homebrew OpenBLAS dylibs
  are copied into the wheel and install names are rewritten.
- Windows wheels should install OpenBLAS through `vcpkg` and be repaired with
  `delvewheel` so required DLLs are included in the wheel.
- Source builds may still require a working system BLAS/OpenBLAS toolchain.

## Artifact policy

Ship:

- Rust crates through crates.io eventually
- Python wheels through PyPI eventually
- native model artifacts as validated bundle directories or zip archives

For the Python package, the default model should be either:

- vendored into the wheel under `charstreamer/models/default/`
- attached to the same GitHub release as `charstreamer-default-<version>.zip`
  and resolved by the Python loader into the local model cache

Do not require:

- ONNX runtime for default first-party models
- Python for Rust-only inference

See [model-artifacts.md](model-artifacts.md) for the manifest format, vendoring
script, runtime resolution order, and release gates.

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
- macOS x64
- macOS arm64
- Windows x64
- Windows arm64

Release workflow shape:

- `prepare`: validate the requested tag and create one normalized model bundle.
- `wheel`: build and repair one wheel per supported OS/architecture from that
  exact model bundle.
- `sdist`: build the source distribution with the same vendored model.
- `publish`: publish only after all platform artifacts pass metadata,
  model-bundle, and offline smoke-test gates.
