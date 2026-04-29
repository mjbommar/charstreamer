# Python, PyO3, And Maturin Best Practices

## Purpose

This document captures best practices for the Python-facing side of
`charstreamer`: packaging, bindings, type information, thread-safety, testing,
 and release discipline.

It is centered on official PyO3 and maturin guidance, plus official Python
 packaging and testing guidance.

## Design goals

The Python layer should optimize for:

- a thin wrapper over the Rust core
- no Python in the hot path
- clear `str` vs `bytes` semantics
- low-copy interop where possible
- thread-safe behavior
- packaging that is simple to develop locally and predictable to ship

## Packaging model

Use PyO3 for bindings and maturin for packaging.

### Why this combination

PyO3’s own guide says the easiest path for generating a native Python module is
 to use maturin. Maturin’s user guide is built around exactly this workflow.

### Build-system policy

Use a modern `pyproject.toml` as the authoritative Python build configuration.

The Python Packaging User Guide states that `pyproject.toml` acts as the
 configuration file for packaging-related tools and that `[build-system]`
 declares the build requirements.

### Project layout

For `charstreamer`, prefer a mixed Rust/Python layout with an explicit Python
 source directory:

```text
charstreamer/
  Cargo.toml
  pyproject.toml
  python/
    charstreamer/
      __init__.py
      ...
  src/ or crates/...
```

And in `pyproject.toml`:

```toml
[tool.maturin]
python-source = "python"
module-name = "charstreamer._charstreamer"
```

Maturin’s guide explicitly recommends the `python-source` layout to avoid a
 common ImportError pitfall.

## Module naming

Use a private native submodule and a public Python package shim.

Recommended structure:

- public package: `charstreamer`
- native extension module: `charstreamer._charstreamer`

Reasons:

- keeps Python package namespace clean
- allows Python helper code alongside the extension
- works well with IDEs and type stubs

### Rust side

Ensure `#[pymodule]` naming matches the configured extension module name.

Maturin’s guide shows either renaming the function itself or adding
 `#[pyo3(name = "_my_project")]`.

## Crate type and linking

PyO3’s guide notes:

- `cdylib` is necessary for Python to import the shared library
- if downstream Rust code, tests, or examples need to `use` the crate, include
  `rlib` alongside `cdylib`

Project rule:

- binding crates should usually use `crate-type = ["cdylib", "rlib"]`

This keeps Rust-side examples and tests practical.

## Extension-module builds

Do not build your workflow around the old PyO3 `extension-module` feature.

PyO3’s current distribution guide explains that:

- extension modules on Unix must not link to `libpython` for manylinux compliance
- maturin sets `PYO3_BUILD_EXTENSION_MODULE` automatically
- the historical `extension-module` feature caused development pain because it
  disabled linking for tests and benchmarks too

Project rule:

- let maturin manage extension-module build mode
- keep Rust tests and benchmarks buildable in normal development workflows

## `abi3` policy

PyO3 supports `abi3`, and maturin can build `abi3` wheels.

### Benefits

- fewer wheels to build and publish
- one wheel can cover multiple Python versions

### Costs

PyO3 documents several limitations under `abi3`, including missing features for
 some Python-version-specific behaviors and fewer opportunities for exact-version
 optimizations.

### Project guidance

Default to **not** using `abi3` in the first performance-sensitive release unless:

- distribution complexity is the bottleneck, and
- required Python features still work, and
- measured performance is unaffected for our workloads

For a pure CPU-heavy extension with tight buffer and feature-path requirements,
 exact-version wheels are often the safer first choice.

## `str` vs `bytes`

The Python API must make this distinction explicit.

### Recommended policy

- APIs accepting `bytes` operate on byte offsets and byte spans
- APIs accepting `str` may expose character spans by default, with an explicit
  byte-span alternative
- docs must state the conversion cost whenever a `str` API requires byte-to-char mapping

Never silently blur these semantics.

## Threading and parallelism

### Releasing the interpreter lock

PyO3’s parallelism guide explicitly recommends using `Python::detach` to allow
 the interpreter to do other work while Rust work is ongoing.

Use `Python::detach` around:

- long-running scans
- feature extraction
- model scoring
- training loops
- Rayon-based chunk/batch parallel work

### Important constraint

PyO3’s guide also notes that if worker threads acquire and hold the GIL, Rayon
 parallelism will not speed up CPU work under regular CPython.

Project rule:

- no CPU-heavy parallel path should hold the interpreter lock
- acquire Python objects at the boundary
- convert into Rust-owned structures
- detach
- run native work
- reacquire only to build Python return values

### Free-threaded Python

PyO3 documents support for free-threaded Python and notes that modules are now
 assumed thread-safe by default in PyO3 0.28.

Project guidance:

- do not rely on the historical GIL for soundness
- assume extension code may run in a freer-threaded future
- write thread-safe binding code now

## `#[pyclass]` design

PyO3’s thread-safety guidance is explicit:

- Python objects may be shared between threads
- `#[pyclass]` types therefore need `Send` and `Sync`

### Rules

- keep `#[pyclass]` wrappers thin
- avoid storing non-thread-safe state directly in `#[pyclass]`
- prefer `#[pyclass(frozen)]` where mutation can be controlled through atomics or locks
- use atomics for simple counters/flags
- use `Mutex` / `RwLock` for structured shared state
- avoid `#[pyclass(unsendable)]` except in rare, tightly justified cases

### Architecture rule

Prefer:

- Rust-native internal types for core logic
- separate thin wrapper types for Python exposure

Do not let PyO3 wrapper concerns contaminate the Rust core model.

## Error handling

PyO3’s guide uses `PyResult<T>` / `Result<T, PyErr>` as the main exception bridge.

### Rules

- every `#[pyfunction]` and `#[pymethods]` API that can fail should return `PyResult<T>`
- map domain errors to concrete Python exception types where possible
- avoid exposing opaque “Rust panic happened” failure modes to Python users
- convert predictable input errors to `ValueError`, `TypeError`, etc.

### Panic policy

- panics should be considered bugs
- normal argument validation and model errors should become Python exceptions

## Type hints and stubs

PyO3’s typing guide explicitly recommends shipping `.pyi` files and `py.typed`.

### Rules

- ship `py.typed`
- ship in-package `.pyi` stubs
- keep Python-visible docs and stub signatures current with Rust exports

### Layout

If there are no other Python files, maturin can include a top-level `<module>.pyi`.

If there is a real Python package, place:

- `__init__.py`
- `_charstreamer.pyi` or `charstreamer.pyi`
- `py.typed`

next to the package code.

### Project recommendation

Because `charstreamer` will likely have Python helper code, use the mixed-package
 layout and keep stubs inside `python/charstreamer/`.

## Local development workflow

### Recommended loop

- create a virtual environment
- install `maturin`
- use `maturin develop` for rapid local iteration
- run tests from the same environment

Maturin’s tutorial states that `maturin develop` skips wheel generation and
 installs directly into the current environment, while `maturin build` produces
 a distributable wheel.

### Important nuance

Maturin’s user guide notes that `maturin develop` is faster, but does not support
 everything that a full `maturin build` + `pip install` path supports.

Project rule:

- use `maturin develop` for day-to-day iteration
- validate packaging with `maturin build` before release work

## Python tooling

### Testing

Use `pytest`.

Pytest’s “Good Integration Practices” recommends:

- using virtual environments
- using `pyproject.toml`
- installing the package in editable mode for development

### Linting and formatting

Use `ruff` for both linting and formatting unless a stronger reason appears.

Ruff’s own formatter documentation positions `ruff format` as the main entrypoint
 and recommends using the linter and formatter together as a unified toolchain.

### Typing

Use stub files plus a checker such as `mypy` in CI for the Python shim layer.

Mypy’s docs confirm `.pyi` stub files as a standard way to provide type
 information for libraries.

### Recommended command surface

```bash
ruff check python
ruff format --check python
pytest
python -m pip install -e .
```

And for the extension package:

```bash
maturin develop
maturin build -r
```

## Testing strategy for the Python layer

### Required categories

- import tests
- API parity tests against Rust expectations
- `str` vs `bytes` semantic tests
- exception mapping tests
- typing/stub smoke tests
- concurrency smoke tests for detached long-running calls

### Packaging tests

- editable install path
- wheel build path
- clean-env import test from built wheel

## Release and distribution

### Wheels

Use `maturin build -r` for release artifacts.

Maturin’s docs note:

- wheels are stored in `target/wheels` by default
- Linux publishing typically needs manylinux Docker or zig
- PyO3/maturin-action is the standard GitHub Actions route

### Publishing discipline

- test local wheel install before publishing
- test on at least one Linux and one macOS target before tagging
- decide explicitly whether `abi3` is on or off
- keep Python package metadata in `pyproject.toml`

## What this means for `charstreamer`

Concretely, implementers should:

- keep Python as a thin shell around Rust-owned data and compute
- use a mixed project layout with `python-source = "python"`
- expose the native extension as a private submodule
- release the interpreter lock for all CPU-heavy work
- make `#[pyclass]` wrappers thread-safe
- ship `.pyi` stubs and `py.typed`
- use `maturin develop` for iteration and `maturin build` for release validation
- avoid `abi3` by default until measured and justified

## Current Charstreamer Binding

The production binding lives in `crates/charstreamer-python` and wraps the
Rust `charstreamer-segmentation` crate.

Build a release wheel:

```bash
uv run --with maturin maturin build --release \
  --manifest-path crates/charstreamer-python/Cargo.toml \
  -i python3
```

Smoke-test the built wheel:

```bash
uv run --with target/wheels/charstreamer-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl python - <<'PY'
import charstreamer

text = 'Case: X\nDocket: 1\n\n# Facts\nOne. Two.\n\n- Item one.\n\n"Hello."'
result = charstreamer.annotate(text)
print(charstreamer.__version__)
print(result["tagged"])
print(charstreamer.Segmenter.default().benchmark(text * 1000, iterations=3))
PY
```

Exposed API:

- `charstreamer.SegmenterConfig`
- `charstreamer.Segmenter`
- `charstreamer.annotate(text) -> {"tagged": str, "spans": list[dict]}`
- `charstreamer.spans(text) -> list[dict]`
- `charstreamer.tagged(text) -> str`
- `charstreamer.render(text, spans) -> str`
- `charstreamer.render_bytes(text, spans) -> str`
- `Segmenter.benchmark(text, iterations=10) -> dict`

Python offset contract:

- `span["start"]` and `span["end"]` are Python character offsets, suitable for `text[start:end]`
- `span["start_byte"]` and `span["end_byte"]` preserve the canonical Rust UTF-8 byte offsets
- `charstreamer.render(text, spans)` accepts Python character offsets
- `charstreamer.render_bytes(text, spans)` accepts canonical Rust byte offsets
- benchmark dictionaries include both byte and character throughput fields

## Official sources

- PyO3 guide: https://pyo3.rs/
- PyO3 error handling: https://pyo3.rs/main/function/error-handling
- PyO3 parallelism: https://pyo3.rs/main/parallelism
- PyO3 building/distribution: https://pyo3.rs/main/building-and-distribution
- PyO3 typing hints: https://pyo3.rs/latest/python-typing-hints
- PyO3 thread safety: https://pyo3.rs/v0.26.0/class/thread-safety.html
- PyO3 free-threading: https://pyo3.rs/main/free-threading.html
- maturin guide: https://www.maturin.rs/
- maturin project layout: https://www.maturin.rs/project_layout
- maturin configuration: https://www.maturin.rs/config.html
- maturin tutorial: https://www.maturin.rs/tutorial
- Python Packaging User Guide: https://packaging.python.org/specifications/declaring-project-metadata/
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Ruff formatter: https://docs.astral.sh/ruff/formatter/
- mypy stubs: https://mypy.readthedocs.io/en/stable/stubs.html
