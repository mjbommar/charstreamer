# AGENTS

## Purpose

This file is the operational standard for anyone implementing `charstreamer`.

It synthesizes the project architecture and the researched best practices from:

- [docs/reference/architecture.md](docs/reference/architecture.md)
- [docs/reference/primitives.md](docs/reference/primitives.md)
- [docs/reference/data-model.md](docs/reference/data-model.md)
- [docs/reference/model-families.md](docs/reference/model-families.md)
- [docs/rust.md](docs/rust.md)
- [docs/python.md](docs/python.md)
- [docs/quality/release-gates.md](docs/quality/release-gates.md)

## Read first

Before changing code, read in this order:

1. `docs/reference/architecture.md`
2. `docs/reference/primitives.md`
3. `docs/reference/data-model.md`
4. `docs/rust.md`
5. `docs/python.md`
6. `docs/plan/work-breakdown.md`

## Project priorities

Optimize for:

- CPU-first training and inference
- byte-first internal representation
- explicit data layout
- predictable performance
- generic reusable primitives
- manifest-driven reproducibility for experiments and docking
- pipeline composition on top of those primitives
- thin Python bindings over a Rust-owned core

Do not optimize for:

- Python-first architecture
- opaque magic abstractions
- premature neural backends
- Unicode-heavy hot paths at the expense of the byte fast path

## Non-negotiable architecture rules

### Canonical text representation

- internal text storage is bytes
- internal positions are byte offsets
- scanners are byte-first
- Unicode-aware views are derived side tables, not the primary hot path

### Primitive-first architecture

Build bottom up.

- optimize low-level primitives once and reuse them many times
- keep portable and optimized implementations behind the same primitive contracts
- do not let task adapters become the place where fast paths live

### Separate execution patterns

Do not merge byte and character logic into one branchy implementation.

- byte path: scanning, chunking, hashing, delimiter detection
- UTF-8 scalar path: Unicode-aware features and Python character spans

### Pipeline shape

Keep the architecture organized around:

1. scan candidates
2. gather context
3. append/extract features
4. predict in batch
5. decode outputs

`build_windows` is conceptual only. The hot path should not require materialized
window objects.

### Experiment discipline

When comparing prior vs current implementations:

- define the run with an explicit manifest, not ad hoc code edits
- hold corpus, split, scanner, feature spec, window sizes, sampling policy, and
  evaluation method constant unless one of them is the deliberate variable
- run feature-row parity checks before trusting model-level comparisons
- change one primary variable at a time

### Hot-path API design

Every important hot path should eventually have both:

- an allocating convenience API
- a low-level `*_into` API using caller-provided buffers

### Config vs code

- every performance-critical feature must be backed by a reusable Rust primitive
- config-defined feature pipelines must compile into the same primitive objects
  used by code-defined pipelines
- do not build a second interpreted feature engine for configuration support

### Non-negotiable primitive rules

- no `Vec<Vec<T>>` in execution kernels
- no per-candidate heap allocation in hot loops
- no separate training-only feature path if inference can reuse the same primitives
- feature blocks must have stable schemas and disjoint column ownership

## Model policy

### First-party native scope

Prioritize, in order:

1. linear regression
2. logistic regression
3. threshold calibration
4. shallow decision trees

### Backend scope

Keep external Rust model backends optional and isolated.

- `Burn` is the active neural backend
- avoid adding classical ML backend dependencies unless there is a documented
  Pareto reason

### Do not start with

- LSTM
- boosted trees
- a home-grown autodiff stack

unless project requirements change and are documented.

## Rust implementation rules

### Workspace

Use a virtual Cargo workspace with:

- explicit `resolver = "3"`
- shared `workspace.package`
- shared `workspace.dependencies`
- shared `workspace.lints`

### API style

Follow the Rust API Guidelines by default:

- use newtypes for semantic distinctions
- keep fields private unless public fields are intentional API
- implement common traits where meaningful
- use clear conversion naming
- avoid boolean parameter soup
- prefer config structs and builders for complex construction

### Documentation

- every public crate gets crate-level docs
- public APIs need rustdoc summaries
- significant public APIs need examples
- docs should include panic, error, and safety sections where relevant
- doc examples should prefer `?` over `unwrap`

### Unsafe code

Unsafe is allowed only where it is justified by performance or system interface needs.

Rules:

- keep unsafe localized to low-level modules, ideally `charstreamer-kernels`
- every unsafe block gets a safety comment
- every `unsafe fn` gets a `# Safety` rustdoc section
- enable `unsafe_op_in_unsafe_fn`
- prefer safe wrappers around tiny unsafe regions

If a safe implementation is close enough in performance, use the safe version.

### Errors

- no `unwrap`/`expect` in reusable library code without a strong invariant reason
- use typed errors for public boundaries
- keep error values small and meaningful in hot-path APIs

### Performance

- avoid `Vec<Vec<T>>` in execution kernels
- prefer contiguous row-major storage
- minimize allocations
- reuse workspaces
- parallelize at chunk or batch boundaries
- keep heavyweight dependencies optional
- benchmark primitives directly, not only end-to-end pipelines

## Python / PyO3 / maturin rules

### Packaging model

Use:

- PyO3 for bindings
- maturin for build and packaging
- `pyproject.toml` as the Python packaging authority

### Layout

Prefer a mixed project layout with a dedicated Python source directory:

```text
python/
  charstreamer/
    __init__.py
    ...
```

And configure:

```toml
[tool.maturin]
python-source = "python"
module-name = "charstreamer._charstreamer"
```

### Native module exposure

- expose the extension as a private submodule, e.g. `_charstreamer`
- keep the public Python package as a thin wrapper layer

### Linking

Binding crates should usually use:

- `crate-type = ["cdylib", "rlib"]`

so Rust tests/examples remain practical.

### Extension build mode

- do not rely on PyO3’s legacy `extension-module` feature as the main workflow
- let maturin manage extension-module distribution behavior

### `abi3`

Default position:

- do not enable `abi3` by default for the first performance-sensitive release

It may be enabled later if distribution simplicity outweighs the documented feature
 and optimization tradeoffs.

### GIL / interpreter lock handling

For CPU-heavy work:

- convert Python inputs into Rust-owned data
- release the interpreter lock with `Python::detach`
- run native work
- reacquire only to create Python return objects

Do not hold the interpreter lock across Rayon or other worker-thread execution.

### Thread safety

Treat `#[pyclass]` types as thread-shared.

Rules:

- keep `#[pyclass]` wrappers thin
- prefer `#[pyclass(frozen)]` when possible
- use atomics or locks for shared mutable state
- avoid `#[pyclass(unsendable)]` except with strong justification

### Typing

- ship `.pyi` stubs
- ship `py.typed`
- keep stubs in sync with Rust exports

### Python semantics

- `bytes` APIs use byte offsets/spans
- `str` APIs must document whether spans are bytes or characters
- do not silently blur `bytes` and `str` semantics

### Binding architecture

- Python wrappers should sit on top of the same primitive-backed pipeline as Rust
- do not create Python-only feature extraction or scoring paths

## Tooling rules

### Rust

Run regularly:

```bash
cargo fmt --all
cargo check --workspace
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
cargo test --workspace --doc
```

### Python

Run regularly:

```bash
ruff check python
ruff format --check python
pytest
maturin develop
```

Before packaging validation:

```bash
maturin build -r
```

## Documentation and planning rules

When code changes:

- update `docs/reference/` if behavior or architecture changes
- update `docs/plan/` if sequencing or backlog changes
- update `docs/quality/` if acceptance criteria change

When meaningful work is done:

- append a short note to `docs/logs/build-diary.md`

When a project-level decision is made:

- append it to `docs/logs/decision-log.md`

## Review checklist

Before considering work done, confirm:

- architecture still matches byte-first design
- primitive contracts are still reusable and narrow
- public API follows Rust idioms
- Python API keeps `str`/`bytes` semantics explicit
- unsafe is documented and localized
- tests cover the new invariant or behavior
- docs are updated
- logs are updated if the work was meaningful

## If scope is unclear

Default to:

- smaller API surface
- fewer dependencies
- native Rust ownership of the core path
- optional adapters for heavyweight ecosystems
- explicit semantics over convenience magic
