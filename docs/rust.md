# Rust Best Practices

## Purpose

This document captures Rust-specific best practices for `charstreamer`, based on
official Rust documentation, the Cargo Book, the rustdoc book, Clippy docs, the
Rust Style Guide, and the Rust API Guidelines.

It is not a generic style guide for all Rust projects. It is a project-specific
operating document for a CPU-focused systems library with a Python binding layer.

## Design goals

Rust code in `charstreamer` should optimize for:

- explicit data layout
- predictable performance
- low allocation pressure
- clear API contracts
- easy benchmarking
- safe defaults with tightly scoped `unsafe`
- documentation that doubles as executable examples

## Language and workspace baseline

### Edition and resolver

Use Rust 2024 edition for new crates.

Use a virtual workspace root with:

- explicit `resolver = "3"`
- shared `workspace.package`
- shared `workspace.dependencies`
- shared `workspace.lints`

Rationale:

- the Cargo Book documents virtual workspaces and explicitly notes that `resolver`
  must be set in virtual workspaces
- shared metadata and dependencies reduce drift across crates
- `workspace.lints` allows cross-crate policy instead of duplicated lint tables

## Repository and crate layout

Use a multi-crate workspace, not one giant crate.

Recommended crate split:

- `charstreamer-core`
- `charstreamer-kernels`
- `charstreamer-models-native`
- optional backend crates
- `charstreamer-py`

Design rules:

- keep hot-path kernels separate from orchestration code
- keep Python binding code out of core crates
- isolate third-party backend adapters behind feature-gated crates
- keep native model serialization in the native model crate

## API design

Use the Rust API Guidelines as the default API review checklist.

### Naming

- use idiomatic Rust naming and casing
- follow `as_`, `to_`, `into_` conversion conventions
- use `iter`, `iter_mut`, and `into_iter` consistently
- use consistent word order across related types

### Type design

- prefer newtypes for meaningful domain distinctions
- prefer explicit config structs over boolean parameter soup
- keep struct fields private unless direct field access is part of the API contract
- derive or implement common traits where meaningful:
  - `Debug`
  - `Clone`
  - `PartialEq` / `Eq`
  - `Default`
  - `Hash`

### Trait design

- keep core traits small and composable
- prefer associated types when they improve readability
- make traits object-safe only when dynamic dispatch is a real use case
- avoid “mega traits” that force unrelated capabilities together

### Primitive factoring

- design reusable low-level primitives before task adapters
- keep hot loops in scanner, gather, feature, and model primitives rather than in high-level pipeline code
- prefer borrowed views and caller-provided buffers over task-specific owned intermediates
- treat materialized window objects as an exception, not the default execution model

### Function design

- methods when there is a clear receiver
- associated functions for constructors
- avoid out-parameters in public APIs unless they are the intentional low-level
  `*_into` escape hatch
- validate arguments at the API boundary
- separate allocating APIs from workspace-driven low-allocation APIs

### Caller control

Favor APIs where the caller controls:

- allocation
- destination buffers
- chunk sizing
- parallelism decisions where practical

This aligns with the Rust API Guidelines’ “caller decides where to copy and place
data” principle and is a direct fit for `charstreamer`.

## Error handling

### Library code

- do not use `unwrap`/`expect` in non-test library code unless an invariant is
  truly impossible to violate without internal bugs
- prefer typed error enums for boundary-crossing public APIs
- keep error messages concrete and actionable
- document panic conditions, error conditions, and safety conditions

### Application / benchmark code

- `unwrap` is acceptable in benchmark harnesses, examples, or short-lived
  tooling where failure should abort immediately

### Result sizing

Be mindful of error size in hot paths. Clippy has `result_large_err` guidance and
 configurable thresholds; large error payloads are often a smell in low-level APIs.

## `unsafe` policy

`charstreamer` will almost certainly need some `unsafe` for SIMD, pointer-based
 gathers, or FFI-like optimization boundaries. The goal is not “no unsafe ever”;
 the goal is “unsafe is explicit, justified, and quarantined.”

### Rules

- keep `unsafe` concentrated in `charstreamer-kernels` or tightly scoped low-level modules
- prefer safe wrappers around tiny `unsafe` blocks
- enable `unsafe_op_in_unsafe_fn`
- document every unsafe block with a safety comment
- document every `unsafe fn` with a `# Safety` section in rustdoc
- avoid exposing `unsafe` in public APIs unless the caller must uphold a real
  memory-safety invariant

### Review standard for unsafe code

Every unsafe block should answer:

- what invariant is being assumed
- why the invariant holds at this call site
- how it is tested
- what benchmark motivated the unsafe path

If a safe implementation is within noise of the unsafe one, keep the safe one.

## Documentation

The rustdoc book explicitly recommends thorough crate-level docs and executable
 examples. `charstreamer` should treat docs as part of the API.

### Public docs expectations

- every public crate has crate-level docs
- every public type and function has a short summary
- important APIs have examples
- functions that can error, panic, or require safety guarantees document those sections
- prose should link to related types and functions where useful

### Examples

- examples should prefer `?` over `unwrap`
- doctests should compile and run where practical
- long examples can live in `examples/` and be referenced from rustdoc

### Doctests

`cargo test` runs documentation tests by default, and rustdoc extracts code
 samples from documentation comments and executes them. Keep doc examples current.

## Formatting and style

Use the default Rust style.

### Formatter

- use `cargo fmt`
- do not bikeshed formatting that rustfmt owns
- avoid custom formatting conventions that fight the formatter

The Rust Style Guide says the default Rust style is the intended reference style,
 and rustfmt uses it as the reference.

### Style conventions

- block indent, not visual indent
- 100-column line width unless tooling forces otherwise
- prefer small diffs and low rightward drift

## Linting

Clippy is the standard linting layer for this project.

### Default policy

- CI should run `cargo clippy --workspace --all-targets --all-features`
- treat warnings as failures in CI once bootstrap is complete
- use lint configuration deliberately, not reactively

### Lints to consider early

- `missing_errors_doc`
- `missing_panics_doc`
- `missing_safety_doc`
- `result_large_err`
- `large_futures`
- `fn_params_excessive_bools`
- `struct_excessive_bools`

### Lint levels

Project guidance:

- `deny` for correctness and safety-sensitive lints
- `warn` for style and maintainability lints during active development
- `allow` only with a reason

### Workspace lint inheritance

Prefer central lint configuration through `workspace.lints` plus per-crate
 `lints.workspace = true`.

## Testing

### Standard command surface

- `cargo test --workspace`
- `cargo test --workspace --doc`
- `cargo test -p <crate>`

### Test categories

- unit tests for local invariants
- integration tests for crate seams
- doctests for public APIs
- benchmark smoke tests that compile in CI
- property tests / fuzz tests where invariants justify them

### Working directory assumptions

The Cargo Book documents that unit and integration tests run with the package
 root as working directory, which means relative fixture access should be based
 on package roots, not arbitrary caller cwd assumptions.

## Benchmarking

Use stable-friendly benchmarking tools for serious measurements. Cargo’s builtin
 `#[bench]` remains nightly-only; the Cargo Book explicitly points stable users
 toward ecosystem tools like Criterion.

Project rules:

- benchmark hot kernels in isolation
- benchmark end-to-end pipelines
- separate single-thread and parallel results
- record CPU, toolchain, and features for every report

## Performance engineering

### Data structures

- prefer contiguous row-major buffers in hot paths
- avoid `Vec<Vec<T>>` in execution kernels
- minimize pointer chasing
- choose `u32` or narrow integers where bandwidth matters and range permits

### Allocation

- default hot paths should support caller-provided buffers
- provide allocating wrappers for convenience only
- reuse workspaces aggressively

### Parallelism

- prefer `rayon` for data-parallel phases
- parallelize at chunk or batch boundaries
- avoid fine-grained locking in hot loops

### Genericity

- use generics where they improve ergonomics or eliminate overhead
- avoid deep generic abstraction in the hottest loops if it hurts compile time,
  debuggability, or autovectorization

## Dependency policy

- default feature set should stay lean
- heavyweight dependencies must be optional
- backend crates must not define the core data model
- new dependencies require a concrete reason: performance, interoperability,
  correctness, or maintenance leverage

## Recommended commands

```bash
cargo fmt --all
cargo check --workspace
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
cargo test --workspace --doc
```

## What this means for `charstreamer`

Concretely, implementers should:

- keep bytes as the canonical storage format
- use newtypes for positions and spans
- keep `unsafe` localized to kernels
- write public docs and examples as code is added
- make every hot API available in both allocating and `*_into` forms
- prefer typed configs over ad-hoc booleans
- keep backend integrations optional and isolated

## Official sources

- Rust documentation hub: https://doc.rust-lang.org/
- Rust Book: https://doc.rust-lang.org/book/
- Cargo workspaces: https://doc.rust-lang.org/cargo/reference/workspaces.html
- Rust Style Guide: https://doc.rust-lang.org/style-guide/
- rustdoc book: https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html
- rustdoc doctests: https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html
- Clippy docs: https://doc.rust-lang.org/clippy/
- Clippy lint config: https://doc.rust-lang.org/clippy/lint_configuration.html
- Rust API Guidelines: https://rust-lang.github.io/api-guidelines/checklist.html
- `unsafe_op_in_unsafe_fn`: https://doc.rust-lang.org/edition-guide/rust-2024/unsafe-op-in-unsafe-fn.html
