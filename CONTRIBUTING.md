# Contributing

CharStreamer is early-stage. Contributions should keep the codebase small,
measurable, and CPU-performance oriented.

## Development Rules

- Keep generated data, model artifacts, local logs, and virtual environments out
  of git.
- Prefer reusable primitives over one-off task-specific hot loops.
- Keep byte-oriented and Unicode-aware paths explicit.
- Add tests for behavior changes.
- Add or update benchmarks when changing hot paths.
- Do not add new external model backends unless the quality/throughput reason is
  documented.

## Validation

Before opening a change, run:

```bash
cargo fmt --all
cargo test --workspace
```

For performance-sensitive changes, also run the relevant benchmark:

```bash
cargo bench -p charstreamer-kernels
cargo bench -p charstreamer-core
cargo bench -p charstreamer-models-native
cargo bench -p charstreamer-segmentation
```

For Python package changes:

```bash
cd crates/charstreamer-python
maturin develop
```

For release-wheel changes:

```bash
uvx maturin build --release --manifest-path crates/charstreamer-python/Cargo.toml --out dist
uvx twine check dist/*
```

For span-generator changes:

```bash
cd tools/span-generator
uv run pytest
```
