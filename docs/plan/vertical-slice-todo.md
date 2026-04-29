# Vertical Slice Todo

## Goal

Build the first narrow primitive-first vertical slice end to end:

- candidate scanner
- reusable feature appenders
- simple native logistic scorer
- threshold decoder
- Criterion microbenches plus one pipeline bench

## Task list

### Foundation

- status: `done`
- create Cargo workspace root
- create `charstreamer-core`
- create `charstreamer-kernels`
- create `charstreamer-models-native`
- wire shared dependencies and lints

### Core primitives

- status: `done`
- implement byte position and range types
- implement `TextBytes` and initial derived views
- implement candidate buffers and slices
- implement feature blocks, schema, matrix, and matrix views
- implement pipeline workspace and scratch buffers

### Execution traits

- status: `done`
- implement scanner trait
- implement feature appender trait
- implement feature kernel trait
- implement batch predictor trait
- implement decoder trait
- implement generic pipeline orchestration

### Narrow kernels

- status: `done`
- implement `ByteSet256`
- implement baseline `memchr`-backed scanner
- implement byte-window appender
- implement ASCII-class appender
- implement boundary-heuristic appender
- implement composite feature kernel

### Narrow model and decoder

- status: `done`
- implement packed logistic scorer
- implement demo boundary-model weights
- implement threshold-to-span decoder

### Demo and verification

- status: `done`
- add end-to-end example
- add integration test for the demo slice
- add Criterion primitive benches
- add Criterion pipeline bench
- run `cargo fmt`
- run `cargo test`
- run `cargo bench --no-run`

## Validation

- `cargo fmt --all`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`
- `cargo bench --workspace --no-run`
- `cargo bench -p charstreamer-kernels --bench primitives -- --sample-size 10`
- `cargo bench -p charstreamer-core --bench pipeline -- --sample-size 10`
- `cargo run -p charstreamer-core --example narrow_slice`

## Exit condition

The slice is done when one demo pipeline can:

1. scan candidates from text
2. produce reusable feature blocks
3. score those blocks with a native model
4. decode byte spans
5. pass tests and compile benchmark targets
