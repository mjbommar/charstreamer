# Generic Scoring Slice Todo

## Goal

Generalize the first narrow boundary slice into a broader primitive-first shape:

- scanners can emit arbitrary position sets, not only punctuation candidates
- feature extraction stays reusable across sparse and dense position tasks
- models can emit one score or many scores per position
- decoders can produce labels or labeled spans, not only thresholded boundaries
- the workspace supports region and change-point tasks

## Task list

### Generic data and traits

- status: `done`
- add reusable position aliases alongside candidate aliases
- add score-matrix owned and borrowed views
- add a reusable scoring workspace
- add a `BatchScorer` trait for matrix outputs
- add a `ScoreDecoder` trait for task-specific matrix decoding

### Generic decoders

- status: `done`
- add argmax per-position label decoding
- add contiguous labeled-span decoding over adjacent positions
- keep binary threshold decoding as a separate adapter

### More generic scanners

- status: `done`
- add a stride scanner for coarse regular sampling
- add a line-start scanner for record and region style tasks
- keep byteset scanning as the sparse candidate baseline

### More generic feature blocks

- status: `done`
- add composable byte-count style appenders for format and region tasks
- keep boundary-specific appenders separate from generic appenders
- add a composite format demo kernel that can distinguish XML-like and CSV-like lines

### More generic model path

- status: `done`
- add a native linear multiclass scorer that writes score matrices
- keep the logistic binary scorer for the original boundary slice

### Second vertical slice

- status: `done`
- add a format-switch example using line starts as positions
- add an integration test for XML-to-CSV region detection
- extend the pipeline benchmark with the second slice

## Validation

- `cargo fmt --all`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`
- `cargo bench --workspace --no-run`
- `cargo run -p charstreamer-core --example narrow_slice`
- `cargo run -p charstreamer-core --example format_switch`

## Exit condition

The generic slice is done when the same primitive stack can support both:

1. sparse boundary classification with scalar scores and boundary decoding
2. region labeling with score matrices and labeled-span decoding
