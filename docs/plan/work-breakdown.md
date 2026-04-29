# Work Breakdown

## Goal

Provide an actionable backlog for implementation. This is the document that
should be updated most frequently during active development.

## Status legend

- `todo`
- `active`
- `blocked`
- `done`
- `deferred`

## Epic E0: Repository foundation

### E0-T1 Workspace bootstrap

- status: `done`
- deliverable: root Cargo workspace and crate skeletons
- dependencies: none
- acceptance:
  - workspace builds
  - placeholder crates compile

### E0-T2 Docs bootstrap

- status: `done`
- deliverable: `docs/` tree and execution docs
- dependencies: none
- acceptance:
  - index exists
  - reference/plan/quality/templates/logs all exist

### E0-T3 Tooling baseline

- status: `done`
- deliverable: formatter, linter, test, and primitive benchmark commands
- dependencies: E0-T1

## Epic E1: Foundational reusable primitives

### E1-T1 Position and range types

- status: `done`
- deliverable: `BytePos`, `ByteSpan`, `ScanRange`, `ChunkRange`, `OwnedRange`
- dependencies: E0-T1

### E1-T2 Text views

- status: `done`
- deliverable: `TextBytes`, `AsciiByteView`, `Utf8ScalarView` scaffolding
- dependencies: E1-T1

### E1-T3 Candidate buffers and matrix views

- status: `done`
- deliverable: candidate buffers, feature matrix views, row writers, score buffers
- dependencies: E1-T1

### E1-T4 Workspace scratch types

- status: `done`
- deliverable: scan, feature, model, and pipeline workspace storage
- dependencies: E1-T1

### E1-T5 Primitive microbenchmark harness

- status: `done`
- deliverable: protected microbench targets for scan, feature append, and scoring primitives
- dependencies: E1-T3

## Epic E2: Scanning primitives

### E2-T1 ByteSet and byte-class tables

- status: `done`
- deliverable: `ByteSet256` plus reusable byte classification tables
- dependencies: E1-T2

### E2-T2 Candidate scanner baseline

- status: `done`
- deliverable: byte-set scanner using portable Rust and `memchr`
- dependencies: E2-T1

### E2-T3 Chunk plan and overlap merge primitives

- status: `todo`
- deliverable: scan-range partitioning and overlap-aware candidate merge logic
- dependencies: E1-T1, E2-T2

### E2-T4 SIMD scanner

- status: `todo`
- deliverable: AVX2/NEON runtime-dispatched scanner kernels
- dependencies: E2-T2

## Epic E3: Feature primitives

### E3-T1 Gather and window indexing helpers

- status: `todo`
- deliverable: reusable window/index helpers with no materialized per-candidate windows
- dependencies: E1-T3

### E3-T2 Encoded byte window appender

- status: `done`
- deliverable: compact categorical byte-window features
- dependencies: E3-T1

### E3-T3 ASCII class appender

- status: `done`
- deliverable: whitespace, punctuation, digit, alpha flags
- dependencies: E3-T1

### E3-T4 Boundary heuristic appender

- status: `done`
- deliverable: quote/list/terminator heuristics comparable to `charboundary`
- dependencies: E3-T1

### E3-T5 Composite kernel

- status: `done`
- deliverable: zero-extra-allocation kernel composition
- dependencies: E3-T2, E3-T3, E3-T4

## Epic E4: Pipeline composition and Unicode support

### E4-T1 Primitive traits and pipeline combinators

- status: `done`
- deliverable: scanner/appender/kernel/predictor/decoder traits plus pipeline wrappers
- dependencies: E1-T3, E1-T4

### E4-T2 Prototype boundary pipeline

- status: `done`
- deliverable: end-to-end inference path with dummy or simple scorer
- dependencies: E2-T2, E3-T5, E4-T1

### E4-T3 UTF-8 scalar side table

- status: `todo`
- deliverable: decoded scalar map and byte<->scalar translation
- dependencies: E1-T2

### E4-T4 Unicode category appender

- status: `todo`
- deliverable: Unicode-aware feature extraction without corrupting byte fast-path
- dependencies: E4-T3, E3-T5

### E4-T5 Rayon chunk parallelism

- status: `todo`
- deliverable: chunked parallel scanning/extraction/scoring
- dependencies: E2-T3, E4-T2

## Epic E5: Native models

### E5-T1 Linear regression

- status: `active`
- deliverable: native inference and training
- dependencies: E4-T1

### E5-T2 Logistic regression

- status: `done`
- deliverable: binary classification training/inference
- dependencies: E5-T1

### E5-T3 Threshold calibration

- status: `done`
- deliverable: threshold tuning and report outputs
- dependencies: E5-T2

### E5-T4 Shallow decision tree

- status: `todo`
- deliverable: native training/inference for shallow trees
- dependencies: E4-T1

### E5-T5 Native serialization

- status: `done`
- deliverable: versioned native file format for first-party models
- dependencies: E5-T2 or E5-T4

## Epic E6: Python bindings

### E6-T1 PyO3 crate

- status: `todo`
- deliverable: extension module skeleton
- dependencies: E0-T1, E5-T2

### E6-T2 High-level Python pipeline

- status: `todo`
- deliverable: `BoundaryPipeline` Python API
- dependencies: E6-T1, E4-T2

### E6-T3 NumPy interop

- status: `todo`
- deliverable: feature/scoring array adapters
- dependencies: E6-T1

### E6-T4 GIL-free compute path

- status: `todo`
- deliverable: compute blocks release GIL
- dependencies: E6-T2

## Epic E7: Optional backends and release

### E7-T1 Burn adapter

- status: `done`
- deliverable: train/infer wrappers for Burn neural baselines
- dependencies: E4-T1

### E7-T1b sklearn docking baseline

- status: `done`
- deliverable: manifest-driven Python/sklearn RF baseline for exact legacy-model docking experiments
- dependencies: E4-T1

### E7-T2 Neural backend adapter

- status: `deferred`
- deliverable: `Burn` adapter for tiny MLP/CNN experiments
- dependencies: E4-T1, E6-T1

### E7-T3 Benchmark suite

- status: `active`
- deliverable: full throughput and latency suite, including manifest sweeps across model families
- dependencies: E1-T5, E4-T2, E5-T2

### E7-T4 Release automation

- status: `todo`
- deliverable: package and wheel build workflow
- dependencies: E6-T2, E7-T3
