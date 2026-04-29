# Roadmap

## Goal

Sequence the project from planning to production-quality release without losing
clarity about scope, dependencies, or quality gates.

## Phase summary

### Phase 00: Foundation

Deliverables:

- repository skeleton
- Cargo workspace
- docs structure
- coding standards
- primitive benchmark harness skeleton

Exit condition:

- project can build empty crates and run placeholder tests

### Phase 01: Foundational primitives and core pipeline

Deliverables:

- byte-first data model
- reusable text, range, buffer, and matrix primitives
- candidate scanner baseline
- feature appenders and composite kernel
- sentence-boundary prototype path assembled from primitives

Exit condition:

- end-to-end pipeline with stub or simple logistic model
- no materialized window requirement in the hot path
- protected primitive microbenchmarks exist

### Phase 02: Native models

Deliverables:

- native logistic/linear training and inference
- native shallow tree training and inference
- native serialization format

Exit condition:

- train and serve first-party native models in Rust only

### Phase 03: Python and backend adapters

Deliverables:

- PyO3/maturin layer
- Python high-level API
- optional Burn backend adapter

Exit condition:

- Python package can train and infer with at least one native and one backend model family

### Phase 04: Optimization and release hardening

Deliverables:

- SIMD scanning kernels
- chunked parallel execution
- quality/perf regressions guarded in CI
- release artifacts

Exit condition:

- release-candidate quality gates pass

## Cross-cutting streams

These run across multiple phases:

- documentation
- primitive contract review
- benchmark harnesses
- corpus/fixture curation
- test suite growth
- decision logging

## Sequencing rules

- do not start Python polish before native Rust pipeline semantics are stable
- do not optimize before correctness is measurable
- do not optimize pipelines before primitive contracts are stable enough to reuse
- do not commit to neural backends before native linear/tree results are benchmarked
- keep byte-first path working before adding Unicode-heavy kernels

## Deliverable map

| Area | Phase 00 | Phase 01 | Phase 02 | Phase 03 | Phase 04 |
| --- | --- | --- | --- | --- | --- |
| Docs | yes | yes | yes | yes | yes |
| Primitives | scaffold | stable baseline | reused | reused | optimized |
| Core traits | scaffold | stable | stable | stable | hardened |
| Scanner | design | baseline | stable | stable | SIMD |
| Feature kernels | design | baseline | expanded | stable | tuned |
| Native models | no | placeholder | yes | yes | tuned |
| Python | no | no | no | yes | hardened |
| Benchmarks | scaffold | baseline | comparative | Python | release |

## Execution expectations

For each phase:

- update the phase doc
- keep the work breakdown current
- record meaningful milestones in the build diary
- record architecture-impacting decisions in the decision log
