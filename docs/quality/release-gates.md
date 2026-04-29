# Release Gates

## Goal

Define explicit go/no-go rules so the library is not “done” based on vibes.

## Gate classes

### G0: Build gate

Must pass:

- workspace builds
- docs build
- lint and format checks

### G1: Correctness gate

Must pass:

- unit tests
- property tests
- portable vs optimized primitive differential tests where optimized paths exist
- serialization round-trip
- chunked vs non-chunked equivalence

### G2: API gate

Must pass:

- public Rust API documented
- Python API documented if present
- byte-vs-character semantics explicitly documented

### G3: Quality gate

Must pass:

- no known crashers on supported platforms
- deterministic fixed-seed training for native models
- error messages for malformed inputs are tested

### G4: Performance gate

Must pass:

- protected primitive benchmarks have no unexplained regressions
- no benchmark regression beyond agreed tolerance
- required throughput targets met or signed off with explanation
- Python overhead within documented range

## Phase gates

### Phase 00 exit

- G0 only

### Phase 01 exit

- G0
- G1 for core pipeline
- protected primitive benchmarks exist for scanner and feature appenders

### Phase 02 exit

- G0
- G1
- G2 for Rust training APIs
- G3 for native model artifacts

### Phase 03 exit

- G0
- G1
- G2 for Python APIs
- G3 for Rust/Python parity

### Phase 04 exit / release candidate

- all gates

## Release candidate checklist

- native pipeline passes corpus suite
- native models serialize and reload cleanly
- benchmark report is current
- build diary includes release hardening summary
- decision log includes any deviations from original architecture
