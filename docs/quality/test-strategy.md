# Test Strategy

## Goal

Make correctness failures hard to ship, especially around byte offsets, Unicode
handling, model parity, and serialization.

The test plan should protect reusable primitives first and task pipelines second.

## Test layers

### 1. Primitive contract tests

Targets:

- bytesets and class tables
- position and range math
- byte class tables
- scanners
- gather helpers
- feature schema layout
- feature matrix views
- feature row writers
- feature matrix indexing
- model math
- serialization helpers
- decoder edge cases

Rules:

- every public type with invariants gets direct unit coverage
- every primitive contract gets a reference-case test
- every bug fix should add a regression test

### 2. Property tests

Targets:

- span coverage
- sorted candidate positions
- feature block non-overlap
- chunk merge equivalence
- serialization round-trip
- train/predict deterministic behavior under fixed seed

Example properties:

- decoded spans cover the input without gaps or overlap
- chunked pipeline output equals non-chunked pipeline output
- composite feature kernels equal the concatenation of their appender blocks
- loading a serialized model preserves predictions within tolerance

### 3. Differential tests

Targets:

- compare portable scanner kernels against ISA-specialized scanner kernels
- compare portable feature appenders against optimized feature appenders
- compare native pipeline against `charboundary` for the initial sentence-boundary problem
- compare native and backend model adapters on the same feature matrix
- compare Python and Rust outputs on shared fixtures

Purpose:

- detect silent semantic drift during migration and optimization

### 4. Pipeline composition tests

Targets:

- scanner -> feature kernel -> model -> decoder assembly
- chunked vs non-chunked pipeline equivalence
- training feature path vs inference feature path equivalence

### 5. Corpus tests

Corpora should include:

- short English prose
- legal text
- bullet and numbered lists
- heavily quoted text
- Unicode punctuation
- newline-heavy documents
- ASCII-only corpora

### 6. Fuzz tests

Targets:

- UTF-8 decoding boundary behavior
- scanner/decoder interactions
- Python boundary adapters
- serialization input validation

### 7. Performance regression tests

Targets:

- protected primitive microbenchmarks
- scanner throughput
- feature extraction throughput
- end-to-end pipeline throughput
- Python overhead

Use as benchmark gates, not correctness gates.

## Protected primitive families

Every optimized implementation should have a protected benchmark and at least one
differential test in these families:

- byte scanning
- candidate merge and overlap handling
- window/gather indexing
- feature appender block writes
- dense linear scoring
- shallow tree scoring

## Required fixture classes

### Text fixtures

- empty text
- one-character text
- no-candidate text
- all-candidate text
- repeated delimiters
- malformed UTF-8 for byte APIs
- valid mixed-language UTF-8

### Primitive fixtures

- small candidate buffers with known offsets
- feature schemas with multiple blocks
- tiny feature matrices with nontrivial row stride
- chunk ranges with overlap edge cases

### Model fixtures

- tiny linear model
- tiny logistic model
- tiny shallow tree
- serialized native artifact

## Phase-specific testing requirements

### Phase 00

- placeholder smoke tests
- primitive benchmark harness smoke build

### Phase 01

- scanner correctness
- matrix view correctness
- feature appender composition
- span coverage
- chunk equivalence

### Phase 02

- training convergence smoke tests
- serialization round-trip
- native vs reference metric checks

### Phase 03

- Python/Rust parity
- GIL-release smoke tests
- buffer ownership tests

### Phase 04

- architecture-specific correctness
- portable vs SIMD primitive parity
- release-candidate corpus suite

## Quality rules

- no new public API without direct tests
- no optimized primitive without a portable baseline and differential coverage
- no new serialization format without round-trip and compatibility tests
- no new optimization path without differential equivalence tests
