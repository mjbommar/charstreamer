# Performance Targets

## Goal

Define measurable throughput and latency targets without overcommitting to fake precision
before baseline implementations exist.

The core rule is to benchmark reusable primitives directly, not only end-to-end
pipelines.

## Measurement policy

Benchmark all claims on:

- one x86_64 AVX2-capable machine
- one arm64 NEON-capable machine

Record:

- CPU model
- core count
- compiler version
- feature flags
- input corpus class
- whether results are single-threaded or parallel

## Primitive-first benchmark policy

Every important optimization should be measurable at two levels:

- isolated primitive benchmark
- end-to-end pipeline benchmark

The primitive benchmark answers "did this kernel get faster."

The pipeline benchmark answers "did users actually benefit."

We need both. Primitive-only wins can hide integration regressions, and
pipeline-only wins make it hard to know which reusable building block improved.

## Metric classes

### Scanner metrics

- bytes scanned per second
- candidate positions emitted per second
- latency on small buffers

### Gather metrics

- candidate windows indexed per second
- branch-miss-sensitive edge handling cost
- allocations per call

### Feature appender metrics

- rows appended per second
- columns written per second
- allocations per call
- ASCII-heavy vs mixed UTF-8 behavior

### Feature metrics

- candidates featurized per second
- bytes featurized per second
- allocations per call

### Model metrics

- rows scored per second
- training samples per second
- serialized model size

### End-to-end metrics

- chars or bytes processed per second
- p50/p95 latency on short inputs
- scaling across cores
- Python wrapper overhead

## Release-oriented targets

These are initial targets and may be tightened once baselines exist.

### Relative targets

- native Rust sentence-boundary pipeline should exceed `charboundary` sklearn path on the same corpus
- optimized native path should target at least parity with `charboundary` ONNX path before release
- Python wrapper overhead should stay below 15% on large-buffer workloads

### Absolute targets

Treat these as directional gates, not promises until reference hardware is fixed:

- byte scanner: >1 GiB/s single-thread on modern desktop/server x64
- byte scanner: >700 MiB/s single-thread on modern arm64
- byte-window and ASCII-class appenders should stay allocation-free in steady-state execution
- full native linear/logistic boundary pipeline: >1 million chars/s single-thread on ASCII-heavy corpora
- chunked parallel pipeline: near-linear scaling through at least 4 cores on sufficiently large corpora

## Benchmark suites

### Suite A: microbenchmarks

- byteset membership and scan kernels
- overlap merge and candidate handling
- gather/index helpers
- individual feature appenders
- dot-product scoring only
- tree scoring only

### Suite B: pipeline benchmarks

- short sentence inputs
- medium legal paragraphs
- large mixed-text corpora

### Suite C: parity benchmarks

- `charboundary` sklearn
- `charboundary` ONNX
- native `charstreamer`
- optional backend adapters

### Suite D: Python overhead benchmarks

- `str` input
- `bytes` input
- NumPy feature scoring

## Protected benchmarks

The following benchmarks should be treated as protected reusable primitives:

- portable byteset scanner
- ISA-specialized byteset scanner
- byte window appender
- ASCII class appender
- composite feature kernel
- native logistic scoring

## Regression policy

- any >5% regression in a protected benchmark requires explanation
- any >10% regression blocks release unless explicitly approved and documented

## Benchmark output requirements

Every benchmark report should include:

- git revision
- date
- hardware details
- corpus details
- exact command
- summary table
- interpretation notes
