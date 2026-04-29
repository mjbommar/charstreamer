# Primitives

## Goal

Define the low-level reusable building blocks that `charstreamer` should be made
from.

This document is the bottom-up counterpart to
[architecture.md](architecture.md). Its purpose is to make sure we build a
small set of optimization-friendly primitives that can be reused across:

- inference and training
- boundary detection and other candidate-based tasks
- Rust and Python APIs
- portable, SIMD, and parallel execution paths

## Design law

Optimize once, reuse many times.

In practice, that means:

- put hot loops behind stable primitive contracts
- keep primitives task-agnostic
- let higher layers compose primitives instead of re-implementing them
- avoid materialized intermediates unless benchmarks justify them

If a fast path only helps one task adapter and cannot be reused by scanners,
feature kernels, or models elsewhere, it probably belongs too high in the stack.

## Primitive stack

### 1. Text and position primitives

These define what all other layers read from.

Core types:

- `TextBytes<'a>`
- `AsciiByteView<'a>`
- `Utf8ScalarView<'a>`
- `ByteToCharMap`
- `BytePos`
- `ScalarPos`
- `ByteSpan`
- `ScanRange`
- `ChunkRange`
- `OwnedRange`

Responsibilities:

- canonical byte storage
- fast ASCII detection
- explicit byte and scalar position semantics
- chunk ownership and overlap accounting

### 2. Scan primitives

These discover candidate positions without knowing anything about features or
models.

Core types:

- `ByteSet256`
- `ByteClassTable`
- `PositionBuffer`
- `PositionSlice<'a>`
- `CandidateBuffer`
- `CandidateSlice<'a>`
- `ChunkPlan`

Responsibilities:

- find candidate byte offsets
- classify bytes cheaply
- emit sorted candidate positions
- support portable and ISA-specialized implementations under one contract

Examples of reusable scan work:

- terminator scanning
- whitespace or delimiter scanning
- newline scanning
- quote scanning
- line-start scanning
- stride-based coarse sampling

### 3. Gather and context primitives

These describe and access context around candidate positions.

Core types:

- `ByteWindowSpec`
- `ScalarWindowSpec`
- `StrideWindowSpec`
- `GatherScratch`
- `PositionBatch<'a>`

Responsibilities:

- translate positions plus window specs into indexed reads
- support left/right boundary clamping or padding policies
- make rolling context access reusable by multiple feature appenders

Important rule:

- windows are described, not materialized as per-candidate heap objects

The hot path should not produce `Vec<Window>` or equivalent unless a benchmark
shows a real win.

### 4. Feature write primitives

These turn gathered context into stable column blocks in a dense feature matrix.

Core types:

- `FeatureBlock`
- `FeatureSchema`
- `FeatureRowMut<'a, T>`
- `FeatureMatrixViewMut<'a, T>`
- `FeatureScratch`

Core traits:

- `FeatureAppender<T>`
- `FeatureKernel<T>`

Responsibilities:

- own stable column layout
- append features into caller-provided storage
- compose multiple appenders without extra allocation
- keep training and inference feature generation identical

Design rule:

- each appender owns a fixed block of columns
- appenders never compete for the same output columns
- composite kernels orchestrate appenders; they do not invent a second feature format

### 5. Matrix and score primitives

These are the numerical substrate shared by models and trainers.

Core types:

- `FeatureMatrix<T>`
- `FeatureMatrixView<'a, T>`
- `FeatureMatrixViewMut<'a, T>`
- `ScoreBuffer<S>`
- `ScoreMatrix<S>`
- `ScoreMatrixView<'a, S>`
- `ScoreMatrixViewMut<'a, S>`
- `DatasetView<'a, F, L>`

Responsibilities:

- contiguous row-major storage
- borrowed views over owned buffers
- shared representation for inference and training

Rules:

- no `Vec<Vec<T>>` on the hot path
- row-major is the default storage contract
- strided views are allowed for slicing, but owned buffers should stay contiguous
- scalar-score and matrix-score paths should share the same feature matrix contract

### 6. Model primitives

These consume matrix views and produce score buffers.

Core types:

- packed linear/logistic models
- packed shallow tree models
- `FitOptions`
- `FitReport`
- `FitScratch`

Core traits:

- `BatchPredictor<F, S>`
- `BatchScorer<F, S>`
- `TrainablePredictor<F, L>`

Responsibilities:

- batch prediction from dense views
- native CPU-friendly training
- packed serialization-friendly parameter storage

Design rule:

- models should only know about feature matrices and labels
- models should not know about text scanning or sentence-specific heuristics

### 7. Decode primitives

These turn scores back into task-level outputs.

Core types:

- threshold configuration
- post-rule configuration
- task-specific output buffers

Core traits:

- `Decoder<S, O>`
- `ScoreDecoder<S, O>`

Responsibilities:

- map positions and scores to output objects
- keep scoring separate from output formatting
- support both scalar-score and score-matrix decode shapes

### 8. Workspace and dispatch primitives

These manage reuse, specialization, and parallel execution.

Core types:

- `ScanScratch`
- `FeatureScratch`
- `ModelScratch`
- `PipelineWorkspace`
- `ScoringWorkspace`
- `KernelDispatch`
- `ThreadLocalWorkspace`

Responsibilities:

- own reusable buffers
- avoid per-call allocations
- isolate runtime CPU dispatch
- give parallel workers disjoint scratch storage

Design rule:

- optimized code paths must conform to the same primitive contracts as portable ones

## Stable contracts before optimization

Every primitive family should have:

- a portable reference implementation
- explicit invariants
- direct tests
- differential tests against optimized implementations
- isolated microbenchmarks

Only after that should we add:

- AVX2 / AVX-512 / NEON kernels
- chunk-parallel scheduling
- specialized packing layouts

## Reuse rules

### What belongs in reusable primitives

- byteset membership and class tables
- scan range math
- overlap merge logic
- window indexing helpers
- feature block layout
- row-major matrix views
- packed linear algebra for native models
- threshold and decode combinators

### What belongs above the primitive layer

- sentence-boundary default configs
- corpus-specific heuristics
- task presets
- Python convenience wrappers
- migration adapters for `charboundary`

## Anti-patterns

Avoid these unless benchmarks and reuse arguments are extremely strong:

- `Vec<Window>` or any other materialized per-candidate window object
- scanners that directly call model code
- feature extractors that allocate one row at a time
- UTF-8 decoding mixed into the byte scanner hot path
- separate training-only and inference-only feature generators
- task adapters that bypass reusable buffers and views

## Review checklist

For any new primitive or optimization, ask:

- can another task reuse this directly
- is the data layout explicit
- can the caller provide the destination buffer
- does it preserve the byte fast path
- does it avoid forcing Unicode work on ASCII workloads
- is there a portable baseline
- is there a microbenchmark for it
- is the contract narrow enough to remain stable
