# Data Model

## Goal

Define the canonical data representations used by `charstreamer` so the library
can stay fast, portable, and predictable across Rust and Python APIs.

## Core principle

The canonical internal representation is bytes.

- source text is stored as `&[u8]`
- internal positions are byte offsets
- hot-path scanners operate on bytes
- Unicode-aware views are derived, not primary

This keeps the main execution model aligned with hardware and makes SIMD,
chunking, and cache behavior simpler.

The design goal is not only correctness. It is to make sure the same data
representations can be reused by:

- portable and SIMD kernels
- inference and training
- Rust and Python APIs
- single-threaded and chunk-parallel execution

## Text views

### `TextBytes<'a>`

Primary raw input view.

Fields:

- `bytes: &'a [u8]`
- `is_ascii: bool`
- `is_utf8: bool`

Responsibilities:

- expose raw byte storage
- support chunking by byte offset
- support cheap ASCII fast-path detection
- avoid forcing UTF-8 validation on raw byte APIs

### `AsciiByteView<'a>`

Specialized view for ASCII-only or ASCII-dominant execution.

Responsibilities:

- byte class lookup
- punctuation and delimiter search
- cheap lowercase/uppercase checks
- direct indexing with no UTF-8 decoding
- serve as the preferred source for the byte fast path

### `Utf8ScalarView<'a>`

Decoded side-table view for Unicode-aware feature kernels.

Suggested contents:

- `scalar_starts: Vec<u32>`
- `scalar_values: Vec<u32>`
- optional category tags
- optional lowercase flags

Responsibilities:

- map byte offsets to scalar indices
- support Unicode-aware kernels
- support Python-facing character span translation
- stay completely out of the byte scanner hot path

### `ByteToCharMap`

Optional mapping for edge APIs.

Responsibilities:

- byte offset -> scalar index
- scalar span -> byte span

Use only when required by:

- Python `str` output semantics
- user-facing span reporting
- Unicode-aware evaluation

Do not require this map for the scanner or dense inference hot path.

## Position and range types

Use explicit newtypes rather than naked integers in core crates.

Suggested types:

```rust
pub struct BytePos(pub u32);
pub struct ScalarPos(pub u32);
pub struct ByteSpan {
    pub start: BytePos,
    pub end: BytePos,
}
pub struct ScanRange {
    pub start: BytePos,
    pub end: BytePos,
}
pub struct ChunkRange {
    pub start: BytePos,
    pub end: BytePos,
    pub left_overlap: u32,
    pub right_overlap: u32,
}
pub struct OwnedRange {
    pub start: BytePos,
    pub end: BytePos,
}
```

Notes:

- `u32` is usually sufficient for in-memory documents and reduces bandwidth
- allow `usize` conversions at boundaries, not everywhere in the API
- use `u64` only if multi-gigabyte buffers become a real target

Use ranges to describe work ownership explicitly:

- `ScanRange` describes where a scanner is allowed to read and emit candidates
- `ChunkRange` describes overlap-aware chunk boundaries
- `OwnedRange` describes which positions or outputs belong to a worker after overlap handling

## Position storage

Position buffers are the canonical bridge between scanners, feature kernels,
models, and decoders.

Use dense contiguous buffers:

- owned `PositionBuffer` for reusable storage
- borrowed `PositionSlice<'a>` for read-only views
- `CandidateBuffer` and `CandidateSlice<'a>` as task-specific aliases when the
  positions represent sparse candidate events

Invariants:

- sorted ascending
- unique within a chunk after overlap deduplication
- positions refer to the original byte slice
- positions remain byte offsets even when Unicode-aware features are used

Recommended conceptual shapes:

```rust
pub struct PositionBuffer {
    pub data: Vec<BytePos>,
}

pub struct PositionSlice<'a> {
    pub data: &'a [BytePos],
}
```

Typical scanner families:

- sparse candidate scanners such as punctuation or delimiter scans
- dense position scanners such as every byte or every `N` bytes
- structural scanners such as line starts, token boundaries, or record starts

## Context and window specifications

### Byte windows

```rust
pub struct ByteWindowSpec {
    pub left: usize,
    pub right: usize,
}
```

Used for:

- punctuation and delimiter context
- byte class windows
- hashed byte ngrams
- most first-generation feature extractors

### Scalar windows

```rust
pub struct ScalarWindowSpec {
    pub left: usize,
    pub right: usize,
}
```

Used only in Unicode-aware kernels.

### Stride windows

```rust
pub struct StrideWindowSpec {
    pub left: usize,
    pub right: usize,
    pub stride: usize,
}
```

Used for:

- sparse context sampling
- lower-dimensional approximations
- benchmark experiments

### Window materialization policy

Window specs describe how to read context. They do not imply a materialized
window object per candidate.

Preferred execution pattern:

- keep candidate positions in one buffer
- keep the raw text in one buffer
- compute indexed loads directly into feature writers

Avoid:

- `Vec<ByteWindow>`
- copied context slices per candidate
- allocating gather objects for every position

## Feature schema and block layout

Feature layout should be explicit and reusable.

Suggested shapes:

```rust
pub struct FeatureBlock {
    pub name: &'static str,
    pub offset: usize,
    pub width: usize,
}

pub struct FeatureSchema {
    pub blocks: Vec<FeatureBlock>,
    pub total_dim: usize,
}
```

Rules:

- each feature appender owns one stable block
- blocks are concatenated in schema order
- schema ids should be serializable with models
- training and inference must use the same schema

## Feature matrices

Avoid nested vectors in the core execution path.

Preferred layout:

```rust
pub struct FeatureMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

pub struct FeatureMatrixView<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub data: &'a [T],
}

pub struct FeatureMatrixViewMut<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub data: &'a mut [T],
}

pub struct FeatureRowMut<'a, T> {
    pub data: &'a mut [T],
}
```

Rules:

- row-major layout
- contiguous storage
- caller-visible shape metadata
- workspace reuse across invocations
- borrowed views are first-class; owned matrices are not the only representation
- appenders should write into mutable views, not demand owned allocation
- row slicing and column-block slicing should be cheap

Supported scalar types:

- `i16` or `i32` for categorical/ordinal features
- `f32` for normalized dense features
- `f64` for training solvers where stability matters

## Score buffers and score matrices

Keep score storage separate from decoded outputs.

The data model should support two score shapes:

- one scalar score per position
- one score vector per position

Recommended conceptual shapes:

```rust
pub struct ScoreBuffer<S> {
    pub data: Vec<S>,
}

pub struct ScoreMatrix<S> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<S>,
}

pub struct ScoreMatrixView<'a, S> {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub data: &'a [S],
}

pub struct ScoreMatrixViewMut<'a, S> {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub data: &'a mut [S],
}
```

Use score vectors for:

- binary classification
- scalar regression
- change-point probability

Use score matrices for:

- multiclass classification
- IOB and related per-position tagging
- region labeling
- format classification over sampled positions or line starts

## Dataset views

Training should consume the same feature representation as inference.

Suggested shape:

```rust
pub struct DatasetView<'a, F, L> {
    pub features: FeatureMatrixView<'a, F>,
    pub labels: &'a [L],
}
```

Rules:

- training code should not require a separate feature container type
- corpus ingestion may be streamed, but fit kernels should still converge on matrix views
- labels must align one-to-one with feature rows

## Decoded outputs

The decoder should produce task-level objects, not mutate the source text.

Suggested output types:

- `Vec<ByteSpan>` for segmentation
- `Vec<BoundaryEvent>`
- `Vec<LabelAtPos>`
- `Vec<LabeledSpan>`
- `Vec<RegressionAtPos>`

Keep “insert marker into string” behavior as an adapter in higher layers only.

## Chunking model

Chunks operate on byte ranges.

Each chunk needs:

- `chunk_start`
- `chunk_end`
- `left_overlap`
- `right_overlap`

Invariants:

- overlap width >= max feature context width
- decoded outputs are emitted only for the owned interior region
- duplicate candidates in overlap are removed before merge
- workers own disjoint scratch and output buffers

## Workspaces and scratch

Reusable workspaces are part of the data model, not an implementation afterthought.

Suggested conceptual pieces:

- `ScanScratch`
- `FeatureScratch`
- `ModelScratch`
- `PipelineWorkspace`
- `ScoringWorkspace`

Responsibilities:

- own reusable candidate buffers
- own reusable feature and score buffers
- hold temporary gather or normalization scratch
- support thread-local reuse in parallel execution

Suggested conceptual shapes:

```rust
pub struct PipelineWorkspace<F, S> {
    pub candidates: CandidateBuffer,
    pub features: FeatureMatrix<F>,
    pub scores: ScoreBuffer<S>,
    pub feature_scratch: FeatureScratch,
}

pub struct ScoringWorkspace<F, S> {
    pub positions: PositionBuffer,
    pub features: FeatureMatrix<F>,
    pub scores: ScoreMatrix<S>,
    pub feature_scratch: FeatureScratch,
}
```

Rule:

- all expensive hot-path allocations should eventually migrate into workspace-owned storage

## Serialization expectations

Serialized model artifacts should store:

- model family and version
- feature schema id
- window spec
- scanner config
- decoder config
- weights or tree payload
- training metadata

Do not serialize transient workspace buffers.
