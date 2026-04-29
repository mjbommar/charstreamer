# API Surface

## Goal

Define the intended public API for Rust and Python so implementation stays aligned
with user-facing ergonomics and low-level performance constraints.

## Design principles

- direct and explicit
- batch-first
- allocation-aware
- primitive-first
- same core semantics in Rust and Python
- convenience layers on top of reusable workspaces
- optimize low-level contracts once and let higher layers inherit the speedup

## Rust API layers

### Layer 0: data views and buffers

Audience:

- internal crates
- performance-sensitive users
- primitive implementers

Examples:

- text views
- position/range types
- candidate buffers
- matrix views
- workspace storage

Rules:

- caller controls buffers
- no hidden text mutation
- byte offsets are canonical
- owned and borrowed forms must share the same semantics

### Layer 1: execution primitives

Audience:

- kernel implementers
- model implementers
- pipeline composers

Examples:

- candidate scanners
- feature appenders
- feature kernels
- batch predictors
- decoders

Rules:

- primitive contracts should stay narrow and stable
- portable and optimized implementations must share the same contract
- primitives should be task-agnostic

### Layer 2: pipeline combinators

Audience:

- normal Rust consumers
- benchmark harnesses
- Python binding layer

Suggested types:

```rust
pub struct Pipeline<S, K, M, D> { ... }
pub struct PipelineWorkspace<F, S> { ... }
```

Methods:

- `scan_candidates`
- `extract_features`
- `predict_scores`
- `decode`
- `run`
- `run_into`

These types should orchestrate primitives. They should not invent a second hidden
execution model.

### Layer 3: task adapters

Audience:

- application code
- migration from `charboundary`

Examples:

- `segment_to_spans`
- `segment_to_sentences`
- `segment_to_boundaries`
- `score_positions`

These are adapters on top of the generic pipeline.

## Rust traits

The primitive traits should be small and composable.

### Scanner

```rust
pub trait CandidateScanner {
    fn scan_into(
        &self,
        text: TextBytes<'_>,
        range: ScanRange,
        out: &mut CandidateBuffer,
    );
}
```

Rules:

- scanner output is sorted by byte offset
- output positions are byte offsets, even for Unicode-aware tasks
- optimized implementations may specialize internally, but not change semantics

### Feature appender

```rust
pub trait FeatureAppender<T> {
    fn block(&self) -> &FeatureBlock;
    fn append_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        out: FeatureMatrixViewMut<'_, T>,
        scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError>;
}
```

Rules:

- an appender owns a stable column block
- `out` is already sliced to the appender's block width
- appenders must not allocate destination storage
- the same appender must be reusable in both training and inference

### Feature kernel

```rust
pub trait FeatureKernel<T> {
    fn schema(&self) -> &FeatureSchema;
    fn extract_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        out: FeatureMatrixViewMut<'_, T>,
        scratch: &mut FeatureScratch,
    ) -> Result<(), FeatureError>;
}
```

Recommended implementation pattern:

- a composite kernel owns ordered appenders
- it slices the destination matrix into block views
- each appender fills its own block

### Model predictor

```rust
pub trait BatchPredictor<F, S> {
    fn predict_into(
        &self,
        features: FeatureMatrixView<'_, F>,
        out: &mut [S],
    ) -> Result<(), PredictError>;
}
```

Rules:

- predictors consume views, not necessarily owned matrices
- predictors write into caller-owned score buffers
- models should be batch-oriented by default

### Matrix scorer

```rust
pub trait BatchScorer<F, S> {
    fn score_dim(&self) -> usize;
    fn score_into(
        &self,
        features: FeatureMatrixView<'_, F>,
        out: ScoreMatrixViewMut<'_, S>,
    ) -> Result<(), PredictError>;
}
```

Use this shape when there are multiple scores or logits per position.

### Trainable predictor

```rust
pub trait TrainablePredictor<F, L> {
    type FitOptions;
    type FitReport;

    fn fit(
        dataset: DatasetView<'_, F, L>,
        options: &Self::FitOptions,
        scratch: &mut FitScratch,
    ) -> Result<(Self, Self::FitReport), FitError>
    where
        Self: Sized;
}
```

Rules:

- native training should operate on the same matrix layout used by inference
- corpus-facing training APIs may stream input, but fitting should terminate in a shared dataset view contract

### Decoder

```rust
pub trait Decoder<S, O> {
    fn decode_into(
        &self,
        positions: CandidateSlice<'_>,
        scores: &[S],
        out: &mut Vec<O>,
    ) -> Result<(), DecodeError>;
}
```

Rules:

- decoding is separate from scoring
- decoders should not mutate source text
- higher-level string adapters should sit above this layer

### Score-matrix decoder

```rust
pub trait ScoreDecoder<S, O> {
    fn decode_scores_into(
        &self,
        text: TextBytes<'_>,
        positions: CandidateSlice<'_>,
        scores: ScoreMatrixView<'_, S>,
        out: &mut Vec<O>,
    ) -> Result<(), DecodeError>;
}
```

Use this shape for:

- multiclass labeling
- region labeling
- change-point tasks

## Workspace APIs

Every hot path should have:

- an allocating convenience method
- a low-level `*_into` method

Examples:

```rust
pipeline.run(text)
pipeline.run_into(text, &mut workspace, &mut outputs)
model.predict(features_view)
model.predict_into(features_view, out)
```

### Workspace contents

`PipelineWorkspace` should eventually own:

- candidate buffers
- feature matrices
- score buffers
- gather scratch
- feature scratch
- model scratch

These buffers should be partitionable for chunk-parallel work.

For multiclass or region tasks, use a workspace that stores a score matrix rather
than a single score buffer.

## Training APIs

### Dataset-facing APIs

Support:

- in-memory dense feature matrices
- streamed text corpus ingestion for feature extraction
- candidate-sampled training corpora

Suggested entry points:

- `CorpusFeaturizer::scan_and_featurize`
- `Trainer::fit_from_features`
- `Trainer::fit_from_corpus`
- `Trainer::cross_validate`
- `Trainer::calibrate_threshold`

Important rule:

- `fit_from_corpus` must route through the same scanner and feature kernel contracts used by inference
- do not create a separate training-only feature path unless it is a thin adapter over the same primitives

### Reports

Fit reports should include:

- loss or objective summary
- training time
- feature dimension
- class distribution
- threshold calibration output
- metrics on holdout or validation data

## Python API

Python should mirror the same layers, but only expose the useful ones.

### High-level objects

Suggested surface:

```python
from charstreamer import BoundaryPipeline

pipe = BoundaryPipeline.default_sentence_model()
spans = pipe.segment_to_spans(text)
scores = pipe.score_positions(text)
```

And for broader tasks, expect a future surface shaped more like:

```python
from charstreamer import RegionPipeline

pipe = RegionPipeline.default_format_classifier()
regions = pipe.label_regions(text)
```

### Lower-level Python APIs

- `scan_candidates(text) -> np.ndarray[np.uint32]`
- `extract_features(text, positions=None) -> np.ndarray`
- `predict_features(features) -> np.ndarray`
- `decode_positions(positions, scores) -> list[tuple[int, int]]`

### Python input types

Accept:

- `str`
- `bytes`
- contiguous NumPy arrays for feature scoring

### Python output rules

- spans returned to Python should be documented as byte spans or character spans, never ambiguous
- if returning character spans for `str`, document the conversion cost
- release the GIL during scanning, feature extraction, prediction, and training
- do not expose an alternate Python-only feature format that diverges from the Rust core

## Serialization-facing APIs

Support:

- `save_native(path)`
- `load_native(path)`
- optional backend export/import adapters

Keep the native format authoritative for first-party models.

## Non-goals

- hidden mutation of source text in core APIs
- materialized per-candidate window objects as a required execution step
- requiring Python for model loading
- callback-heavy inference APIs on the hot path
- storing all intermediate artifacts in public return values
