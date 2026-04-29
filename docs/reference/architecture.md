# CharStreamer Architecture

## Goal

Build a Rust-first library for:

- scanning byte or character streams
- collecting rolling windows around candidate positions
- extracting fixed-width features
- scoring those features with a pluggable model
- decoding the scores into boundaries, labels, or regressions
- exposing the same engine to Rust and Python

This should generalize the current `charboundary` approach without baking in
"sentence segmentation" as the only task.

## Design law

The library should be built from reusable primitives first and pipelines second.

The implementation target is not "one fast sentence segmenter." The target is a
small set of byte-oriented, allocation-aware primitives that can be optimized
once and then reused across:

- inference and training
- single-threaded and parallel execution
- portable and SIMD kernels
- Rust and Python APIs

The detailed primitive inventory lives in [primitives.md](primitives.md).

## What `charboundary` already gets right

- It reduces inference to candidate positions instead of scoring every character.
- It uses small fixed windows, which are cache-friendly.
- It separates encoding, feature extraction, model scoring, and decoding.
- It samples training data aggressively, which matters for boundary detection.

## What should change in the new design

- Make bytes the primary storage unit and UTF-8 character indexing optional.
- Separate the pipeline stages into explicit traits built on reusable primitives.
- Avoid Python/NumPy as the hot path.
- Make model execution pluggable: pure Rust, ONNX, or user-defined.
- Prefer zero-allocation batch APIs over per-position calls.
- Design for runtime CPU dispatch, chunked parallelism, and preallocated output.
- Avoid materialized window objects in the hot path.

## Lessons from `StringZilla` and `NumKong`

`StringZilla` is a strong model for the scanning layer:

- very fast byte-oriented search and split primitives
- SIMD-first kernels with runtime ISA dispatch
- APIs that are useful before any ML is involved

`NumKong` is a good model for the scoring layer:

- explicit hardware-aware kernels
- packed reusable state instead of rebuilding per call
- `out=` style APIs and batch execution
- Python bindings that expose fast native kernels while releasing the GIL

`charstreamer` should combine those ideas:

- `StringZilla` style scanning and byte windowing
- `NumKong` style packed models, reusable workspaces, and batch prediction

## Primitive-first stack

The core stack should be:

1. text and position views
2. scan primitives
3. gather/context primitives
4. feature appenders
5. matrix and score views
6. batch predictors and trainers
7. decoders
8. workspaces and dispatch

That stack matters more than any one task pipeline. A sentence-boundary pipeline
is just one composition of those primitives.

## Recommended workspace layout

```text
charstreamer/
  Cargo.toml
  crates/
    charstreamer-core/
    charstreamer-kernels/
    charstreamer-models-native/
    charstreamer-backend-burn/
    charstreamer-py/
  benches/
  examples/
  python/
    pyproject.toml
    charstreamer/__init__.py
```

## Crate responsibilities

### `charstreamer-core`

Core types and orchestration:

- `BytePos`, `ByteSpan`, range types, window specs
- `TextBytes<'a>` and derived views
- candidate buffers and matrix views
- workspace buffers
- primitive traits
- pipeline combinators

### `charstreamer-kernels`

Hot loops and CPU-specific implementations:

- byte classification tables and bytesets
- candidate search kernels
- rolling gather/indexing helpers
- reusable feature appenders
- SIMD or ISA-specialized primitive implementations
- optional adapters around `StringZilla` where it wins

### `charstreamer-models-native`

Native model training and scoring:

- linear / logistic models
- shallow tree models
- serialization for packed native models

### Optional backend crates

Adapters only:

- `charstreamer-backend-burn`

### `charstreamer-py`

PyO3 bindings:

- Python-facing pipeline classes
- NumPy buffer support
- bytes and `str` adapters
- GIL-released batch inference

## Conceptual pipeline vs physical execution

Conceptually, the library does this:

1. `scan_candidates`
2. `gather_context`
3. `append_features`
4. `predict_batch`
5. `decode_outputs`

Physically, the implementation should avoid materializing a separate window layer.

The intended execution pattern is:

- scanner emits candidate byte offsets into a reusable buffer
- feature schema allocates a dense destination matrix view
- feature appenders read the text and positions directly and write their column blocks
- model scores the matrix view in batch
- decoder emits task outputs from either a score vector or a score matrix

Important rules:

- `build_windows` is a conceptual stage, not a required allocation
- the hot path should not create `Vec<Window>` or equivalent unless benchmarks prove it is better
- training and inference must share the same scan and feature primitives

## Two scoring shapes

The architecture should support both:

- `positions -> features -> one score per position -> decoder`
- `positions -> features -> many scores per position -> decoder`

The first shape covers binary boundary detection.

The second shape covers:

- multiclass tagging
- IOB-style per-position labeling
- region classification
- change-point and format-switch detection

## Reuse boundary

The primitive layer should contain:

- byteset scanning
- byte classification
- window indexing helpers
- feature block appenders
- matrix views
- packed model scoring

The task layer should contain:

- default sentence-boundary scanners
- task-specific heuristic blocks
- threshold presets
- output adapters

Do not let task adapters become the place where fast paths live.

## Data model

### Positions

Internally, use byte offsets.

That gives the fastest scanning story and matches the hardware. For Python and
Unicode-heavy use cases, provide optional mappings:

- byte offset -> UTF-8 scalar index
- byte span -> Python slice-compatible codepoint span

Do not make Unicode scalar indexing the default hot path.

### Bytes vs characters

Treat these as separate execution families, not one implementation with a few
`if unicode` branches.

#### Byte-first path

Use this for:

- candidate scanning
- ASCII-heavy corpora
- delimiter and punctuation detection
- ngram hashing
- most high-throughput inference

Properties:

- offsets are byte offsets
- classes come from byte lookup tables
- SIMD scanning is straightforward
- chunking and overlap logic is simple

#### UTF-8 scalar path

Use this when features need:

- Unicode case/category information
- non-ASCII quote or punctuation logic
- language-specific codepoint-aware heuristics
- Python-facing character spans

Properties:

- requires UTF-8 decoding
- hot loops are different from raw byte loops
- SIMD opportunities exist, but in different places
- should be built on top of a decoded side table, not mixed into the byte scanner

#### Recommended architecture

Keep one canonical storage format:

- input text stored as `&[u8]`

Then add optional derived views:

- `AsciiByteView`
- `Utf8ScalarView`
- `ByteToCharMap`

The scanner should stay byte-first. Unicode-aware feature kernels should consume
the derived UTF-8 view when needed.

## Optimize-once rule

All heavy optimization work should land in reusable primitive implementations:

- portable scan kernels
- SIMD scan kernels
- gather helpers
- reusable feature appenders
- packed linear/tree scorers
- chunk-parallel orchestration

Higher-level pipelines should benefit automatically from those optimizations.

## Anti-patterns

Avoid these shapes:

- task-specific hot loops outside reusable kernels
- materialized per-candidate window objects
- `Vec<Vec<_>>` feature storage
- mandatory Unicode maps in the byte fast path
- separate training-only and inference-only featurization logic

## Concrete first target

The first production-quality target should be a generic boundary detector:

- input: `&[u8]` or `&str`
- candidate scanner: punctuation / quote / newline byteset
- feature kernel: reusable appenders for byte windows, ASCII classes, and heuristic channels
- model: logistic regression or shallow tree ensemble
- decoder: threshold + post-rules -> boundary byte spans

That directly covers `charboundary`, but the same pipeline can handle:

- boundary classification
- per-position regression
- multi-label tagging at candidate positions

The first vertical slice should prove the primitive layer, not bypass it.

## Generic task examples

This architecture should support at least these task families:

- sparse candidate classification
  Example: punctuation positions that may be sentence boundaries

- dense or stride-based tagging
  Example: every byte boundary or every token boundary gets a label

- region labeling and change-point detection
  Example: line starts or fixed-stride offsets labeled as XML vs CSV, then merged into spans

## Scanning strategy

This is where most of the cheap speedup lives.

### Baseline

- use `memchr` / `memchr2` / `memchr3` where possible
- use byte lookup tables for ASCII classes
- gather candidate positions before feature extraction

### Accelerated path

- add runtime dispatch for AVX2 / AVX-512 / NEON
- scan 32 to 64 bytes at a time using byteset masks
- keep chunk overlap only for feature extraction, not scanning

This is the layer most likely to benefit from `StringZilla` integration or from
borrowing the same design pattern.

## Feature extraction strategy

`charboundary` currently computes windows plus a few heuristic flags. The Rust
version should treat those heuristics as one `FeatureKernel`, not as the entire
system.

Recommended feature kernel families:

- `EncodedByteWindowKernel`
- `AsciiClassWindowKernel`
- `BoundaryHeuristicKernel`
- `Utf8CategoryKernel`
- `NgramHashKernel`

Those kernels should compose into a `CompositeKernel` without reallocating.

## Model strategy

Do not make the crate depend on a single ML runtime.

Start with:

- native logistic regression
- native linear regression
- optional ONNX Runtime backend

Then add:

- packed shallow tree ensembles if boundary models still want RF-style behavior

For very small feature vectors, a packed linear model will often beat a general
runtime and may be good enough for many segmentation tasks.

## Model families

For this problem family, the best candidates are not the same for all stages.

### Tier 1: linear / logistic models

Best first implementation target.

Why:

- trivial and very fast inference
- compact model files
- easy to batch and vectorize
- easy to train natively on CPU
- excellent fit for hand-crafted rolling-window features
- supports classification and regression with nearly the same machinery

Use for:

- boundary / non-boundary classification
- probability scoring with calibrated thresholds
- regression on local byte/char windows

Implementation priority:

- binary logistic regression
- multiclass softmax regression
- linear regression / ridge regression

### Tier 2: shallow trees

Good second target.

Why:

- captures feature interactions better than linear models
- still manageable to implement and serialize natively
- inference is predictable for shallow trees

Use for:

- rule-like problems where interactions matter
- cases where handcrafted heuristics have important nonlinear effects

Recommendation:

- support single decision trees first
- keep depth small
- use them mainly as interpretable baselines and teacher models

### Tier 3: random forests / extra trees

Useful, but not the first native target.

Why:

- often work well on tabular hand-engineered features
- low feature-engineering burden
- easy to get decent accuracy quickly

Costs:

- much heavier model files
- weaker cache locality
- slower inference than linear models
- training parallelism is easy, but memory traffic is high
- probability outputs are less clean than logistic models

Recommendation:

- support inference after linear models and shallow trees are solid
- support training if you need parity with `charboundary` style models
- consider Extra Trees as well; they can be simpler to parallelize efficiently

### Tier 4: boosted trees

Potentially strong, but probably not a first-party v1 target.

Why:

- can outperform forests on tabular features
- often compact relative to forests for the same quality

Costs:

- training implementation is substantially more complex
- histogram building and split finding become a major subsystem

Recommendation:

- treat as a future backend, not part of the initial core

### Tier 5: LSTM / small neural sequence models

Use only if there is evidence the feature-based models have hit a ceiling.

Why they are attractive:

- they can learn context directly from sequences
- they reduce manual feature design

Why they are unattractive here:

- training is much more expensive on CPU
- inference latency is worse for small local decisions
- recurrent execution is less SIMD-friendly than fixed-window linear models
- much harder to make a tiny, predictable, allocation-free serving core

Recommendation:

- if neural models are needed, consider small 1D CNNs or tiny MLPs before LSTMs
- LSTM support should come from a backend integration, not from a bespoke v1 implementation

### Practical recommendation

For `charstreamer`, I would prioritize:

1. logistic / linear
2. shallow decision tree
3. random forest / extra trees
4. tiny MLP
5. LSTM only if justified by benchmarks

That ordering matches both implementation cost and likely CPU efficiency.

## Training strategy

Inference and training should both exist in the library, but not every model
family needs the same level of first-party implementation.

### First-party training to own

Own these in Rust from the start:

- linear regression
- logistic regression
- ridge / L2-regularized variants
- possibly softmax regression
- shallow decision tree

Reason:

- small code surface
- easy to optimize for x64 and arm CPUs
- straightforward serialization
- stable and testable numerics

### Training to adopt before reimplementing

Adopt existing Rust crates or backends first for:

- random forest
- extra trees
- neural networks

Reason:

- much more implementation complexity
- less likely to be a differentiator for v1
- easier to replace later behind a model trait

### CPU optimization principles for training

Training code should be designed around:

- row-major contiguous feature matrices
- pre-binned or compact categorical features where possible
- Rayon parallelism over rows, trees, or batches
- explicit workspaces for gradients, histograms, and scratch buffers
- runtime ISA dispatch for hot reductions and dot products

For linear/logistic models specifically:

- implement dense `f32` and `f64` training
- support SGD, minibatch SGD, and L-BFGS or coordinate-style solvers
- add optional feature standardization and class weighting
- make threshold calibration a first-class post-fit step

For tree models:

- parallelize over candidate splits during node growth and over trees in ensembles
- prefer feature-column scratch views or binned histograms to reduce branchy scans
- keep serialization packed and traversal iterative

For neural models:

- do not build an autodiff engine just for this project
- use a backend framework if we decide neural training is necessary

## Parallelization model

Parallelize over independent text chunks, but preserve overlap:

- overlap size = maximum left/right window
- scan chunks independently
- drop duplicate candidate positions in overlap regions
- extract features and score per chunk
- merge decoded outputs in order

Use Rayon for Rust-side parallelism. For Python calls, release the GIL before
the pipeline starts and reacquire only when constructing Python objects.

## Workspace and allocation policy

Every public hot-path API should have two forms:

- convenience API that allocates
- low-level API that writes into caller-provided buffers

Example:

```rust
pipeline.predict(text)
pipeline.predict_into(text, &mut workspace, &mut outputs)
```

That is the `NumKong` pattern worth copying.

## Python API shape

The Python layer should look simple even if the Rust internals are generic.

Suggested Python surface:

```python
from charstreamer import BoundaryPipeline, WindowSpec, LinearBoundaryModel

pipe = BoundaryPipeline.default_sentence_model()
spans = pipe.boundaries(text)

scores = pipe.score_positions(text)
```

Also expose lower-level pieces:

- `scan_candidates(text) -> np.ndarray[np.uint32]`
- `extract_features(text, positions=None) -> np.ndarray`
- `predict_features(features) -> np.ndarray`

For Python users, accept both `str` and `bytes`.

## Serialization

Support a native packed model format first.

The file format should include:

- feature spec
- model type
- packed weights or trees
- threshold / decoder config
- version / ABI marker

This avoids forcing Python, pickle, or ONNX onto the Rust crate.

## Suggested implementation phases

### Phase 1

- workspace layout
- core traits
- byte candidate scanner
- single-threaded window kernel
- native logistic model
- PyO3 bindings

### Phase 2

- reusable workspaces
- Rayon chunk parallelism
- UTF-8 offset mapping
- NumPy array interop

### Phase 3

- SIMD scanning kernels
- SIMD feature kernels
- packed native model format
- benchmark suite against `charboundary`

### Phase 4

- ONNX backend
- tree ensemble inference
- custom kernel registration

## Benchmarks to require before calling it done

- bytes scanned per second
- candidate positions per second
- features extracted per second
- full pipeline latency on short texts
- throughput on large corpora
- Python overhead for `str` and `bytes`
- scaling across core counts

## Non-goals for the first version

- full tokenizer integration
- grapheme-cluster indexing in the hot path
- arbitrary Python callback features during inference
- training infrastructure beyond lightweight native models

## Immediate next step

Implement the minimal vertical slice:

- byte candidate scanner
- one composite feature kernel equivalent to current `charboundary` features
- native logistic model
- sentence-boundary decoder
- PyO3 bindings exposing `segment_to_spans`

That will validate the architecture before adding more model backends.
