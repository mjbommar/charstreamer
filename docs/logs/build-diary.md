# Build Diary

## 2026-04-29

Focus:

- reduce the external Rust model-backend path to Burn only
- prepare the repository for its first public source commit

Changes:

- removed SmartCore and Linfa backend crates from the workspace
- removed SmartCore and Linfa model variants from the manifest runner
- deleted stale SmartCore/Linfa experiment specs and the SmartCore-only example
- simplified the synthetic boundary trainer to a Burn-only MLP path
- removed the stale workspace-level `ndarray` pin; Burn still brings its own
  transitive `ndarray`
- added root release files: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `LICENSE-MIT`, and `LICENSE-APACHE`
- tightened `.gitignore` so generated data, model artifacts, logs, build
  outputs, and local Python environments are not commit candidates
- replaced the span-generator shard helper's local `~/.bashrc` API-key scrape
  with a standard `OPENAI_API_KEY` environment check
- corrected the workspace MSRV from `1.85` to `1.89` because Burn `0.20.0`
  requires Rust `1.89`
- added package metadata and internal dependency versions for publishable crates
- fixed clippy warnings in library code
- added GitHub Actions CI and tag-driven release workflows for the single PyPI
  wheel path

Validation:

- `cargo test -p charstreamer-backend-burn`
- `cargo test -p charstreamer-experiments --example train_synthetic_boundary_burn`
- `cargo test --workspace`
- `cargo clippy --workspace --lib`
- `cargo doc --workspace --no-deps`
- `uv run pytest` in `tools/span-generator`
- `cargo package -p charstreamer-core --allow-dirty`
- `cargo package --list` for dependent publishable crates
- release rerun:
  `cargo run --release -p charstreamer-experiments --example train_synthetic_boundary_burn -- --input data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.jsonl --report /tmp/charstreamer-synthetic-boundary-burn-5k-after-backend-cleanup.json --epochs 32 --batch-size 1024 --hidden-dim 128 --hidden-dim2 64 --encoded-left 7 --encoded-right 7 --count-radius 24 --seed 19`

## 2026-04-27

Focus:

- test 3-5 additional model families on the best current full-corpus feature
  stack under single-thread benchmark settings
- check whether any new backend can pareto-dominate the current directional tree

Changes:

- added a new `charstreamer-backend-linfa` crate with a logistic-regression
  adapter built on `linfa-logistic`
- extended the `smartcore` backend layer with additional adapters for logistic
  regression, KNN, and Gaussian naive Bayes
- wired the new model families into the manifest-driven experiment system
- added checked-in full-corpus experiment manifests for directional-feature
  logistic, RF, KNN, and Gaussian-NB runs
- ran the full-corpus single-thread alternative-model sweep on the current
  directional feature stack

Validation:

- `cargo fmt --all`
- `cargo test -p charstreamer-backend-smartcore -p charstreamer-backend-linfa -p charstreamer-experiments`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-linfa-logistic-directional-full.json /tmp/current-legal-linfa-logistic-directional-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-smartcore-logistic-directional-full.json /tmp/current-legal-smartcore-logistic-directional-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-rf-directional-class-counts-small-full.json /tmp/current-legal-rf-directional-class-counts-small-full-report.json`
- `RUST_MIN_STACK=33554432 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-smartcore-gaussian-nb-directional-full.json /tmp/current-legal-smartcore-gaussian-nb-directional-full-report.json`

Notes:

- `current_legal_smartcore_knn_directional_full` was started under the same
  single-thread settings but abandoned after several minutes of CPU-bound fit
  time; it was not a practical Pareto candidate
- `smartcore` Gaussian-NB training on this full dataset currently aborts with a
  release-mode stack overflow
- none of the newly tested alternative backends beat the current directional
  tree on both SCOTUS F1 and single-thread throughput
- the best full-corpus single-thread default remains
  `current_legal_tree_directional_class_counts_full`
- added a first Burn neural backend using the `ndarray` backend with
  `openblas-system`
- implemented and tested five Burn neural model families on the current
  directional feature stack: shallow MLP, deep MLP, window CNN, window GRU,
  and window LSTM
- none of the tested Burn neural variants pareto-dominated the current
  directional tree; the best Burn compromise was the window CNN, while the best
  Burn SCOTUS F1 came from the LSTM at a much lower throughput
- added a first Candle backend using `candle-core` and `candle-nn` on the CPU
  device
- implemented and tested five Candle neural model families on the current
  directional feature stack: shallow MLP, deep MLP, window CNN, window GRU,
  and window LSTM
- added a second Candle tuning pass through checked-in manifests so hidden
  dimensions, projection dimensions, epochs, batch sizes, learning rates, and
  weight decay stay configuration-driven
- none of the tested Candle variants pareto-dominated the current directional
  tree; the strongest Candle SCOTUS F1 stayed with the recurrent models, but
  their single-thread throughput remained lower than the tree

## 2026-04-26

Focus:

- initial exploration of `charboundary`
- definition of `charstreamer` architecture direction
- establishment of documentation/planning system
- research of official Rust and PyO3/maturin best practices
- refinement of the design around reusable low-level primitives

Changes:

- created architecture reference for byte-first generic pipeline
- created docs index, reference docs, plan docs, quality docs, templates, and logs
- added `docs/rust.md` and `docs/python.md`
- synthesized implementation standards into `AGENTS.md`
- added a dedicated primitive reference and revised architecture, API, backlog,
  and quality docs to enforce optimize-once primitive reuse
- scaffolded the Cargo workspace and the first three crates
- implemented the first narrow primitive-first vertical slice:
  byte positions and views, candidate buffers, feature schema/matrix views,
  `memchr`-backed scanning, reusable feature appenders, a native logistic scorer,
  a threshold span decoder, an end-to-end example, tests, and Criterion benches
- generalized the core beyond binary boundary detection:
  added score-matrix pipelines, multiclass decoding, stride and line-start
  scanners, a multiclass linear scorer, and a second end-to-end region-labeling
  slice for XML-to-CSV format switches
- aligned the reference docs with the generalized implementation:
  documented position buffers, score matrices, scoring workspaces, and added a
  dedicated task list for the generic scoring slice
- added the first native training and evaluation slice:
  ALEA and MultiLegal corpus loaders, candidate-dataset builders, binary
  metrics and throughput utilities, a native logistic fit path, a richer
  legal-boundary feature kernel, and a sample end-to-end training example
  against the legal sentence datasets
- added the first persisted native model artifact and model-level benchmarks:
  versioned logistic JSON save/load with schema metadata, structured sample
  training reports, and Criterion coverage for native logistic scoring and fit
- added manifest-driven experiment and parity infrastructure:
  a legacy UTF-8 candidate scanner, a parity-oriented `CharBoundaryLegacyAppender`,
  a `smartcore` random-forest backend, `BoundaryExperimentSpec` and
  `ParityCheckSpec`, JSON spec loaders with relative-path resolution, runnable
  example drivers, and checked-in spec manifests under `specs/`
- executed the first controlled docking runs:
  a legacy parity manifest with exact Rust/Python feature-row agreement,
  a legacy small-window logistic experiment, a legacy small-window random-forest
  experiment, and a current legal logistic experiment against ALEA plus SCOTUS
- added an exact sklearn baseline path for legacy-model docking:
  installed user-local `scikit-learn`, added a Python/sklearn RF backend to the
  experiment layer, introduced a checked-in sklearn manifest, and changed the
  experiment harness to batch corpus prediction for Python-backed models so the
  baseline is slow but still usable
- added another Rust-side classifier family and a comparison harness:
  a `smartcore` decision-tree backend with probability outputs, a checked-in
  tree manifest, and a sweep runner that prints side-by-side throughput/F1
  summaries across multiple manifests
- expanded the legacy parity sweep space:
  added medium (`7/5`) and large (`9/7`) legacy manifests for logistic and tree,
  plus markdown sweep output
- tested the current fast legal feature stack with a tree model:
  added a `current_legal_tree` manifest and found that it materially improves
  quality over the current logistic fast path while preserving most of the speed
- exposed count-based feature primitives in the manifest layer:
  `SelectedByteCounts` and `LineByteCounts` are now configurable experiment features
- ran the first fast-path feature-engineering sweeps:
  local and line count ablations, then a punctuation-vs-structure split on the
  tree path, then a structure-count check on the logistic path
- checked in a `docs/results.md` reference with real saved outputs and current
  default-candidate recommendations
- added reusable byte-class density primitives:
  a symmetric `ByteClassCountAppender` and an asymmetric
  `DirectionalByteClassCountAppender`, both exposed through manifest specs
- ran another round of fast-tree sweeps:
  symmetric class-count ablation, directional class-count ablation, and a
  tree-hyperparameter sweep on the current best local-structure preset
- confirmed that `current_legal_tree_local_structure` still generalizes best on
  the current reduced SCOTUS sweep, even though some class-density variants
  improve in-split ALEA validation scores
- tightened full-corpus legacy docking:
  fixed ALEA explicit-boundary semantics, widened the legacy scanner to include
  paragraph terminals, aligned the sampler with the original always-draw logic,
  made dataset and sklearn randomness optional/configurable, and reran
  full-corpus legacy manifests plus a direct original-Python baseline under the
  same boundary metric
- added full-corpus hybrid manifests that compose the exact legacy feature block
  with newer structural-count features through config only
- ran full-corpus Rust-native model searches under single-thread benchmark settings:
  current local-structure tree, directional-count tree, structure+directional tree,
  legacy tree, hybrid legacy+structure tree, hybrid legacy+directional tree, and
  directional-tree tuning variants

Validation:

- repo inspected locally
- `cargo fmt --all`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`
- `cargo bench --workspace --no-run`
- `cargo bench -p charstreamer-kernels --bench primitives -- --sample-size 10`
- `cargo bench -p charstreamer-core --bench pipeline -- --sample-size 10`
- `cargo bench -p charstreamer-models-native --bench logistic -- --sample-size 10`
- `cargo run -p charstreamer-core --example narrow_slice`
- `cargo run -p charstreamer-core --example format_switch`
- `cargo run -p charstreamer-core --example train_sample_boundary`
- `CHARSTREAMER_MODEL_OUT=/tmp/charstreamer-sample-model.json CHARSTREAMER_REPORT_OUT=/tmp/charstreamer-sample-report.json cargo run -p charstreamer-core --example train_sample_boundary`
- `cargo run -p charstreamer-experiments --example run_parity_spec -- specs/parity/charboundary-legacy-sample.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-logistic.json /tmp/charboundary-small-logistic-report.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-rf.json /tmp/charboundary-small-rf-report.json`
- `python3 -m pip install --user --break-system-packages scikit-learn`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-python-rf.json /tmp/charboundary-small-python-rf-report.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-tree.json /tmp/charboundary-small-tree-report.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- specs/experiments/charboundary-small-logistic.json specs/experiments/charboundary-small-tree.json specs/experiments/charboundary-small-rf.json specs/experiments/current-legal-logistic.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/charboundary-window-sweep.md specs/experiments/charboundary-small-logistic.json specs/experiments/charboundary-small-tree.json specs/experiments/charboundary-medium-logistic.json specs/experiments/charboundary-medium-tree.json specs/experiments/charboundary-large-logistic.json specs/experiments/charboundary-large-tree.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree.json /tmp/current-legal-tree-report.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- specs/experiments/current-legal-logistic.json specs/experiments/current-legal-tree.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree-counts.json /tmp/current-legal-tree-counts-report.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/current-legal-tree-ablation.md specs/experiments/current-legal-tree.json specs/experiments/current-legal-tree-local-counts.json specs/experiments/current-legal-tree-line-counts.json specs/experiments/current-legal-tree-counts.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/current-legal-tree-local-split.md specs/experiments/current-legal-tree.json specs/experiments/current-legal-tree-local-punct.json specs/experiments/current-legal-tree-local-structure.json specs/experiments/current-legal-tree-local-counts.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- specs/experiments/current-legal-logistic.json specs/experiments/current-legal-logistic-local-structure.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-logistic.json /tmp/current-legal-logistic-report.json`
- `cargo test -p charstreamer-experiments`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/current-legal-tree-class-ablation.md specs/experiments/current-legal-tree.json specs/experiments/current-legal-tree-local-structure.json specs/experiments/current-legal-tree-class-counts.json specs/experiments/current-legal-tree-structure-class-counts.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/current-legal-tree-directional-ablation.md specs/experiments/current-legal-tree.json specs/experiments/current-legal-tree-local-structure.json specs/experiments/current-legal-tree-directional-class-counts.json specs/experiments/current-legal-tree-structure-directional-class-counts.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/current-legal-tree-local-structure-tuning.md specs/experiments/current-legal-tree-local-structure.json specs/experiments/current-legal-tree-local-structure-balanced.json specs/experiments/current-legal-tree-local-structure-shallow.json specs/experiments/current-legal-tree-local-structure-entropy.json`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo bench -p charstreamer-kernels --bench primitives -- --sample-size 10`
- `cargo run --release -p charstreamer-experiments --example run_parity_spec -- /tmp/alea_parity_real.json`
- `cargo run --release -p charstreamer-experiments --example run_parity_spec -- /tmp/scotus_parity_real.json`
- `cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-python-rf-full.json /tmp/charboundary-small-python-rf-full-report.json`
- `PYTHONPATH=/home/mjbommar/projects/personal/charboundary python3 <direct full-corpus charboundary retrain/eval script>`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree-local-structure-full.json /tmp/current-legal-tree-local-structure-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree-directional-class-counts-full.json /tmp/current-legal-tree-directional-class-counts-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree-structure-directional-class-counts-full.json /tmp/current-legal-tree-structure-directional-class-counts-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-tree-full.json /tmp/charboundary-small-tree-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-tree-hybrid-structure-full.json /tmp/charboundary-small-tree-hybrid-structure-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/charboundary-small-tree-hybrid-directional-full.json /tmp/charboundary-small-tree-hybrid-directional-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree-directional-class-counts-deep-full.json /tmp/current-legal-tree-directional-class-counts-deep-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_spec -- specs/experiments/current-legal-tree-directional-class-counts-entropy-full.json /tmp/current-legal-tree-directional-class-counts-entropy-full-report.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release -p charstreamer-experiments --example run_experiment_sweep -- --markdown-out /tmp/current-legal-window-radius-sweep.md specs/experiments/current-legal-tree-directional-class-counts-window-3-1-full.json specs/experiments/current-legal-tree-directional-class-counts-window-3-3-full.json specs/experiments/current-legal-tree-directional-class-counts-window-7-7-full.json specs/experiments/current-legal-tree-directional-class-counts-window-9-7-full.json specs/experiments/current-legal-tree-directional-class-counts-window-3-1-radius-6-full.json specs/experiments/current-legal-tree-directional-class-counts-window-3-1-radius-24-full.json specs/experiments/current-legal-tree-directional-class-counts-window-3-1-radius-36-full.json`

Decisions:

- internal representation will be byte-first
- native first-party focus will start with linear/logistic and shallow trees
- planning artifacts will live under `docs/`
- implementation guidance will follow official Rust docs, PyO3 docs, and maturin docs where practical
- the project will be built from reusable primitives first and task pipelines second
- the first native training target is candidate-first binary logistic regression
  over reusable feature matrices, with threshold calibration separated from fit
- prior-vs-current docking runs must flow through explicit manifests and parity
  checks instead of informal parameter matching
- exact legacy-model docking may use a Python/sklearn backend inside the
  experiment layer, but should batch prediction by corpus instead of spawning
  one Python process per document
- the `smartcore` adapter layer should expose both tree and forest baselines so
  model-family sweeps do not require Python
- larger legacy windows (`7/5`, `9/7`) are not automatically better; on the
  current reduced sweep they materially hurt logistic and only marginally change
  tree quality
- the current legal feature stack combined with a tree is now the strongest
  default candidate seen so far, because it materially improves over the fast
  logistic path while preserving much higher throughput than the legacy tree/RF
- within the fast tree path, local structure-count features are a better
  addition than line counts or the combined punctuation+structure count block
- the same structure-count block hurts the logistic path, so feature additions
  need to be tuned per model family, not assumed to transfer
- symmetric byte-class density features can look attractive on ALEA validation
  while hurting reduced SCOTUS generalization, so they are not a safe default
- directional class-density features are a better reusable primitive than
  symmetric class counts, but still do not beat the simpler structure-count
  preset on the current sweep
- the current best reduced-sweep tree preset is
  `current_legal_tree_local_structure`, but the full-corpus default candidate
  has moved to `current_legal_tree_directional_class_counts_full`
- switched the Unicode feature work from an ad hoc category crate to the same
  ICU General Category stack used in `alea-preprocess`
- added configuration-driven ICU Unicode category and category-group directional
  count appenders to the feature compiler
- kept the byte fast path intact: Unicode decode/category lookup is only paid
  when a Unicode-aware feature block is explicitly configured
- added a full-corpus window sweep on the current feature stack and found that a
  shorter encoded window helps: `3/1` strictly beats the old `5/3` directional
  baseline on both SCOTUS F1 and single-thread throughput
- added a Unicode feature sweep on the same full-corpus tree setup; the new
  ICU-based category/group features are correct and configurable, but they do
  not yet beat the byte/ASCII directional-count default on the current legal
  corpora
- added a follow-up sweep over larger current-stack encoded windows (`7/7`,
  `9/7`) and decoupled directional count radii (`6/6`, `24/24`, `36/36`)
- confirmed that the shorter encoded-window conclusion is stable: larger
  encoded windows improve validation metrics slightly but do not improve SCOTUS
  F1, while too-small count radii lose generalization even when they speed up
  validation throughput
- kept `current_legal_tree_directional_class_counts_window_3_1_full` as the
  default and `current_legal_tree_directional_class_counts_window_3_3_full` as
  the quality-oriented neighboring point after the broader sweep
- researched OpenAI Structured Outputs, Batch, Prompt Caching, Flex, Background
  mode, and Evals as a synthetic-data and weak-labeling stack for semantic
  segmentation
- researched PydanticAI output modes, validators, retries, durable execution,
  and evals as the orchestration layer for typed weak-label generation
- documented a broader `charstreamer` direction beyond sentence boundaries:
  paragraph, section, dialogue, list-item, and entity-aware segmentation with
  candidate-ID or line-index supervision instead of raw LLM byte offsets
- scaffolded a standalone `uv` utility at `tools/span-generator/` for weak-label
  generation from streamed Hugging Face text corpora
- implemented deterministic tagged-text validation so byte offsets are derived
  locally from `<|label|>...<|/label|>` markup rather than guessed by the model
- verified the streaming path against
  `alea-institute/kl3m-data-sample-005-shuffled` without downloading the full
  dataset and wrote sample JSONL output through the OpenAI annotation path

Next step:

- keep improving generic feature primitives and the native model stack:
  feature engineering on top of the `3/1` and `3/3` frontier, SIMD-friendly
  feature kernels, and eventually native tree inference/training to reduce
  adapter overhead

## 2026-04-29

- removed the Candle backend crate from the active workspace
- removed Candle model variants from the experiment spec and compiled model
  dispatch path
- deleted checked-in Candle experiment manifests so new runs cannot select the
  dropped backend accidentally
- kept Burn as the neural backend to validate going forward
- split the Python package into a thin wrapper plus `_native` PyO3 module so
  model artifact resolution can happen before the Rust hot path is called
- added default model bundle validation and vendoring tools under
  `tools/model-artifacts/`
- added release gates that reject model-backed wheels without a validated Burn
  model bundle and offline `require_model=True` startup
- documented the model artifact manifest, runtime resolution order, and
  remaining model-backed release tasks
