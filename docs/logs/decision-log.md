# Decision Log

## 2026-04-29: Burn-only external Rust model backend

Context:

- Candle, SmartCore, and Linfa were evaluated as alternative model backends
- Burn is the only external Rust backend currently worth carrying for neural
  training/inference

Decision:

- remove Candle, SmartCore, and Linfa backend crates from the active workspace
- keep in-tree native models and Python sklearn docking as separate comparison
  utilities, not external Rust backend dependencies

Consequences:

- dependency surface is smaller and easier to productionize
- future backend additions require a documented quality/throughput reason

## 2026-04-26: Byte-first canonical representation

Context:

- the project needs both extreme throughput and optional Unicode-aware behavior

Decision:

- store text canonically as bytes and derive Unicode-aware views only when needed

Consequences:

- scanning and chunking stay simple and fast
- Python character spans require an explicit mapping layer

Related docs:

- [../reference/data-model.md](../reference/data-model.md)
- [../reference/architecture.md](../reference/architecture.md)

## 2026-04-26: Model implementation priority

Context:

- the library must support both training and inference on CPU

Decision:

- prioritize native linear/logistic models first, then shallow trees, then optional backends for heavier families

Consequences:

- early versions stay small and CPU-friendly
- random forests and neural models remain available through adapters later

Related docs:

- [../reference/model-families.md](../reference/model-families.md)

## 2026-04-26: Primitive-first contract layering

Context:

- the initial architecture described a good pipeline, but the reusable primitive
  layer was still too implicit

Decision:

- make low-level primitives first-class reference concepts and treat pipelines as
  compositions of those primitives
- treat `build_windows` as a conceptual stage, not a required materialized data
  structure in the hot path

Consequences:

- optimization work will concentrate in scanners, gather helpers, feature
  appenders, matrix views, and packed model kernels
- training and inference are expected to share the same feature-generation
  primitives
- backlog, test strategy, and benchmarks must all protect primitive reuse

Related docs:

- [../reference/primitives.md](../reference/primitives.md)
- [../reference/architecture.md](../reference/architecture.md)
- [../reference/api-surface.md](../reference/api-surface.md)

## 2026-04-26: Manifest-driven experiment and docking control

Context:

- the project needs configuration-defined features, multiple model families, and
  trustworthy prior-vs-current comparisons without parameter drift

Decision:

- represent controlled runs with explicit experiment manifests and feature parity
  manifests
- resolve spec paths relative to the spec file so checked-in manifests stay
  portable inside the repo
- require feature-row parity checks before claiming legacy-model docking

Consequences:

- feature configuration and code-defined pipelines share one compiled Rust
  execution layer
- experiment reports can now record the exact scanner, feature, model, and
  sampling configuration used for a run
- future comparisons should change one primary variable at a time instead of
  mixing window, feature, model, and sampling differences

Related docs:

- [../../specs/README.md](../../specs/README.md)
- [../reference/api-surface.md](../reference/api-surface.md)

## 2026-04-26: Exact sklearn baseline for legacy RF docking

Context:

- `smartcore` gives a useful Rust RF baseline, but it is not the same model
  implementation as the original sklearn-based `charboundary` RF
- we needed a stricter docking path to tell whether differences came from
  features, sampling, or the RF backend itself

Decision:

- add a Python/sklearn random-forest backend to the experiment layer only
- keep it manifest-driven under the same `BoundaryExperimentSpec` system
- batch prediction by corpus for Python-backed models to reduce subprocess overhead

Consequences:

- we can now compare Rust-native, `smartcore`, and exact sklearn RF runs on the
  same legacy feature/spec configuration
- this backend is appropriate for controlled experiments, not for production
  inference

Related docs:

- [../../specs/README.md](../../specs/README.md)
- [../plan/work-breakdown.md](../plan/work-breakdown.md)

## 2026-04-26: Provisional default candidate is current feature stack + tree

Context:

- the first legacy window sweep showed that wider legacy windows did not improve
  the current reduced benchmark enough to justify becoming the default
- the project still needed a better default tradeoff than either “fast logistic”
  or “legacy tree”

Decision:

- treat the `current_legal_tree` configuration as the current best default
  candidate, pending larger sweeps

Consequences:

- this candidate keeps much of the fast-path throughput of the current feature
  stack while materially improving validation and cross-domain quality over the
  current logistic baseline
- future feature engineering should benchmark against this configuration, not
  only against the older legacy tree manifests

Related docs:

- [../../specs/README.md](../../specs/README.md)

## 2026-04-26: Add probability-capable Rust tree baseline and sweep runner

Context:

- the project needed another classifier family in the Rust path besides logistic
  and hard-label RF
- repeated comparison runs across manifests were becoming manual and error-prone

Decision:

- add a `smartcore` decision-tree backend to the experiment layer
- expose it through the same manifest system as other models
- add a sweep runner that executes multiple experiment manifests and prints one summary table

Consequences:

- we can now compare logistic, decision tree, RF, and exact sklearn RF under the
  same legacy feature manifests
- decision trees provide probability outputs, which makes threshold tuning more
  meaningful than the current hard-label RF adapter

Related docs:

- [../../specs/README.md](../../specs/README.md)
- [../plan/work-breakdown.md](../plan/work-breakdown.md)

## 2026-04-26: Keep local structure counts as the default tree add-on

Context:

- the project added two more reusable feature primitives for generic density-style
  modeling: symmetric byte-class counts and directional left/right byte-class counts
- both primitives improved some in-split ALEA metrics, but the main default choice
  should still be driven by cross-domain behavior on SCOTUS

Decision:

- keep `current_legal_tree_local_structure` as the best default candidate for now
- treat `DirectionalByteClassCounts` as a useful reusable primitive, but not part of
  the default preset yet

Consequences:

- the feature stack stays simple and fast while still outperforming the newer
  class-density variants on the current reduced cross-domain sweep
- future feature engineering should prefer asymmetric, task-aware primitives over
  broad symmetric density blocks when cross-domain quality matters

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Separate exact-legacy and deterministic experiment modes

Context:

- exact prior-vs-current docking needs to match the old `charboundary` scripts as
  closely as possible, including the fact that they do not fix dataset sampling
  or sklearn RF randomness
- normal development still benefits from deterministic manifests with explicit seeds

Decision:

- make `dataset_options.seed` optional in the dataset builder
- make sklearn `random_state` optional in the Python RF experiment backend
- use `null` for exact-legacy manifests and fixed seeds for deterministic sweeps

Consequences:
- the same experiment system now supports strict legacy emulation and reproducible
  engineering runs without mixing the two
- comparisons need to state whether they are “exact-legacy stochastic” or
  “deterministic replay” runs

Related docs:

- [../../specs/README.md](../../specs/README.md)
- [../results.md](../results.md)

## 2026-04-27: Keep the directional-count tree as the default full-corpus preset

Context:

- we added several more configurable backend families on the current strongest
  full-corpus feature stack: `linfa` logistic, `smartcore` logistic, a small
  directional RF, KNN, and Gaussian naive Bayes
- the goal was to see whether any of them could pareto-dominate the current
  directional tree on both cross-domain F1 and single-thread throughput

Decision:

- keep `current_legal_tree_directional_class_counts_full` as the default
  full-corpus single-thread preset
- keep the new alternative backends available behind manifests, but do not
  promote them to the default path

Consequences:

- `smartcore` logistic remains a useful high-speed baseline, but it gives up too
  much SCOTUS F1 to replace the tree
- the small directional RF remains a useful quality-oriented comparison, but it
  is slower and still worse on SCOTUS than the tree
- `linfa` logistic, KNN, and the current `smartcore` Gaussian-NB path are not
  competitive enough to justify default status on this task
- future Pareto attempts should likely focus on a neural backend or a more
  optimized native forest/tree implementation rather than more classical models

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Keep Burn neural models optional, not default

Context:

- the project added a real Burn backend on CPU using the `ndarray` backend with
  system OpenBLAS
- five full-corpus Burn models were run on the same directional feature stack:
  shallow MLP, deep MLP, window CNN, window GRU, and window LSTM

Decision:

- keep the Burn neural models as configurable experiment options only
- do not promote any Burn neural preset to the default path

Consequences:

- the shallow Burn MLP is a practical neural baseline, but it loses too much
  SCOTUS F1 to the tree
- the window CNN is the strongest Burn quality/speed compromise, but still
  trails the tree on both axes
- the GRU and LSTM improve slightly on SCOTUS F1 over the MLPs, but their
  single-thread throughput is too low to compete with the current tree preset
- future neural exploration should likely focus on a different backend such as
  Candle, or on a more optimized Burn backend once CPU matmul/conv performance
  improves

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Keep Candle neural models optional, not default

Context:

- the project added a real Candle backend on CPU using `candle-core` and
  `candle-nn`
- five full-corpus Candle model families were run on the same directional
  feature stack, then followed by a second manifest-driven tuning pass over
  wider MLP/CNN variants and lighter GRU/LSTM variants

Decision:

- keep the Candle neural models as configurable experiment options only
- do not promote any Candle neural preset to the default path

Consequences:

- the shallow Candle MLP is the most practical Candle speed-preserving
  baseline, but it gives up too much SCOTUS F1 to the tree
- the recurrent Candle models are the strongest on SCOTUS quality, but they are
  still slower than the tree in single-thread throughput
- widening the Candle MLP and CNN models did not improve the frontier on this
  feature stack
- future neural exploration should focus on a different backend such as
  `burn-tch`/`tch`, or on a more task-specific sequence architecture, rather
  than more blind width scaling

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Promote full-corpus directional-count tree as the default candidate

Context:

- full-corpus legacy docking is now close enough that the remaining search can
  focus on better Rust-native models rather than reproduction work
- multiple full-corpus tree candidates were run under the same boundary metric
  and single-thread benchmark settings

Decision:

- treat `current_legal_tree_directional_class_counts_full` as the current best
  Rust-native default candidate

Consequences:

- this preset effectively matches the original Python small model on SCOTUS under
  the current evaluator while staying fully inside the Rust training/inference path
- legacy-feature hybrids remain useful experiments, but they are not the default
  path forward

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Promote shorter current-stack window as the full-corpus default

Context:

- the project ran a full-corpus single-thread sweep over the current tree stack
  varying only the encoded byte-window size while holding the directional
  byte-count block, training split, and evaluation contract fixed
- the same pass also added ICU-based Unicode category and category-group
  feature variants aligned with `alea-preprocess`

Decision:

- treat `current_legal_tree_directional_class_counts_window_3_1_full` as the
  new default full-corpus single-thread preset
- keep `current_legal_tree_directional_class_counts_window_3_3_full` as the
  quality-oriented neighboring point on the current Pareto frontier

Consequences:

- the old `5/3` full-corpus directional baseline is no longer the default,
  because `3/1` improves both SCOTUS F1 and single-thread throughput
- current Unicode category/category-group features are now available and
  configuration-driven, but they remain optional because they regress SCOTUS F1
  on the current legal corpora
- future feature engineering should continue to explore current-stack windows
  and derived legal-format features before assuming that more Unicode
  dimensionality is useful for this benchmark

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Keep the short current-stack window after the larger-window and radius follow-up

Context:

- the project ran a second full-corpus single-thread sweep on the same current
  tree stack to test larger encoded windows (`7/7`, `9/7`) and decoupled
  directional byte-count radii (`6/6`, `24/24`, `36/36`) while holding the
  rest of the experiment contract fixed
- the goal was to check whether the earlier `3/1` and `3/3` frontier was only
  a local artifact of the first narrower sweep

Decision:

- keep `current_legal_tree_directional_class_counts_window_3_1_full` as the
  default full-corpus single-thread preset
- keep `current_legal_tree_directional_class_counts_window_3_3_full` as the
  quality-oriented neighboring point on the frontier
- do not promote `7/7`, `9/7`, or any of the radius-only variants to the
  default path

Consequences:

- larger encoded windows remain available as configurable manifests, but they
  now look like validation-only gains rather than cross-domain improvements
- the directional count block still matters, but shrinking or stretching its
  radius away from the current moderate setting does not beat the default
- future feature work should spend effort on better derived legal-format and
  Unicode-aware signals rather than simply making the local context larger

Related docs:

- [../results.md](../results.md)
- [../../specs/README.md](../../specs/README.md)

## 2026-04-27: Treat semantic segmentation and weak labeling as first-class scope

Context:

- sentence boundaries are only one structural task in the target document
  pipeline
- the project needs a path to paragraph, section, dialogue, list-item, and
  entity-aware segmentation without abandoning the byte-first architecture
- OpenAI, PydanticAI, and GLiNER now provide a practical mix of teacher
  labeling, typed orchestration, and local extraction

Decision:

- treat `charstreamer` as a general document segmentation and extraction
  framework, not just a sentence splitter
- prefer candidate-ID, line-index, or inline-tag supervision when generating
  synthetic or weak labels from LLMs; do not use raw LLM byte offsets as the
  primary annotation format
- use OpenAI Structured Outputs for weak labeling and adjudication, PydanticAI
  for orchestration and validation, and GLiNER as a local entity-extraction
  sidecar or weak-label baseline
- do not block segmentation work on a full multi-task NER model; support NER as
  a parallel label layer first

Consequences:

- the next architectural expansion should add hierarchical span corpora and
  weak-label pipelines before jumping to shared neural encoders
- future synthetic-data work should focus on labeling real corpora and covering
  structural edge cases, not on generating large amounts of fully synthetic
  prose
- entity extraction becomes part of the system roadmap, but remains optional
  and layered rather than mandatory in the hot path

Related docs:

- [../reference/semantic-segmentation.md](../reference/semantic-segmentation.md)

## 2026-04-29: Remove Candle from the active neural backend surface

Decision:

- remove `charstreamer-backend-candle` from the workspace and experiment model
  spec enum
- keep `charstreamer-backend-burn` as the supported neural backend
- retain historical Candle benchmark results only as archived reference

Reason:

- Candle did not pareto-dominate the current tree or Burn baselines
- local Candle CNN/LSTM smoke tests were not stable enough for a production
  default build gate
- Burn better matches the project requirement for configurable Rust-native CPU
  training and inference
