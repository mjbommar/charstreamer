# Model Families

## Goal

Define which model families are good fits for `charstreamer`, in what order they
should be implemented, and whether training should be first-party or backend-based.

## Problem shape

`charstreamer` is optimized for:

- local or semi-local decisions
- fixed-width rolling windows
- candidate-first scoring
- CPU-only training and inference
- predictable latency

This biases the project toward compact tabular and small-sequence models.

## Evaluation criteria

Every candidate model family should be judged on:

- training complexity
- inference latency
- batchability
- cache locality
- SIMD friendliness
- memory footprint
- serialization simplicity
- probability quality
- ability to support both classification and regression

## Tier 1: native linear models

### Families

- linear regression
- ridge regression
- binary logistic regression
- multiclass softmax regression

### Why they fit

- excellent match for hand-crafted local features
- dense math is easy to vectorize
- model files are tiny
- inference is just dot products plus small nonlinearities
- training can be written natively with modest code size

### First-party requirement

Yes.

These should be core native implementations.

### Training algorithms to support

- SGD
- minibatch SGD
- L-BFGS for smaller dense problems
- optional AdaGrad/Adam only if experiments justify them

### Inference requirement

- single-example scoring
- batch scoring
- `predict_into` with caller-owned output
- probability output for logistic models

## Tier 2: shallow decision trees

### Families

- binary classification tree
- multiclass classification tree
- regression tree

### Why they fit

- capture nonlinear feature interactions
- interpretable
- useful as teacher/baseline models
- training is still manageable without building a whole GBDT engine

### First-party requirement

Yes, but after linear models.

### Constraints

- start with axis-aligned splits only
- keep inference iterative, not recursive
- optimize for shallow trees first

## Tier 3: random forests and extra trees

### Why they fit

- strong on tabular hand-engineered features
- easy to get good quality quickly
- parallelism across trees is natural

### Costs

- larger model artifacts
- higher memory traffic
- weaker cache behavior than linear models
- slower per-candidate inference

### First-party requirement

Not initially.

Recommended path:

- optional backend first
- native inference later if required
- native training only after linear and tree core is stable

## Tier 4: boosted trees

### Why they are interesting

- can be very strong on structured features
- often better quality/size tradeoff than forests

### Why not early

- histogram building is a major subsystem
- training implementation complexity is much higher
- no need to absorb this cost before validating simpler models

### First-party requirement

No for v1.

Treat as future work or external backend territory.

## Tier 5: tiny neural models

### Candidate families

- tiny MLP
- narrow 1D CNN
- character embedding + pooling

### Why they may matter

- can reduce hand tuning of features
- may beat linear models if useful interactions are difficult to encode manually

### Why they are still secondary

- training complexity jumps significantly
- CPU inference is still heavier than linear models
- model management is harder than classical models

### First-party requirement

No.

Use a backend framework first.

## Tier 6: LSTM / recurrent models

### Why they are least attractive here

- recurrent execution is not a natural fit for the candidate-first local pipeline
- CPU training cost is much higher
- serving is harder to keep tiny and predictable
- SIMD utilization is worse than fixed-window dense kernels

### Position

Only adopt if benchmarks show a clear quality gain that simpler families cannot reach.

## Recommended implementation order

1. native linear/logistic
2. native shallow trees
3. backend-based random forest / extra trees
4. backend-based tiny MLP or CNN
5. anything recurrent only if evidence demands it

## Training ownership policy

### First-party native ownership

Own these from the start:

- linear regression
- logistic regression
- ridge / L2
- softmax regression
- shallow decision tree

### Backend adoption first

Adopt before reimplementing:

- random forest
- extra trees
- neural networks

## Suggested crate boundaries

- `charstreamer-models-native`: first-party linear/tree models
- `charstreamer-backend-burn`: neural backend adapter

Keep the public pipeline generic enough that these are swappable.

## Crate ecosystem notes

As of April 26, 2026:

- `Burn` is the strongest Rust-native training/inference framework for neural models
- `Candle`, `SmartCore`, and `Linfa` were evaluated and removed from the active build; retain Burn as the supported external Rust model backend unless a future requirement documents a Pareto reason to add another backend

Do not let any of those crates define the main data model of `charstreamer`.
